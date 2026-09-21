"""Offline OpenSpatial ARKitScenes subset: exact source filter, explicit scenes.

No downloading, inferred scene IDs, question rewriting or image resizing. A
successful receipt covers only the caller-supplied scene/exclusion mapping, not
unknown aliases or pretraining contamination. Outputs are preparation manifests;
the downstream task-specific chat adapter still needs its own acceptance.
"""
import argparse
import hashlib
import heapq
import io
import json
from pathlib import Path
import random
import sys


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def provenance(shard, row_index):
    return f'{shard}::row::{row_index}'


def load_scene_map(path):
    if path is None:
        return {}
    result = {}
    for row in read_jsonl(path):
        key, scene = row.get('provenance_id'), row.get('scene_id')
        if not isinstance(key, str) or not key or not isinstance(scene, str) or not scene:
            raise ValueError('Scene map requires nonempty provenance_id and scene_id strings')
        if key in result:
            raise ValueError(f'Duplicate scene mapping: {key}')
        result[key] = scene
    return result


def rank_key(key, seed):
    return int.from_bytes(hashlib.blake2b(f'{seed}\0{key}'.encode(), digest_size=16).digest(), 'big')


def select_records(records, count, seed=3407, scene_map=None, excluded_scenes=None):
    """Bounded-memory deterministic bottom-k; no raw image payloads retained."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Count must be positive')
    scene_map, excluded_scenes = scene_map or {}, excluded_scenes or set()
    heap = []
    counts = {'scanned_rows': 0, 'source_rows': 0, 'excluded_scene_rows': 0}
    for row in records:
        counts['scanned_rows'] += 1
        if row.get('data_source') != 'arkitscenes':
            continue
        counts['source_rows'] += 1
        key = row['provenance_id']
        if scene_map.get(key) in excluded_scenes:
            counts['excluded_scene_rows'] += 1
            continue
        # Provenance uniquely identifies physical shard rows even if source IDs
        # repeat. The caller's metadata iterator guarantees this uniqueness.
        entry = (-rank_key(key, seed), key, row)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    selected = [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]
    counts['selected_rows'] = len(selected)
    counts['selected_missing_scene'] = sum(row['provenance_id'] not in scene_map for row in selected)
    return selected, counts


def split_scenes(selected, scene_map, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    scenes = sorted({scene_map[row['provenance_id']] for row in selected})
    if len(scenes) < 2:
        raise ValueError('At least two explicitly mapped source scenes required for split')
    random.Random(seed).shuffle(scenes)
    n_validation = min(len(scenes) - 1, max(1, round(len(scenes) * validation_fraction)))
    validation = set(scenes[:n_validation])
    return {scene: 'validation' if scene in validation else 'train' for scene in scenes}


def shard_inventory(root, pattern):
    root = Path(root).resolve()
    shards = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if not shards:
        raise ValueError('No local Parquet shards found')
    if len(set(shards)) != len(shards):
        raise ValueError('Duplicate resolved shard paths')
    entries = []
    for path in shards:
        relative = path.relative_to(root).as_posix()  # rejects external symlinks
        stat = path.stat()
        entries.append({'shard': relative, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return entries


def metadata_rows(root, inventory):
    import pyarrow.parquet as pq
    for shard in inventory:
        file = pq.ParquetFile(Path(root) / shard['shard'])
        if not {'data_source', 'id'} <= set(file.schema_arrow.names):
            raise ValueError('Parquet must expose data_source and id')
        index = 0
        for batch in file.iter_batches(batch_size=4096, columns=['data_source', 'id']):
            for row in batch.to_pylist():
                yield {'provenance_id': provenance(shard['shard'], index),
                       'shard': shard['shard'], 'row_index': index,
                       'source_id': row['id'], 'data_source': row['data_source']}
                index += 1


def image_payload(image, media_root):
    """Validate decode while preserving original encoded bytes exactly."""
    from PIL import Image
    payload = image.get('bytes')
    if payload is None:
        if media_root is None or not image.get('path'):
            raise ValueError('Path-only image requires explicit media root and real path')
        root = Path(media_root).resolve()
        path = (root / image['path']).resolve()
        path.relative_to(root)  # refuse paths outside the declared data root
        payload = path.read_bytes()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError('Empty or invalid image payload')
    with Image.open(io.BytesIO(payload)) as image_file:
        image_file.load()
        width, height = image_file.size
        extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif',
                     'TIFF': '.tiff', 'BMP': '.bmp'}.get(image_file.format, '.img')
    if min(width, height) < 1:
        raise ValueError('Invalid decoded image dimensions')
    return payload, extension, [width, height]


def export_records(root, output, inventory, selected, scene_map, splits, media_root=None):
    import pyarrow.parquet as pq
    output = Path(output)
    media = output / 'media'
    if media.exists() or any((output / f'{name}.jsonl').exists() for name in ('train', 'validation')):
        raise ValueError('Preserve existing exports: use a new output directory')
    media.mkdir()
    lookup = {row['provenance_id']: row for row in selected}
    exported, images_total = {}, 0
    # Read full payloads only for shards that contain selected records. No large
    # image reservoir and no normalization/resizing/transcoding.
    for shard in inventory:
        wanted = {row['row_index'] for row in selected if row['shard'] == shard['shard']}
        if not wanted:
            continue
        file = pq.ParquetFile(Path(root) / shard['shard'])
        index = 0
        for batch in file.iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                current, index = index, index + 1
                if current not in wanted:
                    continue
                key = provenance(shard['shard'], current)
                chosen = lookup[key]
                if raw.get('data_source') != 'arkitscenes' or raw.get('id') != chosen['source_id']:
                    raise ValueError('Source record changed since selection')
                conversations, source_images = raw.get('conversations'), raw.get('images')
                if not isinstance(conversations, list) or not conversations:
                    raise ValueError(f'Missing raw conversations: {key}')
                if not isinstance(source_images, list) or not source_images:
                    raise ValueError(f'Missing media: {key}')
                name = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
                paths, image_records = [], []
                for image_index, source_image in enumerate(source_images):
                    payload, extension, dimensions = image_payload(source_image, media_root)
                    relative = f'media/{name}-{image_index:03d}{extension}'
                    with (output / relative).open('xb') as handle:
                        handle.write(payload)
                    paths.append(relative)
                    image_records.append({'source_path': source_image.get('path'),
                                          'path': relative, 'size': dimensions, 'bytes': len(payload)})
                    images_total += 1
                scene = scene_map[key]
                exported[key] = {'id': key, 'source_id': raw['id'], 'provenance_id': key,
                                 'source_shard': shard['shard'], 'source_row_index': current,
                                 'scene_id': scene, 'split': splits[scene], 'media': paths,
                                 'images': image_records,
                                 'raw_record': {k: v for k, v in raw.items() if k != 'images'}}
    if set(exported) != set(lookup):
        raise ValueError('Selected rows missing from export')
    counts = {'train': 0, 'validation': 0, 'images': images_total}
    handles = {split: (output / f'{split}.jsonl').open('x', encoding='utf-8')
               for split in ('train', 'validation')}
    try:
        for selected_row in selected:
            row = exported[selected_row['provenance_id']]
            handles[row['split']].write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[row['split']] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-root', type=Path, required=True)
    parser.add_argument('--glob', default='**/*.parquet')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--scene-map', type=Path,
                        help='JSONL: provenance_id (relative-shard::row::index), scene_id')
    parser.add_argument('--excluded-scenes', type=Path,
                        help='Explicit JSON list of benchmark/excluded scene IDs, even when empty')
    parser.add_argument('--validation-fraction', type=float, default=.01)
    parser.add_argument('--media-root', type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / 'receipt.json'
    if receipt_path.exists():
        raise ValueError('Existing preparation receipt: use a new output directory')
    receipt = {'status': 'inventory', 'ready_for_training': False,
               'source_filter': {'data_source': 'arkitscenes'}, 'requested_rows': args.count,
               'seed': args.seed, 'media_verified': False, 'leakage_checked': False,
               'scope': 'Supplied explicit scene IDs/exclusions only; no alias/perceptual audit',
               'downstream_task_adapter_verified': False}
    write_json(receipt_path, receipt)
    try:
        scenes = load_scene_map(args.scene_map)
        exclusions = json.loads(args.excluded_scenes.read_text(encoding='utf-8')) if args.excluded_scenes else []
        if not isinstance(exclusions, list) or any(not isinstance(x, str) or not x for x in exclusions):
            raise ValueError('Excluded scenes must be a JSON list of nonempty scene IDs')
        inventory = shard_inventory(args.shard_root, args.glob)
        write_json(args.output / 'source-inventory.json', inventory)
        selected, counts = select_records(metadata_rows(args.shard_root, inventory), args.count,
                                         args.seed, scenes, set(exclusions))
        with (args.output / 'selection.jsonl').open('x', encoding='utf-8') as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        receipt.update(counts)
        blockers = []
        if len(selected) < args.count:
            blockers.append('insufficient_arkitscenes_records')
        if args.scene_map is None or counts['selected_missing_scene']:
            blockers.append('explicit_scene_mapping_required')
        if args.excluded_scenes is None:
            blockers.append('explicit_benchmark_scene_exclusions_required')
        if blockers:
            receipt.update(status='blocked', blockers=blockers)
            write_json(receipt_path, receipt)
            return 2
        splits = split_scenes(selected, scenes, args.validation_fraction, args.seed)
        write_json(args.output / 'scene-split.json', splits)
        write_json(args.output / 'scene-map-used.json', {row['provenance_id']: scenes[row['provenance_id']] for row in selected})
        write_json(args.output / 'excluded-scenes.json', exclusions)
        receipt.update(status='exporting')
        write_json(receipt_path, receipt)
        exported = export_records(args.shard_root, args.output, inventory, selected, scenes, splits, args.media_root)
        if shard_inventory(args.shard_root, args.glob) != inventory:
            raise ValueError('Input shard inventory changed during export')
        receipt.update(status='prepared', ready_for_training=False, media_verified=True,
                       leakage_checked=True, export=exported,
                       selected_records_ready=True,
                       remaining_gate='task-specific chat adapter and real model input acceptance')
        write_json(receipt_path, receipt)
        return 0
    except Exception as error:
        receipt.update(status='blocked', error=f'{type(error).__name__}: {error}')
        write_json(receipt_path, receipt)
        raise


if __name__ == '__main__':
    sys.exit(main())
