"""Six-source preparation gates: actual inventories first, never infer scene IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def scan_openspatial(acquisition, output):
    import pyarrow.parquet as pq
    state = json.loads(Path(acquisition).read_text())
    counts = Counter()
    keys, meta_keys = set(), set()
    seen = set()
    mappings = []
    for item in state['files']:
        path = Path(item['path'])
        table = pq.read_table(path, columns=['id', 'data_source', 'meta_info', 'images'])
        for index, row in enumerate(table.to_pylist()):
            if row['data_source'] != 'arkitscenes':
                continue
            counts['candidate_rows'] += 1
            counts['duplicate_source_ids'] += row['id'] in seen
            seen.add(row['id'])
            meta = json.loads(row['meta_info']) if row['meta_info'] else []
            entries = meta if isinstance(meta, list) else [meta]
            for entry in entries:
                if isinstance(entry, dict):
                    meta_keys.update(entry)
            paths = [x.get('path') for x in row['images']]
            counts['rows_with_media_path'] += any(paths)
            # Only explicit source scene fields are usable; UUIDs and image hashes are not scenes.
            scenes = {str(e[k]) for e in entries if isinstance(e, dict)
                      for k in ('scene_id', 'scene_name', 'video_id') if e.get(k)}
            if len(scenes) == 1:
                mappings.append(dict(provenance_id=item['source_path']+'::row::'+str(index),
                                     source_id=row['id'], scene_id=next(iter(scenes))))
            else:
                counts['missing_explicit_scene'] += 1
            counts['images'] += len(paths)
        keys.update(pq.read_schema(path).names)
    result = dict(counts=dict(counts), source_columns=sorted(keys), metadata_keys=sorted(meta_keys),
                  scene_mapping_rows=len(mappings), media_path_parser_not_invented=True)
    write(output / 'openspatial-schema-audit.json', result)
    with (output / 'openspatial-scene-map.jsonl').open('w') as f:
        for row in mappings:
            f.write(json.dumps(row)+'\n')
    return result


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.source_root)
    receipt = dict(status='auditing_source_inventory', selection_name='shared-six-source-geothinker-vlm3r-mindcube-v3',
                   ready_for_training=False, train_manifest=None, media_verified=False, leakage_checked=False,
                   counts={}, blockers=[], updated=time.time())
    path = out / 'data-readiness.json'
    write(path, receipt)
    try:
        counts = Counter()
        with (root / 'manifests/sft.train.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                counts[row['dataset']] += 1
        with (root / 'datasets/vsi590k/vsi_590k.jsonl').open() as f:
            vsi_rows = 0
            videos = set()
            images = set()
            for line in f:
                row = json.loads(line)
                vsi_rows += 1
                if row.get('video'):
                    videos.add(row['video'])
                elif row.get('image'):
                    images.add(row['image'])
                else:
                    raise ValueError('VSI record lacks video/image')
        receipt['counts'].update(spar_hound=dict(counts), vsi590k_raw=vsi_rows, vsi_unique_videos=len(videos), vsi_unique_images=len(images))
        write(out / 'vsi-video-inventory.json', sorted(videos))
        write(out / 'vsi-image-inventory.json', sorted(images))
        audit = scan_openspatial(args.openspatial_acquisition, out)
        receipt['counts']['openspatial'] = audit['counts']
        if audit['counts']['missing_explicit_scene']:
            receipt['blockers'].append('OpenSpatial lacks explicit source scene mapping; no UUID/image-hash substitute permitted')
        receipt['component_receipt'] = args.component_receipt
        component = json.loads(Path(args.component_receipt).read_text())
        receipt['component_status'] = component['status']
        if not component.get('component_ready'):
            receipt['blockers'].append('VLM3R/MindCube component not yet complete; rerun into a new output after ready')
        receipt['blockers'].append('VSI590K videos require complete real-frame decoding and coverage receipt before six-source merge')
        receipt['status'] = 'blocked_explicit_data_gates'
        receipt['next_action'] = 'Provide verified OpenSpatial provenance-to-scene map; prepare VSI590K video frames; then merge all six sources with global source-scene split, not per-source splits'
        write(path, receipt)
    except Exception as exc:
        receipt.update(status='failed', error=type(exc).__name__+': '+str(exc))
        write(path, receipt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--component-receipt', required=True)
    p.add_argument('--openspatial-acquisition', required=True)
    run(p.parse_args())
