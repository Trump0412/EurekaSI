"""Prepare the six-source mixture with an explicit OpenSpatial-only scene waiver.

CPU-only immutable-output pipeline. All media must be readable. No unknown scene
is replaced with a UUID. OpenSpatial is train-only; validation uses known sources.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('five_source', HERE/'merge-five-source-geometry.py')
five = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five)


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(value, indent=2));temp.replace(path)


def extract_archive(archive, dest, wanted, trusted_reuse_roots=()):
    extracted = 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = member.name.removeprefix('./')
            if name not in wanted:
                continue
            path = dest/name
            if Path(name).is_absolute() or '..' in Path(name).parts or not member.isfile():
                raise ValueError(f'Unsafe archive member type/path: {name!r} type={member.type!r}')
            if not path.parent.resolve().is_relative_to(dest.resolve()):
                raise ValueError(f'Unsafe archive parent: {name!r} -> {path.parent.resolve()}')
            if path.is_symlink():
                resolved = path.resolve()
                if not any(resolved.is_relative_to(Path(r).resolve()) for r in trusted_reuse_roots):
                    raise ValueError(f'Unapproved reuse symlink: {name!r} -> {resolved}')
                if not resolved.is_file() or resolved.stat().st_size != member.size:
                    raise ValueError(f'Reused media size mismatch: {name!r} -> {resolved}; expected={member.size}')
                continue  # Explicitly approved existing input: never open it for writing.
            if not path.resolve().is_relative_to(dest.resolve()):
                raise ValueError(f'Unsafe archive destination: {name!r}')
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.stat().st_size != member.size:
                    raise ValueError('Existing media size mismatch')
                continue
            part = path.with_suffix(path.suffix+'.part')
            if part.is_symlink():
                raise ValueError(f'Unsafe partial-file symlink: {part}')
            with stream.extractfile(member) as source, part.open('wb') as target:
                shutil.copyfileobj(source, target, length=2**20)
            path.with_suffix(path.suffix+'.part').replace(path)
            extracted += 1
    return extracted


def decode_video(rel, media, frames):
    import cv2
    cv2.setNumThreads(1)
    source = media/rel
    target = frames/hashlib.blake2b(rel.encode(), digest_size=12).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    complete = target/'complete.json'
    if complete.exists():
        d = json.loads(complete.read_text())
        if d['bytes'] == source.stat().st_size and all(Path(p).is_file() for p in d['media']):
            return rel, d['media']
        raise ValueError('Stale video cache')
    cap = cv2.VideoCapture(str(source))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or count < 1:
        raise ValueError('Unreadable video: '+rel)
    n = min(32, count)
    indices = sorted({round(i*(count-1)/max(n-1, 1)) for i in range(n)})
    paths = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Failed requested real frame: '+rel+':'+str(index))
            path = target/f'{index:08d}.png'
            if not cv2.imwrite(str(path), frame):
                raise ValueError('Image write failed')
            paths.append(str(path))
    finally:
        cap.release()
    dump(complete, dict(source=rel, bytes=source.stat().st_size, original_frame_indices=indices, media=paths))
    return rel, paths


def openspatial(args, out):
    import pyarrow.parquet as pq
    from PIL import Image
    acquisition = json.loads(Path(args.openspatial_acquisition).read_text())
    selected = []
    for item in acquisition['files']:
        table = pq.read_table(item['path'], columns=['id', 'data_source'])
        for index, row in enumerate(table.to_pylist()):
            if row['data_source'] == 'arkitscenes':
                identity = item['source_path']+'::row::'+str(index)
                selected.append((hashlib.blake2b(f'{args.seed}:{identity}'.encode(), digest_size=16).hexdigest(), identity))
    if len(selected) < 100000:
        raise ValueError('Fewer than required 100K ARKitScenes records')
    keep = {identity for _, identity in sorted(selected)[:100000]}
    rows = 0
    with (out/'openspatial.train.jsonl').open('w') as dest:
        for item in acquisition['files']:
            for index, row in enumerate(pq.read_table(item['path']).to_pylist()):
                identity = item['source_path']+'::row::'+str(index)
                if identity not in keep:
                    continue
                paths = []
                for image_index, image in enumerate(row['images']):
                    target = out/'media/openspatial'/Path(item['source_path']).stem/f'{index:05d}-{image_index}.png'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(io.BytesIO(image['bytes'])) as im:
                        im.load();im.save(target)
                    paths.append(str(target))
                conversations = row['conversations']
                if len(conversations) != 2 or conversations[0]['from'] != 'human' or conversations[1]['from'] != 'gpt':
                    raise ValueError('Unimplemented multi-turn OpenSpatial schema; no silent truncation')
                result = dict(id='openspatial::'+identity, source_id=row['id'], source_index=index,
                              source_file=item['source_path'], dataset='openspatial', split='train',
                              scene_id=None, source_scene=None, scene_overlap='unknown',
                              question=conversations[0]['value'].replace('<image>', '').strip(),
                              answer=conversations[1]['value'], media=paths,
                              leakage_waiver='explicit_user_authorization_openspatial_only')
                dest.write(json.dumps(result, ensure_ascii=False)+'\n');rows += 1
    if rows != 100000:
        raise ValueError('Incorrect OpenSpatial export count')
    return rows


def run(args):
    root, out, component = Path(args.source_root), Path(args.output), Path(args.component_root)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='extracting_vsi_media', selection_name='shared-six-source-geothinker-vlm3r-mindcube-v4-openspatial-waiver',
                 ready_for_training=False, leakage_checked=False, global_scene_split_verified=False,
                 known_sources_leakage_checked=False,
                 contamination_waiver=dict(authorized=True, scope='openspatial_only',
                                     reason='user authorizes training despite missing source scene mapping',
                                     scene_overlap='unknown', validation_policy='all OpenSpatial train-only'),
                 openspatial_validation_policy='all100000_train_only', media_verified=False, started=time.time())
    def update(**kw):
        state.update(kw, updated=time.time());dump(out/'receipt.json', state)
    update()
    wanted, videos, images = set(), set(), set()
    for row in five.jsonlines(root/'datasets/vsi590k/vsi_590k.jsonl'):
        if row.get('video'):videos.add(row['video'])
        else:images.add(row['image'])
    wanted = videos | images
    media = out/'media/vsi590k';media.mkdir(parents=True)
    if args.reuse_vsi_media_root:
        reuse = Path(args.reuse_vsi_media_root)
        for name in wanted:
            old = reuse/name
            if old.is_file():
                target=media/name;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(old.resolve())
    archives = sorted({Path(name).parts[0]+'.tar.gz' for name in wanted})
    for name in archives:
        if not (root/'datasets/vsi590k'/name).is_file():raise ValueError('Missing source archive: '+name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(extract_archive, root/'datasets/vsi590k'/name, media,
                            {rel for rel in wanted if rel.startswith(name.removesuffix('.tar.gz')+'/')},
                            args.trusted_reuse_root):name for name in archives}
        done = []
        for future in as_completed(jobs):
            done.append(dict(archive=jobs[future], extracted=future.result()))
            update(extracted_archives=done)
    missing = [name for name in wanted if not (media/name).is_file()]
    if missing:
        dump(out/'missing-media.json', missing);raise ValueError('Archive media missing: '+str(len(missing)))
    update(status='decoding_vsi_videos', videos_total=len(videos), images_total=len(images))
    frame_index = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(decode_video, name, media, out/'frames/vsi590k') for name in sorted(videos)]
        for future in as_completed(futures):
            name, paths = future.result();frame_index[name] = paths
            if len(frame_index) % 50 == 0:update(videos_decoded=len(frame_index))
    dump(out/'frames-index.json', frame_index)
    from PIL import Image
    update(status='verifying_vsi_images')
    for index, name in enumerate(sorted(images)):
        with Image.open(media/name) as im:im.load()
        if index % 1000 == 0:update(images_verified=index+1)
    update(status='exporting_openspatial')
    rows = openspatial(args, out);update(openspatial_exported=rows)
    update(status='merging_five_checked_sources')
    merge_args = argparse.Namespace(source_root=str(root), component_root=str(component), output=str(out/'five-source'),
                                    vsi_media_root=str(media), vsi_frames_index=str(out/'frames-index.json'),
                                    seed=args.seed, validation_fraction=.01, wait_hours=48)
    five.run(merge_args)
    five_receipt = json.loads((out/'five-source/receipt.json').read_text())
    if five_receipt['missing_media']:raise ValueError('Five-source media not complete')
    counts = Counter(five_receipt['counts']);counts['openspatial'] = rows
    for split in ['train', 'validation']:
        with (out/(split+'.jsonl')).open('wb') as target:
            with (out/'five-source'/(split+'.candidate.jsonl')).open('rb') as source:shutil.copyfileobj(source, target)
            if split == 'train':
                with (out/'openspatial.train.jsonl').open('rb') as source:shutil.copyfileobj(source, target)
    update(status='prepared_with_user_waiver', ready_for_training=True, media_verified=True,
           known_sources_leakage_checked=True, six_sources_verified=set(counts)=={'spar','hound','vsi590k','vlm3r','mindcube','openspatial'},
           train_manifest=str(out/'train.jsonl'), validation_manifest=str(out/'validation.jsonl'),
           counts=dict(counts), five_source_receipt=str(out/'five-source/receipt.json'),
           validation_rows=five_receipt['splits']['validation'], train_rows=five_receipt['splits']['train']+rows,
           limitation='OpenSpatial benchmark contamination unknown by explicit waiver; not clean generalization.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True);p.add_argument('--component-root', required=True)
    p.add_argument('--openspatial-acquisition', required=True);p.add_argument('--output', required=True)
    p.add_argument('--authorize-openspatial-scene-waiver', action='store_true')
    p.add_argument('--workers', type=int, default=4);p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--reuse-vsi-media-root', help='Read-only completed files from a stopped earlier attempt')
    p.add_argument('--trusted-reuse-root', action='append', default=[], help='Explicit allowed final resolved roots of read-only reused files')
    a = p.parse_args()
    if not a.authorize_openspatial_scene_waiver:p.error('Explicit user authorization required')
    try:run(a)
    except Exception as exc:
        path = Path(a.output)/'receipt.json'
        if path.exists() and not isinstance(exc, FileExistsError):
            state=json.loads(path.read_text());state.update(status='failed',ready_for_training=False,error=str(exc));dump(path,state)
        raise
