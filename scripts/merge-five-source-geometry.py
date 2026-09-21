"""Automatically assemble five-source candidates; never claim six-source readiness.

VSI video inputs need a decoded-frames index (JSON: video -> ordered real frame paths).
Missing media remain in the candidate ledger, not silently excluded or train-ready.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time


def scene_group(scene, source):
    scene = str(scene)
    match = re.fullmatch(r'(scene\d{4})_\d\d', scene)
    if match:
        return 'scannet:' + match[1]
    if source in {'arkitscenes', 'arkit'}:
        return 'arkitscenes:' + scene
    return source + ':' + scene


def split_for(group, seed, fraction):
    value = int(hashlib.blake2b(f'{seed}:{group}'.encode(), digest_size=8).hexdigest(), 16) / 2**64
    return 'validation' if value < fraction else 'train'


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_scene(row):
    scene = str(row['scene_id'])
    if re.fullmatch(r'scene\d{4}_\d\d', scene):
        return scene_group(scene, 'scannet')
    if row['dataset'] == 'mindcube':
        return scene_group(scene, 'mindcube')
    paths = ' '.join(row.get('media', []))
    for source in ('arkitscenes', 'scannetpp', 'structured3d', 'hypersim'):
        if '/'+source+'/' in paths:
            return scene_group(scene, source)
    return scene_group(scene, row['dataset'])


def run(args):
    root, component, out = Path(args.source_root), Path(args.component_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    def receipt(value):
        temp = out/'receipt.tmp'
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(out/'receipt.json')
    state = dict(status='waiting_component', ready_for_training=False,
                 six_sources_verified=False, scope='five-source candidates, no OpenSpatial', counts={})
    receipt(state)
    deadline = time.time() + args.wait_hours * 3600
    while True:
        status = json.loads((component/'data-readiness.json').read_text())
        if status.get('component_ready'):
            break
        if status['status'] in {'failed', 'media_failed'}:
            raise ValueError('Component media/semantics failed; inspect component receipt')
        if time.time() >= deadline:
            raise TimeoutError('Component preparation deadline')
        time.sleep(30)
    frames_path = Path(args.vsi_frames_index)
    frames = json.loads(frames_path.read_text()) if frames_path.exists() else {}
    excluded_groups = set()
    for name in ['revsi32.test.jsonl', 'vsibench32.test.jsonl']:
        for row in jsonlines(root/'manifests'/name):
            scene = row['scene_id']
            if re.fullmatch(r'scene\d{4}_\d\d', scene):
                excluded_groups.add(scene_group(scene, 'scannet'))
            elif str(scene).isdigit():
                excluded_groups.add(scene_group(scene, 'arkitscenes'))
            else:
                excluded_groups.add(source_scene(row))
    def candidates():
        for filename in [root/'manifests/sft.train.jsonl', component/'vlm3r.jsonl', component/'mindcube.train.jsonl']:
            yield from jsonlines(filename)
        for index, row in enumerate(jsonlines(root/'datasets/vsi590k/vsi_590k.jsonl')):
            rel = row.get('video') or row.get('image')
            parts = Path(rel).parts
            source = parts[0]
            scene = Path(parts[1]).stem
            media = frames.get(rel, []) if row.get('video') else [str(Path(args.vsi_media_root)/rel)]
            yield dict(id=f'vsi590k::row::{index}', source_id=row.get('id'), source_index=index,
                       dataset='vsi590k', scene_id=scene, source_scene=scene_group(scene, source),
                       question=row['conversations'][0]['value'].replace('<image>', '').strip(),
                       answer=row['conversations'][1]['value'], media=media,
                       original_media=rel, media_kind='video' if row.get('video') else 'image',
                       question_type=row.get('question_type'))
    state['status'] = 'merging_candidates'; receipt(state)
    counts, splits, groups = Counter(), Counter(), {}
    seen_ids, seen_qa, missing = set(), set(), set()
    total, duplicates, excluded = 0, 0, 0
    with (out/'train.candidate.jsonl').open('w') as train, (out/'validation.candidate.jsonl').open('w') as val, (out/'excluded-ledger.jsonl').open('w') as ledger:
        for row in candidates():
            if row['id'] in seen_ids:
                raise ValueError('Duplicate canonical sample ID: '+row['id'])
            seen_ids.add(row['id'])
            group = row.get('source_scene') or source_scene(row)
            if group in excluded_groups:
                excluded += 1; ledger.write(json.dumps(dict(id=row['id'], reason='benchmark_source_scene', group=group))+'\n');continue
            key = (group, tuple(row.get('media', [])), row.get('original_media'),
                   re.sub(r'\s+', ' ', row['question']).strip(), str(row['answer']))
            if key in seen_qa:
                duplicates += 1;ledger.write(json.dumps(dict(id=row['id'], reason='same_scene_question_answer'))+'\n');continue
            seen_qa.add(key)
            split = split_for(group, args.seed, args.validation_fraction)
            groups[group] = split
            row.update(split=split, source_scene=group)
            if not row.get('media'):
                missing.add(row.get('original_media', row['id']))
            for path in row.get('media', []):
                if not Path(path).is_file():
                    missing.add(path)
            (train if split == 'train' else val).write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[row['dataset']] += 1; splits[split] += 1; total += 1
            if total % 50000 == 0:
                state['counts'] = dict(counts);receipt(state)
    (out/'missing-media.json').write_text(json.dumps(sorted(missing), indent=2))
    (out/'scene-split.json').write_text(json.dumps(groups, indent=2))
    state.update(status='candidates_assembled_not_training_ready', counts=dict(counts), splits=dict(splits),
                 same_scene_qa_duplicates=duplicates, excluded_benchmark_rows=excluded,
                 missing_media=len(missing), candidate_train=str(out/'train.candidate.jsonl'),
                 candidate_validation=str(out/'validation.candidate.jsonl'),
                 blockers=['OpenSpatial verified scene mapping and sixth-source merge missing',
                           'VSI complete decoding/readability gate required'],
                 limitation='Canonical known source-scene IDs; no unknown-alias/perceptual dedup. ID source provenance preserved.')
    receipt(state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--component-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--vsi-media-root', required=True)
    p.add_argument('--vsi-frames-index', required=True)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--validation-fraction', type=float, default=.01)
    p.add_argument('--wait-hours', type=float, default=24)
    a = p.parse_args()
    try:
        run(a)
    except Exception as exc:
        path = Path(a.output)/'receipt.json'
        if path.exists() and not isinstance(exc, FileExistsError):
            state = json.loads(path.read_text()); state.update(status='failed', error=str(exc));path.write_text(json.dumps(state, indent=2))
        raise
