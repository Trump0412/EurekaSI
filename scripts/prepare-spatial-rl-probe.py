#!/usr/bin/env python3
"""Freeze a small scene-disjoint spatial RL exploration; never edits shared inputs.

This is not an official benchmark. Validation scenes are unseen only when the
initializer has not trained on the source SFT manifest (use the original VLM).
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)')
NUMERIC_TYPES = {
    f'{kind}_prediction_{relation}{suffix}'
    for kind in ('distance', 'depth')
    for relation in ('oc', 'oo') for suffix in ('', '_mv')
}
BOOLEAN_TYPES = {'obj_spatial_relation_oo', 'obj_spatial_relation_oo_mv'}


def balanced_take(rows, count, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[row['task']].append(row)
    for bucket in groups.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < count:
        progressed = False
        for task in sorted(groups):
            if groups[task] and len(selected) < count:
                selected.append(groups[task].pop())
                progressed = True
        if not progressed:
            raise ValueError(f'Only {len(selected)} eligible rows for requested {count}')
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--out', '--output', dest='out', required=True, type=Path)
    parser.add_argument('--train-count', type=int, default=512)
    parser.add_argument('--val-count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--answer-kind', choices=('boolean', 'numeric', 'both'), default='boolean')
    args = parser.parse_args()
    if args.train_count < 1 or args.val_count < 1:
        raise ValueError('Positive train and val counts required')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    root = args.source_root.resolve()
    annotations = json.loads((root/'datasets/vg-llm/train/spar_234k.json').read_text())
    heldout = {json.loads(line)['scene_id'] for line in
               (root/'manifests/revsi32.test.jsonl').open()}
    heldout |= {scene.split('/')[-1] for scene in heldout}
    candidates, rejects, seen = [], Counter(), set()
    for line in (root/'manifests/sft.train.jsonl').open():
        row = json.loads(line)
        if row['id'] in seen:
            raise ValueError('Duplicate input ID')
        seen.add(row['id'])
        if row['dataset'] != 'spar':
            rejects['not_spar'] += 1
            continue
        if row['scene_id'] in heldout:
            rejects['revsi_scene'] += 1
            continue
        if not 1 <= len(row['media']) <= 3:
            rejects['frame_budget'] += 1
            continue
        raw = annotations[row['source_index']]
        if str(raw['id']) != row['source_id']:
            raise ValueError('source_index/source_id mismatch')
        info = raw.get('spar_info') or {}
        if isinstance(info, str):
            info = json.loads(info)
        task = info.get('type')
        answer = row['answer'].strip()
        numeric = task in NUMERIC_TYPES and NUMBER.fullmatch(answer) is not None
        boolean = task in BOOLEAN_TYPES and answer.lower() in {'yes', 'no'}
        allowed = (boolean and args.answer_kind in {'boolean', 'both'}) or (numeric and args.answer_kind in {'numeric', 'both'})
        if not allowed:
            rejects['unsupported_task_or_answer'] += 1
            continue
        row = dict(row, task=task, metric='numeric' if numeric else 'exact')
        row['metadata'] = dict(row.get('metadata') or {},
            source_id=row['source_id'], source_index=row['source_index'],
            spar_info_type=task,
            selection_reason='SPAR marked 1-3 frame objective scalar/boolean QA',
            answer_kind='meter_scalar' if numeric else 'yes_no',
            reward_contract='strict full response, no freeform number extraction',
            unit='meter' if numeric else None)
        candidates.append(row)
    rng = random.Random(args.seed)
    scenes = sorted({r['scene_id'] for r in candidates})
    rng.shuffle(scenes)
    val_scenes = set(scenes[:max(1, round(len(scenes)*0.15))])
    train = balanced_take([r for r in candidates if r['scene_id'] not in val_scenes], args.train_count, rng)
    val = balanced_take([r for r in candidates if r['scene_id'] in val_scenes], args.val_count, rng)
    assert not ({r['scene_id'] for r in train} & {r['scene_id'] for r in val})
    for rows in (train, val):
        for row in rows:
            if len(row['media']) != len(row['geometry_media']):
                raise ValueError('Frame alignment mismatch')
            if any(not Path(p).is_file() for p in row['media']):
                raise FileNotFoundError(row['id'])
    report = {'purpose': 'exploratory spatial RL, not official benchmark',
        'seed': args.seed, 'source_root': str(root), 'candidates': len(candidates), 'answer_kind': args.answer_kind,
        'candidate_types': dict(Counter(r['task'] for r in candidates)),
        'rejected_reasons': dict(rejects), 'split_method': 'seeded scene partition 15% val, task-balanced fixed sample caps',
        'scene_overlap': 0, 'revsi_scene_overlap': 0,
        'initializer_requirement': 'Original VLM, not source-SFT checkpoint; source-SFT has exposed val data',
        'limitations': 'Foundation pretraining exposure and image near-duplicates not excluded; labels may be noisy',
        'reward_requirement': 'Lock explicit full-response numeric tolerance or yes/no exact before running; no test tuning'}
    args.out.mkdir(parents=True)
    for split, rows in [('train', train), ('val', val)]:
        for row in rows:
            row['split'] = split
        (args.out/f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        report[split] = {'rows': len(rows), 'ids': [r['id'] for r in rows],
            'scenes': sorted({r['scene_id'] for r in rows}),
            'types': dict(Counter(r['task'] for r in rows))}
    (args.out/'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('candidates', 'scene_overlap', 'revsi_scene_overlap')}))
    print(f'Prepared {len(train)} train + {len(val)} val rows at {args.out}')


if __name__ == '__main__':
    main()
