import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/prepare-spatial-rl-probe.py'
spec = importlib.util.spec_from_file_location('rl_selection', SCRIPT)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def test_task_balanced_sampling_is_fixed_and_refuses_shortage():
    rows = [{'id': str(i), 'task': 'a' if i < 8 else 'b'} for i in range(16)]
    first = selection.balanced_take(rows, 10, random.Random(3407))
    second = selection.balanced_take(rows, 10, random.Random(3407))
    assert first == second
    assert len({r['id'] for r in first}) == 10
    assert sum(r['task'] == 'a' for r in first) == 5
    with pytest.raises(ValueError, match='Only 16'):
        selection.balanced_take(rows, 17, random.Random(3407))


def test_cli_scene_split_keeps_labels_markers_and_source_ids(tmp_path):
    root = tmp_path/'source'
    (root/'manifests').mkdir(parents=True)
    (root/'datasets/vg-llm/train').mkdir(parents=True)
    media = root/'marker.png'
    media.touch()  # Selection validates paths, deliberately does not decode pixels.
    rows, raw = [], []
    for scene in range(20):
        for sample in range(4):
            i = len(rows)
            task = 'obj_spatial_relation_oo' if sample % 2 else 'obj_spatial_relation_oo_mv'
            frames = 1 if sample % 2 else 3
            raw.append({'id': f'original-{i}', 'spar_info': json.dumps({'type': task})})
            rows.append({'id': f'spar::row::{i}', 'source_id': f'original-{i}',
                         'source_index': i, 'dataset': 'spar', 'scene_id': f'scene-{scene}',
                         'split': 'train', 'question': 'Is red right of blue?',
                         'answer': 'Yes' if sample < 2 else 'No',
                         'media': [str(media)]*frames, 'geometry_media': [str(media)]*frames})
    (root/'datasets/vg-llm/train/spar_234k.json').write_text(json.dumps(raw))
    (root/'manifests/sft.train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'manifests/revsi32.test.jsonl').write_text(json.dumps({'scene_id': 'scene-0'})+'\n')
    out = tmp_path/'selected'
    command = [sys.executable, str(SCRIPT), '--source-root', str(root), '--output', str(out),
               '--train-count', '24', '--val-count', '8']
    subprocess.run(command, check=True, capture_output=True, text=True)
    train = [json.loads(l) for l in (out/'train.jsonl').read_text().splitlines()]
    val = [json.loads(l) for l in (out/'val.jsonl').read_text().splitlines()]
    assert len(train) == 24 and len(val) == 8
    assert not ({r['scene_id'] for r in train} & {r['scene_id'] for r in val})
    assert all(r['scene_id'] != 'scene-0' for r in train+val)
    for row in train+val:
        original = rows[row['source_index']]
        for key in ('id', 'source_id', 'answer', 'question', 'media', 'geometry_media'):
            assert row[key] == original[key]
        assert row['metadata']['source_index'] == row['source_index']
        assert row['metadata']['spar_info_type'] == row['task']
    assert {r['split'] for r in train} == {'train'}
    assert {r['split'] for r in val} == {'val'}
    assert subprocess.run(command, capture_output=True).returncode != 0
