import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'prepare_openspatial_arkit', Path(__file__).resolve().parents[1] / 'scripts/prepare-openspatial-arkit.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows(n=12):
    return [{'provenance_id': MODULE.provenance('data/001.parquet', i),
             'source_id': 'repeated-id', 'row_index': i, 'shard': 'data/001.parquet',
             'data_source': 'arkitscenes'} for i in range(n)]


def test_exact_source_filter_and_repeated_source_ids_preserved():
    candidates = rows(4)
    candidates += [dict(candidates[0], data_source='ARKitScenes'),
                   dict(candidates[0], data_source='arkitscenes_extra')]
    chosen, counts = MODULE.select_records(candidates, 10)
    assert len(chosen) == 4
    assert len({row['provenance_id'] for row in chosen}) == 4
    assert all(row['source_id'] == 'repeated-id' for row in chosen)
    assert counts['source_rows'] == 4 and counts['scanned_rows'] == 6
    assert counts['selected_missing_scene'] == 4


def test_selection_deterministic_independent_of_iteration_order():
    candidates = rows(100)
    first, _ = MODULE.select_records(candidates, 7)
    reverse, _ = MODULE.select_records(reversed(candidates), 7)
    assert first == reverse
    other_seed, _ = MODULE.select_records(candidates, 7, seed=3408)
    assert first != other_seed


def test_explicit_exclusions_applied_before_count():
    candidates = rows(8)
    mapping = {r['provenance_id']: 'exclude' if r['row_index'] < 4 else 'keep' for r in candidates}
    chosen, counts = MODULE.select_records(candidates, 6, scene_map=mapping, excluded_scenes={'exclude'})
    assert len(chosen) == 4
    assert counts['excluded_scene_rows'] == 4
    assert counts['selected_missing_scene'] == 0


def test_scene_split_keeps_entire_source_scene_together():
    candidates = rows(12)
    mapping = {r['provenance_id']: f'scene-{r["row_index"] // 3}' for r in candidates}
    split = MODULE.split_scenes(candidates, mapping, .25, 3407)
    assert list(split.values()).count('validation') == 1
    assert split == MODULE.split_scenes(list(reversed(candidates)), mapping, .25, 3407)


def test_missing_scenes_never_guessed_from_uuid():
    with pytest.raises(KeyError):
        MODULE.split_scenes(rows(2), {}, .01, 3407)
    with pytest.raises(ValueError, match='two explicitly mapped'):
        MODULE.split_scenes(rows(2), {r['provenance_id']: 'same' for r in rows(2)}, .01, 3407)


def test_duplicate_mapping_rejected(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    row = {'provenance_id': 'shard::row::0', 'scene_id': 'real-scene'}
    path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        MODULE.load_scene_map(path)


def test_inventory_rejects_absent_shards(tmp_path):
    with pytest.raises(ValueError, match='No local'):
        MODULE.shard_inventory(tmp_path, '**/*.parquet')


def tiny_parquet(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    image_module = pytest.importorskip('PIL.Image')
    buffer = io.BytesIO()
    image_module.new('RGB', (4, 5), 'red').save(buffer, format='PNG')
    payload = buffer.getvalue()
    source = tmp_path / 'input'
    source.mkdir()
    records = [{'id': 'same-id', 'data_source': 'arkitscenes',
                'conversations': [{'from': 'human', 'value': '<image> unchanged question'},
                                  {'from': 'gpt', 'value': 'unchanged answer'}],
                'images': [{'bytes': payload, 'path': None}], 'meta_info': '{"unchanged":true}'}
               for _ in range(3)]
    pq.write_table(pa.Table.from_pylist(records), source / '001.parquet')
    return source, records, payload


def test_insufficient_and_missing_mapping_block_not_ready(tmp_path):
    source, _, _ = tiny_parquet(tmp_path)
    output = tmp_path / 'output'
    assert MODULE.main(['--shard-root', str(source), '--output', str(output), '--count', '4']) == 2
    receipt = json.loads((output / 'receipt.json').read_text())
    assert not receipt['ready_for_training']
    assert 'insufficient_arkitscenes_records' in receipt['blockers']
    assert 'explicit_scene_mapping_required' in receipt['blockers']
    assert not (output / 'media').exists()


def test_real_parquet_export_preserves_raw_text_ids_bytes_and_scene_split(tmp_path):
    source, raw, payload = tiny_parquet(tmp_path)
    mapping = tmp_path / 'scenes.jsonl'
    mapping.write_text('\n'.join(json.dumps({'provenance_id': f'001.parquet::row::{i}',
                                              'scene_id': f'scene-{i}'}) for i in range(3)))
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    output = tmp_path / 'output'
    assert MODULE.main(['--shard-root', str(source), '--output', str(output), '--count', '3',
                        '--scene-map', str(mapping), '--excluded-scenes', str(excluded)]) == 0
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['selected_records_ready'] and receipt['media_verified']
    assert not receipt['ready_for_training']  # no downstream adapter/model acceptance implied
    train = list(MODULE.read_jsonl(output / 'train.jsonl'))
    validation = list(MODULE.read_jsonl(output / 'validation.jsonl'))
    assert len(train) + len(validation) == 3
    assert not ({row['scene_id'] for row in train} & {row['scene_id'] for row in validation})
    for row in train + validation:
        assert row['source_id'] == 'same-id'
        assert row['raw_record']['id'] == 'same-id'
        assert row['raw_record']['conversations'] == raw[0]['conversations']
        assert (output / row['media'][0]).read_bytes() == payload


def test_path_only_media_cannot_escape_explicit_root(tmp_path):
    pytest.importorskip('PIL.Image')
    root = tmp_path / 'media'
    root.mkdir()
    with pytest.raises(ValueError):
        MODULE.image_payload({'path': '../secret.png', 'bytes': None}, root)
