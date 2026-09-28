"""Node replicas must not silently reuse bad IDs or mutate source manifests."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Linux supervisor uses flock')


def module():
    pytest.importorskip('fcntl')
    spec = importlib.util.spec_from_file_location('node_sft', Path(__file__).parents[1] / 'scripts/run-node-sft.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(root, *, duplicate=False, overlap=False):
    (root / 'manifests').mkdir(parents=True)
    (root / 'receipts').mkdir()
    row = {'id': 'spar::row::0', 'source_index': 0, 'dataset': 'spar', 'scene_id': 'test' if overlap else 'train'}
    rows = [row, row] if duplicate else [row]
    (root / 'manifests/sft.train.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (root / 'manifests/revsi32.test.jsonl').write_text(json.dumps({'scene_id': 'test'}) + '\n')
    (root / 'manifests/train-exclusions.jsonl').write_text('')
    (root / 'receipts/train-prepared.json').write_text(json.dumps({
        'status': 'complete', 'scene_overlap': 0, 'rows': len(rows), 'counts': {'spar': len(rows)}}))


def test_snapshot_is_independent(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    fixture(source)
    original = (source / 'manifests/sft.train.jsonl').read_bytes()
    assert module().snapshot(source, target) == 1
    assert (target / 'manifests/sft.train.jsonl').read_bytes() == original
    assert not (target / 'manifests/sft.train.jsonl').is_symlink()
    assert (source / 'manifests/sft.train.jsonl').read_bytes() == original


@pytest.mark.parametrize('options', [{'duplicate': True}, {'overlap': True}])
def test_reject_bad_manifest(tmp_path, options):
    source, target = tmp_path / 'source', tmp_path / 'target'
    fixture(source, **options)
    with pytest.raises(ValueError, match='Duplicate ID or heldout'):
        module().snapshot(source, target)


def test_seed_override_keeps_original_argument_type(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from spatial_intelligence import study
    class OriginalArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    observed = {}
    fake = SimpleNamespace(TrainingArguments=OriginalArguments, set_seed=lambda value: observed.update(seed=value))
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    def training(*args):
        original = fake.TrainingArguments(seed=3407, data_seed=3407)
        assert type(original) is OriginalArguments
        assert original.seed == original.data_seed == 3408
        observed['training'] = args
    monkeypatch.setattr(study, 'train', training)
    module().worker(SimpleNamespace(root=str(tmp_path), seed=3408, model='model',
        name='run', micro=4, ga=2, max_steps=-1))
    assert observed['seed'] == 3408
    assert observed['training'][3:] == (4, 2, -1)
