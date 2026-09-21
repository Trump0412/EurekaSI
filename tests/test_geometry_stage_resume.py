"""Immutable-root continuation contract checks (not a GPU resume acceptance)."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('matrix_stage_resume',Path(__file__).resolve().parents[1]/'scripts/train-geometry-stage.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def fixture(tmp_path):
    old=tmp_path/'old'; old.mkdir()
    checkpoint=old/'checkpoint-20'; checkpoint.mkdir()
    manifest=tmp_path/'manifest.jsonl'; manifest.write_text('{"id": "one"}\n')
    contract=dict(diagnostic=False,stage='sft',world=8,micro=1,ga=48,global_batch=384,
                  manifest=str(manifest),deepspeed=None,epochs=1,rows=297899,seed=3407)
    (old/'contract.json').write_text(json.dumps(contract))
    for name in ('trainer_state.json','optimizer.pt','rng_state_0.pth'):
        (checkpoint/name).write_text('{}')
    return checkpoint,contract


def test_resume_allows_batch_runtime_not_budget_change(tmp_path):
    checkpoint,contract=fixture(tmp_path)
    contract.update(micro=2,ga=24,save_steps=20,throughput_policy={'encoder_batch_size':2})
    assert worker.validate_resume_checkpoint(checkpoint,contract)==str(checkpoint)
    contract['epochs']=2
    with pytest.raises(ValueError,match='budget contract'):
        worker.validate_resume_checkpoint(checkpoint,contract)


def test_resume_rejects_reordered_data_and_missing_optimizer(tmp_path):
    checkpoint,contract=fixture(tmp_path)
    changed=tmp_path/'other.jsonl'; changed.write_text('{"id": "other"}\n')
    contract['manifest']=str(changed)
    with pytest.raises(ValueError,match='manifest contents'):
        worker.validate_resume_checkpoint(checkpoint,contract)
    (checkpoint/'optimizer.pt').unlink()
    with pytest.raises(ValueError,match='optimizer state'):
        worker.validate_resume_checkpoint(checkpoint,contract)


def test_resume_rejects_diagnostics_and_world_change(tmp_path):
    checkpoint,contract=fixture(tmp_path)
    contract['diagnostic']=True
    with pytest.raises(ValueError,match='Diagnostic'):
        worker.validate_resume_checkpoint(checkpoint,contract)
    contract.update(diagnostic=False,world=4,ga=96)
    with pytest.raises(ValueError,match='budget contract'):
        worker.validate_resume_checkpoint(checkpoint,contract)
