import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('geofits_stage',Path(__file__).resolve().parents[1]/'scripts/train-geofits-stage.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def test_schedule_preserves_rows_and_seed():
    rows=[dict(id=str(i)) for i in range(65)]
    actual,count=worker.arrange(rows)
    assert len(actual)==128 and count['tail_padding']==63
    assert actual==worker.arrange(rows)[0]
    assert {r['id'] for r in actual}=={r['id'] for r in rows}


def test_data_and_runtime_gate_fail_closed(tmp_path):
    data=tmp_path/'data.json';data.write_text(json.dumps(dict(status='ready',media_verified=True,leakage_checked=True)))
    gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(status='pending')))
    plan=dict(use_lora=False,data_receipt=str(data),runtime_acceptance=str(gate),architecture={})
    assert worker.validate_plan(plan,2)['status']=='ready'
    with pytest.raises(ValueError,match='actual full-model'): worker.validate_plan(plan)
    with pytest.raises(ValueError,match='Only explicit'): worker.validate_plan(dict(plan,variant='dense'),2)
