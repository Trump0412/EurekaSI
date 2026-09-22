import importlib.util
import json
from pathlib import Path
import pytest

REPO=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/name)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_evaluation_identity_and_frame_contract(tmp_path):
    worker=module('evaluate-geofits.py')
    rows=[dict(id='one',media=['real.jpg'])]
    worker.check_rows(rows)
    with pytest.raises(ValueError):worker.check_rows(rows*2)
    with pytest.raises(ValueError):worker.check_rows([dict(id='one',media=[])])
    worker.lock(tmp_path/'lock.json',dict(a=1))
    worker.lock(tmp_path/'lock.json',dict(a=1))
    with pytest.raises(ValueError):worker.lock(tmp_path/'lock.json',dict(a=2))


def test_runtime_accepts_only_real_variant_updates(tmp_path):
    worker=module('verify-geofits-runtime.py')
    plan=dict(architecture=dict(variant='dense'))
    receipt=dict(status='complete',accepted=True,diagnostic=True,finite_loss=True,
        nonzero_update=True,reload_verified=True,optimizer_rng_files_present=True,
        generation_smoke_tokens=2,steps=2,component_updates=dict(language=True,native_visual=True),
        contract=dict(architecture=plan['architecture'],world=8,global_batch=64))
    path=tmp_path/'completion.json';path.write_text(json.dumps(receipt))
    assert worker.accepted_stage(path,plan,8)['accepted']
    receipt['component_updates']['native_visual']=False;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):worker.accepted_stage(path,plan,8)
    receipt['component_updates']['native_visual']=True;receipt['diagnostic']=False
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):worker.accepted_stage(path,plan,8)


def test_ready_gate_not_accepted_as_formal_training(tmp_path):
    queue=module('run-geometry-followups.py')
    path=tmp_path/'gate.json';path.write_text(json.dumps(dict(status='ready',full_model_verified=True)))
    assert not queue.accepted(path)
    assert queue.accepted(path,dict(status='ready',full_model_verified=True))


def test_full_benchmark_merge_recomputes_scores_and_rejects_contract_changes(tmp_path):
    worker=module('evaluate-geofits.py')
    rows=[dict(id=str(i),question='Which?',media=['image.jpg'],choices={'A':'left','B':'right'},
               answer='A',question_type='spatial') for i in range(1000)]
    checkpoint=str((tmp_path/'checkpoint').resolve())
    contract=dict(checkpoint=checkpoint,benchmark='mmsi',world=2,manifest=rows)
    for rank in range(2):
        worker.lock(tmp_path/f'contract.rank{rank}.json',contract)
        (tmp_path/f'predictions.rank{rank}.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],
            response='<answer>A</answer>',truncated=False,score=-123))+'\n' for r in rows[rank::2]))
    assert worker.merge(rows,tmp_path,2,checkpoint,'mmsi')['accuracy']==1
    changed=[dict(r) for r in rows];changed[0]['answer']='B'
    with pytest.raises(ValueError,match='manifest'):worker.merge(changed,tmp_path,2,checkpoint,'mmsi')


def test_georoute_merge_keeps_same_source_and_model(tmp_path,monkeypatch):
    worker=module('evaluate-georoute.py')
    source=[dict(id=str(i),question='Which?',media=['image.jpg'],choices={'A':'left','B':'right'},
                 answer='A',question_type='spatial') for i in range(1000)]
    plan=tmp_path/'plan.json';plan.write_text('{}')
    manifest=tmp_path/'source.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in source))
    checkpoint=str((tmp_path/'checkpoint').resolve())
    contract=dict(checkpoint=checkpoint,benchmark='mmsi',world=1,ids=[r['id'] for r in source])
    (tmp_path/'contract.rank0.json').write_text(json.dumps(contract))
    (tmp_path/'manifest.rank0.jsonl').write_bytes(manifest.read_bytes())
    (tmp_path/'predictions.rank0.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],response='A',
        truncated=False,score={'score':-123}))+'\n' for r in source))
    monkeypatch.setattr('sys.argv',['evaluate','--plan',str(plan),'--checkpoint',checkpoint,
        '--manifest',str(manifest),'--output',str(tmp_path),'--benchmark','mmsi','--merge'])
    worker.main()
    assert json.loads((tmp_path/'completion.json').read_text())['accuracy']==1
    contract['checkpoint']='different';(tmp_path/'contract.rank0.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='contract'):worker.main()
