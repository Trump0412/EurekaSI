import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('runtime', Path(__file__).parents[1] / 'scripts/verify-georoute-runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def test_selection_preserves_real_frames(tmp_path):
    rows = [dict(id='single', media=['a']), dict(id='two', media=['a','b']),
            dict(id='long', media=[str(i) for i in range(32)])]
    p = tmp_path / 'rows.jsonl'
    p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = module.select_rows(p)
    assert result['mixed'] == rows[1:]
    assert result['pressure32'] == [rows[2]]
    p.write_text(json.dumps(rows[1]))
    with pytest.raises(ValueError, match='32-frame'):
        module.select_rows(p)


def test_preflight_requires_data_and_finished_eval(tmp_path):
    data = dict(status='ready', media_verified=True, leakage_checked=True)
    plan = dict(variant='final_only', use_lora=False, data_receipt=save(tmp_path/'data.json',data))
    dep = save(tmp_path/'dep.json',dict(status='running',gpu_work_finished=False))
    with pytest.raises(ValueError,match='predecessor'):
        module.preflight(plan,[dep])
    save(Path(dep),dict(status='complete',gpu_work_finished=True))
    assert module.preflight(plan,[dep]) == data
    data['media_verified']=False;save(tmp_path/'data.json',data)
    with pytest.raises(ValueError,match='Audited'):
        module.preflight(plan,[dep])


def test_receipt_rejects_placeholder_or_missing_updates(tmp_path):
    p=tmp_path/'completion.json'
    v=dict(status='complete',accepted=True,diagnostic=True,reload_verified=True,
        finite_loss=True,nonzero_update=True,optimizer_rng_files_present=True,
        generation_smoke_tokens=8,contract={'stage':'sft'},steps=2,
        component_updates={'language':True,'native_visual':True,'routing_exit3':True})
    save(p,v);assert module.accepted_stage(p,'sft') == v
    v['component_updates']['routing_exit3']=False;save(p,v)
    with pytest.raises(ValueError,match='Missing real'):
        module.accepted_stage(p,'sft')
    v['component_updates']['routing_exit3']=True;v['diagnostic']=False;save(p,v)
    with pytest.raises(ValueError):module.accepted_stage(p,'sft')


def test_nonzero_gradient_is_not_effect_acceptance():
    report=dict(graph=dict(edges=10),tensor_comparisons={'exit3.post_exit':dict(changed_elements=0,relative_delta_l2=0)},
        next_token_logits=dict(changed_elements=1))
    with pytest.raises(ValueError,match='vanished'):module.accepted_effect(report)
    report['tensor_comparisons']['exit3.post_exit'].update(changed_elements=1,relative_delta_l2=1e-5)
    assert module.accepted_effect(report)['weak_effect_warning']
    report['next_token_logits']['changed_elements']=0
    with pytest.raises(ValueError,match='output influence'):module.accepted_effect(report)
