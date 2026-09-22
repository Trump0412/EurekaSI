import importlib.util
import json
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builder',repo/'scripts/build-geometry-followup-plans.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

def test_all_roles_real_commands_and_cross_node_barrier():
    bindings=dict(nodes={role:dict(root='/jobs/'+role,python='/env/bin/python',gpus=list(range(n))) for role,n in [('primary',8),('control_a',4),('control_b',8)]},
        data_receipt='/data/ready.json',shared_cache='/data/cache',common=dict(model='/model',processor='/model'),
        geofits_architecture=dict(hidden_size=2048),dependencies=[dict(path='/previous.json',equals=dict(status='complete'))],
        eval_manifests=dict(revsi='/data/revsi.jsonl',vsibench='/data/vsi.jsonl'))
    science=json.loads((repo/'configs/geometry-followup-studies.json').read_text())
    plans=builder.build(bindings,science)
    assert all('matched_rgb_sft.json' not in p['stage_plans'] for p in plans.values())
    assert set(plans['control_a']['stage_plans'])=={'one_stb.json','geofits-3d_only.json','geofits-4d_only.json'}
    for plan in plans.values():
        assert '\\' not in plan['root']
        assert all(s['commands'] and s['requirements'] for s in plan['stages'])
        assert all(s['use_lora'] is False for s in plan['stages'])
        for v in plan['stage_plans'].values():assert '\\' not in v['root']
    fits=next(s for s in plans['primary']['stages'] if s['name']=='geofits-full-sft')
    assert len([r for r in fits['requirements'] if r['path'].endswith('route-paper-complete.json')])==3
    assert plans['control_a']['stage_plans']['one_stb.json']['architecture']['blocks_per_exit']==1
    assert plans['control_b']['stage_plans']['final_only.json']['architecture']['active_exits']==[3]
    assert plans['control_b']['stage_plans']['geofits-single_layer.json']['architecture']['fusion_layers']==[3]
    variants={v.removeprefix('geofits-').removesuffix('.json') for p in plans.values()
              for v in p['stage_plans'] if v.startswith('geofits-')}
    assert variants=={'full','3d_only','4d_only','dense','no_gate','single_layer'}
