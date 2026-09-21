import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('followups',Path(__file__).resolve().parents[1]/'scripts/run-geometry-followups.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def plan():
    return {'root':'r','python':'python','dependencies':[{'path':'prior'}],'gpus':[0,1],
        'stages':[{'name':'route','study':'georoute','use_lora':False,'requirements':[{'path':'data'}],
            'commands':[['python','train.py']],'receipt':'route.json'},
            {'name':'fits','study':'geofits','use_lora':False,'requirements':[{'path':'data'}],
             'after':['route'],'commands':[],'receipt':'fits.json'}]}


def test_order_and_no_lora():
    assert module.validate(plan())
    value=plan(); value['stages'].reverse()
    with pytest.raises(ValueError): module.validate(value)
    value=plan(); value['stages'][0]['use_lora']=True
    with pytest.raises(ValueError): module.validate(value)


def test_launch_requires_explicit_command(monkeypatch):
    monkeypatch.setattr('sys.argv',['runner','--plan','nonexistent-plan.json'])
    with pytest.raises(SystemExit) as exc: module.main()
    assert exc.value.code==2


def test_shell_strings_and_forward_dependencies_rejected():
    value=plan(); value['stages'][0]['commands']=['python train.py']
    with pytest.raises(ValueError): module.validate(value)
    value=plan(); value['stages'][0]['after']=['fits']
    with pytest.raises(ValueError): module.validate(value)


def test_receipt_file_does_not_mean_readiness(tmp_path):
    path=tmp_path/'data.json'; req={'path':str(path),'equals':{'status':'ready','media_verified':True}}
    assert not module.requirement_status(req)[0]
    path.write_text(json.dumps({'status':'ready'}))
    assert not module.requirement_status(req)[0]
    path.write_text(json.dumps({'status':'ready','media_verified':True}))
    assert module.requirement_status(req)[0]
    assert not module.accepted(path)


def test_command_failure_blocks_dependent_without_starting_it(tmp_path,monkeypatch):
    value=plan(); value['root']=str(tmp_path)
    queue=module.Queue(value)
    monkeypatch.setattr(queue,'wait',lambda *a,**kw:None)
    monkeypatch.setattr(queue,'idle',lambda:True)
    def fail(*args): raise RuntimeError('real failure')
    monkeypatch.setattr(queue,'launch',fail)
    if __import__('os').name=='nt': pytest.skip('Linux flock integration')
    queue.execute()
    assert queue.states['route']['status']=='failed'
    assert queue.states['fits']['status']=='blocked_dependency'
