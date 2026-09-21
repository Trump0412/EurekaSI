import importlib.util
from pathlib import Path
import tempfile
import time

spec=importlib.util.spec_from_file_location('gap',Path(__file__).resolve().parents[1]/'scripts/run-georoute-gap.py')

def test_priority_wins_over_ready_route():
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.choose_action(True,True)=='priority_rft'
    assert module.choose_action(False,True)=='georoute_gate'
    assert module.choose_action(False,False)=='wait_data_or_priority'
    with tempfile.TemporaryDirectory() as tmp:
        receipt=Path(tmp)/'state.json';requirement=[{'path':str(receipt),'equals':{'status':'complete','accepted':True}}]
        assert not module.ready(requirement)
        module.write(receipt,{'status':'paused','accepted':False})
        assert not module.ready(requirement)
        module.write(receipt,{'status':'complete','accepted':True})
        assert module.ready(requirement)


def test_fresh_pause_receipt_not_generic_launcher_failure():
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);q=module.Queue(dict(root=tmp,gpus=[0],code=tmp))
        q.last_started=time.time()-1
        module.write(q.pause,{'reason':'priority'})
        checkpoint=root/'checkpoint-2';checkpoint.mkdir()
        for name in ('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors'):
            (checkpoint/name).touch()
        receipt=root/'pause.json'
        stage={'pause_acceptance':[{'path':str(receipt),'equals':{'status':'paused','exit_code':75,'completed':False,'optimizer_boundary':True}}]}
        assert not q.paused(stage)
        module.write(receipt,dict(status='paused',exit_code=75,completed=False,optimizer_boundary=True,
                                 request=str(q.pause),checkpoint=str(checkpoint)))
        assert q.paused(stage)
        q.last_started=time.time()+10
        assert not q.paused(stage)
