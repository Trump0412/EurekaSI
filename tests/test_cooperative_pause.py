"""Actual CPU Trainer checkpoint pause/resume; no GPU or pretrained model."""
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from torch import nn
from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
from spatial_intelligence.cooperative_pause import (
    PauseProtocol,TrainingPaused,persistent_probe_baselines,validate_full_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.Linear(3,2);self.dropout=nn.Dropout(.15)
    def forward(self,input_ids,labels):
        logits=self.layer(self.dropout(input_ids.float()))
        return dict(loss=(logits-labels).square().mean(),logits=logits)


class Callback(TrainerCallback):
    def __init__(self,protocol,request_on_step=None):self.protocol=protocol;self.ended=0;self.request_on_step=request_on_step
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==self.request_on_step:self.protocol.request_path.touch()
        return self.protocol.step_end(control,args.device)
    def on_save(self,args,state,control,**kwargs):self.protocol.saved(state.global_step)
    def on_train_end(self,*args,**kwargs):self.ended+=1


def trainer(path,callback):
    set_seed(42)
    rows=[dict(input_ids=torch.tensor([i/20.,.3,.7]),labels=torch.tensor([.2,.8])) for i in range(16)]
    args=TrainingArguments(output_dir=str(path),use_cpu=True,per_device_train_batch_size=2,
        gradient_accumulation_steps=2,num_train_epochs=1,learning_rate=.01,warmup_ratio=.25,
        lr_scheduler_type='cosine',save_steps=100,logging_steps=1,report_to=[],seed=42,
        dataloader_num_workers=0,save_total_limit=2,disable_tqdm=True)
    model=TinyModel()
    return Trainer(model=model,args=args,train_dataset=rows,callbacks=[callback])


@pytest.mark.parametrize('pause_at',[1,4])
def test_actual_trainer_pause_does_not_end_and_resume_preserves_trajectory(tmp_path,pause_at):
    torch.set_num_threads(2)
    reference=trainer(tmp_path/'reference',Callback(PauseProtocol(None,tmp_path/'reference')))
    reference.train()
    request=tmp_path/'pause-request'
    output=tmp_path/'resumed'
    callback=Callback(PauseProtocol(request,output),request_on_step=pause_at)
    interrupted=trainer(output,callback)
    with pytest.raises(TrainingPaused):interrupted.train()
    assert callback.ended==0
    checkpoint=output/f'checkpoint-{pause_at}'
    validate_full_checkpoint(checkpoint)
    assert not (output/'final').exists() and not (output/'completion.json').exists()
    import json
    assert json.loads((output/'pause.json').read_text())['completed'] is False
    request.unlink()
    resumed_callback=Callback(PauseProtocol(request,output))
    resumed=trainer(output,resumed_callback)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed_callback.ended==1 and resumed.state.global_step==4
    for left,right in zip(reference.model.parameters(),resumed.model.parameters()):
        torch.testing.assert_close(left,right,atol=1e-7,rtol=1e-6)
    assert reference.lr_scheduler.state_dict()==resumed.lr_scheduler.state_dict()
    for key,value in reference.optimizer.state_dict()['state'].items():
        for name,tensor in value.items():
            other=resumed.optimizer.state_dict()['state'][key][name]
            torch.testing.assert_close(tensor,other,atol=1e-7,rtol=1e-6)


def test_original_probe_audit_survives_resume(tmp_path):
    path=tmp_path/'probe.pt'
    before={'route':torch.tensor([0.,1.])}
    persistent_probe_baselines(path,before)
    restored=persistent_probe_baselines(path,{'route':torch.tensor([.2,1.5])})
    torch.testing.assert_close(restored['route'],before['route'])
    with pytest.raises(ValueError):persistent_probe_baselines(path,{'other':torch.ones(2)})


def test_incomplete_checkpoint_rejected(tmp_path):
    (tmp_path/'trainer_state.json').write_text('{}')
    with pytest.raises(ValueError,match='optimizer'):validate_full_checkpoint(tmp_path)


def test_actual_torchrun_two_rank_pause_receipt_and_full_rng(tmp_path):
    import json,os,subprocess,sys
    request=tmp_path/'pause-request';request.touch()
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
        '--nproc_per_node=2',str(Path(__file__).resolve()),'--cpu-pause-rank',str(tmp_path)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        capture_output=True,text=True,timeout=180)
    assert result.returncode!=0
    assert '75' in result.stderr,result.stderr
    receipt=json.loads((tmp_path/'distributed/pause.json').read_text())
    assert receipt['exit_code']==75 and receipt['completed'] is False
    validate_full_checkpoint(receipt['checkpoint'],world_size=2)
    assert not (tmp_path/'distributed/final').exists()


if __name__=='__main__':
    import sys
    assert sys.argv[1]=='--cpu-pause-rank'
    root=Path(sys.argv[2]);output=root/'distributed'
    callback=Callback(PauseProtocol(root/'pause-request',output))
    training=trainer(output,callback)
    try:training.train()
    except TrainingPaused:
        assert callback.ended==0
        raise SystemExit(75)
    raise RuntimeError('The distributed worker did not pause')
