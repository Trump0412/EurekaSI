"""Optimizer-boundary cooperative pause; no signals or resource ownership."""
import json
from pathlib import Path

class TrainingPaused(RuntimeError):
    exit_code=75

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)

def validate_full_checkpoint(path,world_size=1):
    path=Path(path)
    if not (path/'trainer_state.json').is_file():raise ValueError('Missing Trainer state')
    shards=list(path.rglob('*optim_states.pt'))
    if not (path/'optimizer.pt').is_file() and not shards:raise ValueError('Missing optimizer state')
    if not (path/'scheduler.pt').is_file() and not shards:raise ValueError('Missing scheduler state')
    if not list(path.glob('rng_state*.pth')):raise ValueError('Missing RNG state')
    if world_size>1 and any(not (path/f'rng_state_{rank}.pth').is_file() for rank in range(world_size)):
        raise ValueError('Missing rank-local RNG state')
    if shards and len(shards)<world_size:raise ValueError('Missing sharded optimizer state')
    if not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')) and not list(path.rglob('*model_states.pt')):
        raise ValueError('Missing model state')

class PauseProtocol:
    def __init__(self,request_path,output_dir):
        self.request_path=Path(request_path) if request_path else None
        self.output_dir=Path(output_dir);self.pending=False
    def step_end(self,control,device):
        import torch
        dist=torch.distributed.is_initialized();rank=torch.distributed.get_rank() if dist else 0
        value=int(rank==0 and self.request_path is not None and self.request_path.is_file())
        flag=torch.tensor(value,dtype=torch.int64,device=device)
        if dist:torch.distributed.broadcast(flag,src=0)
        if flag.item():self.pending=True;control.should_save=True
        return control
    def saved(self,step):
        if not self.pending:return
        import torch
        dist=torch.distributed.is_initialized()
        if dist:torch.distributed.barrier()
        checkpoint=self.output_dir/f'checkpoint-{step}'
        error=None
        try:validate_full_checkpoint(checkpoint,torch.distributed.get_world_size() if dist else 1)
        except Exception as exc:error=repr(exc)
        errors=[error]
        if dist:
            errors=[None]*torch.distributed.get_world_size();torch.distributed.all_gather_object(errors,error)
        if any(errors):raise ValueError('Pause checkpoint validation failed on ranks: '+repr(errors))
        error=None
        if not dist or torch.distributed.get_rank()==0:
            try:
                atomic_json(self.output_dir/'pause.json',dict(status='paused',step=step,
                    checkpoint=str(checkpoint),request=str(self.request_path),exit_code=75,
                    completed=False,optimizer_boundary=True))
            except Exception as exc:error=repr(exc)
        errors=[error]
        if dist:
            errors=[None]*torch.distributed.get_world_size();torch.distributed.all_gather_object(errors,error)
        if any(errors):raise ValueError('Pause receipt publication failed: '+repr(errors))
        raise TrainingPaused(f'Checkpoint saved; cooperative pause at step {step}')

def persistent_probe_baselines(path,current):
    """Compare final values to original run start, not the last resumed segment."""
    import torch
    path=Path(path);dist=torch.distributed.is_initialized()
    rank=torch.distributed.get_rank() if dist else 0
    if rank==0 and not path.exists():
        temporary=path.with_suffix('.tmp');torch.save(current,temporary);temporary.replace(path)
    if dist:torch.distributed.barrier()
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if set(saved)!=set(current):raise ValueError('Changed original parameter audit groups')
    for key,value in saved.items():
        if value.shape!=current[key].shape or not torch.isfinite(value).all():raise ValueError('Invalid original audit baseline')
    return saved
