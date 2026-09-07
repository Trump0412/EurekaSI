"""Synchronous data parallel reference execution via torchrun.

Every rank issues the same collectives, including unused-parameter gradients.
No DDP wrapper is placed around generation-capable custom backends.
"""
import copy
from datetime import timedelta
import os
import torch
import torch.distributed as dist


class Distributed:
    def __init__(self,cfg):
        self.world=int(os.environ.get('WORLD_SIZE','1'));self.rank=int(os.environ.get('RANK','0'));self.local_rank=int(os.environ.get('LOCAL_RANK','0'))
        self.main=self.rank==0;self.owns_group=False
        self.cfg=copy.deepcopy(cfg)
        cuda=str(cfg['model'].get('device','cpu')).startswith('cuda')
        self.device=torch.device(f'cuda:{self.local_rank}' if cuda and self.world>1 else cfg['model'].get('device','cpu'))
        if self.world>1:
            if cuda:torch.cuda.set_device(self.device)
            self.cfg['model']['device']=str(self.device)
            self.cfg['teacher']['device']=str(self.device)
            if not dist.is_initialized():
                dist.init_process_group('nccl' if cuda else 'gloo',timeout=timedelta(minutes=15));self.owns_group=True
    def barrier(self):
        if self.world>1:dist.barrier()
    def objects(self,value):
        if self.world==1:return [value]
        result=[None]*self.world;dist.all_gather_object(result,value);return result
    def broadcast_parameters(self,model):
        if self.world>1:
            with torch.no_grad():
                for p in model.parameters():dist.broadcast(p,0)
    def average_gradients(self,params):
        if self.world==1:return
        for p in params:
            # All ranks participate even when a local geometry branch has no gradient.
            used=torch.tensor(int(p.grad is not None),device=p.device,dtype=torch.int32)
            dist.all_reduce(used,op=dist.ReduceOp.SUM)
            grad=p.grad if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(grad,op=dist.ReduceOp.SUM)
            p.grad=grad/self.world if used.item() else None
    def all_finite(self,value):
        flag=torch.tensor(int(value),device=self.device,dtype=torch.int32)
        if self.world>1:dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        return bool(flag.item())
    def close(self):
        if self.owns_group:dist.destroy_process_group()
