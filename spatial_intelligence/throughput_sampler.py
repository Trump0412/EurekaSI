"""Trainer/Accelerate adapter preserving the seedable baseline's global steps."""
from torch.utils.data import Sampler
from accelerate.data_loader import SeedableRandomSampler
from .throughput import balanced_order


class EffectiveBatchSampler(Sampler):
    def __init__(self,dataset,costs,*,seed,micro_batch,world_size,accumulation):
        if len(dataset)!=len(costs):raise ValueError('Cost coverage mismatch')
        self.base=SeedableRandomSampler(dataset,data_seed=seed)
        self.costs=costs;self.micro=micro_batch;self.world=world_size;self.ga=accumulation
    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):self.base.set_epoch(epoch)
    def __iter__(self):
        return iter(balanced_order(list(self.base),self.costs,micro_batch=self.micro,
                                   world_size=self.world,accumulation=self.ga))
