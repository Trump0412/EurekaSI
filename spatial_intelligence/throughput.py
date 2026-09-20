"""Opt-in throughput primitives. Never change samples across optimizer steps."""
from contextlib import contextmanager
import math


def row_cost(row):
    """Cheap scheduling proxy, NOT an exact token count or a filtering rule."""
    return 256*len(row['media'])+math.ceil((len(row['question'])+len(row['answer']))/4)


def balanced_order(indices, costs, *, micro_batch, world_size, accumulation):
    """Group similar lengths, then balance grouped work over ranks within a step.

    Output is the global order expected by non-split BatchSamplerShard:
    microstep -> rank -> sample. A partial final optimizer step is left unchanged.
    First global step stays unchanged: Accelerate may reuse its prefix to pad
    the final distributed batch. No resampling, cross-step reordering,
    truncation or media reordering.
    """
    if min(micro_batch,world_size,accumulation)<1:raise ValueError('Positive sizes required')
    size=micro_batch*world_size*accumulation;result=[]
    for start in range(0,len(indices),size):
        block=list(indices[start:start+size])
        if start==0 or len(block)<size:result.extend(block);continue
        ordered=sorted(block,key=lambda i:(costs[i],i),reverse=True)
        groups=[ordered[i:i+micro_batch] for i in range(0,size,micro_batch)]
        ranks=[[] for _ in range(world_size)];loads=[0]*world_size
        for group in groups:
            rank=min((r for r in range(world_size) if len(ranks[r])<accumulation),key=lambda r:(loads[r],r))
            ranks[rank].append(group);loads[rank]+=max(costs[i] for i in group)*micro_batch
        for step in range(accumulation):
            for rank in range(world_size):result.extend(ranks[rank][step])
    return result


@contextmanager
def delta_backend(name):
    """Control only DeltaNet kernels; keep norm/conv identical between candidates.

    Modules capture these functions on construction. No mutation of installed
    packages, no fallback when FLA was explicitly requested.
    """
    if name not in ('reference','fla'):raise ValueError(name)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    keys=('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule','FusedRMSNormGated','causal_conv1d_fn','causal_conv1d_update')
    old={k:getattr(module,k) for k in keys}
    try:
        for k in keys:setattr(module,k,None)
        if name=='fla':
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
            module.chunk_gated_delta_rule=chunk_gated_delta_rule
            module.fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule
        yield
    finally:
        for key,value in old.items():setattr(module,key,value)


def kernel_inventory(model):
    return [{'name':name,'chunk':f'{layer.chunk_gated_delta_rule.__module__}.{layer.chunk_gated_delta_rule.__name__}',
             'norm':type(layer.norm).__name__,'conv':str(layer.causal_conv1d_fn)}
            for name,layer in model.named_modules() if hasattr(layer,'chunk_gated_delta_rule')]
