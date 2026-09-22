"""Small eval-only GeoFits diagnostics, including BF16 addition-rounding loss.

All returned values are JSON scalars/lists; no activations survive the call.
Intended residual means the computed residual immediately BEFORE addition,
not hypothetical FP32 recomputation of the entire adapter.
"""
import math
import torch


def accumulate_fusion_statistics(previous,current,fused,diagnostics):
    if torch.is_grad_enabled(): raise ValueError('Telemetry requires no_grad')
    if current.shape!=fused.shape: raise ValueError('Diagnostic region shape mismatch')
    gate=diagnostics['gate'].detach().float()
    weights=diagnostics['weights'].detach().float()
    indices=diagnostics['indices']
    residual=diagnostics['intended_residual'].detach().float()
    original=current.detach().float();actual=fused.detach().float()-original
    if residual.shape!=current.shape or weights.shape[:-1]!=current.shape[:-1]:
        raise ValueError('Diagnostic bank/residual alignment mismatch')
    if not all(torch.isfinite(x).all() for x in (original,actual,residual,gate,weights)):
        raise ValueError('Nonfinite fusion diagnostics')
    regions=math.prod(current.shape[:-1]);elements=current.numel()
    selected=torch.zeros_like(weights,dtype=torch.bool).scatter_(-1,indices,True)
    sums=dict(region_count=regions,element_count=elements,prefill_sample_events=1,
        gate_sum=float(gate.sum()),gate_min=float(gate.min()),gate_max=float(gate.max()),
        original_squared_norm=float(original.square().sum()),intended_squared_norm=float(residual.square().sum()),
        actual_squared_norm=float(actual.square().sum()),
        changed_elements=int((actual!=0).sum()),changed_regions=int((actual!=0).any(-1).sum()),
        intended_nonzero_elements=int((residual!=0).sum()),
        swallowed_elements=int(((residual!=0)&(actual==0)).sum()),
        entry_weight_sums=weights.flatten(0,-2).sum(0).cpu().tolist(),
        entry_selection_counts=selected.flatten(0,-2).sum(0).cpu().tolist())
    if previous is not None:
        if len(previous['entry_weight_sums'])!=len(sums['entry_weight_sums']):
            raise ValueError('Cannot aggregate different banks')
        for key,value in list(sums.items()):
            if key=='gate_min': sums[key]=min(previous[key],value)
            elif key=='gate_max': sums[key]=max(previous[key],value)
            elif isinstance(value,list): sums[key]=[a+b for a,b in zip(previous[key],value)]
            else: sums[key]=previous[key]+value
    denominator=sums['region_count'];norm=max(sums['original_squared_norm'],1e-30)
    sums.update(gate_mean=sums['gate_sum']/denominator,
        entry_mean_weights=[x/denominator for x in sums['entry_weight_sums']],
        entry_selection_frequency=[x/denominator for x in sums['entry_selection_counts']],
        intended_relative_residual_norm=math.sqrt(sums['intended_squared_norm']/norm),
        actual_relative_residual_norm=math.sqrt(sums['actual_squared_norm']/norm),
        actual_changed_fraction=sums['changed_elements']/sums['element_count'],
        actual_changed_region_fraction=sums['changed_regions']/denominator,
        swallowed_nonzero_residual_fraction=sums['swallowed_elements']/max(1,sums['intended_nonzero_elements']))
    return sums
