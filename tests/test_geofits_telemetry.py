import json
import pytest
import torch
from spatial_intelligence.geofits_telemetry import accumulate_fusion_statistics
from test_geofits_model import setup


def test_bf16_swallowed_residual_is_not_reported_as_effective_change():
    current=torch.ones(1,1,4,dtype=torch.bfloat16)
    residual=torch.full_like(current,.001)
    fused=current+residual
    assert torch.equal(current,fused)
    diagnostics=dict(gate=torch.ones(1,1,1),weights=torch.tensor([[[.25,.75,0.]]]),
        indices=torch.tensor([[[0,1]]]),intended_residual=residual)
    with torch.no_grad(): result=accumulate_fusion_statistics(None,current,fused,diagnostics)
    assert result['intended_relative_residual_norm']>0
    assert result['actual_relative_residual_norm']==0
    assert result['actual_changed_fraction']==0
    assert result['swallowed_nonzero_residual_fraction']==1
    assert result['entry_selection_frequency']==[1.,1.,0.]
    json.dumps(result,allow_nan=False)


def test_optin_disabled_by_default_and_eval_only():
    model,inputs,features=setup();model.eval()
    assert model._diagnostics is None
    with model.feature_context(features),torch.no_grad(): model(**inputs,use_cache=False)
    assert model._diagnostics is None
    with model.fusion_diagnostics(),model.feature_context(features),pytest.raises(ValueError,match='no_grad'):
        model(**inputs,use_cache=False)
    model.train()
    with pytest.raises(ValueError,match='eval context'):
        with model.fusion_diagnostics(): pass


@pytest.mark.parametrize('variant',['full','no_gate','single_layer'])
def test_per_layer_generation_counts_and_gate_behavior(variant):
    model,inputs,features=setup(variant);model.eval()
    if variant=='full':
        # A true zero gate should leave the visual representation unchanged.
        with torch.no_grad():
            for module in model.geofits.layers:
                module.gate[-1].weight.zero_();module.gate[-1].bias.fill_(-1000.)
    prompt={k:v for k,v in inputs.items() if k!='labels'}
    for key in ('input_ids','attention_mask','mm_token_type_ids'): prompt[key]=prompt[key][:,:8]
    features[0].pop('labels')
    with model.fusion_diagnostics() as summary,model.feature_context(features),torch.no_grad():
        model.generate(**prompt,max_new_tokens=3,min_new_tokens=3,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
    assert set(summary['layers'])==({'3'} if variant=='single_layer' else {'1','2','3'})
    for layer in summary['layers'].values():
        assert layer['prefill_sample_events']==1 # three decoding tokens do not reinject geometry
        assert layer['region_count']==2
        assert abs(sum(layer['entry_mean_weights'])-1)<1e-6
        assert sum(layer['entry_selection_frequency'])==2
        if variant=='full':
            assert layer['gate_mean']==0 and layer['actual_changed_fraction']==0
        if variant=='no_gate':
            assert layer['gate_mean']==layer['gate_min']==layer['gate_max']==1
            assert layer['actual_changed_fraction']>0
    json.dumps(summary,allow_nan=False)
