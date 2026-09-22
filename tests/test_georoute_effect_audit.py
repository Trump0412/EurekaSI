import importlib.util
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')

SPEC = importlib.util.spec_from_file_location('audit_effect', Path(__file__).resolve().parents[1] / 'scripts/audit-georoute-effect.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metrics_detect_identity_projection_annihilation_and_transmission():
    reference = torch.tensor([[1., 2.]])
    changed = reference + torch.tensor([[.5, 0.]])
    result = MODULE.tensor_comparison(reference, changed)
    assert result['changed_elements'] == 1 and result['relative_delta_l2'] > 0
    merger = torch.tensor([[0.], [1.]])
    collapsed = MODULE.tensor_comparison(reference @ merger, changed @ merger)
    assert collapsed['changed_elements'] == 0
    assert MODULE.tensor_comparison(reference, reference)['cosine'] == pytest.approx(1.)


def test_zero_norm_explicit_undefined_and_nonfinite_rejected():
    result = MODULE.tensor_comparison(torch.zeros(3), torch.ones(3))
    assert result['relative_delta_l2'] is None and result['cosine'] is None
    with pytest.raises(ValueError, match='Nonfinite'):
        MODULE.tensor_comparison(torch.ones(1), torch.tensor([float('nan')]))


def test_compare_reports_logits_and_each_exit_not_just_gate():
    states0 = {}; states1 = {}
    names = ['pre_before_route', 'pre_merger', 'post_merger_before_post_route', 'post_exit']
    for index in range(4):
        for name in names:
            states0[f'exit{index}.{name}'] = torch.ones(2, 4)
            states1[f'exit{index}.{name}'] = torch.ones(2, 4) + (0. if name == 'pre_before_route' else .1)
    logits0 = torch.tensor([[[1., 2.]]]); logits1 = torch.tensor([[[3., 2.]]])
    result = MODULE.compare_passes((states0, logits0, []), (states1, logits1, []))
    assert len(result['exits']) == 4
    assert result['argmax_zero'] != result['argmax_one']
    assert result['next_token_kl_zero_to_one_mean'] > 0
    assert all(row['pre_changed'] and row['post_changed'] for row in result['exits'])


def test_bf16_rounded_addition_is_not_automatically_dead_gradient():
    hidden = torch.ones(2, dtype=torch.bfloat16)
    alpha = torch.tensor(.0001, dtype=torch.bfloat16, requires_grad=True)
    result = hidden + alpha
    assert torch.equal(result, hidden)  # residual rounded away in forward
    result.float().sum().backward()
    assert alpha.grad is not None and alpha.grad != 0  # local backward still connected


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('placement',['pre','post'])
def test_real_tiny_qwen_hooks_detect_effect_and_remove_cleanly(dtype,placement):
    from spatial_intelligence.georoute import install_georoute
    spec=importlib.util.spec_from_file_location('effect_tiny_fixture',Path(__file__).with_name('test_georoute_integration.py'))
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    model=fixtures.model.__wrapped__()
    route=install_georoute(model,dict(exits=[0,1,2,3],bottleneck=8,placement=placement))
    model.to(dtype=dtype)
    inputs=fixtures.batch();inputs.pop('labels');inputs['pixel_values']=inputs['pixel_values'].to(dtype)
    graph=fixtures.graph()
    mergers=list(model.model.visual.deepstack_merger_list)+[model.model.visual.merger]
    hooks_before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in mergers]
    zero=MODULE.capture_forward(model,inputs,graph,0.)
    initial=MODULE.capture_forward(model,inputs,graph,1.)
    assert MODULE.compare_passes(zero,initial)['next_token_logits']['changed_elements']==0
    # Synthetic activation of an existing checkpoint gate is only a diagnostic
    # fixture, not a production alpha initialization or a measured accuracy gain.
    with torch.no_grad():
        for blocks in route.routes:
            for block in blocks: block.alpha.fill_(.2)
    active=MODULE.capture_forward(model,inputs,graph,1.)
    report=MODULE.compare_passes(zero,active)
    assert report['next_token_logits']['changed_elements']>0
    stage='pre_merger' if placement=='pre' else 'post_exit'
    assert all(report['tensor_comparisons'][f'exit{i}.{stage}']['changed_elements']>0 for i in range(4))
    assert hooks_before==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in mergers]
