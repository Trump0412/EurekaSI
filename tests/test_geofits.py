"""CPU semantic and gradient tests, not real-teacher/HF runtime acceptance."""
import io
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from spatial_intelligence.geofits import (GeoFitsConfig, GeoFitsFusion, GeometryBank,
    FrozenGeometryTeacher, FusionContext, install_geofits, relative_time,
    enable_full_parameter_training)
from spatial_intelligence.geofits_recipe import architecture_for_variant,VARIANTS
from dataclasses import asdict


def config(**changes):
    return GeoFitsConfig(**dict(dict(hidden_size=8, vggt_width=4, pi3_width=6,
        temporal_bottleneck=3, temporal_group_size=1, temporal_group_reduction='mean',
        timestamp_encoding='normalized_linear_sincos', pooling='average_2x2', retrieval_width=4), **changes))


def features():
    return ({n: torch.randn(1, 2, 4, 4, 4) for n in (11, 17, 23)},
            {n: torch.randn(1, 2, 4, 4, 6) for n in (17, 26, 35)})


def context(bank):
    return FusionContext(torch.randn(1, 8, 8), bank, torch.arange(8)[None, :],
                         torch.tensor([9]), torch.tensor([10]),
                         torch.tensor([[-100] * 10 + [3, 4]]))


def test_six_entries_shape_and_trainable_temporal_adapters():
    module = GeometryBank(config()); v, p = features()
    out = module(v, p, torch.tensor([[3., 7.]]), native_grid=(2, 2, 2))
    assert out.shape == (1, 8, 6, 8)
    out.square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in module.parameters())
    assert module.temporal[0].down.weight.grad.abs().sum() > 0


def test_bank_fails_closed_for_missing_level_and_wrong_native_grid():
    module = GeometryBank(config()); v, p = features()
    with pytest.raises(ValueError, match='merger grid'):
        module(v, p, torch.tensor([[0., 1.]]), native_grid=(1, 2, 2))
    v.pop(11)
    with pytest.raises(ValueError, match='levels'):
        module(v, p, torch.tensor([[0., 1.]]), native_grid=(2, 2, 2))


def test_actual_time_shift_scale_invariance_and_order_only_provenance():
    assert torch.allclose(relative_time(torch.tensor([[3., 5., 7.]])),
                          relative_time(torch.tensor([[30., 50., 70.]])))
    with pytest.raises(ValueError): relative_time(torch.tensor([[2., 1.]]))
    with pytest.raises(ValueError): relative_time(torch.tensor([[1., 1.]]))
    module = GeometryBank(config(timestamp_encoding='order_only_sincos')); v, p = features()
    assert module(v, p, None, native_grid=(2, 2, 2)).shape == (1, 8, 6, 8)
    with pytest.raises(ValueError, match='Order-only'):
        module(v, p, torch.tensor([[0., 1.]]), native_grid=(2, 2, 2))


def test_topk_two_normalized_gate_only_modifies_visual_indices():
    module = GeoFitsFusion(config()); hidden = torch.randn(1, 12, 8)
    ctx = context(torch.randn(1, 8, 6, 8))
    out, diag = module(hidden, ctx.original_visual, ctx.bank, visual_indices=ctx.visual_indices,
        question_indices=ctx.question_indices, prefix_lengths=ctx.prefix_lengths, labels=ctx.labels, layer=1)
    assert torch.equal(out[:, 8:], hidden[:, 8:])
    assert ((diag['weights'] > 0).sum(-1) == 2).all()
    assert torch.allclose(diag['weights'].sum(-1), torch.ones(1, 8))
    assert ((diag['gate'] > 0) & (diag['gate'] < 1)).all()


def test_answer_index_and_answer_prefix_rejected():
    module = GeoFitsFusion(config()); ctx = context(torch.randn(1, 8, 6, 8)); hidden = torch.randn(1, 12, 8)
    args = dict(visual_indices=ctx.visual_indices, question_indices=torch.tensor([11]),
                prefix_lengths=ctx.prefix_lengths, labels=ctx.labels, layer=1)
    with pytest.raises(ValueError, match='Question index'):
        module(hidden, ctx.original_visual, ctx.bank, **args)
    args['question_indices'] = ctx.question_indices; args['labels'][0, 9] = 42
    with pytest.raises(ValueError, match='leaked'):
        module(hidden, ctx.original_visual, ctx.bank, **args)


class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(8, 8)

    def forward(self, x):
        count = torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        return x + self.linear(x.cumsum(1) / count).tanh()


def test_hooks_checkpoint_gradients_and_teacher_forcing_prefix_equivalence():
    torch.manual_seed(9)
    module = GeoFitsFusion(config()); blocks = nn.ModuleList([CausalBlock() for _ in range(3)])
    hooks = install_geofits(blocks, module)
    ctx = context(torch.randn(1, 8, 6, 8)); base = torch.randn(1, 12, 8)
    def run(use_checkpoint, change_answers=False):
        module.zero_grad(); blocks.zero_grad()
        x = base.clone().requires_grad_(True)
        if change_answers: x = torch.cat((x[:, :10], x[:, 10:] + 500), 1)
        with hooks.context(ctx):
            for block in blocks:
                x = checkpoint(block, x, use_reentrant=False) if use_checkpoint else block(x)
            x[:, :10].square().mean().backward()
        return x.detach(), module.layers[0].query.weight.grad.clone()
    plain, grad = run(False)
    checked, checked_grad = run(True)
    altered, _ = run(False, True)
    assert torch.allclose(plain, checked)
    assert torch.allclose(grad, checked_grad, atol=1e-7)
    assert torch.equal(plain[:, :10], altered[:, :10])
    assert grad.abs().sum() > 0
    # Unbound incremental decoding must not rerun the bank fusion.
    assert blocks[0](torch.randn(1, 1, 8)).shape == (1, 1, 8)
    hooks.remove(); assert not blocks[0]._forward_hooks


def test_full_training_ownership_frozen_teacher_and_state_roundtrip():
    module = GeoFitsFusion(config()); lm = nn.Linear(8, 8); rgb = nn.Linear(8, 8)
    teacher = FrozenGeometryTeacher(nn.Sequential(nn.Linear(8, 8), nn.Dropout(.9)))
    enable_full_parameter_training(lm, rgb, module, [teacher]); teacher.train()
    assert not teacher.backbone.training
    assert all(p.requires_grad for obj in (lm, rgb, module) for p in obj.parameters())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher(torch.randn(1, 8, requires_grad=True)).requires_grad
    buffer = io.BytesIO(); torch.save(module.state_dict(), buffer); buffer.seek(0)
    restored = GeoFitsFusion(config()); restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in module.state_dict().items())


def test_real_qwen_deepstack_prefill_cache_matches_full_teacher_forcing():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    torch.manual_seed(19)
    cfg = Qwen3VLTextConfig(vocab_size=48, hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=1, num_key_value_heads=1, head_dim=8,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'mrope_section': [1, 1, 2], 'mrope_interleaved': True})
    cfg._attn_implementation = 'sdpa'
    model = Qwen3VLTextModel(cfg).eval()
    fusion = GeoFitsFusion(config()).eval(); hooks = install_geofits(model.layers, fusion)
    hidden = torch.randn(1, 11, 8); bank = torch.randn(1, 8, 6, 8)
    ctx = context(bank); ctx.labels = None
    deepstack = [torch.randn(8, 8) for _ in range(3)]
    mask = torch.tensor([[True] * 8 + [False] * 3])
    with torch.no_grad():
        with hooks.context(ctx):
            full = model(inputs_embeds=hidden, visual_pos_masks=mask,
                         deepstack_visual_embeds=deepstack, use_cache=False)
        with hooks.context(ctx):
            prefix = model(inputs_embeds=hidden[:, :10], visual_pos_masks=mask[:, :10],
                           deepstack_visual_embeds=deepstack, use_cache=True)
        decoded = model(inputs_embeds=hidden[:, 10:], past_key_values=prefix.past_key_values,
                        attention_mask=torch.ones(1, 11), use_cache=True)
    assert torch.allclose(full.last_hidden_state[:, -1:], decoded.last_hidden_state, atol=1e-5)
    hooks.remove()


@pytest.mark.parametrize('variant',VARIANTS)
def test_real_structural_ablation_bank_gate_and_gradients(variant):
    settings=architecture_for_variant(asdict(config()),variant)
    for key in ('bank_sources','fusion_layers'): settings[key]=tuple(settings[key])
    cfg=GeoFitsConfig(**settings);module=GeoFitsFusion(cfg)
    v,p=features()
    if 'vggt' not in cfg.bank_sources: v={}
    if 'pi3' not in cfg.bank_sources: p={}
    bank=module.bank(v,p,torch.tensor([[0.,1.]]),native_grid=(2,2,2))
    assert bank.shape[2]==3*len(cfg.bank_sources)
    assert len(module.bank.projectors)==bank.shape[2]
    assert len(module.bank.temporal)==(3 if 'pi3' in cfg.bank_sources else 0)
    assert len(module.layers)==len(cfg.fusion_layers)
    hidden=torch.randn(1,12,8,requires_grad=True);ctx=context(bank)
    out,diag=module(hidden,ctx.original_visual,bank,visual_indices=ctx.visual_indices,
        question_indices=ctx.question_indices,prefix_lengths=ctx.prefix_lengths,
        labels=ctx.labels,layer=cfg.fusion_layers[0])
    assert ((diag['weights']>0).sum(-1)==(bank.shape[2] if variant=='dense' else 2)).all()
    if variant=='no_gate':
        assert not any('gate' in name for name,_ in module.named_parameters())
        assert torch.equal(diag['gate'],torch.ones_like(diag['gate']))
    out.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.bank.parameters())
    if variant=='3d_only':
        with pytest.raises(ValueError,match='levels'): module.bank(v,features()[1],torch.tensor([[0.,1.]]),native_grid=(2,2,2))
    if variant=='single_layer': assert cfg.fusion_layers==(3,)
