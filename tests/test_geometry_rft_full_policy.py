"""Small CPU contract tests; not 2B/full-GPU memory or RFT acceptance."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from spatial_intelligence.geometry_rft import (FULL_SCOPE,configure_full_policy,
    save_policy,restore_full_policy,reference_context,CPUStateAdamW,policy_base,
    match_reference_precision,rft_autocast)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.language_model=nn.Linear(3,3)
        self.model.visual=nn.Linear(3,3)
        self.geometry_backbone=nn.Linear(3,3)
        self.geometry_adapter=nn.Linear(3,3)
        self.lm_head=nn.Linear(3,3)


def test_no_grad_rollout_does_not_poison_checkpointed_backward():
    from torch.utils.checkpoint import checkpoint
    model=nn.Sequential(nn.Linear(3,8),nn.SiLU(),nn.Linear(8,3))
    x=torch.randn(2,3,requires_grad=True)
    with rft_autocast('cpu'):
        assert not torch.is_autocast_cache_enabled()
        with torch.no_grad(): model(x)
        checkpoint(model,x,use_reentrant=False).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())


def test_reference_precision_preserves_fixed_values_and_independence():
    reference=Base().bfloat16().requires_grad_(False)
    before={n:p.float().clone() for n,p in reference.named_parameters()}
    actor=configure_full_policy(Base().bfloat16())
    match_reference_precision(reference,actor)
    for name,p in reference.named_parameters():
        assert p.dtype==dict(actor.named_parameters())[name].dtype
        assert torch.equal(p.float(),before[name])
        assert not p.requires_grad
        assert p.data_ptr()!=dict(actor.named_parameters())[name].data_ptr()


def test_scope_and_exact_fp32_save_reload(tmp_path):
    model=configure_full_policy(Base().bfloat16())
    assert policy_base(model) is model
    for name,p in model.named_parameters():
        expected=any(part in name for part in ('language_model','lm_head','geometry_adapter'))
        assert p.requires_grad==expected
        if expected: assert p.dtype==torch.float32
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad: p.add_(1e-6)
    plan=dict(policy_scope=FULL_SCOPE,model_checkpoint=str(tmp_path/'sft'))
    save_policy(model,tmp_path/'policy',plan)
    restored=restore_full_policy(Base().bfloat16(),tmp_path/'policy',plan)
    for name,p in model.named_parameters():
        if p.requires_grad: assert torch.equal(p,dict(restored.named_parameters())[name])
    with pytest.raises(ValueError,match='lineage'):
        restore_full_policy(Base(),tmp_path/'policy',dict(plan,model_checkpoint=str(tmp_path/'other')))


def test_full_reference_cannot_disable_actor_or_share_trainable_model():
    actor=configure_full_policy(Base())
    with pytest.raises(ValueError,match='distinct frozen'):
        with reference_context(actor): pass
    with pytest.raises(ValueError,match='distinct frozen'):
        with reference_context(actor,actor): pass
    reference=Base().requires_grad_(False)
    before=reference.model.language_model.weight.clone()
    with torch.no_grad(): actor.model.language_model.weight.add_(1.)
    with reference_context(actor,reference) as actual:
        assert actual is reference
        assert torch.equal(actual.model.language_model.weight,before)


def test_cpu_optimizer_matches_adam_and_restores_moments():
    torch.manual_seed(7)
    a=nn.Parameter(torch.randn(5));b=nn.Parameter(a.detach().clone())
    direct=torch.optim.AdamW([a],lr=1e-3,weight_decay=0.)
    offload=CPUStateAdamW([b],lr=1e-3,weight_decay=0.)
    for _ in range(3):
        a.square().sum().backward();b.square().sum().backward()
        direct.step();offload.step();direct.zero_grad();offload.zero_grad()
        assert torch.equal(a,b)
    state=offload.state_dict()
    restored=CPUStateAdamW([b],lr=1e-3,weight_decay=0.)
    restored.load_state_dict(state)
    a.square().sum().backward();b.square().sum().backward()
    direct.step();restored.step()
    assert torch.equal(a,b)
