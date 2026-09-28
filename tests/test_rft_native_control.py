"""Native controls must not masquerade as trained geometry checkpoints."""
import json
import pytest
import torch
from torch import nn
from spatial_intelligence.geometry_rft import (
    FULL_SCOPE, configure_full_policy, save_policy, restore_full_policy, cached_geometry)
from spatial_intelligence.rft_initialization import validate_native_initialization


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.language_model=nn.Linear(3,3)
        self.model.visual=nn.Linear(3,3)
        self.lm_head=nn.Linear(3,3)


def test_native_trainable_scope_and_exact_reload(tmp_path):
    model=configure_full_policy(Native().bfloat16())
    assert not hasattr(model,'geometry_adapter')
    assert not any(p.requires_grad for p in model.model.visual.parameters())
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.model.language_model.parameters())
    plan=dict(policy_scope=FULL_SCOPE,model_kind='native_qwen3vl',model_checkpoint=str(tmp_path/'released'))
    save_policy(model,tmp_path/'policy',plan)
    restored=restore_full_policy(Native().bfloat16(),tmp_path/'policy',plan)
    for name,p in model.named_parameters():
        if p.requires_grad: assert torch.equal(p,dict(restored.named_parameters())[name])
    with cached_geometry(model,{'input_ids':torch.tensor([[1]])}) as features:
        assert features is None
    with pytest.raises(ValueError,match='geometry inputs'):
        with cached_geometry(model,{'geometry_images':[]}): pass


def test_native_initialization_is_not_fake_sft(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(model_type='qwen3_vl')))
    (tmp_path/'model.safetensors').write_bytes(b'shape-only-fixture')
    plan=dict(model_kind='native_qwen3vl',initialization='released_native',
              model_checkpoint=str(tmp_path),use_lora=False,policy_scope=FULL_SCOPE)
    assert validate_native_initialization(plan)['sft_performed'] is False
    with pytest.raises(ValueError,match='SFT or VGGT'):
        validate_native_initialization(dict(plan,sft_receipt='fake.json'))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'x':'missing.safetensors'})))
    with pytest.raises(ValueError,match='shard'):
        validate_native_initialization(plan)


def test_explicit_native_prompt_preserves_legacy_and_rejects_geometry():
    from spatial_intelligence.geometry_rft import (resolve_rft_prompt,PROMPT_VERSION,
        NATIVE_FORMAT_PROMPT_VERSION,STRUCTURED_INSTRUCTION)
    assert resolve_rft_prompt({}, {})==(PROMPT_VERSION,STRUCTURED_INSTRUCTION)
    recipe={'prompt_version':NATIVE_FORMAT_PROMPT_VERSION}
    with pytest.raises(ValueError,match='geometry'):
        resolve_rft_prompt(recipe,{'model_kind':'geometry'})
    version,instruction=resolve_rft_prompt(recipe,dict(model_kind='native_qwen3vl',initialization='released_native'))
    assert version==NATIVE_FORMAT_PROMPT_VERSION and instruction.startswith(STRUCTURED_INSTRUCTION)
    assert 'literal output text' in instruction
