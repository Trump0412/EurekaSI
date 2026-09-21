"""Tiny real Qwen integration with a registered differentiable fake VGGT.

The fake tests parameter ownership/gradient/export paths, not VGGT accuracy.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig

from spatial_intelligence.qwen3vl_geometry_matrix import Qwen3VLGeometryMatrix, insert_slots
from spatial_intelligence.qwen35 import completion_loss


class TinyRegisteredVGGT(nn.Module):
    def __init__(self, source, weights=None, trainable=False):
        super().__init__()
        self.projection = nn.Linear(3, 2048)
        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        self.trainable = bool(trainable)
        self.requires_grad_(self.trainable)
        return self

    def forward(self, images):
        features = self.projection(images.mean((-1, -2)))
        return features[:, None].expand(-1, 1024, -1)

    def forward_batch(self, sequences, max_batch=1):
        # Match the registered backbone protocol while keeping this tiny fake
        # focused on model ownership/export rather than batching performance.
        return [self(images) for images in sequences]


@pytest.fixture
def model(monkeypatch):
    import spatial_intelligence.geometry_backbone as backbone
    monkeypatch.setattr(backbone, 'RegisteredVGGT', TinyRegisteredVGGT)
    torch.manual_seed(41)
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=48, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256, pad_token_id=0,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                'mrope_section': [1, 1, 2], 'mrope_interleaved': True}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32,
            num_heads=2, in_channels=3, patch_size=2, spatial_merge_size=2,
            temporal_patch_size=1, out_hidden_size=32, num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1]),
        image_token_id=6, video_token_id=7, vision_start_token_id=4,
        vision_end_token_id=5, pad_token_id=0, eos_token_id=2)
    config._attn_implementation = 'sdpa'
    result = Qwen3VLGeometryMatrix(config)
    result.attach_geometry({'source': 'unused-test-source', 'adapter': 'query64', 'version': 1})
    return result


def inputs():
    ids = torch.tensor([[1, 4, 6, 5, 12, 13, 2]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        labels=torch.tensor([[-100, -100, -100, -100, -100, 13, 2]]),
        mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0, 0]]),
        pixel_values=torch.randn(4, 12), image_grid_thw=torch.tensor([[1, 2, 2]]))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0,
        convert_tokens_to_ids=lambda _: 5))
    result = insert_slots(batch, processor, [1], 'query64', max_context=256)
    result['geometry_images'] = [torch.randn(1, 3, 4, 4)]
    return result


def has_nonzero_gradient(module):
    return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())


@pytest.mark.parametrize('stage,train_vggt', [('align', False), ('sft', False), ('sft', True)])
def test_stage_gradient_and_actual_update_ownership(model, stage, train_vggt):
    model.configure_stage(stage, train_vggt)
    # Frozen language weights must still permit checkpointed backward into the
    # trainable interface; switching the entire LM to eval disables that path.
    assert model.model.language_model.training
    assert model.model.visual.training == (stage == 'sft')
    baseline_hooks = set(model.get_input_embeddings()._forward_hooks)
    groups = {'interface': model.geometry_adapter, 'vggt': model.geometry_backbone,
              'rgb': model.model.visual, 'language': model.model.language_model}
    before = {key: [p.detach().clone() for p in module.parameters()] for key, module in groups.items()}
    loss, _ = completion_loss(model, inputs(), reduction='sample_mean')
    assert torch.isfinite(loss)
    loss.backward()
    expected = {'interface': True, 'vggt': train_vggt, 'rgb': stage == 'sft', 'language': stage == 'sft'}
    for key, module in groups.items():
        assert has_nonzero_gradient(module) == expected[key], key
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    optimizer.step()
    for key, module in groups.items():
        changed = any(not torch.equal(old, current) for old, current in zip(before[key], module.parameters()))
        assert changed == expected[key], key
    assert not model.geometry_adapter.null_tokens.requires_grad
    assert set(model.get_input_embeddings()._forward_hooks) == baseline_hooks


def test_saved_checkpoint_preserves_updated_registered_backbone(model, tmp_path):
    model.configure_stage('sft', True)
    batch = inputs()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loss, _ = completion_loss(model, batch, reduction='sample_mean')
    loss.backward(); optimizer.step(); model.eval()
    with torch.no_grad():
        expected = model(**batch, use_cache=False).logits
    model.save_pretrained(tmp_path / 'model')
    restored = Qwen3VLGeometryMatrix.from_pretrained(tmp_path / 'model', attn_implementation='sdpa').eval()
    assert torch.equal(model.geometry_backbone.projection.weight, restored.geometry_backbone.projection.weight)
    with torch.no_grad():
        actual = restored(**batch, use_cache=False).logits
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_generation_geometry_only_runs_during_prefill(model):
    model.configure_stage('eval')
    batch = inputs(); batch.pop('labels')
    calls = []
    handle = model.geometry_backbone.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            output = model.generate(**batch, max_new_tokens=3, min_new_tokens=3,
                do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0)
    finally:
        handle.remove()
    assert output.shape[1] == batch['input_ids'].shape[1] + 3
    assert len(calls) == 1
    assert not model.get_input_embeddings()._forward_hooks


def test_bfloat16_cast_preserves_native_rope_and_reload(model, tmp_path):
    before = {name: value.detach().clone() for name, value in model.named_buffers()
              if name.endswith('inv_freq') or name.endswith('original_inv_freq')}
    assert before, 'The real tiny Qwen must expose native rotary frequency buffers'
    assert all(value.dtype == torch.float32 for value in before.values())
    model.bfloat16()
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    after = dict(model.named_buffers())
    for name, expected in before.items():
        assert after[name].dtype == torch.float32, name
        assert torch.equal(after[name], expected), name
    model.save_pretrained(tmp_path / 'bf16-model')
    restored = Qwen3VLGeometryMatrix.from_pretrained(tmp_path / 'bf16-model',
        dtype=torch.bfloat16, attn_implementation='sdpa').eval()
    reloaded = dict(restored.named_buffers())
    for name, expected in before.items():
        assert reloaded[name].dtype == torch.float32, name
        assert torch.equal(reloaded[name], expected), name
