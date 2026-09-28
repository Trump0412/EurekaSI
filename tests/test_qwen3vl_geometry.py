"""Tiny real Qwen3-VL CPU integration; no pretrained download or GPU needed."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from spatial_intelligence.qwen3vl_geometry import Qwen3VLGeometry, add_geometry_slots
from spatial_intelligence.qwen35 import completion_loss


def tiny_model():
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
        vision_end_token_id=5, pad_token_id=0, eos_token_id=2,
        geometry_interface=dict(input_dim=12, latent_dim=16, output_dim=32,
            num_tokens=64, num_heads=4, dropout=0.))
    config._attn_implementation = 'sdpa'
    return Qwen3VLGeometry(config).eval()


def tiny_inputs():
    ids = torch.tensor([[1, 4, 6, 5, 12, 13, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 13, 2]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                 mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0, 0]]),
                 pixel_values=torch.randn(4, 12), image_grid_thw=torch.tensor([[1, 2, 2]]))
    token = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: 5)
    return add_geometry_slots(batch, SimpleNamespace(tokenizer=token),
        torch.randn(1, 2, 3, 12), torch.zeros(1, 2, 3, dtype=torch.bool), max_context=256)


def test_real_forward_preserves_deepstack_visual_mask_and_native_rope(monkeypatch):
    model, batch = tiny_model(), tiny_inputs()
    captured = {}
    language = model.model.language_model
    original = language.forward

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(language, 'forward', capture)
    result = model(**batch, use_cache=False)
    assert result.logits.shape == (1, 71, 48)
    assert torch.isfinite(result.loss)
    assert captured['visual_pos_masks'].sum().item() == 1
    assert captured['visual_pos_masks'][0, 2]
    assert not captured['visual_pos_masks'][0, batch['geometry_positions'][0]].any()
    assert len(captured['deepstack_visual_embeds']) == 2
    assert all(t.shape == (1, 32) for t in captured['deepstack_visual_embeds'])
    expected, _ = model.model.get_rope_index(input_ids=batch['input_ids'],
        mm_token_type_ids=batch['mm_token_type_ids'], image_grid_thw=batch['image_grid_thw'],
        attention_mask=batch['attention_mask'])
    assert torch.equal(captured['position_ids'], expected)
    result.loss.backward()
    assert model.geometry_adapter.input_projection[0].weight.grad.abs().sum() > 0
    assert len(model.get_input_embeddings()._forward_hooks) == 0


def test_selected_completion_logits_loss_and_gradients_match_full():
    model, batch = tiny_model(), tiny_inputs()
    full = model(**batch, use_cache=False)
    full.loss.backward()
    expected_grad = model.geometry_adapter.input_projection[0].weight.grad.clone()
    model.zero_grad(set_to_none=True)
    selected_loss, selected = completion_loss(model, batch)
    positions = (batch['labels'][:, 1:] != -100).any(0).nonzero().flatten()
    assert torch.allclose(selected.logits, full.logits[:, positions], atol=1e-6)
    assert torch.allclose(selected_loss, full.loss, atol=1e-6)
    selected_loss.backward()
    assert torch.allclose(model.geometry_adapter.input_projection[0].weight.grad,
                          expected_grad, atol=1e-6, rtol=1e-5)


def test_save_and_explicit_class_reload_preserve_geometry_logits(tmp_path):
    model, batch = tiny_model(), tiny_inputs()
    with torch.no_grad():
        expected = model(**batch, use_cache=False).logits
    model.save_pretrained(tmp_path/'tiny')
    restored = Qwen3VLGeometry.from_pretrained(tmp_path/'tiny', attn_implementation='sdpa').eval()
    assert restored.config.geometry_interface == model.config.geometry_interface
    with torch.no_grad():
        actual = restored(**batch, use_cache=False).logits
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_load_vanilla_checkpoint_initializes_new_geometry_parameters(tmp_path):
    configured = tiny_model().config
    geometry_config = configured.geometry_interface
    vanilla_config = configured.to_dict()
    vanilla_config.pop('geometry_interface')
    vanilla = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**vanilla_config)).eval()
    vanilla.save_pretrained(tmp_path/'vanilla')
    loaded_config = Qwen3VLConfig.from_pretrained(tmp_path/'vanilla')
    loaded_config.geometry_interface = geometry_config
    restored = Qwen3VLGeometry.from_pretrained(tmp_path/'vanilla', config=loaded_config,
                                               attn_implementation='sdpa').eval()
    assert torch.equal(restored.get_input_embeddings().weight, vanilla.get_input_embeddings().weight)
    for name, parameter in restored.geometry_adapter.named_parameters():
        assert parameter.device.type == 'cpu', name
        assert torch.isfinite(parameter).all(), name
    assert .001 < restored.geometry_adapter.queries.std() < .1
    assert .001 < restored.geometry_adapter.null_tokens.std() < .1
    assert restored.geometry_adapter.resampler.in_proj_weight.std() > .01
    result = restored(**tiny_inputs(), use_cache=False)
    assert torch.isfinite(result.logits).all() and torch.isfinite(result.loss)
    result.loss.backward()
    assert restored.geometry_adapter.input_projection[0].weight.grad.abs().sum() > 0
    assert restored.geometry_adapter.queries.grad.abs().sum() > 0
    assert torch.isfinite(restored.geometry_adapter.resampler.in_proj_weight.grad).all()


def test_generate_geometry_is_consumed_only_on_prefill():
    model, batch = tiny_model(), tiny_inputs()
    batch.pop('labels')
    calls = []
    hook = model.geometry_adapter.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=3, min_new_tokens=3,
                do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0)
    finally:
        hook.remove()
    assert generated.shape == (1, batch['input_ids'].shape[1]+3)
    assert len(calls) == 1
    assert len(model.get_input_embeddings()._forward_hooks) == 0


def test_no_features_fail_closed():
    model, batch = tiny_model(), tiny_inputs()
    batch.pop('geometry_features')
    with pytest.raises(ValueError, match='requires explicit'):
        model(**batch)


def test_stale_positions_are_rejected_before_insertion():
    batch = tiny_inputs()
    batch['position_ids'] = torch.zeros(3, 1, batch['input_ids'].shape[1], dtype=torch.long)
    token = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: 5)
    with pytest.raises(ValueError, match='stale position_ids'):
        add_geometry_slots(batch, SimpleNamespace(tokenizer=token),
            batch['geometry_features'], batch['geometry_padding_mask'])


def test_default_collator_uses_online_media_without_dense_cache(monkeypatch):
    import spatial_intelligence.qwen3vl_geometry as implementation
    from spatial_intelligence.qwen35 import Collator
    ids = torch.tensor([[1, 4, 6, 5, 12, 2]])
    raw = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
           'mm_token_type_ids': torch.tensor([[0, 0, 1, 0, 0, 0]])}
    monkeypatch.setattr(Collator, '__call__', lambda self, rows: raw)

    def cache_forbidden(*args):
        raise AssertionError('Default online collator must not read a dense cache')

    monkeypatch.setattr(implementation, 'load_features', cache_forbidden)
    token = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: 5)
    collator = implementation.GeometryCollator(SimpleNamespace(tokenizer=token))
    result = collator([{'media': ['original-0.jpg', 'original-1.jpg']}])
    assert result['geometry_media'] == [['original-0.jpg', 'original-1.jpg']]
    assert 'geometry_features' not in result and 'geometry_padding_mask' not in result
    assert result['geometry_positions'].shape == (1, 64)
