import pytest
import torch

from spatial_intelligence.geometry_tokens import (
    GeometryTokenAdapter, insert_geometry_tokens, replace_geometry_embeddings,
)


def adapter(**kwargs):
    torch.manual_seed(7)
    return GeometryTokenAdapter(input_dim=12, latent_dim=16, output_dim=20,
                                num_heads=4, **kwargs)


def test_64_slots_gradients_and_explicit_frame_order():
    model = adapter(dropout=0.)
    features = torch.randn(2, 3, 5, 12)
    output = model(features)
    assert output.shape == (2, 64, 20)
    (output*torch.randn_like(output)).sum().backward()
    for name, parameter in model.named_parameters():
        if name == 'null_tokens':
            continue
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    model.eval()
    reordered = model(features.flip(1))
    assert not torch.allclose(output, reordered, atol=1e-6, rtol=1e-6)


def test_mask_blocks_invalid_features_and_all_invalid_uses_null():
    model = adapter(dropout=0.).eval()
    features = torch.randn(2, 2, 3, 12)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[0, 1, 1:] = True
    mask[1] = True
    expected = model(features, mask)
    changed = features.clone()
    changed[mask] = 1000*torch.randn_like(changed[mask])
    assert torch.allclose(expected, model(changed, mask), atol=1e-6)
    assert torch.equal(expected[1], model.null_tokens)
    assert torch.isfinite(expected).all()


def test_whole_sample_dropout_training_only_and_forced_intervention(monkeypatch):
    model = adapter(dropout=.2)
    features = torch.randn(2, 2, 3, 12)
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: torch.tensor([.1, .9]))
    trained = model(features)
    assert torch.equal(trained[0], model.null_tokens)
    assert not torch.equal(trained[1], model.null_tokens)
    trained[0].sum().backward()
    assert torch.equal(model.null_tokens.grad, torch.ones_like(model.null_tokens))
    model.eval()

    def random_forbidden(*args, **kwargs):
        raise AssertionError('Eval must not sample geometry dropout')

    monkeypatch.setattr(torch, 'rand', random_forbidden)
    evaluated = model(features)
    assert not torch.equal(evaluated[0], model.null_tokens)
    forced = model(features, force_null=torch.tensor([False, True]))
    assert torch.equal(forced[1], model.null_tokens)
    # Active batch size differs (1 versus 2), allowing ordinary GEMM roundoff.
    assert torch.allclose(forced[0], evaluated[0], atol=1e-6, rtol=1e-6)


def test_adapter_rejects_invalid_shapes_masks_and_nonfinite():
    model = adapter()
    with pytest.raises(ValueError, match='features must'):
        model(torch.randn(2, 3, 12))
    with pytest.raises(ValueError, match='boolean'):
        model(torch.randn(2, 3, 4, 12), torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match='finite'):
        model(torch.full((2, 3, 4, 12), float('nan')))


def test_autocast_and_null_paths_share_compatible_dtype():
    model = adapter(dropout=0.)
    features = torch.randn(2, 2, 3, 12)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(features, force_null=torch.tensor([False, True]))
    assert torch.isfinite(output).all()
    assert output.dtype == model.null_tokens.dtype
    output.square().mean().backward()
    assert model.input_projection[0].weight.grad is not None


@pytest.mark.parametrize('null_mode', ['random_all_drop', 'all_masked', 'forced_null', 'none'])
def test_all_parameters_remain_in_graph_for_ddp_microbatch_one(null_mode):
    model = adapter(dropout=1. if null_mode == 'random_all_drop' else 0.)
    features = torch.randn(1, 2, 3, 12)
    mask = torch.full((1, 2, 3), null_mode == 'all_masked', dtype=torch.bool)
    forced = torch.tensor([null_mode == 'forced_null'])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(features, mask, force_null=forced)
    assert torch.isfinite(output).all()
    if null_mode != 'none':
        assert torch.equal(output[0], model.null_tokens)
    output.sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if null_mode != 'none' and name != 'null_tokens':
            assert (parameter.grad == 0).all(), name
    if null_mode == 'none':
        assert (model.null_tokens.grad == 0).all()


def test_insert_after_last_visual_boundary_padding_labels_and_mapping():
    ids = torch.tensor([[0, 7, 9, 11, 9, 21, 0], [8, 9, 22, 0, 0, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0]])
    labels = ids.clone()
    types = torch.tensor([[0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
    inserted = insert_geometry_tokens(ids, mask, labels,
        vision_end_token_id=9, placeholder_token_id=2, mm_token_type_ids=types)
    assert inserted.input_ids.shape == (2, 69)
    assert inserted.geometry_positions[0].tolist() == list(range(4, 68))
    assert inserted.geometry_positions[1].tolist() == list(range(2, 66))
    assert inserted.attention_mask.sum(1).tolist() == [69, 67]
    assert (inserted.labels.gather(1, inserted.geometry_positions) == -100).all()
    assert (inserted.labels[~inserted.attention_mask.bool()] == -100).all()
    assert inserted.geometry_mask.sum(1).tolist() == [64, 64]
    assert (inserted.mm_token_type_ids[inserted.geometry_mask] == 0).all()
    for batch in range(2):
        old = mask[batch].bool()
        positions = inserted.original_positions[batch, old]
        assert torch.equal(inserted.input_ids[batch, positions], ids[batch, old])
        assert torch.equal(inserted.labels[batch, positions], labels[batch, old])
        assert torch.equal(inserted.mm_token_type_ids[batch, positions], types[batch, old])
        assert (inserted.original_positions[batch, ~old] == -1).all()
        # Ordered native visual identity survives; a caller can map its original
        # DeepStack mask without inserting any geometry slots into that mask.
        native = old & ((ids[batch] == 7) | (ids[batch] == 8) | (ids[batch] == 11))
        new_native = inserted.original_positions[batch, native]
        assert torch.equal(inserted.input_ids[batch, new_native], ids[batch, native])
        assert not torch.isin(new_native, inserted.geometry_positions[batch]).any()


def test_embedding_replacement_gradients_preserve_non_geometry_positions():
    token = torch.randn(2, 10, 6, requires_grad=True)
    geometry = torch.randn(2, 3, 6, requires_grad=True)
    positions = torch.tensor([[3, 4, 5], [1, 2, 3]])
    result = replace_geometry_embeddings(token, geometry, positions)
    for batch in range(2):
        assert torch.equal(result[batch, positions[batch]], geometry[batch])
        keep = torch.ones(10, dtype=torch.bool)
        keep[positions[batch]] = False
        assert torch.equal(result[batch, keep], token[batch, keep])
    result.sum().backward()
    assert torch.equal(geometry.grad, torch.ones_like(geometry))
    for batch in range(2):
        assert (token.grad[batch, positions[batch]] == 0).all()


def test_insertion_and_replacement_reject_ambiguous_inputs():
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='no valid vision_end'):
        insert_geometry_tokens(ids, torch.ones_like(ids), vision_end_token_id=9, placeholder_token_id=3)
    with pytest.raises(ValueError, match='unique'):
        replace_geometry_embeddings(torch.zeros(1, 3, 4), torch.zeros(1, 2, 4), torch.tensor([[1, 1]]))
    with pytest.raises(ValueError, match='bounds'):
        replace_geometry_embeddings(torch.zeros(1, 3, 4), torch.zeros(1, 1, 4), torch.tensor([[3]]))
