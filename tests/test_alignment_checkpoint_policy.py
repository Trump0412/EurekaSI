"""Pure policy tests: do not claim actual Qwen checkpoint parity or GPU fit."""
from types import SimpleNamespace

import pytest

from spatial_intelligence.alignment_checkpoint_policy import apply_alignment_checkpoint_policy


def fake_model():
    parameters = [SimpleNamespace(requires_grad=False), SimpleNamespace(requires_grad=False)]
    language = SimpleNamespace(
        layers=[SimpleNamespace(gradient_checkpointing=True, training=True) for _ in range(3)],
        parameters=lambda: iter(parameters),
    )
    return SimpleNamespace(_matrix_stage='align', training=True,
                           model=SimpleNamespace(language_model=language)), parameters


def flags(model):
    return [layer.gradient_checkpointing for layer in model.model.language_model.layers]


def test_default_is_strict_noop_even_for_other_models():
    assert apply_alignment_checkpoint_policy(object(), None) is None


def test_short_boundary_and_long_restore_without_changing_ownership():
    model, parameters = fake_model()
    assert apply_alignment_checkpoint_policy(model, 128, 256) is False
    assert flags(model) == [False] * 3
    assert apply_alignment_checkpoint_policy(model, 256, 256) is False
    assert apply_alignment_checkpoint_policy(model, 257, 256) is True
    assert flags(model) == [True] * 3
    assert all(not parameter.requires_grad for parameter in parameters)
    assert all(layer.training for layer in model.model.language_model.layers)


@pytest.mark.parametrize('threshold', [-1, True, 1.5, '128'])
def test_invalid_threshold(threshold):
    model, _ = fake_model()
    with pytest.raises(ValueError):
        apply_alignment_checkpoint_policy(model, 128, threshold)
    assert flags(model) == [True] * 3


@pytest.mark.parametrize('width', [0, -1, True, 1.5, None])
def test_invalid_width(width):
    model, _ = fake_model()
    with pytest.raises(ValueError):
        apply_alignment_checkpoint_policy(model, width, 256)


@pytest.mark.parametrize('stage', ['sft', 'eval', None])
def test_never_applies_outside_alignment(stage):
    model, _ = fake_model()
    model._matrix_stage = stage
    with pytest.raises(ValueError, match='restricted to alignment'):
        apply_alignment_checkpoint_policy(model, 128, 256)
    assert flags(model) == [True] * 3


def test_trainable_language_rejected():
    model, parameters = fake_model()
    parameters[-1].requires_grad = True
    with pytest.raises(ValueError, match='must be frozen'):
        apply_alignment_checkpoint_policy(model, 128, 256)
    assert flags(model) == [True] * 3


def test_unknown_layer_api_rejected_atomically():
    model, _ = fake_model()
    del model.model.language_model.layers[-1].gradient_checkpointing
    with pytest.raises(ValueError, match='per-layer'):
        apply_alignment_checkpoint_policy(model, 128, 256)
    assert model.model.language_model.layers[0].gradient_checkpointing


@pytest.mark.parametrize('target', ['model', 'layer'])
def test_eval_mode_rejected(target):
    model, _ = fake_model()
    if target == 'model':
        model.training = False
    else:
        model.model.language_model.layers[-1].training = False
    with pytest.raises(ValueError, match='training mode'):
        apply_alignment_checkpoint_policy(model, 128, 256)
    assert flags(model) == [True] * 3


def test_missing_decoder_rejected():
    model, _ = fake_model()
    model.model.language_model.layers = []
    with pytest.raises(ValueError, match='nonempty'):
        apply_alignment_checkpoint_policy(model, 128, 256)
