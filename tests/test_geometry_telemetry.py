"""Probe selection must not mistake BF16 LayerNorm rounding for no training."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    'geometry_stage_telemetry', Path(__file__).resolve().parents[1] / 'scripts/train-geometry-stage.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_adapter_skips_unit_scale_vector_and_selects_matrix():
    vector = SimpleNamespace(requires_grad=True, shape=(16,))
    matrix = SimpleNamespace(requires_grad=True, shape=(16, 16))
    assert not stage.update_probe_candidate('geometry_adapter.norm.weight', vector, 'geometry_adapter')
    assert stage.update_probe_candidate('geometry_adapter.linear.weight', matrix, 'geometry_adapter')


def test_partitioned_matrix_uses_original_shape():
    shard = SimpleNamespace(requires_grad=True, shape=(0,), ds_shape=(16, 16))
    assert stage.update_probe_candidate('geometry_adapter.linear.weight', shard, 'geometry_adapter')


def test_frozen_unused_and_non_qkv_weights_are_not_candidates():
    parameter = SimpleNamespace(requires_grad=False, shape=(16, 16))
    assert not stage.update_probe_candidate('geometry_adapter.linear.weight', parameter, 'geometry_adapter')
    parameter.requires_grad = True
    assert not stage.update_probe_candidate('geometry_adapter.null_tokens', parameter, 'geometry_adapter')
    assert not stage.update_probe_candidate('geometry_backbone.camera_token', parameter, 'geometry_backbone')
    assert stage.update_probe_candidate('geometry_backbone.block.qkv.weight', parameter, 'geometry_backbone')
