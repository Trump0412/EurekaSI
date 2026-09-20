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


def test_large_downsample_matrix_probe_does_not_round_endpoint_out_of_bounds():
    import torch
    size=6144*18432
    broken=torch.linspace(0,size-1,1024).long()
    assert broken[-1].item()==size, 'Reproduce original FP32 telemetry failure'
    indices=stage.update_probe_indices(size)
    assert indices.dtype==torch.long
    assert indices[0].item()==0 and indices[-1].item()==size-1
    assert bool((indices[1:]>indices[:-1]).all())


def test_probe_indices_small_and_partitioned_large_sizes():
    import pytest
    for size in (1,7,1024,2**24+1,2**28,2**31):
        indices=stage.update_probe_indices(size)
        assert len(indices)==min(1024,size)
        assert indices.min().item()==0
        assert indices.max().item()==size-1
    with pytest.raises(ValueError): stage.update_probe_indices(0)
