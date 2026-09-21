"""Precision boundary regression: FP32 VGGT residuals into a BF16 interface."""
import pytest
import torch
from spatial_intelligence.geometry_downsample import DownsampleGeometryAdapter, space_to_depth_3x3


@pytest.mark.parametrize('parameter_dtype', [torch.float32, torch.bfloat16])
def test_explicit_interface_precision_without_outer_autocast(parameter_dtype):
    torch.manual_seed(7)
    module = DownsampleGeometryAdapter(4, 8).to(parameter_dtype)
    features = torch.randn(2, 2, 16, 4, dtype=torch.float32, requires_grad=True)
    output = module(features)
    reference = module.layers(space_to_depth_3x3(
        features.to(parameter_dtype).reshape(4, 16, 4))).reshape(2, 8, 8)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert output.dtype == parameter_dtype
    output.float().square().mean().backward()
    assert features.grad.dtype == torch.float32
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())
    assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    restored = DownsampleGeometryAdapter(4, 8).to(parameter_dtype)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(features), output, rtol=0, atol=0)
