import pytest
torch = pytest.importorskip('torch')
from spatial_intelligence.geometry_downsample import (
    DownsampleGeometryAdapter, downsample_token_count, space_to_depth_3x3,
)


def test_spatial_channel_order_and_padding():
    x = torch.arange(4 * 4 * 2).reshape(1, 16, 2).float()
    actual = space_to_depth_3x3(x)
    grid = torch.nn.functional.pad(x.reshape(4, 4, 2), (0, 0, 0, 2, 0, 2))
    expected = torch.stack([grid[r:r+3, c:c+3].reshape(-1)
                            for r in (0, 3) for c in (0, 3)])[None]
    torch.testing.assert_close(actual, expected)
    assert downsample_token_count() == 121
    with pytest.raises(ValueError):
        downsample_token_count(15)


def test_adapter_grad_null_padding_and_reload():
    torch.manual_seed(3)
    adapter = DownsampleGeometryAdapter(4, 8)
    x = torch.randn(2, 3, 16, 4, requires_grad=True)
    mask = torch.zeros(2, 3, 16, dtype=torch.bool)
    mask[1, 2] = True
    out = adapter(x, mask)
    assert out.shape == (2, 12, 8)
    assert out[1, -4:].count_nonzero() == 0
    out.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in adapter.parameters())
    reloaded = DownsampleGeometryAdapter(4, 8)
    reloaded.load_state_dict(adapter.state_dict(), strict=True)
    torch.testing.assert_close(out, reloaded(x, mask))
    assert adapter(x, mask, force_null=True).count_nonzero() == 0
