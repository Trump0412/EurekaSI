import sys
import types
import pytest
torch = pytest.importorskip('torch')
from spatial_intelligence.geometry_backbone import RegisteredVGGT


class FakeAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.depth = 1
        self.cached_layer_indices = {0}
        self.register_buffer('_resnet_mean', torch.zeros(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('_resnet_std', torch.zeros(1, 1, 3, 1, 1), persistent=False)

    def forward(self, x):
        value = x.mean() * self.weight
        return [value.expand(1, x.shape[1], 1025, 2048)], 1


def test_registration_freezing_and_gradient(monkeypatch, tmp_path):
    fake = types.ModuleType('vggt.models.aggregator')
    fake.Aggregator = FakeAggregator
    monkeypatch.setitem(sys.modules, 'vggt.models.aggregator', fake)
    model = RegisteredVGGT(tmp_path, trainable=True)
    assert 'aggregator.weight' in model.state_dict()
    x = torch.ones(1, 3, 448, 448)
    model(x).mean().backward()
    assert model.aggregator.weight.grad != 0
    model.set_trainable(False).train()
    assert not model.aggregator.training
    assert not model(x).requires_grad
    model.set_trainable(True).train()
    assert model.aggregator.training
    restored = RegisteredVGGT(tmp_path, trainable=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), model(x))
    restored.restore_preprocessing_buffers()
    torch.testing.assert_close(restored.aggregator._resnet_mean.flatten(), torch.tensor([.485,.456,.406]))
    torch.testing.assert_close(restored.aggregator._resnet_std.flatten(), torch.tensor([.229,.224,.225]))
    assert '_resnet_mean' not in restored.aggregator.state_dict()
    restored.bfloat16()
    assert restored.aggregator.weight.dtype == torch.bfloat16
    assert restored.aggregator._resnet_mean.dtype == torch.float32
    torch.testing.assert_close(restored.aggregator._resnet_std.flatten(), torch.tensor([.229,.224,.225]))
