import torch
from spatial_intelligence.backends.fusion import GeometryFusion


def test_zero_gate_parity_then_geometry_gradient():
    torch.manual_seed(0);m=GeometryFusion(16);text=torch.randn(1,5,16);geo=torch.randn(1,8,7)
    assert torch.equal(m(text,geo),text)
    m(text,geo).square().sum().backward();assert m.gate.grad.abs()>0
    with torch.no_grad():m.gate.fill_(0.5)
    m.zero_grad();m(text,geo).square().sum().backward()
    assert m.project[0].weight.grad.norm()>0
    assert not torch.equal(m(text,geo),m(text,geo+1))
