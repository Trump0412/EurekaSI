from types import SimpleNamespace
import torch
from spatial_intelligence.georoute import aggregate_transport,GeoRoute


def test_weighted_sum_and_gradients_match_reference():
    graph=SimpleNamespace(src=torch.tensor([0,1,2,0,3]),dst=torch.tensor([2,2,3,3,2]),weight=torch.tensor([.2,.3,.5,.5,.5],dtype=torch.float64))
    z=torch.randn(5,7,dtype=torch.float64,requires_grad=True)
    original=aggregate_transport(z,graph)
    actual=aggregate_transport(z,graph,'segment_fp32_v1')
    torch.testing.assert_close(actual,original,rtol=1e-12,atol=1e-12)
    a=torch.autograd.grad(actual.square().sum(),z,retain_graph=True)[0]
    b=torch.autograd.grad(original.square().sum(),z)[0]
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert torch.equal(actual[4],torch.zeros_like(actual[4]))


def test_empty_support_and_legacy_config():
    graph=SimpleNamespace(src=torch.tensor([],dtype=torch.long),dst=torch.tensor([],dtype=torch.long),weight=torch.tensor([]))
    z=torch.randn(3,4,requires_grad=True)
    out=aggregate_transport(z,graph,'segment_fp32_v1');out.sum().backward()
    assert torch.equal(out,torch.zeros_like(out))
    assert torch.equal(z.grad,torch.zeros_like(z))
    assert GeoRoute(8,{'active_exits':[3]}).routes[3][0].transport_reduction=='legacy_atomic'
    assert GeoRoute(8,{'active_exits':[3],'transport_reduction':'segment_fp32_v1'}).routes[3][0].transport_reduction=='segment_fp32_v1'
