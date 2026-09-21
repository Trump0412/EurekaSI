"""Routing diagnostics, not proof of a downstream benchmark improvement."""
import math

import pytest
import torch

from spatial_intelligence.georoute import GeoRoute, SparseTransportBlock, build_graph


def simple_graph():
    return build_graph([0,0,1,1],[0,1],[2,3],[1.,1.])


def test_default_matches_original_arithmetic_exactly_and_telemetry_is_detached():
    torch.manual_seed(13)
    graph=simple_graph();block=SparseTransportBlock(8,4,alpha_init=.2)
    hidden=torch.randn(4,8,requires_grad=True)
    z=block.down(block.norm(hidden));messages=torch.zeros_like(z)
    messages.index_add_(0,graph.dst,z[graph.src]*graph.weight.to(z.dtype)[:,None])
    connected=torch.tensor([False,False,True,True])
    expected=torch.where(connected[:,None],hidden+block.alpha*block.up(torch.nn.functional.silu(messages)),hidden)
    records=[]
    actual=block(hidden,graph,telemetry=records.append)
    assert torch.equal(expected,actual)
    assert actual.requires_grad
    assert len(records)==1 and not any(isinstance(value,torch.Tensor) for value in records[0].values())
    measured=records[0]
    assert measured['connected_nodes']==2 and measured['total_nodes']==4
    delta=actual.detach()[2:]-hidden.detach()[2:]
    assert measured['applied_delta_relative_l2']==pytest.approx(float(delta.norm()/hidden.detach()[2:].norm()))
    actual.sum().backward()
    assert hidden.grad is not None


def test_single_block_strength_sweep_and_zero_identity():
    torch.manual_seed(14)
    graph=simple_graph();block=SparseTransportBlock(8,4,alpha_init=.3)
    hidden=torch.randn(4,8)
    baseline=block(hidden,graph)
    for strength in (0.,.5,1.,2.):
        actual=block(hidden,graph,strength=strength)
        torch.testing.assert_close(actual-hidden,(baseline-hidden)*strength,atol=2e-7,rtol=2e-5)
        assert torch.equal(actual[:2],hidden[:2])
    assert torch.equal(block(hidden,graph,strength=0.),hidden)


def test_zero_alpha_first_learns_gate_then_projection_gets_gradient():
    graph=simple_graph();block=SparseTransportBlock(4,2)
    with torch.no_grad():
        block.down.weight.fill_(.2);block.up.weight.fill_(.3)
    hidden=torch.ones(4,4)
    target=torch.full((2,4),2.)
    first=block(hidden,graph)
    assert torch.equal(first,hidden) and block.alpha.item()==0.
    (first[2:]-target).square().mean().backward()
    assert block.alpha.grad.abs()>0
    assert block.down.weight.grad.abs().sum()==0
    assert block.up.weight.grad.abs().sum()==0
    with torch.no_grad(): block.alpha.add_(block.alpha.grad,alpha=-.1)
    block.zero_grad(set_to_none=True)
    second=block(hidden,graph)
    assert not torch.equal(second[2:],hidden[2:])
    (second[2:]-target).square().mean().backward()
    assert block.down.weight.grad.abs().sum()>0
    assert block.up.weight.grad.abs().sum()>0


def test_route_context_labels_each_block_restores_and_does_not_change_checkpoint():
    graph=simple_graph();route=GeoRoute(4,{'bottleneck':2,'alpha_init':.2})
    hidden=torch.randn(4,4);keys=set(route.state_dict());records=[]
    original=route.route(2,hidden,graph)
    with route.routing_diagnostics(strength=0.,telemetry=records.append):
        assert torch.equal(route.route(2,hidden,graph),hidden)
        with route.routing_diagnostics(strength=1.):
            assert torch.equal(route.route(2,hidden,graph),original)
        assert torch.equal(route.route(2,hidden,graph),hidden)
    assert torch.equal(route.route(2,hidden,graph),original)
    assert set(route.state_dict())==keys
    assert [(row['exit_index'],row['block_index']) for row in records]==[(2,0),(2,1)]*2
    assert all(row['applied_delta_relative_l2']==0. for row in records)
    with pytest.raises(RuntimeError):
        with route.routing_diagnostics(strength=2.): raise RuntimeError('test restoration')
    assert route._diagnostic_strength==1. and route._diagnostic_telemetry is None


def test_empty_support_and_zero_hidden_have_explicit_undefined_ratios():
    block=SparseTransportBlock(4,2)
    records=[]
    graph=build_graph([0,0],[],[],[])
    hidden=torch.ones(2,4)
    assert torch.equal(block(hidden,graph,telemetry=records.append),hidden)
    assert records[0]['connected_nodes']==0
    assert records[0]['applied_delta_relative_l2'] is None
    assert records[0]['max_node_delta_ratio'] is None
    block(torch.zeros(4,4),simple_graph(),telemetry=records.append)
    assert records[-1]['connected_nodes']==2 and records[-1]['nonzero_hidden_nodes']==0
    assert records[-1]['applied_delta_relative_l2'] is None


@pytest.mark.parametrize('strength', [-1.,math.inf,math.nan,True,'2'])
def test_invalid_strength_rejected(strength):
    block=SparseTransportBlock(4,2)
    with pytest.raises(ValueError,match='finite nonnegative'):
        block(torch.ones(4,4),simple_graph(),strength=strength)


def test_invalid_sink_rejected():
    block=SparseTransportBlock(4,2)
    with pytest.raises(ValueError,match='callable'):
        block(torch.ones(4,4),simple_graph(),telemetry=[])
