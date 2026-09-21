import pytest
import torch
from spatial_intelligence.georoute import (
    build_graph,batch_graphs,patch_layout,tracks_to_graph,SparseTransportBlock,
    GeoRoute,make_tip_intervention,
)


def graph():
    return build_graph([0,0,0,0,1,1,1,1],[0,1],[4,5],[1.,1.])


def test_graph_deduplicate_topk_normalize_and_empty_identity():
    g=build_graph([0,0,0,1],[0,0,1,2,3],[3,3,3,3,3],[.3,.6,.4,.2,10.],topk=2)
    assert g.src.tolist()==[0,1] and g.dst.tolist()==[3,3]
    assert torch.allclose(g.weight,torch.tensor([.6,.4]))
    empty=build_graph([0,0],[0],[1],[1.])
    block=SparseTransportBlock(8,4,alpha_init=1.)
    x=torch.randn(2,8)
    assert torch.equal(block(x,empty),x)
    with pytest.raises(ValueError,match='Cross-sample'):
        build_graph([0,1],[0],[1],[1.],sample_ids=[0,1])


def test_zero_indegree_source_identity_twohop_and_graph_batch():
    g=build_graph([0,1,2],[0,1],[1,2],[1.,1.])
    a=SparseTransportBlock(4,4,alpha_init=1.)
    x=torch.randn(3,4,requires_grad=True)
    once=a(x,g);twice=a(once,g)
    assert torch.equal(once[0],x[0]) and torch.equal(twice[0],x[0])
    # A two-block destination2 can depend on source0 through node1.
    grad=torch.autograd.grad(twice[2].sum(),x)[0]
    assert grad[0].abs().sum()>0
    combined=batch_graphs([g,g])
    assert combined.src.tolist()==[0,1,3,4]
    assert combined.dst.tolist()==[1,2,4,5]
    assert torch.equal(a(torch.cat([x.detach(),x.detach()]),combined),torch.cat([once.detach(),once.detach()]))
    nested=batch_graphs([combined,g])
    assert nested.sample_ids.tolist()==[0]*3+[1]*3+[2]*3


def test_actual_qwen_merge_group_order_and_track_quantization():
    layout=patch_layout([(4,4),(4,4)],patch_size=16)
    assert layout['raster_to_node'][0].tolist()==[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
    assert layout['centers'][:4].tolist()==[[8.,8.],[24.,8.],[8.,24.],[24.,24.]]
    g=tracks_to_graph([0,1,2,3],[1,1,1,1],[[40,8],[64,8],[8,24],[8,8]],[1,1,.1,1],[1,1,1,.1],[(4,4),(4,4)])
    assert g.src.tolist()==[0] and g.dst.tolist()==[20]


def test_tip_invalid_neighbors_rejected_and_routing_update():
    torch.manual_seed(3);g=graph()
    mask,bad=make_tip_intervention(g,generator=torch.Generator().manual_seed(3))
    route=GeoRoute(8,dict(bottleneck=4,alpha_init=.1))
    clean=[torch.randn(8,8,requires_grad=True) for _ in range(4)]
    loss,terms=route.tip_loss(clean,g,mask,bad)
    assert torch.isfinite(loss) and set(terms)=={'reconstruction','substitution','preservation'}
    loss.backward()
    assert all(x.grad is None for x in clean)
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in r.parameters()) for r in route.routes)
    with pytest.raises(ValueError,match='non-neighbor'):
        route.tip_loss(clean,g,mask,g)
    with pytest.raises(ValueError,match='No supported'):
        make_tip_intervention(build_graph([0,0],[],[],[]))
