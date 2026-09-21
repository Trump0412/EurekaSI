import torch
from torch import nn
import pytest
from spatial_intelligence.geometry_backbone import RegisteredVGGT


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__(); self.calls=[]

    def forward(self,x):
        self.calls.append(tuple(x.shape))
        # Distinct sample/time values detect accidental B/T flattening.
        value=x[:,:,0,0,0,None,None].expand(-1,-1,1025,2048)
        return [value],1


def teacher():
    model=RegisteredVGGT.__new__(RegisteredVGGT)
    nn.Module.__init__(model)
    model.aggregator=FakeAggregator(); model.trainable=False
    return model


def test_grouping_keeps_independent_time_and_original_order():
    m=teacher()
    sequences=[torch.full((t,3,448,448),float(i)) for i,t in enumerate([1,2,1,1])]
    serial=m.forward_batch(sequences,1)
    m.aggregator.calls=[]
    batched=m.forward_batch(sequences,2)
    assert [shape[:2] for shape in m.aggregator.calls]==[(2,1),(1,1),(1,2)]
    for a,b in zip(serial,batched): assert torch.equal(a,b)
    assert all(not x.requires_grad for x in batched)


def test_invalid_batch_rejected_and_trainable_remains_serial():
    m=teacher()
    with pytest.raises(ValueError): m.forward_batch([],0)
    m.trainable=True
    x=torch.ones(1,3,448,448,requires_grad=True)
    outputs=m.forward_batch([x,x],4)
    assert len(m.aggregator.calls)==2
    assert all(y.requires_grad for y in outputs)
