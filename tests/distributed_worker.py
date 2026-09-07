"""Called only by the two-process contract test."""
import torch
from spatial_intelligence.distributed import Distributed

context=Distributed({'model':{'device':'cpu'},'teacher':{'device':'cpu'}})
try:
    p=torch.nn.Parameter(torch.tensor(2.));q=torch.nn.Parameter(torch.tensor(3.));unused=torch.nn.Parameter(torch.tensor(4.))
    loss=p if context.rank==0 else q
    loss.backward();context.average_gradients([p,q,unused])
    assert p.grad.item()==.5 and q.grad.item()==.5 and unused.grad is None
finally:context.close()
