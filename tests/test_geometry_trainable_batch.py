"""Deterministic opt-in batching, gradients, ownership, and rank call order."""
import copy
import pytest
import torch
from torch import nn
from spatial_intelligence.geometry_backbone import RegisteredVGGT, synchronized_sequence_chunks


class SmallTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2)
        self.calls = []

    def forward(self, images):
        self.calls.append(tuple(images.shape[:2]))
        value = self.proj(images.mean((-1, -2)))
        return [value.repeat_interleave(1024, -1).unsqueeze(2).expand(-1, -1, 1025, -1)], 1


def model():
    result = RegisteredVGGT.__new__(RegisteredVGGT)
    nn.Module.__init__(result)
    result.aggregator = SmallTrunk()
    result.trainable = True
    result.trainable_batching = True
    return result


def test_shared_rank_plan_preserves_order_and_identical_call_count():
    assert synchronized_sequence_chunks([[1,1,2,2], [1,2,2,2]], 4) == [[0],[1],[2,3]]
    assert synchronized_sequence_chunks([[1,1,2,2], [8,8,32,32]], 4) == [[0,1],[2,3]]
    assert synchronized_sequence_chunks([[1,1,1,1]], 2) == [[0,1],[2,3]]
    with pytest.raises(ValueError):
        synchronized_sequence_chunks([[1], [1,2]], 2)


def test_batched_gradient_and_optimizer_resume_equivalence():
    torch.manual_seed(11)
    serial = model()
    batched = copy.deepcopy(serial)
    images = [torch.randn(1,3,448,448), torch.randn(1,3,448,448)]
    opts = [torch.optim.AdamW(m.parameters(), lr=.001) for m in (serial, batched)]
    for m, opt, size in zip((serial, batched), opts, (1,2)):
        result = m.forward_batch(images, size)
        loss = sum((index+1)*x.square().mean() for index,x in enumerate(result))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert serial.aggregator.calls == [(1,1),(1,1)]
    assert batched.aggregator.calls == [(2,1)]
    for a,b in zip(serial.parameters(), batched.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    saved = copy.deepcopy({'model': batched.state_dict(), 'optimizer': opts[1].state_dict()})
    restored = model(); restored.load_state_dict(saved['model'])
    restored_opt = torch.optim.AdamW(restored.parameters(),lr=.001)
    restored_opt.load_state_dict(saved['optimizer'])
    for m,opt in ((batched,opts[1]),(restored,restored_opt)):
        opt.zero_grad()
        sum(x.square().mean() for x in m.forward_batch(images,2)).backward()
        opt.step()
    for a,b in zip(batched.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_stochastic_trunk_is_not_silently_changed():
    m=model(); m.aggregator.dropout=nn.Dropout(.1)
    with pytest.raises(ValueError,match='zero stochastic'):
        m.forward_batch([torch.zeros(1,3,448,448)]*2,2)


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(2)
    torch.distributed.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        for trainable in (False, True):
            m=model(); m.trainable=trainable
            lengths=[1,1] if rank==0 else [1,2]
            images=[torch.ones(t,3,448,448) for t in lengths]
            values=m.forward_batch(images,2)
            assert m.aggregator.calls==[(1,t) for t in lengths]
            if trainable:
                sum(x.square().mean() for x in values).backward()
                for p in m.parameters():
                    assert p.grad is not None
                    torch.distributed.all_reduce(p.grad)
                    assert torch.isfinite(p.grad).all()
            else:
                assert not any(x.requires_grad for x in values)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_actual_collective_plan_for_frozen_and_trainable(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker,
        args=((tmp_path/'rendezvous').resolve().as_uri(),), nprocs=2, join=True)
