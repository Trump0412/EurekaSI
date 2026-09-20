from collections import Counter
import pytest
from spatial_intelligence.throughput import balanced_order,row_cost


@pytest.mark.parametrize('micro',[1,2,4,8,16])
def test_same_effective_batches_and_tail(micro):
    ids=list(range(139));costs=[(i*37)%103+1 for i in ids]
    result=balanced_order(ids,costs,micro_batch=micro,world_size=4,accumulation=16//micro)
    assert result==balanced_order(ids,costs,micro_batch=micro,world_size=4,accumulation=16//micro)
    for start in (0,64,128):assert Counter(result[start:start+64])==Counter(ids[start:start+64])
    assert result[128:]==ids[128:]


def test_bucketing_reduces_padding_proxy():
    costs=[1,100]*64;ids=list(range(128))
    result=balanced_order(ids,costs,micro_batch=2,world_size=4,accumulation=8)
    padded=lambda order:sum(max(costs[i] for i in order[s:s+2])*2 for s in range(64,128,2))
    assert padded(result)==sum(costs[64:])<padded(ids)


def test_cost_is_only_metadata_proxy():
    assert row_cost({'media':['missing.jpg'],'question':'abcd','answer':'efgh'})==258


def test_accelerate_sharding_preserves_optimizer_step_members():
    pytest.importorskip('torch');pytest.importorskip('accelerate')
    from torch.utils.data import BatchSampler
    from accelerate.data_loader import SeedableRandomSampler,BatchSamplerShard
    from spatial_intelligence.throughput_sampler import EffectiveBatchSampler
    dataset=list(range(193));costs=[i%23+1 for i in dataset]
    for epoch in (0,1):
        base=SeedableRandomSampler(dataset,data_seed=3407);base.set_epoch(epoch);expected=list(base)
        for micro in (1,2,4):
            ga=16//micro
            ranks=[]
            for rank in range(4):
                sampler=EffectiveBatchSampler(dataset,costs,seed=3407,micro_batch=micro,world_size=4,accumulation=ga)
                sampler.set_epoch(epoch)
                ranks.append(list(BatchSamplerShard(BatchSampler(sampler,micro,False),num_processes=4,process_index=rank)))
            for step in range(3):
                actual=[i for rank in ranks for b in rank[step*ga:(step+1)*ga] for i in b]
                assert Counter(actual)==Counter(expected[step*64:(step+1)*64])
            baseline_ranks=[]
            for rank in range(4):
                sampler=SeedableRandomSampler(dataset,data_seed=3407);sampler.set_epoch(epoch)
                baseline_ranks.append(list(BatchSamplerShard(BatchSampler(sampler,micro,False),num_processes=4,process_index=rank)))
            tail=lambda values:Counter(i for rank in values for b in rank[3*ga:] for i in b)
            assert tail(ranks)==tail(baseline_ranks)


def test_sample_mean_loss_matches_accumulated_examples():
    torch=pytest.importorskip('torch')
    from spatial_intelligence.qwen35 import completion_loss
    from types import SimpleNamespace
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.randn(5,7))
        def forward(self,input_ids,logits_to_keep,**kw):
            return SimpleNamespace(logits=self.weight[input_ids][:,logits_to_keep])
    torch.manual_seed(9);model=Toy()
    ids=torch.tensor([[0,1,2,3],[1,2,3,4]]);labels=torch.tensor([[-100,-100,1,2],[-100,3,4,5]])
    each=sum(completion_loss(model,{'input_ids':ids[i:i+1],'labels':labels[i:i+1]})[0] for i in range(2))/2
    expected=torch.autograd.grad(each,model.weight)[0]
    batch=completion_loss(model,{'input_ids':ids,'labels':labels},reduction='sample_mean')[0]
    actual=torch.autograd.grad(batch,model.weight)[0]
    torch.testing.assert_close(batch,each);torch.testing.assert_close(actual,expected)


def test_controlled_stop_is_diagnostic_only():
    pytest.importorskip('torch');pytest.importorskip('transformers')
    from pathlib import Path
    from spatial_intelligence.study import train
    with pytest.raises(ValueError,match='requires --diagnostic'):
        train(Path('missing'),'unused','formal',stop_after_steps=2)
    with pytest.raises(ValueError,match='strictly inside'):
        train(Path('missing'),'unused','diagnostic-test',diagnostic=True,max_steps=3,stop_after_steps=3)
