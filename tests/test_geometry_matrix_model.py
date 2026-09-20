"""Variable geometry insertion: native media and answer labels must survive."""
from types import SimpleNamespace
import torch
import pytest
from spatial_intelligence.qwen3vl_geometry_matrix import insert_slots


def processor():
    return SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0,convert_tokens_to_ids=lambda _:5))


def batch():
    ids=torch.tensor([[1,4,6,5,4,6,5,9,10],[1,4,6,5,9,10,0,0,0]])
    return dict(input_ids=ids,attention_mask=(ids!=0).long(),
        mm_token_type_ids=(ids==6).long(),labels=torch.where(ids==10,ids,-100))


@pytest.mark.parametrize('adapter',['query64','downsample'])
def test_variable_slots_preserve_native_and_labels(adapter):
    original=batch(); value=insert_slots(original,processor(),[2,1],adapter)
    for b,frames in enumerate([2,1]):
        pos=value['geometry_positions'][b];pos=pos[pos>=0]
        assert len(pos)==(64 if adapter=='query64' else frames*121)
        keep=value['attention_mask'][b].bool().clone();keep[pos]=False
        assert torch.equal(value['input_ids'][b,keep],original['input_ids'][b,original['attention_mask'][b].bool()])
        assert value['mm_token_type_ids'][b].sum()==frames
        assert (value['labels'][b,pos]==-100).all()
        assert (value['labels'][b]==10).sum()==1


def test_downsample_temporal_groups_pair_two_frames():
    original={k:v[1:,:6] for k,v in batch().items()}
    value=insert_slots(original,processor(),[2],'downsample')
    assert value['geometry_positions'].shape==(1,242)


def test_context_limit_is_not_silent_frame_truncation():
    with pytest.raises(ValueError,match='no silent truncation'):
        insert_slots(batch(),processor(),[2,1],'downsample',max_context=20)


def test_incompatible_frame_boundaries_rejected():
    with pytest.raises(ValueError,match='cannot be paired'):
        insert_slots(batch(),processor(),[3,1],'downsample')
