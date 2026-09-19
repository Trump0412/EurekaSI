import torch
import pytest
from spatial_intelligence.backends.qwen35 import Qwen35Backend


def backend():
    obj=object.__new__(Qwen35Backend)
    obj.device='cpu'; obj.max_context=32
    return obj


def test_response_alignment_and_visual_types():
    b=backend()
    batch={'input_ids':torch.tensor([[1,2,3]]), 'attention_mask':torch.tensor([[0,1,1]]),
           'mm_token_type_ids':torch.tensor([[0,1,0]]), 'pixel_values':torch.ones(2,4)}
    inp,pos=b.response_inputs(batch,torch.tensor([[7,8]]))
    assert pos.tolist()==[2,3]
    assert inp['attention_mask'].tolist()==[[0,1,1,1,1]]
    assert inp['mm_token_type_ids'].tolist()==[[0,1,0,0,0]]
    assert inp['pixel_values'] is batch['pixel_values']
    assert batch['input_ids'].shape[1]==3


def test_teacher_prompt_length_has_independent_positions():
    b=backend()
    for length in [3,9]:
        inp,pos=b.response_inputs({'input_ids':torch.ones(1,length,dtype=torch.long),
                                  'attention_mask':torch.ones(1,length,dtype=torch.long)},torch.tensor([[7,8]]))
        assert pos.tolist()==[length-1,length]
    with pytest.raises(ValueError):
        b.response_inputs({'input_ids':torch.ones(1,31,dtype=torch.long)},torch.tensor([[7,8]]))
