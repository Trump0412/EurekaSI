import pytest
from spatial_intelligence.geometry_rft_reward import score_response,RewardConfig,parse_independent_final_tag

CHOICES={'A':'left','B':'right'}

def score(text,gold='B',**kwargs):
    return score_response(text,gold,choices=CHOICES,config=RewardConfig(version='geopsro-final-answer-v4'),**kwargs)

def test_correctness_independent_of_think_wrapper_not_format_bonus():
    text='Spatial Observation: Object is on the right.\n<answer>B</answer>'
    result=score(text)
    assert result['answer']==1 and result['total']==1
    assert result['structure']==result['words']==0
    assert score(text,'A')['total']==0
    assert parse_independent_final_tag(text,'mcq',CHOICES)=='B'
    legacy=score_response(text,'B',choices=CHOICES,config=RewardConfig(version='geopsro-independent-answer-v3'))
    assert legacy['total']==0  # No silent redefinition of previous experiments.

@pytest.mark.parametrize('text',[
    '<answer>A</answer><answer>B</answer>',
    '<think>unfinished <answer>B</answer>',
    'reason </think><answer>B</answer>',
    'reason <answer>B</answer> trailing claim',
    'reason <answer>B',
    'reason </answer><answer>B</answer>',
    'The final answer is A.\n<answer>B</answer>',
    'The correct choice is A.\n<answer>B</answer>',
    'reason <answer>B. right</answer>',
    'reason <answer>Z</answer>',
])
def test_ambiguous_or_malformed_final_is_not_repaired(text):
    assert score(text)['total']==0

def test_truncation_and_numeric_tolerance_preserved():
    assert score('Reason. <answer>B</answer>',truncated=True)['total']==0
    result=score_response('Explanation. <answer>10</answer>','10',task_type='numeric',
        config=RewardConfig(version='geopsro-final-answer-v4'))
    assert result['answer']==1 and result['structure']==0
