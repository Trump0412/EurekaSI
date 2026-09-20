import pytest
from spatial_intelligence.answer_extraction import extract_answer

C={'A':'chair','B':'table','C':'sofa','D':'lamp'}


@pytest.mark.parametrize('text,answer',[
    ('B','B'),('B.','B'),('The answer is B.','B'),
    ('I considered A and C.\nFinal answer: B','B'),
    ('<think>A?</think>\n<answer>B</answer>','B'),
    ('{"answer":"B"}','B'),('\\boxed{B}','B'),('table','B'),('B. table','B')])
def test_mcq_explicit(text,answer):
    assert extract_answer(text,choices=C)['answer']==answer


@pytest.mark.parametrize('text',[
    'A or B','B. chair','<answer>A</answer><answer>B</answer>',
    'The scene contains a table, chair and sofa.',
    '<think>Final answer: B','To determine which object is closest'])
def test_no_oracle_or_ambiguous_choice(text):
    assert extract_answer(text,choices=C)['answer'] is None


@pytest.mark.parametrize('text,answer',[
    ('3','3'),('Final answer: 3.5 meters','3.5'),
    ('{"answer":12}','12'),('There are 2 chairs and 3 tables.\n<answer>5</answer>','5'),
    ('<answer>1,200</answer>','1200')])
def test_numeric(text,answer):
    assert extract_answer(text)['answer']==answer


@pytest.mark.parametrize('text',['2 or 3','There are 3 chairs and 2 tables.','nan','inf','3-5','Final answer: 3 or 4'])
def test_numeric_ambiguity(text):
    assert extract_answer(text)['answer'] is None


def test_truncated_without_final_not_invented():
    assert extract_answer('To determine',choices=C,truncated=True)['status']=='truncated_no_answer'
