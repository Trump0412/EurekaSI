import copy
import json
from pathlib import Path

import pytest
from spatial_intelligence.rft_coldstart import (
    allocate, select, temporal_issue, teacher_row, annotation_check, validate_coldstart_initialization)
from spatial_intelligence.rft_reward_activity import activity_gate, group_activity


def sample(i, source='4drl'):
    return dict(id=f'{source}:{i}',source=source,source_group=f'{source}-g{i%10}',
        question='Which direction?',question_type='direction',answer_type='mcq',answer='A',
        choices={'A':'left','B':'right'},media=['a.png']*8,input_mode='images',raw_annotation={'Answer_cot':'secret'})


def test_exact_mixture_unique_reproducible():
    pool=[sample(i) for i in range(100)]+[sample(i,'spatialladder') for i in range(100)]
    result=select(pool,100)
    assert sum(x['source']=='4drl' for x in result)==70
    assert len({x['id'] for x in result})==100
    assert result==select(pool,100)
    assert pool[0]['id']=='4drl:0'


def test_no_silent_shrink():
    with pytest.raises(ValueError): select([sample(i) for i in range(20)],100)


def test_quotas_largest_remainder():
    assert allocate({'a':3,'b':7},7)=={'a':2,'b':5}


def test_time_origin_flag_is_not_automatic_repair():
    row=dict(sample(0),input_mode='video',fps=30,total_num_frames=300,question='Between 3s and 15s?')
    original=copy.deepcopy(row)
    assert temporal_issue(row)=='question_time_exceeds_clip_duration_unresolved_origin'
    assert row==original
    row['question']='At 3.2s?'
    assert temporal_issue(row) is None


def test_teacher_never_receives_gold_or_source_cot():
    row=sample(0); row.update(raw_answer='A',Bbox_cot='secret')
    safe=teacher_row(row)
    assert not {'answer','raw_answer','raw_annotation','Bbox_cot'} & safe.keys()
    assert safe['choices']==row['choices']


def test_structure_not_enough_and_numeric_exact_required():
    response='<think>Spatial Observation: a box is left. Spatial Transition: positions stay fixed. Answer Derivation: therefore choose left.</think><answer>A</answer>'
    row=sample(0)
    annotation=dict(id=row['id'],response=response,truncated=False,student_token_count=50)
    assert annotation_check(row,annotation,{})['accepted']
    row['answer']='B'
    assert not annotation_check(row,annotation,{})['accepted']
    row.update(answer_type='numeric',answer='100',choices={})
    annotation['response']=response.replace('<answer>A','<answer>99')
    assert 'numeric_not_exact' in annotation_check(row,annotation,{})['reasons']


def test_known_good_format_saturation_is_not_advantage_evidence():
    config=dict(structure_weight=.5,words_weight=0)
    scores=[dict(answer=0,structure=1,words=0,total=.5),dict(answer=1,structure=1,words=0,total=1.5)]
    activity=group_activity(scores,config)
    assert not activity_gate(activity,config)['accepted']
    checked=activity_gate(activity,config,allow_saturated_format=True)
    assert checked['accepted']
    assert checked['saturated_terms']==['format_already_satisfied_no_advantage_effect']
    assert activity['format_advantage_groups']==0


def test_coldstart_lineage_fails_closed(tmp_path):
    p=tmp_path/'receipt.json'; p.write_text(json.dumps(dict(status='complete',diagnostic_only=False,
        finite_loss=True,nonzero_update=True,reload_verified=False)))
    with pytest.raises(ValueError,match='reload'):
        validate_coldstart_initialization(dict(coldstart_policy=str(tmp_path/'policy'),coldstart_receipt=str(p)))
