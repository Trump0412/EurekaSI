import pytest
from spatial_intelligence.rft_dynamic_sampling import informative_group, replacement_row, select_group


def test_total_reward_not_answer_only():
    assert informative_group([{'answer':0,'total':0}, {'answer':0,'total':.5}])
    assert not informative_group([{'total':1}]*8)
    assert not informative_group([{'total':0}]*8)
    with pytest.raises(ValueError): informative_group([{'total':float('nan')},{'total':0}])


def test_refill_logs_every_candidate_and_counts_all_tokens():
    records=[]
    def generate(row):
        return row, [{'tokens':[1,2], 'total':0}, {'tokens':[3], 'total':row['n']}]
    selected,counts=select_group({'n':0},generate,lambda r,row:r,lambda a:{'n':a},
        lambda *args:records.append(args))
    assert selected[0]['n']==1
    assert counts=={'candidate_groups':2,'discarded_groups':1,'generated_tokens':6}
    assert [r[-1] for r in records]==[False,True]


def test_exhaustion_is_bounded_and_never_fills_with_constant_groups():
    result,counts=select_group({'n':0},lambda row:(row,[{'tokens':[1],'total':0}]*8),
        lambda r,row:r,lambda a:{'n':a},lambda *args:None,max_attempts=3)
    assert result is None and counts['discarded_groups']==3
    assert counts['generated_tokens']==24


def test_no_blacklist_source_preservation_and_deterministic_retry():
    pools={'a':[{'id':'one','source':'a'}], 'b':[{'id':'two','source':'b'}]}
    for step in range(20):
        assert replacement_row(pools,'a',3407,step,1)==pools['a'][0]
    assert replacement_row(pools,'a',3407,3,1)==replacement_row(pools,'a',3407,3,1)
