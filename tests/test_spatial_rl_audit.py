import json
import pytest
from spatial_intelligence.spatial_rl_audit import audited_reward,audit_update


def test_reward_excludes_heldout_and_uses_answer_not_format(tmp_path,monkeypatch):
    monkeypatch.setenv('EUREKASI_REWARD_TRACE',str(tmp_path))
    row={'id':'r','dataset':'spar','split':'train','scene_id':'scene','metric':'exact','task':'direction','answer':'left'}
    cfg={'numeric_atol':0.,'numeric_rtol':0.}
    assert audited_reward('<answer>left</answer>',row,cfg)==1.
    assert audited_reward('<answer>right</answer>',row,cfg)==0.
    with pytest.raises(ValueError):audited_reward('left',{**row,'split':'val'},cfg)


def test_audit_requires_within_prompt_variance(tmp_path):
    (tmp_path/'reward-trace').mkdir();(tmp_path/'training').mkdir()
    trace=tmp_path/'reward-trace/rewards.rank0.jsonl'
    trace.write_text('\n'.join(json.dumps({'id':'r','reward':r}) for r in [0,0,0,0]))
    (tmp_path/'training/metrics.jsonl').write_text(json.dumps({'loss':0.,'grad_norm':1.}))
    with pytest.raises(ValueError,match='No validated policy'):audit_update(tmp_path,4)
    trace.write_text('\n'.join(json.dumps({'id':'r','reward':r}) for r in [0,1,0,1]))
    assert audit_update(tmp_path,4)['nonconstant_reward_groups']==1


def strict_fixture(tmp_path):
    (tmp_path/'reward-trace').mkdir();(tmp_path/'training').mkdir()
    metrics=[{'step':s,'loss':0.1,'grad_norm':1.} for s in (1,2)]
    for rank in range(2):
        trace=tmp_path/f'reward-trace/rewards.rank{rank}.jsonl'
        trace.write_text('\n'.join(json.dumps({'id':f'r-{s}','reward':r}) for s in (1,2) for r in (0,1,0,1)))
        name='metrics.jsonl' if rank==0 else f'metrics.rank{rank}.jsonl'
        (tmp_path/'training'/name).write_text('\n'.join(json.dumps(row) for row in metrics))
    (tmp_path/'training/completion.json').write_text(json.dumps({
        'status':'optimizer_training_completed','steps':2,'world_size':2}))


def test_strict_audit_requires_all_ranks_and_steps(tmp_path):
    strict_fixture(tmp_path)
    assert audit_update(tmp_path,4,expected_world=2,expected_steps=2)['groups']==4
    with pytest.raises(ValueError,match='rank coverage'):
        audit_update(tmp_path,4,expected_world=4,expected_steps=2)
    path=tmp_path/'training/metrics.rank1.jsonl'
    path.write_text(json.dumps({'step':2,'loss':0.1,'grad_norm':1.}))
    with pytest.raises(ValueError,match='steps incomplete for rank 1'):
        audit_update(tmp_path,4,expected_world=2,expected_steps=2)


def test_strict_audit_rejects_extra_rollouts_and_wrong_completion(tmp_path):
    strict_fixture(tmp_path)
    path=tmp_path/'training/completion.json'
    path.write_text(json.dumps({'status':'optimizer_training_completed','steps':1,'world_size':2}))
    with pytest.raises(ValueError,match='Completion step count'):
        audit_update(tmp_path,4,expected_world=2,expected_steps=2)
    trace=tmp_path/'reward-trace/rewards.rank1.jsonl'
    with trace.open('a') as stream:
        stream.write('\n'+'\n'.join(json.dumps({'id':'extra','reward':r}) for r in (0,1,0,1)))
    with pytest.raises(ValueError,match='group count mismatch'):
        audit_update(tmp_path,4,expected_world=2,expected_steps=2)
