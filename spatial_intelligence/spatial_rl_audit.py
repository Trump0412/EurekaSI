"""Auditable exact-answer reward for a scene-held-out spatial QA pilot.

This is a generic research reward, not the official ReVSI benchmark metric.
Labels are consumed by the reward only, never inserted into student prompts.
"""
import json
import math
import os
from pathlib import Path
import re
import time

from .evaluation import score_one


def audited_reward(response, row, cfg):
    if row.get('split') != 'train' or row.get('dataset') != 'spar':
        raise ValueError('Only declared SPAR training rows may receive training rewards')
    if row.get('metric') not in {'exact','numeric','choice'}:
        raise ValueError('Undeclared spatial reward metric')
    result=score_one(response,row,cfg['numeric_atol'],cfg['numeric_rtol'])
    reward=float(result['correct'])
    folder=Path(os.environ['EUREKASI_REWARD_TRACE'])
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/f"rewards.rank{os.environ.get('RANK','0')}.jsonl").open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'id':row['id'],'scene_id':row['scene_id'],
            'reward':reward,'response':response,'parsed':result['parsed'],
            'unix_time':time.time()},ensure_ascii=False)+'\n')
    return reward


def audit_update(run,group_size,*,expected_world=None,expected_steps=None):
    """Require actual within-prompt reward diversity plus finite nonzero gradients."""
    if type(group_size) is not int or group_size < 2:
        raise ValueError('group_size must be an integer >= 2')
    for value in (expected_world,expected_steps):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Expected world/steps must be positive integers')
    run=Path(run);groups=0;variable=0
    paths=sorted((run/'reward-trace').glob('rewards.rank*.jsonl'))
    ranks={}
    for path in paths:
        match=re.fullmatch(r'rewards\.rank(\d+)\.jsonl',path.name)
        if not match:raise ValueError('Malformed reward rank filename')
        rank=int(match.group(1))
        if rank in ranks:raise ValueError('Duplicate reward rank')
        ranks[rank]=path
    if expected_world is not None and set(ranks)!=set(range(expected_world)):
        raise ValueError(f'Reward rank coverage mismatch: {sorted(ranks)}')
    rank_groups={}
    for rank,path in ranks.items():
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(rows)%group_size:raise ValueError('Incomplete rollout group')
        rank_groups[rank]=len(rows)//group_size
        for start in range(0,len(rows),group_size):
            chunk=rows[start:start+group_size]
            if len({row['id'] for row in chunk})!=1:raise ValueError('Group mixes prompt identities')
            rewards=[row['reward'] for row in chunk]
            if any(not math.isfinite(value) for value in rewards):raise ValueError('Nonfinite reward')
            groups+=1;variable+=len(set(rewards))>1
    metrics=[json.loads(line) for line in (run/'training/metrics.jsonl').read_text().splitlines()]
    if not metrics or any(not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) for row in metrics):
        raise ValueError('Missing/nonfinite optimizer evidence')
    nonzero=sum(row['grad_norm']>1e-8 for row in metrics)
    if expected_steps is not None:
        steps=list(range(1,expected_steps+1))
        if [row.get('step') for row in metrics]!=steps:
            raise ValueError('Optimizer steps missing, repeated, or out of order')
        for rank,path in ranks.items():
            rank_path=run/'training'/('metrics.jsonl' if rank==0 else f'metrics.rank{rank}.jsonl')
            rank_metrics=[json.loads(line) for line in rank_path.read_text().splitlines()]
            if [row.get('step') for row in rank_metrics]!=steps:
                raise ValueError(f'Optimizer steps incomplete for rank {rank}')
            if any(not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) for row in rank_metrics):
                raise ValueError(f'Nonfinite rank {rank} optimizer evidence')
            # This four-GPU pilot uses one local prompt per step, no GA.
            if rank_groups[rank]!=expected_steps:
                raise ValueError(f'Rollout group count mismatch for rank {rank}')
    if expected_world is not None or expected_steps is not None:
        completion=json.loads((run/'training/completion.json').read_text())
        if completion.get('status')!='optimizer_training_completed':
            raise ValueError('Training completion not confirmed')
        if expected_world is not None and completion.get('world_size')!=expected_world:
            raise ValueError('Completion world size mismatch')
        if expected_steps is not None and completion.get('steps')!=expected_steps:
            raise ValueError('Completion step count mismatch')
    if not groups or not variable or not nonzero:
        raise ValueError(f'No validated policy learning: groups={groups}, variable={variable}, nonzero_steps={nonzero}')
    return {'groups':groups,'nonconstant_reward_groups':variable,'nonzero_gradient_steps':nonzero,
            'steps':len(metrics),'last_loss':metrics[-1]['loss'],'last_grad_norm':metrics[-1]['grad_norm'],
            'rank_groups':rank_groups,'expected_world':expected_world,'expected_steps':expected_steps}
