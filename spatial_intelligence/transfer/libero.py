"""LIBERO closed-loop evaluator. Must run in the simulator's isolated environment."""
from pathlib import Path
import time
import numpy as np
from ..io import symbol,write_json,file_digest,digest,environment


def valid_action(action):
    action=np.asarray(action,dtype=np.float32)
    if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Policy must return seven finite OSC_POSE controls')
    if np.any(np.abs(action)>1.00001):raise ValueError('Action outside normalized [-1,1]; apply the policy training normalization explicitly')
    return action


def rollout(env,policy,initial_state,instruction,seed,horizon,warmup=10):
    env.seed(seed);env.reset();obs=env.set_init_state(initial_state)
    policy.reset(instruction=instruction,seed=seed)
    success=False
    for _ in range(warmup):
        obs,_,done,_=env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))
        if env.check_success():success=True;break
        if done:raise RuntimeError('Environment terminated during warmup without success')
    actions=[];latencies=[]
    for _ in range(0 if success else horizon):
        start=time.perf_counter();action=valid_action(policy.act(obs,instruction));latencies.append(time.perf_counter()-start)
        obs,_,done,_=env.step(action);actions.append(action.tolist());success=bool(env.check_success())
        if done or success:break
    return {'success':success,'steps':len(actions),'actions':actions,'latency_s':latencies,'seed':seed,
            'initial_state_sha256':digest(np.asarray(initial_state).tolist())}


def evaluate(cfg):
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    if cfg.get('action_convention')!='libero_osc_pose_7_normalized':raise ValueError('Declare the LIBERO action convention')
    if cfg.get('policy')=='spatial_intelligence.transfer.policies:ZeroPolicy' and not cfg.get('smoke_only'):
        raise ValueError('Zero policy is only a smoke test')
    if cfg.get('horizon',0)<1 or cfg.get('episodes_per_task',0)<1:raise ValueError('Positive horizon and episodes required')
    out=Path(cfg['output'])
    if out.exists() and any(out.iterdir()):raise FileExistsError('LIBERO output is nonempty')
    suite=get_benchmark_dict()[cfg['suite']](task_order_index=cfg.get('task_order_index',0))
    ids=cfg.get('task_ids',list(range(suite.get_num_tasks())))
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Nonempty unique task IDs required')
    policy=symbol(cfg['policy'])(cfg.get('policy_options',{}))
    if not hasattr(policy,'identity'):raise ValueError('Policy must implement identity() with model/head/normalization hashes')
    write_json(out/'config.json',cfg);write_json(out/'environment.json',environment());write_json(out/'policy.json',policy.identity())
    records=[]
    for task_id in ids:
        task=suite.get_task(task_id);states=suite.get_task_init_states(task_id)
        if cfg['episodes_per_task']>len(states):raise ValueError('Not enough unique official initial states')
        env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),camera_heights=cfg.get('image_size',256),camera_widths=cfg.get('image_size',256),horizon=cfg['horizon']+cfg.get('warmup_steps',10)+1)
        try:
            for episode in range(cfg['episodes_per_task']):
                rec=rollout(env,policy,states[episode],task.language,cfg['seed']+episode,cfg['horizon'],cfg.get('warmup_steps',10))
                rec.update(task_id=task_id,task_name=task.name,episode=episode,bddl_sha256=file_digest(suite.get_task_bddl_file_path(task_id)))
                write_json(out/'episodes'/f'{task_id}-{episode}.json',rec);records.append(rec)
        finally:env.close()
    report={'suite':cfg['suite'],'episodes':len(records),'success_rate':sum(r['success'] for r in records)/len(records),
       'task_success_rate':{str(i):np.mean([r['success'] for r in records if r['task_id']==i]).item() for i in ids},
       'status':'smoke_only_not_model_result' if cfg.get('smoke_only') else 'closed_loop_completed',
       'protocol':{k:v for k,v in cfg.items() if k not in ['policy','policy_options','output']}}
    write_json(out/'report.json',report);return report
