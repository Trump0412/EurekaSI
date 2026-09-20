"""Native VERL/Ray/vLLM GSPO diagnostic, gated on assets and isolated install.

Uses Qwen3-VL (legacy geometry backbone), NOT Qwen3.5. A saved checkpoint alone
does not prove a nonzero update: full acceptance additionally needs actor logs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--detach',action='store_true');p.add_argument('--prepare',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if a.prepare:
        import pandas as pd
        rows=[json.loads(s) for s in (root/'manifests/sft-probe.train.jsonl').read_text().splitlines()][:4]
        records=[]
        for row in rows:
            records.append({'data_source':'hound_diagnostic_train',
                'prompt':[{'role':'user','content':'<image>'*len(row['media'])+'\n'+row['question']}],
                # VERL 0.7's nested byte-dict conversion leaves a dict at the
                # qwen-vl-utils boundary; shared absolute paths decode correctly.
                'images':[str(Path(path).resolve(strict=True)) for path in row['media']],
                'ability':'image_caption_diagnostic',
                'reward_model':{'style':'rule','ground_truth':row['answer']},
                'extra_info':{'index':row['id'],'split':'train','diagnostic_only':True}})
        pd.DataFrame(records).to_parquet(out/'train.parquet')
        (out/'manifest.json').write_text(json.dumps(rows,indent=2));return
    if a.detach:
        with (root/'logs/verl-gspo-diagnostic.log').open('ab') as stream:
            child=subprocess.Popen([sys.executable,__file__,*[s for s in sys.argv[1:] if s!='--detach']],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'output':str(out)}));return
    lock=(root/'state/verl-gspo-diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True,exist_ok=False)
    state=out/'status.json'
    def record(status,**kw):
        state.write_text(json.dumps({'status':status,'diagnostic_only':True,'backbone':'Qwen3-VL-2B-Instruct',
            'qwen35_rl_validated':False,'updated':time.time(),**kw},indent=2))
    record('waiting_for_environment_and_model')
    deadline=time.time()+14400
    while True:
        env_receipt=root/'state/extension-env-verl.json'
        model_receipt=root/'receipts/geometry-qwen3vl2b.json'
        statuses=[json.loads(p.read_text()).get('status') if p.exists() else 'missing' for p in (env_receipt,model_receipt)]
        if 'failed' in statuses:record('blocked_dependency',dependency_status=statuses);return
        if statuses==['complete','complete']:break
        if time.time()>deadline:record('blocked_dependency_timeout');return
        time.sleep(15)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    if len(devices)!=2:raise ValueError('Diagnostic reserves exactly two explicitly allocated GPUs')
    while True:
        memory=subprocess.check_output(['nvidia-smi','--id='+','.join(devices),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if all(int(x.strip())<500 for x in memory.splitlines()):break
        record('waiting_for_idle_allocated_gpus')
        if time.time()>deadline:record('blocked_gpu_timeout');return
        time.sleep(15)
    py=str(root/'envs/verl-legacy/bin/python')
    subprocess.run([py,__file__,'--root',str(root),'--output',str(out),'--prepare'],check=True)
    cmd=[py,'-m','verl.trainer.main_ppo',
        'algorithm.adv_estimator=grpo',f'data.train_files={out}/train.parquet',f'data.val_files={out}/train.parquet',
        'data.train_batch_size=2','data.max_prompt_length=4096','data.max_response_length=64',
        'data.filter_overlong_prompts=False','data.truncation=error',
        f'actor_rollout_ref.model.path={root}/models/Qwen3-VL-2B-Instruct',
        'actor_rollout_ref.model.use_remove_padding=False','actor_rollout_ref.model.enable_gradient_checkpointing=True',
        '+actor_rollout_ref.model.override_config.attn_implementation=sdpa',
        'actor_rollout_ref.actor.optim.lr=1e-6','actor_rollout_ref.actor.ppo_mini_batch_size=2',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1','actor_rollout_ref.actor.use_torch_compile=False',
        'actor_rollout_ref.actor.policy_loss.loss_mode=gspo','actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean',
        'actor_rollout_ref.actor.clip_ratio_low=0.0003','actor_rollout_ref.actor.clip_ratio_high=0.0004',
        'actor_rollout_ref.actor.use_kl_loss=False',
        'actor_rollout_ref.actor.fsdp_config.param_offload=True','actor_rollout_ref.actor.fsdp_config.optimizer_offload=True',
        'actor_rollout_ref.rollout.name=vllm','actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.rollout.max_model_len=4160',
        '+actor_rollout_ref.rollout.limit_images=8',
        'actor_rollout_ref.rollout.n=4','actor_rollout_ref.rollout.temperature=1.0','actor_rollout_ref.rollout.top_p=1.0',
        'actor_rollout_ref.rollout.top_k=-1','actor_rollout_ref.rollout.gpu_memory_utilization=0.35',
        'actor_rollout_ref.rollout.enforce_eager=True','actor_rollout_ref.rollout.free_cache_engine=True',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
        f'custom_reward_function.path={REPO}/scripts/verl-hound-reward.py','custom_reward_function.name=compute_score',
        'trainer.logger=[console]','trainer.project_name=eurekasi-diagnostic','trainer.experiment_name=gspo-hound',
        'trainer.n_gpus_per_node=2','trainer.nnodes=1','trainer.total_epochs=1','trainer.total_training_steps=1',
        'trainer.val_before_train=False','trainer.test_freq=-1','trainer.save_freq=1',
        f'trainer.default_local_dir={out}/checkpoints']
    (out/'command.json').write_text(json.dumps(cmd,indent=2));record('native_verl_running',command=cmd)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
             WANDB_MODE='disabled',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None)
    with (out/'trainer.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            returncode=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            record('failed_timeout',pid=child.pid);return
    checkpoints=[str(p.relative_to(out)) for p in (out/'checkpoints').rglob('*.pt')]
    record('checkpoint_created_needs_update_audit' if returncode==0 and checkpoints else 'failed',
           returncode=returncode,checkpoints=checkpoints,
           acceptance_remaining=['nonconstant group rewards','finite nonzero actor grad_norm','checkpoint reload'])


if __name__=='__main__':main()
