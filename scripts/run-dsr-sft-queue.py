"""Append a durable, serial SFT comparison after existing GPU owners release."""
import argparse
import itertools
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from spatial_intelligence.dsr_sft_eval import (read, rows, write, validate_rows,
    model_identity, paired_difference)


def dependency_reasons(plan):
    reasons = []
    for dep in plan['dependencies']:
        path = Path(dep['path'])
        if not path.is_file():
            reasons.append('missing '+str(path))
            continue
        value = read(path)
        if value.get('status') not in dep.get('statuses', ['complete']):
            reasons.append(f"{path}: status={value.get('status')}")
        if dep.get('require_gpu_release') and value.get('gpu_work_finished') is not True:
            reasons.append(str(path)+': GPU work not released')
    return reasons


def gpus_idle(gpus):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    memory = {int(a): int(b) for a, b in (line.split(',') for line in result.stdout.splitlines())}
    return all(g in memory and memory[g] < 512 for g in gpus)


def snapshot(plan):
    root = Path(plan['root'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        if read(root/'plan.json') != plan:
            raise ValueError('Changed armed plan; use a new version')
        return
    if (root/'code').exists():
        raise ValueError('Partial deployment; inspect before retrying')
    validate_rows(rows(plan['source_manifest']))
    if Path(plan['manifest']) != root/'inputs/dsr.test.jsonl':
        raise ValueError('Manifest must be an independent snapshot inside this study')
    (root/'inputs').mkdir(exist_ok=True)
    shutil.copy2(plan['source_manifest'], plan['manifest'])
    for name in ('scripts', 'spatial_intelligence', 'configs', 'tests'):
        shutil.copytree(REPO/name, root/'code'/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write(root/'plan.json', plan)


def execute(plan):
    import fcntl
    root = Path(plan['root'])
    (root/'state').mkdir(exist_ok=True)
    states = {}
    def status(state, **extra):
        write(root/'state/queue.json', dict(status=state, pid=os.getpid(), time=time.time(), arms=states, **extra))
    with (root/'state/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            reasons = dependency_reasons(plan)
            if not reasons and gpus_idle(plan['gpus']):
                break
            status('waiting_predecessors' if reasons else 'waiting_gpus', waiting_for=reasons)
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, plan['gpus'])),
                   PYTHONPATH=str(root/'code'), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        for arm in plan['arms']:
            name = arm['name']
            try:
                receipt = read(arm['completion'])
                if receipt.get('status') != 'complete' or Path(receipt['checkpoint']).resolve() != Path(arm['checkpoint']).resolve():
                    raise ValueError('Completed SFT lineage not verified')
                if arm['kind'] != 'legacy' and receipt.get('reload_verified') is not True:
                    raise ValueError('Matrix SFT reload acceptance missing')
                model_identity(arm['checkpoint'])
                for smoke in (True, False):
                    stage = 'smoke' if smoke else 'full'
                    target = root/'runs'/name/stage/'completion.json'
                    if target.exists() and read(target).get('accepted') is True:
                        continue
                    if dependency_reasons(plan):
                        raise ValueError('Predecessor state changed; refuse overlapping jobs')
                    while not gpus_idle(plan['gpus']):
                        status('waiting_gpus', arm=name)
                        time.sleep(30)
                    worker = [str(root/'code/scripts/evaluate-dsr-sft.py'), '--plan', str(root/'plan.json'), '--arm', name]
                    if smoke:
                        worker.append('--smoke')
                    commands = [[plan['python'], '-m', 'torch.distributed.run', '--standalone',
                                 f"--nproc_per_node={len(plan['gpus'])}"]+worker,
                                [plan['python']]+worker+['--merge']]
                    for index, command in enumerate(commands):
                        with (root/'logs'/f'{name}-{stage}-{index}.log').open('ab') as log:
                            child = subprocess.Popen(command, cwd=root/'code', env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            try:
                                while child.poll() is None:
                                    status('evaluating', arm=name, phase=stage, child_pid=child.pid)
                                    time.sleep(15)
                                if child.returncode:
                                    raise RuntimeError(f'{stage} command {index} exited {child.returncode}')
                            finally:
                                if child.poll() is None:
                                    os.killpg(child.pid, signal.SIGTERM)
                                    try:
                                        child.wait(timeout=20)
                                    except subprocess.TimeoutExpired:
                                        os.killpg(child.pid, signal.SIGKILL)
                                        child.wait()
                states[name] = dict(status='complete', metrics=read(root/'runs'/name/'full/metrics.json'))
            except Exception as error:
                # An independent checkpoint failure must not erase or suppress other arms.
                states[name] = dict(status='failed', error=repr(error))
            status('between_arms')
        completed = {name: [r for rank in range(len(plan['gpus']))
                     for r in rows(root/'runs'/name/'full'/f'predictions.rank{rank}.jsonl')]
                     for name, state in states.items() if state['status'] == 'complete'}
        comparison = {left+'__vs__'+right: paired_difference(completed[left], completed[right])
                      for left, right in itertools.combinations(completed, 2)}
        write(root/'comparison.json', dict(arms=states, pairs=comparison,
            caveat='Preserved direct baseline differs in stage count and visual freezing; not a single-factor ablation'))
        status('complete' if len(completed) == len(plan['arms']) else 'complete_with_failures', gpu_work_finished=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    if len(plan['arms']) != 4 or len({a['name'] for a in plan['arms']}) != 4:
        raise ValueError('Exactly four unique SFT arms required')
    snapshot(plan)
    root = Path(plan['root'])
    (root/'logs').mkdir(exist_ok=True)
    if args.detach:
        with (root/'logs/supervisor.log').open('ab') as log:
            child = subprocess.Popen([plan['python'], str(root/'code/scripts/run-dsr-sft-queue.py'),
                '--plan', str(root/'plan.json')], cwd=root/'code', stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(dict(status='waiting_queue_started_not_inference', pid=child.pid, root=str(root)), flush=True)
    else:
        def stop(*unused):
            raise InterruptedError('Supervisor interrupted')
        signal.signal(signal.SIGTERM, stop)
        execute(plan)


if __name__ == '__main__':
    main()
