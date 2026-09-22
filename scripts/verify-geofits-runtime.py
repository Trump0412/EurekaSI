"""Real GeoFits multi-GPU update/reload gate; never a benchmark result.

Run only under the allocation owner after predecessor train/eval releases GPUs.
Each attempt has a fresh output. The gate does not launch formal training or
reuse diagnostic weights as initialization.
"""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('route_runtime', REPO / 'scripts/verify-georoute-runtime.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
read, write = common.read, common.write


def accepted_stage(receipt, plan, world):
    value = read(receipt)
    contract = value.get('contract', {})
    updates = value.get('component_updates', {})
    if not (value.get('accepted') is True and value.get('status') == 'complete'
            and value.get('diagnostic') is True and value.get('finite_loss') is True
            and value.get('nonzero_update') is True and value.get('reload_verified') is True
            and value.get('optimizer_rng_files_present') is True
            and value.get('generation_smoke_tokens', 0) > 0 and value.get('steps', 0) >= 2):
        raise ValueError('Missing actual update/checkpoint/reload/generation evidence')
    if not updates or not all(updates.values()):
        raise ValueError('Some required trainable components did not update')
    if contract.get('architecture') != plan['architecture'] or contract.get('world') != world or contract.get('global_batch') != 64:
        raise ValueError('Runtime contract mismatch')
    return value


def execute(args):
    import json
    from spatial_intelligence.followup_data_policy import validate_leakage_policy
    plan = read(args.plan); data = read(plan['data_receipt'])
    if Path(args.receipt).exists():
        raise ValueError('Preserve existing acceptance; use a new attempt')
    if plan.get('use_lora') is not False or data.get('status') != 'ready' or data.get('media_verified') is not True:
        raise ValueError('Full-parameter plan and verified ready data required')
    validate_leakage_policy(data)
    for path in args.dependency_receipt:
        state = read(path)
        if state.get('status') != 'complete' or not (state.get('gpu_work_finished') is True or state.get('accepted') is True):
            raise ValueError('Predecessor still owns GPU work')
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0 or 64 % (len(gpus) * args.micro):
        raise ValueError('Unique GPU IDs and world*micro dividing global64 required')
    used = subprocess.check_output(['nvidia-smi', '--id=' + args.gpus,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = [int(x.strip()) for x in used.splitlines()]
    if len(memory) != len(gpus) or any(x >= 512 for x in memory):
        raise ValueError('Allocation is not idle')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONPATH=str(REPO),
        PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    groups = common.select_rows(data['train_manifest'])
    # Single images are a real source condition too, especially for 4D-only.
    with Path(data['train_manifest']).open(encoding='utf-8') as stream:
        singles = []
        for line in stream:
            row = json.loads(line)
            if len(row['media']) == 1:
                singles.append(row)
            if len(singles) >= 16:
                break
    if not singles:
        raise ValueError('Single-frame diagnostic examples missing')
    groups['single'] = singles
    stages = []
    for name, rows in groups.items():
        subset = output / name; subset.mkdir()
        manifest = subset / 'train.jsonl'
        manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        write(subset / 'data.json', dict(data, train_manifest=str(manifest.resolve()),
            diagnostic_only=True, parent_data_receipt=plan['data_receipt']))
        worker = copy.deepcopy(plan)
        worker.update(root=str(subset.resolve()), data_receipt=str((subset / 'data.json').resolve()))
        worker_plan = subset / 'plan.json'; write(worker_plan, worker)
        base = [str(REPO / 'scripts/train-geofits-stage.py'), '--plan', str(worker_plan.resolve()),
            '--micro', str(args.micro), '--diagnostic-steps', '2']
        write(output / 'progress.json', dict(status='running', subset=name, pid=os.getpid(), updated=time.time()))
        common.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=' + str(len(gpus)), *base], subset / 'train.log', env, args.timeout_seconds)
        common.run([sys.executable, *base, '--verify-saved'], subset / 'reload.log', env, args.timeout_seconds)
        receipt = subset / f'diagnostic/micro{args.micro}/completion.json'
        accepted_stage(receipt, worker, len(gpus))
        stages.append(dict(subset=name, receipt=str(receipt.resolve()), ids=[r['id'] for r in rows]))
    result = dict(status='ready', full_model_verified=True, architecture=plan['architecture'],
        variant=plan['variant'], model=plan['model'], data_receipt=plan['data_receipt'],
        micro=args.micro, global_batch=64, world=len(gpus), stages=stages,
        diagnostic_only=True, finished=time.time(),
        limitations=['Not a downstream benchmark or exact resumed-trajectory proof',
            'First eligible source rows are not a representative throughput estimate'])
    write(output / 'completion.json', result); write(args.receipt, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'output', 'receipt', 'gpus'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--dependency-receipt', action='append', required=True)
    parser.add_argument('--micro', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--authorize-run', action='store_true')
    args = parser.parse_args()
    if not args.authorize_run:
        parser.error('Explicit --authorize-run required')
    def stop(signum, frame):
        raise InterruptedError('Runtime gate interrupted')
    signal.signal(signal.SIGTERM, stop)
    try:
        execute(args)
    except BaseException as exc:
        # Do not mutate someone else's preexisting output on validation failure.
        if not Path(args.output).exists():
            print(repr(exc), file=sys.stderr)
        else:
            print('GeoFits gate failed: ' + repr(exc), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
