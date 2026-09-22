"""Real GPU runtime gate; diagnostic weights never initialize formal training.

The fleet scheduler owns the allocation mutex and RFT priority. This foreground
command checks local predecessor receipts and idle GPUs, but is not a scheduler.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def preflight(plan, dependencies):
    from spatial_intelligence.followup_data_policy import validate_leakage_policy
    if plan.get('variant') not in ('full','no_tip','one_stb','final_only','post_merger'):
        raise ValueError('Unknown GeoRoute variant')
    if plan.get('use_lora') is not False:
        raise ValueError('Explicit full-parameter recipe required')
    data = read(plan['data_receipt'])
    if data.get('status') != 'ready' or data.get('media_verified') is not True:
        raise ValueError('Audited full training data must be ready before diagnostics')
    validate_leakage_policy(data)
    if not dependencies:
        raise ValueError('Local SFT/evaluation completion dependency required')
    for path in dependencies:
        state = read(path)
        if state.get('status') != 'complete' or state.get('gpu_work_finished') is not True:
            raise ValueError('Local predecessor has not completed and released GPUs: ' + str(path))
    return data


def select_rows(path, count=128, pressure_count=16):
    """Fixed first eligible source rows, not a representative throughput sample."""
    mixed, pressure = [], []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            frames = len(row.get('media', []))
            if 2 <= frames <= 32 and len(mixed) < count:
                mixed.append(row)
            if frames == 32 and len(pressure) < pressure_count:
                pressure.append(row)
            if len(mixed) == count and len(pressure) == pressure_count:
                break
    if not mixed or not pressure:
        raise ValueError('Need actual multi-image rows and actual 32-frame rows; no frame duplication')
    for rows in (mixed, pressure):
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate diagnostic IDs')
    return {'mixed': mixed, 'pressure32': pressure}


def accepted_stage(path, expected_stage):
    value = read(path)
    updates = value.get('component_updates', {})
    if not (value.get('status') == 'complete' and value.get('accepted') is True
            and value.get('diagnostic') is True and value.get('reload_verified') is True
            and value.get('finite_loss') is True and value.get('nonzero_update') is True
            and value.get('optimizer_rng_files_present') is True
            and value.get('generation_smoke_tokens', 0) > 0
            and value.get('contract', {}).get('stage') == expected_stage
            and value.get('steps', 0) >= 2 and updates and all(updates.values())):
        raise ValueError('Missing real diagnostic update/reload/inference evidence: ' + str(path))
    if expected_stage == 'sft' and not {'language', 'native_visual'}.issubset(updates):
        raise ValueError('Missing full-model ownership updates')
    return value


def run(command, log, env, timeout):
    with Path(log).open('ab') as stream:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            result = child.wait(timeout=timeout)
            if result:
                raise RuntimeError('GPU gate command failed: ' + str(result))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def accepted_effect(report):
    """Connectivity floor, NOT a requirement that large changes imply accuracy."""
    graph=report.get('graph', {})
    if graph.get('edges',0)<=0:
        raise ValueError('No real cross-frame graph support in influence probe')
    comparisons=report.get('tensor_comparisons', {})
    transmitted=[v for k,v in comparisons.items() if k.endswith('.post_exit')]
    if not transmitted or not any(v.get('changed_elements',0)>0 for v in transmitted):
        raise ValueError('STB influence vanished at every merged visual exit')
    if report.get('next_token_logits',{}).get('changed_elements',0)<=0:
        raise ValueError('No observable BF16 output influence; do not silently accept a dead branch')
    return dict(status='measured_nonzero',
        weak_effect_warning=max(v.get('relative_delta_l2') or 0 for v in transmitted)<1e-3,
        warning_threshold=1e-3,threshold_scope='engineering warning, not a paper hyperparameter or accuracy claim')


def execute(args):
    plan = read(args.plan)
    if Path(args.receipt).exists():
        raise ValueError('Runtime receipt already exists; inspect it instead of overwriting evidence')
    data = preflight(plan, args.dependency_receipt)
    gpus = [int(v) for v in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or any(v < 0 for v in gpus) or 64 % len(gpus):
        raise ValueError('Unique GPU IDs must divide global64')
    memory = subprocess.check_output(['nvidia-smi', '--id=' + args.gpus,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    values = [int(v.strip()) for v in memory.splitlines()]
    if len(values) != len(gpus) or any(v >= 512 for v in values):
        raise ValueError('Allocation not idle; do not interrupt current work')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONPATH=str(REPO),
        PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    stages = []
    for name, rows in select_rows(data['train_manifest']).items():
        subset = output / name
        subset.mkdir()
        manifest = subset / 'train.jsonl'
        manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        diagnostic_data = dict(data, train_manifest=str(manifest.resolve()),
            parent_data_receipt=str(Path(plan['data_receipt']).resolve()), diagnostic_only=True,
            diagnostic_selection='first source-order eligible rows; repeated optimizer schedule, never duplicated frames')
        write(subset / 'data.json', diagnostic_data)
        worker = copy.deepcopy(plan)
        worker.update(root=str(subset.resolve()), data_receipt=str((subset / 'data.json').resolve()),
            diagnostic_tip_checkpoint=str((subset / 'diagnostic/tip/micro1/final').resolve()),
            diagnostic_tip_receipt=str((subset / 'diagnostic/tip/micro1/completion.json').resolve()))
        worker_plan = subset / 'plan.json'
        write(worker_plan, worker)
        for phase in (('sft',) if plan['variant']=='no_tip' else ('tip', 'sft')):
            if args.yield_request and Path(args.yield_request).is_file():
                write(output/'yield.json',dict(status='yielded_to_priority',full_model_verified=False,time=time.time()))
                raise SystemExit(75)
            base = [str(REPO / 'scripts/train-georoute-stage.py'), '--plan', str(worker_plan.resolve()),
                '--stage', phase, '--micro', '1', '--diagnostic-steps', '2']
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=' + str(len(gpus)), *base]
            write(output / 'progress.json', dict(status='running', subset=name, stage=phase,
                pid=os.getpid(), time=time.time()))
            run(command, subset / (phase + '.log'), env, args.timeout_seconds)
            if args.yield_request and Path(args.yield_request).is_file():
                write(output/'yield.json',dict(status='yielded_to_priority',full_model_verified=False,time=time.time()))
                raise SystemExit(75)
            run([sys.executable, *base, '--verify-saved'], subset / (phase + '-reload.log'), env, args.timeout_seconds)
            receipt = subset / f'diagnostic/{phase}/micro1/completion.json'
            accepted = accepted_stage(receipt, phase)
            contract = accepted['contract']
            if (contract.get('architecture') != worker['architecture']
                    or contract.get('graph') != worker['graph']
                    or contract.get('variant') != worker['variant']
                    or contract.get('world') != len(gpus)
                    or contract.get('global_batch') != 64):
                raise ValueError('Runtime evidence belongs to a different variant or allocation')
            stages.append(dict(subset=name, stage=phase, receipt=str(receipt.resolve()), row_ids=[row['id'] for row in rows]))
        effect_path=subset/'sft-effect-bf16.json'
        stage_root=subset/'diagnostic/sft/micro1'
        run([sys.executable,str(REPO/'scripts/audit-georoute-effect.py'),
            '--checkpoint',str(stage_root/'final'),'--bundle',str(stage_root/'reload-evidence.pt'),
            '--output',str(effect_path),'--device','cuda:0','--dtype','bfloat16'],
            subset/'effect.log',env,args.timeout_seconds)
        effect=accepted_effect(read(effect_path))
        stages.append(dict(subset=name,stage='bf16_influence',receipt=str(effect_path.resolve()),**effect))
    result = dict(status='ready', full_model_verified=True, variant=plan['variant'],
        architecture=plan['architecture'], graph=plan['graph'], actual_32frame_pressure_verified=True,
        model=plan['model'], data_receipt=plan['data_receipt'], stages=stages, gpus=gpus,
        micro=1, global_batch=64, runtime='TIP DDP; SFT ZeRO3 per worker plan',
        diagnostic_only=True, formal_initialization='released model; never these diagnostic weights',
        limitations=['Micro2/4 throughput selection not tested', 'Optimizer/RNG presence is not resumed trajectory parity',
            'ReVSI/VSI scoring and format gates are separate', 'Diagnostic first eligible rows are not full-mixture ETA'],
        finished=time.time())
    write(output / 'completion.json', result)
    write(args.receipt, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--dependency-receipt', action='append', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--authorize-run', action='store_true')
    parser.add_argument('--yield-request',help='Yield at a diagnostic command boundary for priority GPU work')
    args = parser.parse_args()
    if not args.authorize_run:
        parser.error('Explicit --authorize-run required; this command consumes GPUs')
    def stop(signum, frame):
        raise InterruptedError('Runtime gate interrupted')
    signal.signal(signal.SIGTERM, stop)
    try:
        execute(args)
    except BaseException as exc:
        write(Path(args.output) / 'failure.json', dict(status='failed', full_model_verified=False,
            error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
