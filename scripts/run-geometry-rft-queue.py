"""Immutable after-SFT geometry RFT queue; never preempt existing GPU work.

This is the custom HF geometry reference backend, not verified VERL support.
One node may execute a subset of a paired design, but initialization and budgets
are fixed by the same pair ID, scientific recipe and audited data receipt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ARMS = {'mixed': (.7, .3), 'four_d_only': (1., 0.)}
TERMINAL = {'complete', 'complete_with_failures', 'failed', 'blocked', 'training_complete_evaluation_blocked'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8'); temporary.replace(path)


def validate_plan(plan):
    for field in ('root', 'python', 'model_checkpoint', 'processor', 'vggt_source', 'sft_receipt', 'data_receipt',
                  'scientific_config', 'pair_id'):
        if not plan.get(field): raise ValueError('Missing explicit ' + field)
    if plan.get('group_size', 8) != 8:
        raise ValueError('Logical rollout group must remain eight')
    if plan.get('prompts_per_update', 16) != 16:
        raise ValueError('Paired design uses a fixed global prompt batch of sixteen')
    if plan.get('seed', 3407) != 3407:
        raise ValueError('Paired design seed must match')
    devices = plan.get('gpus', [])
    if not devices or len({str(x) for x in devices}) != len(devices) or any(not str(x).isdigit() for x in devices):
        raise ValueError('Distinct explicit GPU indices required')
    if 16 % len(devices): raise ValueError('World size must divide global prompt batch sixteen')
    jobs = plan.get('jobs', [])
    names = [job.get('name') for job in jobs]
    if not names or len(names) != len(set(names)) or any(name not in ARMS for name in names):
        raise ValueError('Jobs must be a unique nonempty subset of mixed/four_d_only')
    for job in jobs:
        if any(key in job for key in ('model_checkpoint', 'updates', 'seed', 'group_size', 'prompt_budget')):
            raise ValueError('Arms cannot override shared initialization or budget')
        expected = ARMS[job['name']]
        for key, wanted in zip(('four_d_rl_fraction', 'spatial_fraction'), expected):
            if key in job and job[key] != wanted: raise ValueError('Declared arm mixture changed')
    if not plan.get('dependencies'):
        raise ValueError('Explicit preceding-pipeline dependencies required')
    for dep in plan['dependencies']:
        if not dep.get('path') or not dep.get('statuses'):
            raise ValueError('Each dependency needs a receipt path and allowed terminal statuses')
        if not set(dep['statuses']) <= TERMINAL:
            raise ValueError('Only terminal predecessor statuses can release the RFT queue')
    return plan


def dependency_ready(dep):
    if not Path(dep['path']).exists(): return False
    value = read(dep['path'])
    return value.get('status') in dep['statuses'] and value.get('gpu_work_finished') is True


def validate_sft(plan):
    value = read(plan['sft_receipt'])
    for field in ('finite_loss', 'nonzero_update', 'reload_verified'):
        if value.get(field) is not True: raise ValueError('SFT acceptance missing: ' + field)
    if value.get('status') != 'complete' or value.get('diagnostic_only') is not False:
        raise ValueError('A verified formal SFT checkpoint is required, not diagnostic weights')
    target = Path(plan['model_checkpoint']).resolve()
    if Path(value['checkpoint']).resolve() != target or not (target / 'config.json').exists():
        raise ValueError('SFT checkpoint identity/config mismatch')
    contract = read(target.parent / 'contract.json')
    if contract.get('stage') != 'sft' or contract.get('diagnostic') is not False:
        raise ValueError('Source contract is not formal SFT')
    if contract.get('max_steps') != -1 or contract.get('epochs') != 1:
        raise ValueError('Source SFT is not the declared full one-epoch run')
    if value.get('steps', 0) < 1: raise ValueError('Source SFT has no optimizer updates')
    if not read(target / 'config.json').get('geometry_matrix'):
        raise ValueError('Registered geometry checkpoint required by this backend')
    return value


def validate_data(value):
    if value.get('status') not in ('ready', 'complete') or value.get('ready_for_training') is not True:
        raise ValueError('Data are not explicitly accepted for training')
    if value.get('train_test_overlap') != 0:
        raise ValueError('Missing clean train/test overlap audit')
    if value.get('leakage_checked') is not True or value.get('media_verified') is not True:
        raise ValueError('Leakage and real-media readiness must both be explicitly verified')
    rows = value.get('accepted_train_rows')
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError('A positive audited accepted_train_rows count is required')
    manifests = value.get('manifests', {})
    if not manifests or not all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in manifests.values()):
        raise ValueError('Training manifests missing')
    if not Path(value.get('eval_manifest', '')).is_file() or Path(value['eval_manifest']).stat().st_size == 0:
        raise ValueError('Fixed paired evaluation manifest missing')
    return value


def budget(rows):
    requested = 2 * rows
    updates = math.ceil(requested / 16)
    return dict(accepted_train_rows=rows, reference_passes=2,
                requested_prompt_draws=requested, prompt_budget=updates * 16,
                tail_padding_prompt_draws=updates * 16 - requested,
                updates=updates, prompts_per_update=16, group_size=8,
                rollout_budget=updates * 16 * 8)


def checkpoint_identity(checkpoint):
    root = Path(checkpoint)
    files = sorted(set(root.glob('*.safetensors')) | set(root.glob('*.bin')) |
                   set(root.glob('*.index.json')) | {root / 'config.json'})
    if not any(path.suffix in ('.safetensors', '.bin') for path in files):
        raise ValueError('SFT checkpoint contains no weight files')
    return {path.name: {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in files}


def manifest_ids(path):
    values = [json.loads(line)['id'] for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError('Paired manifests require nonempty unique sample IDs')
    return values


def evaluation_sources(data):
    """Held-out source QA is named distinctly from external benchmarks."""
    result = dict(data.get('eval_manifests', {}))
    for source, path in data.get('validation_manifests', {}).items():
        name = 'validation_' + source
        if name in result and result[name] != path:
            raise ValueError('Conflicting validation evaluation manifest: ' + name)
        result[name] = path
    return result


def pair_contract(plan, data, recipe):
    train_ids = {name: manifest_ids(path) for name, path in sorted(data['manifests'].items())}
    if sum(len(values) for values in train_ids.values()) != data['accepted_train_rows']:
        raise ValueError('Accepted row count differs from the actual source manifests')
    evaluation = {name: manifest_ids(path) for name, path in sorted(evaluation_sources(data).items())}
    return dict(pair_id=plan['pair_id'], model_checkpoint=str(Path(plan['model_checkpoint']).resolve()),
        checkpoint_identity=checkpoint_identity(plan['model_checkpoint']), scientific_recipe=recipe,
        source_data_receipt=data, source_train_ids=train_ids, evaluation_ids=evaluation,
        primary_evaluation_ids=manifest_ids(data['eval_manifest']),
        **budget(data['accepted_train_rows']), seed=3407)


def verify_shared_pair(path, contract):
    """Cross-node equality under a shared-storage file lock; no node paths inside."""
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if read(path) != contract:
                raise ValueError('Cross-node paired scientific contract mismatch')
        # Keep actual manifest byte copies, not only IDs: a changed question or
        # media ordering with unchanged IDs must not pass as a matched dataset.
        inputs = path.parent / (path.stem + '-inputs'); inputs.mkdir(exist_ok=True)
        data = contract['source_data_receipt']
        sources = ([('train-' + name, source) for name, source in sorted(data['manifests'].items())] +
                   [('eval-' + name, source) for name, source in sorted(evaluation_sources(data).items())] +
                   [('primary-eval', data['eval_manifest'])])
        for index, (_, source) in enumerate(sources):
            destination = inputs / f'{index}.jsonl'
            if destination.exists():
                if destination.read_bytes() != Path(source).read_bytes():
                    raise ValueError('Cross-node paired manifest content changed')
            else:
                shutil.copy2(source, destination)
        if not path.exists(): write(path, contract)


def snapshot(plan, source=REPO):
    if not (source / 'scripts/train-geometry-rft.py').is_file():
        raise ValueError('RFT worker must exist before arming an immutable queue')
    root = Path(plan['root']).resolve(); root.mkdir(parents=True, exist_ok=True)
    saved = root / 'plan.json'
    if saved.exists() and read(saved) != plan: raise ValueError('Plan changed; use a new versioned root')
    write(saved, plan)
    code = root / 'code'
    if not code.exists():
        pending = root / 'code.pending'
        if pending.exists(): raise ValueError('Incomplete code snapshot exists; inspect without overwriting')
        pending.mkdir()
        for name in ('scripts', 'spatial_intelligence', 'configs', 'catalog'):
            shutil.copytree(source / name, pending / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        pending.rename(code)
    recipe = root / 'inputs/scientific-config.json'; recipe.parent.mkdir(exist_ok=True)
    source_bytes = Path(plan['scientific_config']).read_bytes()
    if recipe.exists() and recipe.read_bytes() != source_bytes: raise ValueError('Scientific recipe changed')
    if not recipe.exists(): recipe.write_bytes(source_bytes)
    return code


def resolve_inputs(plan):
    """Snapshot accepted manifests and derive the identical paired budget once."""
    root = Path(plan['root']); folder = root / 'inputs'; folder.mkdir(exist_ok=True)
    target = root / 'runtime-plan.json'
    if target.exists():
        runtime = read(target)
        if checkpoint_identity(plan['model_checkpoint']) != runtime['checkpoint_identity']:
            raise ValueError('Source checkpoint changed after acceptance')
        for key in ('sft_receipt', 'data_receipt'):
            if Path(plan[key]).read_bytes() != (folder / (key + '.json')).read_bytes():
                raise ValueError('Accepted source receipt changed')
        if plan.get('pair_contract_path'):
            verify_shared_pair(plan['pair_contract_path'], read(root / 'state/pair-contract.json'))
        return runtime
    validate_sft(plan); data = validate_data(read(plan['data_receipt']))
    recipe = read(folder / 'scientific-config.json')
    if recipe.get('group_size') != 8 or recipe.get('global_prompt_batch') != 16:
        raise ValueError('Scientific recipe disagrees with group/budget contract')
    if plan.get('training') is not None and plan['training'] != recipe.get('training'):
        raise ValueError('Private training override disagrees with immutable scientific recipe')
    paired = pair_contract(plan, data, recipe)
    if plan.get('pair_contract_path'):
        verify_shared_pair(plan['pair_contract_path'], paired)
    for key in ('sft_receipt', 'data_receipt'):
        shutil.copy2(plan[key], folder / (key + '.json'))
    manifests = {}
    for index, (name, source) in enumerate(sorted(data['manifests'].items())):
        destination = folder / f'train-{index}.jsonl'
        shutil.copy2(source, destination); manifests[name] = str(destination)
    evaluation = folder / 'evaluation.jsonl'; shutil.copy2(data['eval_manifest'], evaluation)
    evaluations = {}
    for index, (name, source) in enumerate(sorted(evaluation_sources(data).items())):
        destination = folder / f'evaluation-{index}.jsonl'
        shutil.copy2(source, destination); evaluations[name] = str(destination)
    prepared_data = dict(data, manifests=manifests, train_manifests=manifests,
                         validation_manifests={name: evaluations['validation_' + name]
                                               for name in data.get('validation_manifests', {})},
                         eval_manifest=str(evaluation), eval_manifests=evaluations)
    prepared_receipt = folder / 'prepared-data.json'; write(prepared_receipt, prepared_data)
    runtime = dict(plan, **budget(data['accepted_train_rows']), seed=3407,
        backend='hf_geometry_reference', scientific_config=str(folder / 'scientific-config.json'),
        manifests=manifests, train_manifests=manifests, eval_manifest=str(evaluation),
        eval_manifests=evaluations, data_root=data.get('data_root'),
        data_receipt=str(prepared_receipt), training=recipe.get('training', {}),
        sft_receipt=str(folder / 'sft_receipt.json'),
        jobs=[dict(job, four_d_rl_fraction=ARMS[job['name']][0],
                   spatial_fraction=ARMS[job['name']][1]) for job in plan['jobs']],
        checkpoint_identity=checkpoint_identity(plan['model_checkpoint']))
    write(target, runtime)
    write(root / 'state/pair-contract.json', paired)
    return runtime


def validate_mode_receipt(receipt, runtime, arm, mode):
    if receipt.get('status') != 'complete': raise ValueError('Worker mode did not complete')
    if receipt.get('arm') != arm or receipt.get('mode') != mode:
        raise ValueError('Worker receipt identity mismatch')
    if Path(receipt.get('initial_checkpoint', '')).resolve() != Path(runtime['model_checkpoint']).resolve():
        raise ValueError('Arms must use the same declared SFT checkpoint')
    if mode in ('gate', 'train'):
        required = ['finite_loss', 'nonzero_update', 'reload_verified', 'geometry_preserved']
        if mode == 'gate':
            required += ['reference_initial_parity', 'all_group_responses_verified']
            variation = receipt.get('within_group_reward_variation')
            if not isinstance(variation, (int, float)) or not math.isfinite(variation) or variation <= 0:
                raise ValueError('Missing real runtime gate: within_group_reward_variation')
        for field in required:
            if receipt.get(field) is not True: raise ValueError('Missing real runtime gate: ' + field)
        if receipt.get('group_size') != 8: raise ValueError('Logical rollout group was changed')
    if mode == 'train':
        for field in ('updates', 'prompt_budget'):
            if receipt.get(field) != runtime[field]: raise ValueError('Actual paired budget mismatch: ' + field)
        if not receipt.get('checkpoint') or not Path(receipt['checkpoint']).is_dir():
            raise ValueError('RFT checkpoint missing')
    if mode == 'evaluate':
        if receipt.get('paired_ids_verified') is not True or receipt.get('reload_verified') is not True:
            raise ValueError('Paired evaluation/reload evidence missing')
        if Path(receipt.get('eval_manifest', '')).resolve() != Path(runtime['eval_manifest']).resolve():
            raise ValueError('Evaluation IDs changed')


class Queue:
    def __init__(self, plan):
        self.plan = plan; self.root = Path(plan['root'])
        for name in ('state', 'logs', 'runs'): (self.root / name).mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, plan['gpus'])),
            PYTHONPATH=str(REPO), PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', TORCHINDUCTOR_COMPILE_THREADS='2',
            TORCH_EXTENSIONS_DIR=str(self.root / 'cache/torch-extensions'),
            TRITON_CACHE_DIR=str(self.root / 'cache/triton'))

    def status(self, phase, **extra):
        write(self.root / 'state/rft-queue.json', dict(status=phase, pid=os.getpid(), updated=time.time(), **extra))

    def idle(self):
        values = subprocess.check_output(['nvidia-smi', '--id=' + self.env['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        return len(values) == len(self.plan['gpus']) and all(int(value) < 500 for value in values)

    def wait(self):
        deadline = time.monotonic() + self.plan.get('wait_timeout_seconds', 30 * 86400)
        while True:
            if time.monotonic() > deadline: raise TimeoutError('Prerequisite wait exceeded explicit deadline')
            for dep in self.plan['dependencies']:
                if Path(dep['path']).exists():
                    prior = read(dep['path'])
                    if prior.get('status') in TERMINAL and prior.get('status') not in dep['statuses']:
                        raise ValueError('Predecessor terminated outside the allowed release policy')
            failure_path = self.plan.get('sft_failure_state')
            if failure_path and Path(failure_path).exists() and read(failure_path).get('status') == 'failed':
                source = read(self.plan['sft_receipt']) if Path(self.plan['sft_receipt']).exists() else {}
                if source.get('status') != 'complete' or source.get('reload_verified') is not True:
                    raise ValueError('Declared SFT producer failed without a verified formal checkpoint')
            if not all(dependency_ready(dep) for dep in self.plan['dependencies']):
                self.status('waiting_for_all_prior_gpu_work')
            elif not Path(self.plan['sft_receipt']).exists():
                self.status('waiting_for_formal_sft_checkpoint')
            elif read(self.plan['sft_receipt']).get('status') == 'complete' and read(self.plan['sft_receipt']).get('reload_verified') is not True:
                self.status('waiting_for_sft_reload_acceptance')
            elif not Path(self.plan['data_receipt']).exists():
                self.status('waiting_for_data')
            elif read(self.plan['data_receipt']).get('status') not in TERMINAL | {'ready'}:
                self.status('waiting_for_data')
            else:
                validate_sft(self.plan); validate_data(read(self.plan['data_receipt']))
                if self.idle(): return
                self.status('waiting_for_allocated_gpus')
            time.sleep(30)

    def run_mode(self, runtime, arm, mode):
        receipt = self.root / 'runs' / arm / mode / 'completion.json'
        if receipt.exists():
            validate_mode_receipt(read(receipt), runtime, arm, mode); return
        command = [self.plan['python'], '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=' + str(len(self.plan['gpus'])), str(REPO / 'scripts/train-geometry-rft.py'),
                   '--plan', str(self.root / 'runtime-plan.json'), '--arm', arm, '--mode', mode]
        self.status('running_' + mode, arm=arm)
        process_path = self.root / 'state' / f'process-{arm}-{mode}.json'
        if process_path.exists():
            previous = read(process_path)
            if previous.get('status') == 'running' and Path('/proc', str(previous['pid'])).exists():
                raise ValueError('Previous child still exists; refuse duplicate optimizer execution')
        with (self.root / 'logs' / f'{arm}-{mode}.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=REPO, env=self.env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(process_path, dict(status='running', pid=child.pid, command=command, started=time.time()))
            try:
                code = child.wait(timeout=self.plan.get(mode + '_timeout_seconds', 30 * 86400))
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                write(process_path, dict(status='failed', pid=child.pid, reason='interrupted_or_timeout'))
                raise
        write(process_path, dict(status='complete' if code == 0 else 'failed', returncode=code))
        if code: raise subprocess.CalledProcessError(code, command)
        validate_mode_receipt(read(receipt), runtime, arm, mode)

    def execute(self):
        import fcntl
        with (self.root / 'state/rft-queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.wait(); runtime = resolve_inputs(self.plan)
            except Exception as exc:
                self.status('blocked', error=repr(exc), gpu_work_finished=True); raise
            outcomes = {}
            for job in self.plan['jobs']:
                name = job['name']
                try:
                    # Revalidate fixed source identity before each independent arm.
                    resolve_inputs(self.plan)
                    while not self.idle():
                        self.status('waiting_for_allocated_gpus', arm=name); time.sleep(30)
                    for mode in ('gate', 'train', 'evaluate'): self.run_mode(runtime, name, mode)
                    result = dict(status='complete', finished=time.time())
                except Exception as exc:
                    result = dict(status='failed', error=repr(exc), finished=time.time())
                write(self.root / 'state' / f'arm-{name}.json', result); outcomes[name] = result
            self.status('complete' if all(x['status'] == 'complete' for x in outcomes.values()) else 'complete_with_failures',
                        arms=outcomes, gpu_work_finished=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--detach', action='store_true')
    args = p.parse_args(); plan = validate_plan(read(args.plan)); root = Path(plan['root'])
    if args.detach:
        frozen = snapshot(plan); (root / 'logs').mkdir(exist_ok=True)
        with (root / 'logs/rft-supervisor.log').open('ab') as log:
            child = subprocess.Popen([plan['python'], str(frozen / 'scripts' / Path(__file__).name),
                '--plan', str(root / 'plan.json')], cwd=frozen, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(status='waiting_queue_launched_not_training_acceptance', pid=child.pid, root=str(root))))
    else:
        def interrupted(signum, frame): raise InterruptedError('Supervisor interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        Queue(plan).execute()


if __name__ == '__main__': main()
