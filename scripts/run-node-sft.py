"""Isolated node SFT replica: immutable shared inputs, independent seed/env/outputs.

Requires an independently cloned qwen35 conda prefix. No installation into the
shared source environment, no data preparation or writes to shared input root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(path)


def worker(args):
    import transformers
    from spatial_intelligence import study
    original_class = transformers.TrainingArguments
    original_init = original_class.__init__
    transformers.set_seed(args.seed)
    def seeded_arguments(self, *pos, **kw):
        kw['seed'] = args.seed
        kw['data_seed'] = args.seed
        # Patch the resolved class, not a LazyModule export that a later import
        # can silently restore. Keep the original pickleable dataclass type.
        original_init(self, *pos, **kw)
    original_class.__init__ = seeded_arguments
    study.train(Path(args.root), args.model, args.name, args.micro, args.ga, args.max_steps)


def snapshot(source, root):
    """Copy manifests/receipts once; audit IDs and disjoint scenes independently."""
    for relative in ['manifests/sft.train.jsonl', 'manifests/revsi32.test.jsonl',
                     'manifests/train-exclusions.jsonl', 'receipts/train-prepared.json']:
        src, dst = source / relative, root / relative
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            temp = dst.with_suffix('.copying')
            shutil.copyfile(src, temp)
            temp.replace(dst)
    receipt = json.loads((root / 'receipts/train-prepared.json').read_text())
    if receipt['status'] != 'complete' or receipt['scene_overlap'] != 0:
        raise ValueError('Shared training preparation is not complete')
    heldout = set()
    with (root / 'manifests/revsi32.test.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            heldout.add(row['scene_id'])
    ids, counts = set(), {}
    with (root / 'manifests/sft.train.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['id'] in ids or row['scene_id'] in heldout:
                raise ValueError('Duplicate ID or heldout scene in training manifest')
            if row['id'] != f"{row['dataset']}::row::{row['source_index']}":
                raise ValueError('Expected repaired pinned-source row identity')
            ids.add(row['id'])
            counts[row['dataset']] = counts.get(row['dataset'], 0) + 1
    if len(ids) != receipt['rows'] or counts != receipt['counts']:
        raise ValueError('Manifest/receipt count mismatch')
    write(root / 'receipts/node-input-audit.json', {
        'status': 'complete', 'rows': len(ids), 'counts': counts,
        'source_root': str(source), 'copied_manifest': True,
        'media_policy': 'Original absolute shared paths, read-only reuse',
        'source_ID_policy': 'dataset::row::source_index; never deduplicate source IDs'})
    return len(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--seed', type=int, default=3408)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--detach', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--model'); p.add_argument('--name')
    p.add_argument('--micro', type=int, default=1)
    p.add_argument('--ga', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=-1)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    root, source = Path(args.root).resolve(), Path(args.source_root).resolve()
    if root == source or source.is_relative_to(root):
        raise ValueError('Node root must not contain/overwrite the shared input root')
    for folder in ['logs', 'state', 'receipts', 'runs', 'cache']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (root / 'logs/node-supervisor.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                *[x for x in sys.argv[1:] if x != '--detach']], stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print('Node SFT supervisor PID', process.pid)
        return
    lock = (root / 'state/node-sft.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('supervisor', REPO / 'scripts/run-qwen35-study.py')
    supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
    py = str(root / 'envs/qwen35/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1', HF_ENDPOINT='https://hf-mirror.com',
        HF_HOME=str(root / 'cache/huggingface'), PYTHONPATH=str(REPO))
    model = str(source / 'models/Qwen3.5-2B')
    n = len(args.gpus.split(','))
    if n != 8:
        raise ValueError('This node replica is locked to eight GPUs')
    stop = threading.Event()
    threading.Thread(target=supervisor.telemetry, args=(root, stop), daemon=True).start()
    def run(stage, command, override=None):
        supervisor.run(root, stage, command, env if override is None else override)
    try:
        write(root / 'state/node-study.json', {'status': 'waiting_environment', 'pid': os.getpid()})
        # conda --copy clone may still be copying untracked pip dependencies.
        while True:
            cloning = False
            for proc in Path('/proc').iterdir():
                try:
                    cmd = proc.joinpath('cmdline').read_bytes().decode().split('\0')
                    if '--clone' in cmd and '--prefix' in cmd:
                        target = Path(cmd[cmd.index('--prefix') + 1]).resolve()
                        if target == root / 'envs/qwen35':
                            cloning = True
                except (OSError, UnicodeError):
                    pass
            if not cloning and Path(py).exists():
                break
            time.sleep(15)
        run('node-install-editable', [py, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', str(REPO)])
        run('node-pip-check', [py, '-m', 'pip', 'check'])
        run('node-cpu-tests', [py, '-m', 'pytest', 'tests/test_qwen35_study.py',
            'tests/test_qwen35_posttraining.py', 'tests/test_marker_cache.py', '-q'])
        count = snapshot(source, root)
        write(root / 'receipts/node-experiment.json', {'seed': args.seed, 'rows': count,
            'world_size': n, 'global_batch': 64, 'epochs': 1, 'freeze_vision': True,
            'model': model, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'scope': 'Independent seed replica; 8-rank vs 4-rank numerical/tail-padding differences apply'})
        usage = subprocess.check_output(['nvidia-smi', '--id=' + args.gpus,
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if any(int(v.strip()) >= 500 for v in usage.splitlines()):
            raise RuntimeError('Allocated GPU occupied; refuse collision')
        def profile(item):
            idx, batch = item
            run(f'node-profile-b{batch}', [py, '-m', 'spatial_intelligence.study', 'profile',
                '--root', str(root), '--model', model, '--batch-size', str(batch)],
                dict(env, CUDA_VISIBLE_DEVICES=args.gpus.split(',')[idx]))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(profile, enumerate([1, 2, 4])))
        profiles = [json.loads((root / f'receipts/profile-b{b}.json').read_text()) for b in [1, 2, 4]]
        safe = [x for x in profiles if x['status'] == 'complete' and x['peak_reserved_gib'] < .92 * x['gpu_total_gib']]
        if not safe:
            raise RuntimeError('No safe measured micro-batch')
        selected = max(safe, key=lambda x: x['samples_per_second'])
        micro = selected['batch_size']; ga = 64 // (n * micro)
        write(root / 'receipts/batch-selection.json', {'status': 'complete', 'selected': selected,
            'candidates': profiles, 'world_size': n, 'global_batch': 64, 'ga': ga})
        launch = [py, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={n}']
        trainer = launch + [str(Path(__file__).resolve()), '--worker', '--root', str(root),
            '--source-root', str(source), '--seed', str(args.seed), '--model', model, '--micro', str(micro)]
        name = f'sft-spar234k-hound64k-seed{args.seed}'
        run('node-sft-smoke', trainer + ['--name', 'node-sft-smoke', '--ga', '1', '--max-steps', '2'])
        write(root / 'state/node-study.json', {'status': 'training', 'pid': os.getpid(), 'run': name})
        run('node-sft', trainer + ['--name', name, '--ga', str(ga)])
        evaluated = f'sft-revsi32-seed{args.seed}'
        run('node-revsi-eval', launch + ['-m', 'spatial_intelligence.study', 'eval', '--root', str(root),
            '--model', str(root / 'runs' / name / 'final'), '--name', evaluated, '--batch-size', '2'])
        run('node-revsi-score', [py, '-m', 'spatial_intelligence.study', 'merge-eval', '--root', str(root), '--name', evaluated])
        before = json.loads((source / 'runs/baseline-revsi32/metrics.json').read_text())
        after = json.loads((root / 'runs' / evaluated / 'metrics.json').read_text())
        write(root / 'runs/comparison.json', {'baseline': before, 'sft': after,
            'baseline_source': str(source / 'runs/baseline-revsi32'), 'seed': args.seed,
            'delta': after['overall_acc'] - before['overall_acc']})
        write(root / 'state/node-study.json', {'status': 'complete', 'finished': time.time()})
    except Exception as exc:
        write(root / 'state/node-study.json', {'status': 'failed', 'error': str(exc), 'time': time.time()})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
