"""Isolated node bootstrap -> real-image native VERL GSPO gate (never a full run claim).

The environment must be an independent --copy Conda clone. Shared source models
and manifests are read-only inputs; node state, logs and checkpoints are private.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--name', default='diagnostic-verl-node-v1')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root, source = Path(args.root).resolve(), Path(args.source_root).resolve()
    if root == source:
        raise ValueError('Node root must not be the shared source root')
    for folder in ('logs', 'state', 'receipts', 'runs'):
        (root / folder).mkdir(exist_ok=True, parents=True)
    if args.detach:
        with (root / 'logs/native-rl-supervisor.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, __file__, *[v for v in sys.argv[1:] if v != '--detach']],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'root': str(root)}))
        return
    import fcntl
    lock = (root / 'state/native-rl-supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **kwargs):
        record = {'stage': stage, 'time': time.time(), 'pid': os.getpid(),
                  'devices': args.devices, 'source_root_read_only': str(source), **kwargs}
        (root / 'state/native-rl-supervisor.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    def command(argv, logfile):
        with (root / 'logs' / logfile).open('a') as log:
            subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        status('waiting_for_independent_conda_clone')
        prefix = root / 'envs/verl-legacy'
        deadline = time.monotonic() + 10800
        while True:
            cloning = False
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    tokens = proc.read_bytes().decode().split('\0')
                except (OSError, UnicodeError):
                    continue
                if '--clone' in tokens and '--prefix' in tokens:
                    destination = tokens[tokens.index('--prefix') + 1]
                    if Path(destination).resolve() == prefix:
                        cloning = True
            if not cloning:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Conda clone exceeded 3 hours; inspect clone log')
            time.sleep(15)
        py = prefix / 'bin/python'
        if not py.is_file():
            raise FileNotFoundError(f'Independent environment missing: {py}')
        status('validating_independent_environment')
        command([str(py), '-m', 'pip', 'install', '--no-deps', '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',
                 '-r', str(REPO / 'requirements/verl-node-compat.txt')], 'native-rl-install.log')
        command([str(py), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', str(REPO)], 'native-rl-install.log')
        command([str(py), '-m', 'pip', 'check'], 'native-rl-install.log')
        command([str(py), str(REPO / 'scripts/patch-verl-context.py')], 'native-rl-install.log')
        command([str(py), str(REPO / 'scripts/patch-verl-executor.py')], 'native-rl-install.log')
        for name in ('models', 'data'):
            destination = root / name
            if not destination.exists() and not destination.is_symlink() and (source / name).exists():
                destination.symlink_to(source / name, target_is_directory=True)
        (root / 'manifests').mkdir(exist_ok=True)
        for name in ('sft-probe.train.jsonl', 'revsi32.test.jsonl', 'sft.train.jsonl'):
            destination = root / 'manifests' / name
            if not destination.exists() and not destination.is_symlink():
                destination.symlink_to(source / 'manifests' / name)
        receipt = json.loads((source / 'receipts/geometry-qwen3vl2b.json').read_text())
        if receipt.get('status') != 'complete':
            raise ValueError('Source Qwen3-VL receipt incomplete')
        (root / 'receipts/geometry-qwen3vl2b.json').write_text(json.dumps(receipt, indent=2))
        (root / 'state/extension-env-verl.json').write_text(json.dumps({'status': 'complete',
            'environment': str(prefix), 'independent_copy': True, 'repo': str(REPO), 'time': time.time()}, indent=2))
        status('starting_real_image_gspo_gate')
        out = root / 'runs' / args.name
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices, PYTHONNOUSERSITE='1')
        subprocess.run([sys.executable, str(REPO / 'scripts/verify-node-verl-gspo.py'), '--root', str(root),
                        '--output', str(out)], cwd=REPO, env=env, check=True)
        result = json.loads((out / 'status.json').read_text())
        status('gate_finished_starting_update_and_reload_audit', result=result)
        subprocess.run([str(py), str(REPO / 'scripts/audit-verl-gate.py'), '--root', str(root),
                        '--output', str(out)], cwd=REPO, env=env, check=True)
        status('native_gate_accepted', result=json.loads((out / 'acceptance.json').read_text()))
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
