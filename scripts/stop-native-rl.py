"""Stop only an explicitly verified native RL tree, preserving all artifacts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        base = Path('/proc') / str(pid)
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'starttime': stat[19], 'state': stat[0],
                'cmdline': (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
    except OSError:
        return None


def same_alive(item):
    now = identity(item['pid'])
    return now and now['starttime'] == item['starttime'] and now['state'] != 'Z'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--supervisor-pid', type=int, required=True)
    p.add_argument('--watchdog-pid', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    spec = importlib.util.spec_from_file_location('watch', Path(__file__).with_name('watch-native-rl.py'))
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    for pid, script in [(args.supervisor_pid, 'run-verl-spatial.py'), (args.watchdog_pid, 'watch-native-rl.py')]:
        item = identity(pid)
        if not item or script not in item['cmdline'] or str(out) not in item['cmdline']:
            raise RuntimeError(f'Refusing unverified owner PID {pid}')
    targets = [identity(pid) for pid in [args.watchdog_pid, *watch.descendants(args.supervisor_pid)]]
    targets = [item for item in targets if item]
    receipt = {'time': time.time(), 'targets': targets, 'execute': args.execute,
               'previous_health': json.loads((out / 'health.json').read_text()),
               'checkpoint_files': [str(v) for v in (out / 'checkpoints').rglob('*') if v.is_file()]}
    if not args.execute:
        print(json.dumps(receipt, indent=2)); return
    receipt_path = out / f'stop-receipt-{int(time.time())}.json'
    receipt_path.write_text(json.dumps(receipt, indent=2))
    # Stop spawning before taking a final descendant snapshot. Every signal is
    # identity-checked; never address a process group or global Ray namespace.
    for item in targets:
        if same_alive(item):
            try: os.kill(item['pid'], signal.SIGSTOP)
            except ProcessLookupError: pass
    known = {item['pid'] for item in targets}
    for pid in watch.descendants(args.supervisor_pid):
        if pid not in known:
            item = identity(pid)
            if item: targets.append(item)
    for item in targets:
        if same_alive(item):
            try:
                os.kill(item['pid'], signal.SIGTERM)
                os.kill(item['pid'], signal.SIGCONT)
            except ProcessLookupError: pass
    deadline = time.time() + 10
    while time.time() < deadline and any(same_alive(item) for item in targets):
        time.sleep(0.5)
    for item in targets:
        if same_alive(item):
            try: os.kill(item['pid'], signal.SIGKILL)
            except ProcessLookupError: pass
    time.sleep(2)
    completed = watch.optimizer_steps((out / 'trainer.log').read_text(errors='replace'))
    receipt.update(targets=targets, completed_optimizer_steps=completed,
                   surviving_pids=[item['pid'] for item in targets if same_alive(item)],
                   formal_checkpoint_saved=bool(receipt['checkpoint_files']))
    receipt_path.write_text(json.dumps(receipt, indent=2))
    state = json.loads((out / 'status.json').read_text())
    state.update(status='cancelled_by_user', reason='Reallocate this node to 8-GPU Qwen3VL SFT and VGGT fusion SFT',
                 completed_optimizer_steps=completed, stop_receipt=str(receipt_path),
                 unsaved_updates_lost=completed if not receipt['checkpoint_files'] else None)
    (out / 'status.json').write_text(json.dumps(state, indent=2))
    (out / 'health.json').write_text(json.dumps({'time': time.time(), 'stage': 'cancelled_by_user',
        'supervisor_alive': False, 'optimizer_steps': completed, 'planned_steps': 100,
        'remaining_hours_training_only': None, 'stop_receipt': str(receipt_path)}, indent=2))
    print(json.dumps({key: value for key, value in receipt.items() if key not in ('targets', 'previous_health')}, indent=2))


if __name__ == '__main__':
    main()
