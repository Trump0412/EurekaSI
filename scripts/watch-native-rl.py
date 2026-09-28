"""Bounded detached health/ETA observer for one explicitly named native run.

No GPU utilization is mistaken for optimizer progress. Timeouts terminate only
the verified supervisor's descendant process tree, never global Ray or GPUs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time


def optimizer_steps(text):
    """Only the named training counter is progress; step durations are not."""
    return max((int(v) for v in re.findall(r'(?<![\w/])training/global_step:\s*(\d+)(?=\s|$)', text)), default=0)


def descendants(root_pid):
    parents = {}
    for entry in Path('/proc').glob('[0-9]*/status'):
        try:
            text = entry.read_text()
            parents[int(entry.parent.name)] = int(re.search(r'^PPid:\s+(\d+)', text, re.M).group(1))
        except (OSError, AttributeError):
            continue
    result = [root_pid]
    for parent in result:
        result.extend(pid for pid, ppid in parents.items() if ppid == parent and pid not in result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--supervisor-pid', required=True, type=int)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max-hours', type=float, default=24)
    parser.add_argument('--stale-minutes', type=float, default=90)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if args.detach:
        with (out / 'watchdog.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, __file__, *[v for v in sys.argv[1:] if v != '--detach']],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'output': str(out)})); return
    started, progressed, last_step = time.time(), time.time(), 0
    identity = Path(f'/proc/{args.supervisor_pid}/cmdline')
    try:
        original = identity.read_bytes()
    except OSError:
        original = b''
    if original and str(out).encode() not in original:
        raise ValueError('PID does not own the exact requested output path')
    while True:
        text = (out / 'trainer.log').read_text(errors='replace') if (out / 'trainer.log').exists() else ''
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        completed = optimizer_steps(text)
        durations = [float(v) for v in re.findall(r'timing_s/step:([0-9.eE+-]+)', text)]
        if completed > last_step:
            progressed, last_step = time.time(), completed
        state = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
        try:
            alive = bool(original) and identity.read_bytes() == original
        except OSError:
            alive = False
        health = {'time': time.time(), 'supervisor_alive': alive, 'optimizer_steps': completed,
            'planned_steps': args.steps, 'stage': state.get('status'), 'warming_up': completed < 10,
            'remaining_hours_training_only': statistics.median(durations[-5:]) * max(0, args.steps - completed) / 3600 if durations else None,
            'observed_step_seconds': durations[-5:], 'max_wall_hours': args.max_hours, 'stale_optimizer_minutes': args.stale_minutes}
        (out / 'health.json').write_text(json.dumps(health, indent=2))
        if not alive:
            if state.get('status') not in ('native_training_exited_needs_audit', 'failed', 'blocked_gate_failed', 'blocked_gate_timeout', 'blocked_gpu_timeout', 'cancelled_by_user'):
                state.update(status='failed_supervisor_exit', watchdog_time=time.time(),
                    error='Supervisor exited before recording completion; inspect supervisor/trainer logs')
                (out / 'status.json').write_text(json.dumps(state, indent=2))
            return
        timed_out = time.time() - started > args.max_hours * 3600 or time.time() - progressed > args.stale_minutes * 60
        if timed_out:
            targets = descendants(args.supervisor_pid)
            state.update(status='failed_watchdog_timeout', watchdog_time=time.time(), terminated_pids=targets)
            (out / 'status.json').write_text(json.dumps(state, indent=2))
            for pid in reversed(targets):
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
