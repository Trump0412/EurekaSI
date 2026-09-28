"""Contract checks for the recovery-only runtime cap, not GPU acceptance."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]


def test_cap_keeps_logical_group_and_probes_reachable():
    text=(ROOT/'scripts/train-geometry-rft.py').read_text()
    ast.parse(text)
    assert "rollout_cap=int(plan.get('rollout_micro_cap',8))" in text
    assert 'if m<=rollout_cap' in text
    assert "plan.get('rollout_probe_indices',[])" in text
    assert 'sampled_row(datasets,arm,seed,int(probe_index))' in text
    assert 'group,rollout_micro,max_new' in text
    assert 'rollout_micro>rollout_cap' in text


def test_fatal_training_log_monitor_scoped_to_child():
    text=(ROOT/'scripts/run-geometry-rft-queue.py').read_text()
    ast.parse(text)
    assert "if mode=='train'" in text
    assert "b'CUDA out of memory'" in text
    assert 'os.killpg(child.pid, signal.SIGTERM)' in text
    assert 'monitor.seek(offset)' in text
