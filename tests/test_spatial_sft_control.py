"""CPU supervisor regression: no SSH, GPU access, or model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]


def load_supervisor(monkeypatch):
    # fcntl does not exist on the Windows orchestration host. Mock only the lock;
    # on Linux this also avoids acquiring any real deployment lock in this test.
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(
        LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None))
    spec = importlib.util.spec_from_file_location(
        'sft_control_test', REPO/'scripts/run-spatial-sft-control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.mark.parametrize('training_returncode', [0, 7])
def test_gpu_gate_enters_train_preserves_matched_control(tmp_path, monkeypatch, training_returncode):
    supervisor = load_supervisor(monkeypatch)
    root, reference = tmp_path/'node', tmp_path/'reference'
    for folder in ('state', 'logs', 'runs'):
        (root/folder).mkdir(parents=True)
    cfg = yaml.safe_load((REPO/'configs/rl.yaml').read_text(encoding='utf-8'))
    cfg['train'].update(steps=100, batch_size=1, gradient_accumulation=1,
                        mode='rl', group_size=4, algorithm='gspo', seed=3410)
    cfg['model']['backend'] = 'spatial_intelligence.backends.qwen35:Qwen35Backend'
    cfg['model']['lora'].update(rank=8, alpha=16, target_modules=['q_proj', 'v_proj'])
    rows = [{'dataset': 'spar', 'id': 'a', 'answer': 'Yes'},
            {'dataset': 'spar', 'id': 'b', 'answer': 'No'}]
    row_path = tmp_path/'val.jsonl'
    row_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg['data']['eval'] = str(row_path)
    write_json(reference/'pilot.json', cfg)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg['train']['mode'] = 'sft'
    eval_cfg['protocol']['generation'] = {'do_sample': False, 'max_new_tokens': 64}
    write_json(reference/'baseline-val.json', eval_cfg)
    metric = {'protocol_hash': 'p', 'dataset_hash': 'd', 'scorer_hash': 's',
              'model': {}, 'accuracy': .5, 'n': 2, 'parse_rate': 1.}
    write_json(reference/'baseline-val/metrics.json', metric)
    details = [{'id': 'spar::a', 'parsed': 'yes', 'correct': True},
               {'id': 'spar::b', 'parsed': 'yes', 'correct': False}]
    (reference/'baseline-val/metrics.details.jsonl').write_text(
        ''.join(json.dumps(r)+'\n' for r in details))
    gpu_outputs = iter(['12000\n0\n', '0\n0\n'])
    gpu_calls, sleeps, stages = [], [], []

    def fake_gpu(command, **kwargs):
        assert not stages, 'No subprocess training may start before idle GPU gate'
        assert '--id=2,3' in command
        gpu_calls.append(command)
        return next(gpu_outputs)

    def fake_run(command, **kwargs):
        stage = command[command.index('spatial_intelligence')+1]
        stages.append(stage)
        assert '--nproc_per_node=2' in command
        assert command[0] == str(root/'envs/qwen35/bin/python')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2,3'
        state = json.loads((root/'state/spatial-sft-control.json').read_text())
        assert state['status'] == 'running'
        assert state['stage'] == ('train' if stage == 'train' else 'post-val')
        # Reaching this line catches the original status(stage, stage=...) clash.
        if stage == 'train':
            return SimpleNamespace(returncode=training_returncode)
        post = root/'runs/control/post-val'
        write_json(post/'metrics.json', {**metric, 'accuracy': 1.})
        updated = [details[0], {**details[1], 'parsed': 'no', 'correct': True}]
        (post/'metrics.details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in updated))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(supervisor.subprocess, 'check_output', fake_gpu)
    monkeypatch.setattr(supervisor.subprocess, 'run', fake_run)
    monkeypatch.setattr(supervisor.time, 'sleep', sleeps.append)
    monkeypatch.setattr(sys, 'argv', ['run-spatial-sft-control.py', '--root', str(root),
                                     '--reference-run', str(reference), '--name', 'control'])
    if training_returncode:
        with pytest.raises(RuntimeError, match='train failed'):
            supervisor.main()
        assert stages == ['train']
        assert json.loads((root/'state/spatial-sft-control.json').read_text())['status'] == 'failed'
    else:
        supervisor.main()
        assert stages == ['train', 'infer']
        assert json.loads((root/'state/spatial-sft-control.json').read_text())['status'] == 'complete'
        comparison = json.loads((root/'runs/control/comparison.json').read_text())
        assert comparison['balanced_accuracy_delta'] == .5
    assert len(gpu_calls) == 2 and sleeps == [10]
    trained = json.loads((root/'runs/control/train.json').read_text())
    assert trained['train']['mode'] == 'sft'
    assert trained['train']['gradient_accumulation'] == 2
    assert trained['train']['batch_size'] == 1
    assert 2*trained['train']['batch_size']*trained['train']['gradient_accumulation'] == 4
    assert 4*cfg['train']['batch_size']*cfg['train']['gradient_accumulation'] == 4
    assert trained['train']['steps'] == 100
    assert trained['train']['reward_plugin'] is None
    for key in ('model', 'data', 'protocol'):
        assert trained[key] == cfg[key]
    experiment = json.loads((root/'runs/control/experiment.json').read_text())
    assert experiment['prompt_draws'] == 400
    assert experiment['compute_matched'] is False
