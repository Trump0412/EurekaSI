"""CPU orchestration tests: subprocesses and GPU queries are mocked."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_queue(monkeypatch):
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None))
    spec = importlib.util.spec_from_file_location('geometry_queue_test', REPO/'scripts/run-qwen3vl-geometry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_queue(tmp_path, monkeypatch, *, fault=None):
    module = load_queue(monkeypatch)
    baseline = tmp_path/'baseline'
    root = tmp_path/'geometry'
    state = baseline/'state/baseline.json'
    store(state, {'status': 'running'})
    for benchmark in ('revsi', 'vsibench'):
        store(baseline/f'manifests/{benchmark}32.test.jsonl', {'id': benchmark})
    store(baseline/'receipts/train-prepared.json', {'status': 'complete'})
    gate = tmp_path/'gate.json'
    store(gate, {'status': 'accepted' if fault != 'gate' else 'failed'})
    args = SimpleNamespace(root=root, detach=False, gpus='0,1,2,3,4,5,6,7',
        baseline_state=state, gate=gate, model='/released/Qwen3-VL-2B-Instruct',
        source='/source/vggt', weights='/weights/vggt.pt', name='geometry-sft-test')
    calls, sleeps, gpu_calls = [], [], []

    def sleep(seconds):
        sleeps.append(seconds)
        store(state, {'status': 'failed' if fault == 'baseline' else 'complete', 'gpu_work_finished': True})

    def query(command, **kwargs):
        assert json.loads(state.read_text())['gpu_work_finished']
        assert not calls
        gpu_calls.append(command)
        return ('12000\n'+'0\n'*7) if len(gpu_calls) == 1 else '0\n'*8

    def run(command, **kwargs):
        assert len(gpu_calls) == 2, 'Wait for all allocated GPUs before launching'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == args.gpus
        assert kwargs['start_new_session'] is True
        mode = command[command.index('--mode')+1]
        name = command[command.index('--name')+1]
        model = command[command.index('--model')+1]
        calls.append((mode, name, model, command))
        if mode == 'train-worker':
            assert model == args.model, 'Never initialize geometry SFT from baseline final'
            if '--max-steps' in command:
                assert command[command.index('--ga')+1] == '1'
                store(root/f'runs/{name}/completion.json',
                      {'status': 'complete', 'steps': 1 if fault == 'smoke' else 2})
            elif fault == 'train':
                return SimpleNamespace(pid=12345, wait=lambda **kwargs: 1)
            else:
                (root/f'runs/{name}/final').mkdir(parents=True)
        elif '--merge' in command:
            metrics = {'parse_rate': .95, 'truncation_rate': .0}
            if fault == 'format':
                metrics['parse_rate'] = .5
            if fault == 'missing_metric':
                metrics.pop('truncation_rate')
            store(root/f'runs/{name}/metrics.json', metrics)
        return SimpleNamespace(pid=12345, wait=lambda **kwargs: 0)

    monkeypatch.setattr(module.time, 'sleep', sleep)
    monkeypatch.setattr(module.subprocess, 'check_output', query)
    monkeypatch.setattr(module.subprocess, 'Popen', run)
    monkeypatch.setattr(module.subprocess, 'run',
                        lambda *args, **kwargs: pytest.fail('Unexpected real subprocess.run path'))
    return module, args, calls, sleeps, gpu_calls


def test_queue_waits_then_trains_from_base_and_runs_normal_null_evaluation(tmp_path, monkeypatch):
    module, args, calls, sleeps, gpu_calls = setup_queue(tmp_path, monkeypatch)
    module.queue(args)
    assert sleeps == [30, 30] and len(gpu_calls) == 2
    assert [call[0] for call in calls[:2]] == ['train-worker', 'train-worker']
    assert len(calls) == 14  # smoke+train, two benchmarks each smoke/normal/null plus merges.
    final = str(args.root/'runs'/args.name/'final')
    for _, name, model, command in calls[2:]:
        assert model == final
        assert ('--null-geometry' in command) == name.endswith('-null')
    contract = json.loads((args.root/'state/experiment-contract.json').read_text())
    assert contract['global_batch'] == 64 and contract['ga'] == 8
    assert contract['initialization'] == 'released base, not baseline SFT'
    assert contract['vggt'] == 'frozen-online-final-patches'
    assert json.loads((args.root/'state/geometry-queue.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('fault,exception,expected_calls', [
    ('baseline', RuntimeError, 0), ('gate', ValueError, 0),
    ('smoke', ValueError, 1), ('train', subprocess.CalledProcessError, 2),
    ('missing_metric', KeyError, 4),
])
def test_queue_fails_closed_and_records_exception(tmp_path, monkeypatch, fault, exception, expected_calls):
    module, args, calls, _, gpu_calls = setup_queue(tmp_path, monkeypatch, fault=fault)
    with pytest.raises(exception):
        module.queue(args)
    assert len(calls) == expected_calls
    state = json.loads((args.root/'state/geometry-queue.json').read_text())
    assert state['status'] == 'failed' and state['error']
    if fault == 'baseline':
        assert not gpu_calls


def test_bad_format_blocks_full_benchmark_without_using_accuracy(tmp_path, monkeypatch):
    module, args, calls, _, _ = setup_queue(tmp_path, monkeypatch, fault='format')
    module.queue(args)
    assert len(calls) == 4
    state = json.loads((args.root/'state/geometry-queue.json').read_text())
    assert state['status'] == 'training_complete_evaluation_blocked'
    assert state['gpu_work_finished'] is True


def test_empty_gpu_inventory_fails_closed(tmp_path, monkeypatch):
    module, args, calls, _, _ = setup_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: '')
    with pytest.raises(ValueError, match='eight memory measurements'):
        module.queue(args)
    assert not calls
    assert json.loads((args.root/'state/geometry-queue.json').read_text())['status'] == 'failed'


def test_completed_training_receipts_skip_smoke_and_training_on_restart(tmp_path, monkeypatch):
    module, args, calls, _, _ = setup_queue(tmp_path, monkeypatch)
    store(args.root/'runs/diagnostic-geometry-ddp/completion.json', {'status': 'complete', 'steps': 2})
    store(args.root/'runs'/args.name/'completion.json', {'status': 'complete', 'steps': 4655})
    (args.root/'runs'/args.name/'final').mkdir()
    module.queue(args)
    assert len(calls) == 12 and all(call[0] == 'eval-worker' for call in calls)


def test_timeout_terminates_only_the_started_process_group(tmp_path, monkeypatch):
    module, args, _, _, _ = setup_queue(tmp_path, monkeypatch)
    waits, signals = [], []

    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is True
        def wait(timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired(command, timeout)
            return 0
        return SimpleNamespace(pid=12345, wait=wait)

    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: signals.append((pid, sig)), raising=False)
    with pytest.raises(subprocess.TimeoutExpired):
        module.queue(args)
    assert waits == [172800, 30]
    assert len(signals) == 1 and signals[0][0] == 12345
    assert json.loads((args.root/'state/geometry-queue.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('null_geometry', [False, True])
def test_eval_mounts_final_processor_and_passes_online_geometry_and_null(tmp_path, monkeypatch, null_geometry):
    module = load_queue(monkeypatch)
    recorded = {}

    class TensorStub:
        shape = (32, 1024, 2048)
        def __getitem__(self, key):
            return self

    class ProcessorStub:
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            recorded['processor_path'] = path
            return 'final-processor'

    class ExtractorStub:
        def __init__(self, source, weights, device):
            recorded['extractor'] = (source, weights, device)
        def extract(self, media):
            recorded['media'] = media
            return TensorStub(), {}

    def add_slots(native, processor, features, padding, **kwargs):
        recorded['null'] = kwargs['force_null']
        assert processor == 'final-processor' and native == {'native_video': True}
        return {'geometry_slots': 64}

    fake_transformers = SimpleNamespace(AutoProcessor=ProcessorStub)
    def zeros(shape, **kwargs):
        recorded.setdefault('zeros', []).append((shape, kwargs.get('dtype')))
        return TensorStub()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bool='bool', bfloat16='bfloat16', zeros=zeros))
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setitem(sys.modules, 'spatial_intelligence.qwen3vl_geometry', SimpleNamespace(add_geometry_slots=add_slots))
    monkeypatch.setitem(sys.modules, 'spatial_intelligence.vggt_features', SimpleNamespace(FrozenVGGT=ExtractorStub))
    monkeypatch.setattr(module, 'install', lambda args: recorded.update(installed=args.model))
    fake_evaluator = SimpleNamespace(native_video_inputs=lambda *args: {'native_video': True})

    def evaluate():
        processor = fake_transformers.AutoProcessor.from_pretrained('/wrong/shared/model')
        result = fake_evaluator.native_video_inputs(processor, {'media': ['a.jpg', 'b.jpg']}, 'tagged')
        assert result['geometry_slots'] == 64
        assert '--merge' in sys.argv and '--benchmark' in sys.argv

    fake_evaluator.main = evaluate
    monkeypatch.setattr(module, 'module', lambda *args: fake_evaluator)
    monkeypatch.setattr(sys, 'argv', ['placeholder'])
    monkeypatch.setenv('LOCAL_RANK', '3')
    args = SimpleNamespace(root=tmp_path, model='/geometry/final', source='/vggt/code',
                           weights='/vggt/weights', name='geometry-null', null_geometry=null_geometry)
    module.evaluation(args, ['--benchmark', 'revsi', '--merge'])
    assert recorded['installed'] == '/geometry/final'
    assert recorded['processor_path'] == '/geometry/final'
    assert recorded['null'] is null_geometry
    if null_geometry:
        assert 'extractor' not in recorded and 'media' not in recorded
        assert recorded['zeros'][0] == ((2, 1024, 2048), 'bfloat16')
    else:
        assert recorded['extractor'] == ('/vggt/code', '/vggt/weights', 'cuda:3')
        assert recorded['media'] == ['a.jpg', 'b.jpg']
