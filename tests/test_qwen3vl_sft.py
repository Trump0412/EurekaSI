"""Model-family dispatch must not silently keep the other family's processor."""
import importlib.util
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip('fcntl')
pytest.importorskip('torch')


def entry():
    spec = importlib.util.spec_from_file_location('qwen3vl_sft', Path(__file__).parents[1] / 'scripts/run-qwen3vl-sft.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_wrong_family_rejected(monkeypatch):
    fake = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(model_type='qwen3_5')),
        Qwen3VLForConditionalGeneration=object)
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    with pytest.raises(ValueError, match='only accepts Qwen3-VL'):
        entry().load_qwen3vl('model')


def test_training_freezes_visual_only(monkeypatch):
    import torch
    visual, language = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True),
        named_parameters=lambda: iter([('model.visual.patch.weight', visual), ('model.language_model.weight', language)]),
        gradient_checkpointing_enable=lambda **kw: None, train=lambda: None)
    fake = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(model_type='qwen3_vl')),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    assert entry().load_qwen3vl('model', training=True) is model
    assert not visual.requires_grad and language.requires_grad
    assert not model.config.use_cache


def test_eval_redirects_processor_to_declared_model(monkeypatch, tmp_path):
    from spatial_intelligence import qwen35
    mod = entry()
    calls = []
    class Original:
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            calls.append(path)
            return path
    fake = SimpleNamespace(AutoProcessor=Original)
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    # Ensure process-local loader replacement is restored after the test.
    monkeypatch.setattr(qwen35, 'load_model', qwen35.load_model)
    monkeypatch.setattr(sys, 'argv', [])
    def evaluate():
        assert fake.AutoProcessor.from_pretrained('legacy-wrong-family-path') == tmp_path / 'model'
    monkeypatch.setattr(mod, 'module', lambda *args: SimpleNamespace(main=evaluate))
    args = SimpleNamespace(root=tmp_path, model=tmp_path / 'model', name='paired')
    mod.eval_worker(args, ['--benchmark', 'revsi'])
    assert calls == [tmp_path / 'model']


def test_real_lazy_import_seed_and_checkpoint_pickle(tmp_path):
    # A fresh interpreter is intentional: earlier mock transformer modules must
    # not hide LazyModule export restoration, the production failure mode.
    code = '''
import importlib.util, pathlib, sys, torch
root = pathlib.Path.cwd()
spec = importlib.util.spec_from_file_location('qwen3vl_sft', root/'scripts/run-qwen3vl-sft.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.install_loader(3408)
from transformers import AutoProcessor, Trainer, TrainingArguments
args = TrainingArguments(output_dir=sys.argv[1], seed=3407, data_seed=3407, use_cpu=True, report_to=[])
assert args.seed == args.data_seed == 3408
path = pathlib.Path(sys.argv[1])/'args.bin'
torch.save(args, path)
restored = torch.load(path, weights_only=False)
assert type(restored) is TrainingArguments and restored.seed == restored.data_seed == 3408
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
