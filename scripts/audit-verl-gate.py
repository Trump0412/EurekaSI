"""Audit real rollouts, reward variation, actor gradient, exported weights and reload.

Run with the isolated VERL environment after its Ray job has exited. The output
acceptance.json gates larger experiments; failed evidence never becomes success.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    receipt = out / 'status.json'
    deadline = time.monotonic() + 14400
    while args.wait:
        state = json.loads(receipt.read_text()) if receipt.exists() else {}
        if state.get('status') in ('checkpoint_created_needs_update_audit', 'failed', 'failed_timeout'):
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Native diagnostic did not finish')
        time.sleep(15)
    result = {'status': 'failed', 'time': time.time(), 'diagnostic_only': True}
    try:
        state = json.loads(receipt.read_text())
        assert state['status'] == 'checkpoint_created_needs_update_audit', state
        samples = json.loads((out / 'manifest.json').read_text())
        # Hound prompts may repeat for different videos. Gold text is diagnostic
        # metadata only; require unique gold IDs here rather than merging prompts.
        golds = [sample['answer'] for sample in samples]
        assert len(golds) == len(set(golds)), 'Ambiguous diagnostic gold identity; explicit rollout sample IDs required'
        groups = defaultdict(list)
        for path in (out / 'rollouts').glob('*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                assert row['gts'] in golds
                groups[(row['input'], row['gts'])].append(float(row['score']))
        assert groups, 'No real saved rollouts'
        assert all(len(values) == 4 for values in groups.values()), 'Expected four responses per unique diagnostic sample'
        varied = sum(len(set(values)) > 1 for values in groups.values())
        assert varied > 0, 'All rollout group rewards constant; no policy signal verified'
        log = re.sub(r'\x1b\[[0-9;]*m', '', (out / 'trainer.log').read_text())
        gradients = [float(v) for v in re.findall(r'actor/grad_norm[\s\x27\x22:=]+([0-9.eE+-]+)', log)]
        assert any(math.isfinite(v) and v > 0 for v in gradients), 'No finite nonzero actor gradient'
        actor = out / 'checkpoints/global_step_1/actor'
        exported = out / 'exported'
        if not list(exported.glob('*.safetensors')):
            subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                            '--local_dir', str(actor), '--target_dir', str(exported)], check=True)
        import torch
        # Hundreds of medium tensors are much slower with this host's 128+
        # default CPU threads; this affects audit throughput, not model values.
        torch.set_num_threads(8)
        from safetensors import safe_open
        base_dir = root / 'models/Qwen3-VL-2B-Instruct'
        from tokenizers import Tokenizer
        assert json.loads(Tokenizer.from_file(str(base_dir / 'tokenizer.json')).to_str()) == json.loads(Tokenizer.from_file(str(exported / 'tokenizer.json')).to_str()), 'Export changed tokenizer semantics'
        result['tokenizer_backend_semantics_equal'] = True
        def index(directory):
            result = {}
            for path in directory.glob('*.safetensors'):
                with safe_open(path, framework='pt', device='cpu') as source:
                    result.update({key: path for key in source.keys()})
            return result
        before, after = index(base_dir), index(exported)
        result['parameter_key_difference'] = {'missing_from_export': sorted(before.keys() - after.keys()),
                                              'extra_in_export': sorted(after.keys() - before.keys())}
        extra = after.keys() - before.keys()
        assert before and not before.keys() - after.keys(), 'Exported parameters missing'
        if extra:
            base_config = json.loads((base_dir / 'config.json').read_text())
            export_config = json.loads((exported / 'config.json').read_text())
            assert extra == {'lm_head.weight'}, 'Unexpected exported parameter keys'
            assert base_config.get('tie_word_embeddings') is True and export_config.get('tie_word_embeddings') is True
            embedding_keys = [key for key in before if key.endswith('.embed_tokens.weight')]
            assert len(embedding_keys) == 1, embedding_keys
            embedding_key = embedding_keys[0]
            with safe_open(after['lm_head.weight'], framework='pt', device='cpu') as heads, safe_open(after[embedding_key], framework='pt', device='cpu') as embeddings:
                assert torch.equal(heads.get_tensor('lm_head.weight'), embeddings.get_tensor(embedding_key)), 'Exported tied head differs from embedding'
            result['verified_serialization_alias'] = {'lm_head.weight': embedding_key, 'bitwise_equal': True}
        changed, maximum = 0, 0.0
        for key in sorted(before):
            with safe_open(before[key], framework='pt', device='cpu') as a, safe_open(after[key], framework='pt', device='cpu') as b:
                original, updated = a.get_tensor(key), b.get_tensor(key)
                assert torch.isfinite(updated).all(), key
                delta = float((original.float() - updated.float()).abs().max())
                changed += delta > 0
                maximum = max(maximum, delta)
        assert changed > 0, 'Saved model numerically unchanged'
        from spatial_intelligence.backends.hf import HFBackend
        model = HFBackend({'path': str(exported), 'device': 'cuda:0', 'attention': 'sdpa', 'max_context': 8192})
        sample = json.loads((out / 'manifest.json').read_text())[0]
        sample.pop('answer', None)
        protocol = {'max_frames': 8, 'image_max_side': 448, 'instruction': '',
                    'generation': {'max_new_tokens': 16, 'do_sample': False}}
        batch = model.encode(sample, protocol)
        ids = model.sample_ids(batch, protocol['generation'], 3407)
        assert ids.numel() > 0
        text = model.tokenizer.decode(ids[0], skip_special_tokens=True)
        result.update(status='accepted', sampled_groups=len(groups), varied_reward_groups=varied,
            gradients=gradients, changed_parameter_tensors=changed, parameter_max_abs_change=maximum,
            reloaded_prediction=text, image_count=len(sample['media']), exported=str(exported),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), completed=time.time())
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        (out / 'acceptance.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
