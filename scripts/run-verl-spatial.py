"""Native Qwen3-VL GSPO exploratory study, gated by separately audited 2-GPU run.

Each backbone has its own baseline/final heldout evaluation. This does not claim
Qwen3.5 support in legacy VERL, official SPAR scores, or a matched-backbone
comparison with the reference trainer.
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
    parser.add_argument('--gate', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root, gate, out = Path(args.root), Path(args.gate), Path(args.output)
    if args.prepare:
        import asyncio
        import pandas as pd
        sys.path.insert(0, str(REPO))
        from spatial_intelligence.verl_data import BoundedSpatialDataset
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(root / 'models/Qwen3-VL-2B-Instruct')
        patch_size = processor.image_processor.patch_size
        splits = {}
        image_audit = []
        for name, manifest in (('train', args.train), ('val', args.val)):
            rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
            splits[name] = rows
            records = []
            for row in rows:
                if row['answer'].strip().lower().strip('.。') not in ('yes', 'no'):
                    raise ValueError('Only audited binary spatial QA is supported')
                if len(row['media']) not in (1, 3):
                    raise ValueError('Preserve audited 1/3-frame marked spatial inputs')
                if row['split'] != name:
                    raise ValueError('Manifest split mismatch')
                prompt = '<image>' * len(row['media']) + '\n' + row['question'] + '\nAnswer only Yes or No.'
                records.append({'data_source': 'spar_spatial_yesno', 'prompt': [{'role': 'user', 'content': prompt}],
                    'images': [str(Path(p).resolve(strict=True)) for p in row['media']],
                    'ability': 'spatial_relation', 'reward_model': {'style': 'rule', 'ground_truth': row['answer']},
                    'extra_info': {'index': row['id'], 'split': name, 'scene_id': row['scene_id']}})
                if len(image_audit) < 2 and len(row['media']) not in {entry['image_count'] for entry in image_audit}:
                    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': p} for p in row['media']]}]
                    images, _ = asyncio.run(BoundedSpatialDataset.process_vision_info(messages, patch_size,
                        {'image_kwargs': {'min_pixels': 3136, 'max_pixels': 200704}}))
                    assert len(images) == len(row['media']) and all(im.width * im.height <= 200704 for im in images)
                    image_audit.append({'id': row['id'], 'image_count': len(images), 'decoded_sizes': [im.size for im in images]})
            pd.DataFrame(records).to_parquet(out / f'{name}.parquet')
            (out / f'{name}.manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        a = {r['scene_id'] for r in splits['train']}
        b = {r['scene_id'] for r in splits['val']}
        if None in a | b or a & b:
            raise ValueError('Explicit disjoint scenes required')
        (out / 'split_audit.json').write_text(json.dumps({'train_n': len(splits['train']), 'val_n': len(splits['val']),
            'scene_overlap': [], 'source_manifests': [args.train, args.val]}, indent=2))
        (out / 'image_protocol_audit.json').write_text(json.dumps({'max_pixels': 200704, 'patch_size': patch_size,
            'cases': image_audit, 'status': 'passed'}, indent=2))
        return
    if args.detach:
        with (root / 'logs/verl-spatial-supervisor.log').open('ab') as stream:
            proc = subprocess.Popen([sys.executable, __file__, *[v for v in sys.argv[1:] if v != '--detach']],
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'output': str(out)}))
        return
    import fcntl
    lock = (root / 'state/verl-spatial.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if out.exists():
        pending = out / 'status.json'
        if set(out.iterdir()) != {pending} or json.loads(pending.read_text()).get('status') != 'waiting_for_native_gspo_gate':
            raise ValueError('Existing started/failed experiment must use a new output directory')
    else:
        out.mkdir(parents=True)
    def status(value, **extra):
        (out / 'status.json').write_text(json.dumps({'status': value, 'updated': time.time(),
            'backbone': 'Qwen3-VL-2B-Instruct', 'exploratory': True, 'qwen35_rl': False, **extra}, indent=2))
    status('waiting_for_native_gspo_gate')
    deadline = time.monotonic() + 14400
    while True:
        receipt = gate / 'acceptance.json'
        if receipt.exists():
            if json.loads(receipt.read_text()).get('status') != 'accepted':
                status('blocked_gate_failed'); return
            break
        if time.monotonic() > deadline:
            status('blocked_gate_timeout'); return
        time.sleep(20)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 4:
        raise ValueError('This study requires four explicitly allocated GPUs')
    idle_deadline = time.monotonic() + 14400
    while True:
        memory = subprocess.check_output(['nvidia-smi', '--id=' + ','.join(devices),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        used = [int(value.strip()) for value in memory.splitlines()]
        if len(used) == 4 and all(value < 500 for value in used):
            break
        status('waiting_for_idle_allocated_gpus', devices=devices, used_mib=used)
        if time.monotonic() > idle_deadline:
            status('blocked_gpu_timeout', devices=devices, used_mib=used)
            return
        time.sleep(20)
    py = str(root / 'envs/verl-legacy/bin/python')
    subprocess.run([py, __file__, '--root', str(root), '--gate', str(gate), '--train', args.train,
                    '--val', args.val, '--output', str(out), '--prepare'], check=True)
    original = json.loads((gate / 'command.json').read_text())
    updates = {
        'data.train_files': str(out / 'train.parquet'), 'data.val_files': str(out / 'val.parquet'),
        'data.train_batch_size': '16', 'data.max_response_length': '16', 'data.val_batch_size': '32',
        '+data.image_kwargs.max_pixels': '200704', '+data.image_kwargs.min_pixels': '3136',
        'data.custom_cls.path': str(REPO / 'spatial_intelligence/verl_data.py'),
        'data.custom_cls.name': 'BoundedSpatialDataset',
        'actor_rollout_ref.actor.ppo_mini_batch_size': '16',
        'actor_rollout_ref.rollout.max_model_len': '4112', '+actor_rollout_ref.rollout.limit_images': '3',
        'actor_rollout_ref.rollout.val_kwargs.temperature': '0', 'actor_rollout_ref.rollout.val_kwargs.do_sample': 'False',
        'actor_rollout_ref.rollout.val_kwargs.n': '1', 'trainer.n_gpus_per_node': '4',
        'trainer.total_epochs': '10', 'trainer.total_training_steps': str(args.steps),
        'trainer.project_name': 'eurekasi-spatial-exploration', 'trainer.experiment_name': 'qwen3vl-gspo-spatial-yesno',
        'trainer.val_before_train': 'True', 'trainer.test_freq': str(args.steps), 'trainer.save_freq': '25',
        'trainer.default_local_dir': str(out / 'checkpoints'), 'trainer.resume_mode': 'auto',
        'trainer.rollout_data_dir': str(out / 'rollouts'), 'trainer.validation_data_dir': str(out / 'validation'),
        'custom_reward_function.path': str(REPO / 'scripts/verl-spatial-reward.py'),
    }
    command = original[:3] + [arg for arg in original[3:] if arg.split('=', 1)[0] not in updates]
    command.extend(f'{key}={value}' for key, value in updates.items())
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    status('running_native_gspo', steps=args.steps, prompt_batch=16, rollout_group=4)
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    with (out / 'trainer.log').open('w') as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    status('native_training_exited_needs_audit' if result.returncode == 0 else 'failed', returncode=result.returncode)
    if result.returncode == 0:
        subprocess.run([sys.executable, str(REPO / 'scripts/summarize-verl-spatial.py'), '--output', str(out)], check=True)


if __name__ == '__main__':
    main()
