"""Qwen3-VL full-language SFT with shared data/trainer/evaluation contracts.

All machine locations are runtime arguments. Disposable interface checks never
initialize the formal run; training starts from the declared released weights.
"""
import argparse
import contextlib
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def load_qwen3vl(path, training=False, freeze_vision=True):
    import torch
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    if AutoConfig.from_pretrained(path).model_type != 'qwen3_vl':
        raise ValueError('This adapter only accepts Qwen3-VL weights')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation='sdpa')
    if training:
        model.config.use_cache = False
        if freeze_vision:
            for name, param in model.named_parameters():
                if '.visual.' in name:
                    param.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
        from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
        # Transformers 5.3 shares the timestamp-separated grid iterator issue
        # across these two model families. The helper only changes the private
        # position-indexing copy; real video input is verified by interface_gate.
        model._eurekasi_video_fix = install_video_rope_compat(model)
    return model


def install_loader(seed=3408, audit_path=None):
    import transformers
    from spatial_intelligence import qwen35
    qwen35.load_model = load_qwen3vl
    # Patch the resolved dataclass method, not LazyModule's exported symbol:
    # subsequent lazy imports may restore exported classes on the module.
    original_class = transformers.TrainingArguments
    original_init = original_class.__init__
    transformers.set_seed(seed)
    def arguments(self, *pos, **kw):
        kw['seed'] = seed
        kw['data_seed'] = seed
        original_init(self, *pos, **kw)
        if self.seed != seed or self.data_seed != seed:
            raise ValueError('Runtime TrainingArguments ignored the declared seed')
        if audit_path is not None and int(os.environ.get('RANK', '0')) == 0:
            dump(audit_path, dict(seed=self.seed, data_seed=self.data_seed,
                micro_batch=self.per_device_train_batch_size, ga=self.gradient_accumulation_steps,
                world_size=self.world_size, global_batch=self.per_device_train_batch_size * self.gradient_accumulation_steps * self.world_size,
                epochs=self.num_train_epochs, bf16=self.bf16,
                evidence='Actual constructed TrainingArguments fields, not CLI intent'))
    original_class.__init__ = arguments


def interface_gate(args):
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.qwen35 import Collator, completion_loss
    from spatial_intelligence.study import read_rows
    from spatial_intelligence.spatial_eval import native_video_inputs
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError('Gate output must be fresh; retain previous diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.source_root / 'manifests/sft.train.jsonl')
    selected = {}
    for row in rows:
        selected.setdefault(len(row['media']), row)
        if all(n in selected for n in (1, 2, 3, 8, 32)):
            break
    processor = AutoProcessor.from_pretrained(args.model)
    shapes = {}
    for n in (1, 2, 3, 8, 32):
        batch = Collator(processor)([selected[n]])
        if int(batch['image_grid_thw'].shape[0]) != n:
            raise ValueError('Image count/order contract failed')
        if not bool((batch['labels'][batch['attention_mask'] == 0] == -100).all()):
            raise ValueError('Padding labels are not masked')
        shapes[str(n)] = {'sample_id': selected[n]['id'], 'tokens': batch['input_ids'].shape[1],
            'answer_tokens': int((batch['labels'] != -100).sum()), 'grid': batch['image_grid_thw'].tolist()}
    inputs = {k: v.cuda() for k, v in Collator(processor)([selected[1]]).items()}
    model = load_qwen3vl(args.model, training=True).cuda()
    if args.mode == 'fp32-parity':
        torch.backends.cuda.matmul.allow_tf32 = False
        model.float()
    vision = [p for n, p in model.named_parameters() if '.visual.' in n]
    if not vision or any(p.requires_grad for p in vision):
        raise ValueError('Vision freeze gate failed')
    model.eval()
    labels = inputs['labels']
    plain = {k: v for k, v in inputs.items() if k != 'labels'}
    positions = (labels[:, 1:] != -100).any(0).nonzero().flatten()
    parameter = next(p for n, p in model.named_parameters() if 'q_proj.weight' in n and p.requires_grad)
    full = model(**plain, use_cache=False).logits
    full_loss = torch.nn.functional.cross_entropy(full[:, :-1].float().reshape(-1, full.shape[-1]),
        labels[:, 1:].reshape(-1), ignore_index=-100)
    reference = full[:, positions].detach().clone()
    full_loss.backward()
    reference_grad = parameter.grad.detach().float().clone()
    full_value = float(full_loss.detach())
    del full_loss, full
    model.zero_grad(set_to_none=True)
    loss, outputs = completion_loss(model, inputs)
    torch.testing.assert_close(reference, outputs.logits.detach(), atol=.05, rtol=.01)
    if abs(float(loss.detach()) - full_value) > .005:
        raise ValueError('Selected/full loss mismatch')
    loss.backward()
    observed_grad = parameter.grad.float()
    gradient_relative_l2 = float((reference_grad - observed_grad).norm() / reference_grad.norm().clamp_min(1e-12))
    gradient_cosine = float(torch.nn.functional.cosine_similarity(reference_grad.flatten(), observed_grad.flatten(), dim=0))
    # BF16 full-vocabulary versus selected-position GEMMs can use different
    # accumulation kernels. Elementwise relative error near zero is ill posed;
    # require both a small global error and aligned update direction instead.
    if gradient_relative_l2 > .03 or gradient_cosine < .999:
        raise ValueError(f'Selected/full BF16 gradient mismatch: relative L2={gradient_relative_l2}, cosine={gradient_cosine}')
    if args.mode == 'fp32-parity':
        if gradient_relative_l2 > .0001 or gradient_cosine < .999999:
            raise ValueError(f'FP32 semantic parity failed: {gradient_relative_l2}, {gradient_cosine}')
        dump(out / 'completion.json', dict(status='complete', precision='float32-tf32-disabled',
            full_loss=full_value, selected_loss=float(loss.detach()), gradient_relative_l2=gradient_relative_l2,
            gradient_cosine=gradient_cosine, purpose='Disambiguate BF16 accumulation noise from selected-logit semantic errors'))
        return
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
        raise ValueError('No finite nonzero optimizer gradient')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=True)
    before = parameter.detach().float().clone()
    optimizer.step()
    parameter_change = float((before - parameter.detach().float()).abs().sum())
    if parameter_change <= 0:
        raise ValueError('Optimizer did not change the language parameters')
    loss_value = float(loss.detach())
    del outputs, loss, before, reference_grad, observed_grad, reference, optimizer, parameter
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        expected = model(**plain, use_cache=False, logits_to_keep=positions).logits.detach().cpu()
    checkpoint = out / 'checkpoint'
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    del model
    gc.collect(); torch.cuda.empty_cache()
    restored = load_qwen3vl(checkpoint).cuda()
    with torch.no_grad():
        actual = restored(**plain, use_cache=False, logits_to_keep=positions).logits.detach().cpu()
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    dump(out / 'math-gate.json', dict(loss=loss_value, full_loss=full_value, grad_norm=float(norm),
        gradient_relative_l2=gradient_relative_l2, gradient_cosine=gradient_cosine,
        reload_max_abs=float((expected - actual).abs().max()), parameter_change=parameter_change))
    # The real native-video forward is an input/position gate, not a score claim.
    video_row = read_rows(args.source_root / 'manifests/revsi32.test.jsonl')[0]
    video = {k: v.cuda() for k, v in native_video_inputs(processor, video_row, 'tagged').items()}
    with torch.no_grad():
        generated = restored.generate(**video, max_new_tokens=4, do_sample=False)
    dump(out / 'completion.json', {'status': 'complete', 'model_type': 'qwen3_vl',
        'loss': loss_value, 'full_loss': full_value, 'grad_norm': float(norm),
        'gradient_relative_l2': gradient_relative_l2, 'gradient_cosine': gradient_cosine,
        'language_parameter_l1_change': parameter_change, 'reload_max_abs': float((expected - actual).abs().max()),
        'vision_frozen': True, 'frame_strata': shapes, 'native_video_grid': video['video_grid_thw'].tolist(),
        'video_generated_tokens': generated.shape[1] - video['input_ids'].shape[1],
        'scope': 'Real-image mask/logits/gradient/update/save-reload/native-video interface gate, not benchmark performance'})
    print(json.dumps(json.loads((out / 'completion.json').read_text())), flush=True)


def train_worker(args):
    install_loader(args.seed, args.root / 'receipts' / f'{args.name}-actual-arguments.json')
    from spatial_intelligence.study import train
    train(args.root, str(args.model), args.name, args.micro, args.ga, args.max_steps)


def eval_worker(args, extra):
    """Reuse the public native-video evaluator, changing only model dispatch.

    Its historical fixed processor path is redirected explicitly to this
    checkpoint's tokenizer/processor; never create a misleading model symlink.
    """
    import transformers
    from spatial_intelligence import qwen35
    processor_class = transformers.AutoProcessor
    original_from_pretrained = processor_class.from_pretrained
    def from_pretrained(cls, path, *pos, **kw):
        return original_from_pretrained(args.model, *pos, **kw)
    processor_class.from_pretrained = classmethod(from_pretrained)
    qwen35.load_model = load_qwen3vl
    evaluator = module('shared_native_video_evaluator', REPO / 'scripts/evaluate-spatial.py')
    sys.argv = ['evaluate-spatial.py', '--root', str(args.root), '--model', str(args.model),
                '--name', args.name, *extra]
    evaluator.main()


def profile_worker(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoProcessor
    from spatial_intelligence.qwen35 import Collator, completion_loss
    from spatial_intelligence.study import read_rows
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); torch.set_num_threads(4)
    dist.init_process_group('nccl')
    if dist.get_world_size() != 4:
        raise ValueError('The locked mixed-data selection uses four ranks')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = read_rows(args.output / 'samples.jsonl')[rank::4]
    processor = AutoProcessor.from_pretrained(args.model)
    loader = torch.utils.data.DataLoader(rows, batch_size=args.micro, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=Collator(processor))
    model = DistributedDataParallel(load_qwen3vl(args.model, training=True).cuda(),
        device_ids=[local], find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=.01, fused=True)
    iterator = iter(loader); ga = 16 // args.micro; records = []
    for step in range(21):
        dist.barrier(); torch.cuda.synchronize(); started = time.monotonic()
        optimizer.zero_grad(set_to_none=True); data_wait = 0.; loss_sum = 0.
        for micro in range(ga):
            tick = time.monotonic(); cpu = next(iterator); data_wait += time.monotonic() - tick
            batch = {k: v.cuda(non_blocking=True) for k, v in cpu.items()}
            with (model.no_sync() if micro < ga - 1 else contextlib.nullcontext()), torch.autocast('cuda', dtype=torch.bfloat16):
                loss, outputs = completion_loss(model, batch)
                (loss / ga).backward()
            loss_sum += float(loss.detach()) / ga
            del loss, outputs, batch, cpu
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite profiling gradient')
        optimizer.step(); torch.cuda.synchronize()
        values = torch.tensor([time.monotonic() - started, data_wait,
            torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30], device='cuda')
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        elapsed, wait, allocated, reserved = values.tolist()
        record = dict(step=step + 1, seconds=elapsed, data_wait_max_seconds=wait,
            peak_allocated_gib=allocated, peak_reserved_gib=reserved, rank0_loss=loss_sum, stress_batch=step == 20)
        records.append(record)
        if rank == 0:
            dump(args.output / f'b{args.micro}-progress.json', record)
            print(json.dumps(record), flush=True)
    if rank == 0:
        measured = records[4:20]
        dump(args.output / f'b{args.micro}.json', dict(status='complete', micro_batch=args.micro,
            ga=ga, world_size=4, effective_batch=64, warmup_steps=4, timed_steps=16,
            samples_per_second=64 * 16 / sum(x['seconds'] for x in measured),
            peak_reserved_gib=max(x['peak_reserved_gib'] for x in records),
            gpu_total_gib=torch.cuda.get_device_properties(local).total_memory / 2**30,
            records=records, diagnostic_only=True))
    dist.destroy_process_group()


def pipeline(args):
    from spatial_intelligence.study import read_rows
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'state', 'receipts', 'runs'):
        (root / folder).mkdir(exist_ok=True)
    if args.detach:
        with (root / 'logs/pipeline.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                *[x for x in sys.argv[1:] if x != '--detach']], cwd=REPO, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'root': str(root)})); return
    lock = (root / 'state/pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    public = module('node_snapshot_helpers', REPO / 'scripts/run-node-sft.py')
    supervisor = module('stage_runner', REPO / 'scripts/run-qwen35-study.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1',
        HF_ENDPOINT='https://hf-mirror.com')
    py = args.python or sys.executable
    visible = args.gpus.split(',')
    if len(visible) != 8:
        raise ValueError('Full matched replica requires eight GPUs')
    stop = threading.Event()
    threading.Thread(target=supervisor.telemetry, args=(root, stop), daemon=True).start()
    def status(value, **details):
        dump(root / 'state/pipeline.json', dict(status=value, pid=os.getpid(), updated=time.time(), **details))
    def run(stage, command, run_env=None, required=True):
        try:
            supervisor.run(root, stage, command, env if run_env is None else run_env)
            return True
        except RuntimeError:
            if required:
                raise
            return False
    common = [str(Path(__file__).resolve()), '--root', str(root), '--source-root', str(args.source_root),
        '--model', str(args.model), '--seed', str(args.seed)]
    launch = [py, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8']
    try:
        if args.adopt_profile_pid:
            path = Path(f'/proc/{args.adopt_profile_pid}/cmdline')
            if path.exists() and path.read_bytes():
                command = path.read_bytes()
                if b'profile-worker' not in command or str(root).encode() not in command:
                    raise ValueError('Refuse to wait on unrelated profile process')
                status('adopting_disposable_profile', profile_pid=args.adopt_profile_pid)
                end = time.monotonic() + 900
                while path.exists() and path.read_bytes():
                    if time.monotonic() > end:
                        raise TimeoutError('Adopted profile did not exit; no duplicate started')
                    time.sleep(5)
        status('snapshotting_inputs')
        count = public.snapshot(args.source_root, root)
        for name in ['vsibench32.test.jsonl']:
            source = args.source_root / 'manifests' / name
            target = root / 'manifests' / name
            if source.exists() and not target.exists():
                shutil.copyfile(source, target)
        dump(root / 'receipts/experiment.json', dict(model_family='qwen3_vl', seed=args.seed,
            rows=count, epochs=1, global_batch=64, world_size=8, freeze_vision=True,
            language_training='full parameter, not LoRA', frame_policy='All prescribed ordered frames, original rendered markers',
            eval_protocol='native-video-32/tagged-final/512-greedy/blind-extractor-v1',
            model=str(args.model), profile_optimizer_seed=args.profile_seed,
            comparison_scope='Matched recipe to the other 2B backbone; not a paper LoRA recipe'))
        status('waiting_interface_gate')
        deadline = time.monotonic() + 1800
        while not (args.gate / 'completion.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Real-image interface gate not accepted')
            time.sleep(10)
        if json.loads((args.gate / 'completion.json').read_text())['status'] != 'complete':
            raise ValueError('Interface gate rejected')
        status('waiting_allocated_gpus')
        while True:
            usage = subprocess.check_output(['nvidia-smi', '--id=' + args.gpus,
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if all(int(x) < 500 for x in usage.splitlines()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Allocated GPUs did not become idle')
            time.sleep(10)
        profile = root / 'runs/mixed-batch-selection-v1'; profile.mkdir(exist_ok=True)
        if not (profile / 'samples.jsonl').exists():
            rows = read_rows(root / 'manifests/sft.train.jsonl')
            indices = list(range(len(rows))); random.Random(3407).shuffle(indices)
            selected = [rows[i] for i in indices[:1280]]
            selected += sorted(rows, key=lambda r: (len(r['media']), len(r['question']) + len(r['answer'])), reverse=True)[:64]
            with (profile / 'samples.jsonl').open('w') as stream:
                for row in selected:
                    stream.write(json.dumps(row) + '\n')
            del rows, selected
        status('profiling_mixed_data')
        results = []
        for micro in (1, 2, 4, 8):
            receipt = profile / f'b{micro}.json'
            if not receipt.exists():
                profile_common = common.copy()
                profile_common[profile_common.index('--seed') + 1] = str(args.profile_seed)
                command = [py, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', *profile_common,
                    '--mode', 'profile-worker', '--micro', str(micro), '--output', str(profile)]
                log_path = profile / f'b{micro}.log'
                with log_path.open('ab') as log:
                    child = subprocess.Popen(command, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=','.join(visible[:4])),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    ends = time.monotonic() + 900; failure = None
                    while child.poll() is None:
                        if 'OutOfMemoryError' in log_path.read_text(errors='replace')[-32000:]:
                            failure = 'oom'
                        elif time.monotonic() > ends:
                            failure = 'timeout'
                        if failure:
                            os.killpg(child.pid, signal.SIGTERM)
                            try: child.wait(timeout=15)
                            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                            break
                        time.sleep(2)
                    code = child.wait()
                if code or not receipt.exists():
                    dump(receipt, dict(status='failed', micro_batch=micro, returncode=code, reason=failure))
            result = json.loads(receipt.read_text()); results.append(result)
            if result.get('reason') == 'oom':
                break
        safe = [r for r in results if r['status'] == 'complete' and r['peak_reserved_gib'] < .92 * r['gpu_total_gib']]
        if not safe:
            raise RuntimeError('No accepted mixed-data batch')
        selected = max(safe, key=lambda r: r['samples_per_second'])
        micro = selected['micro_batch']; ga = 64 // (8 * micro)
        dump(root / 'receipts/batch-selection.json', dict(selected=selected, candidates=results,
            formal_world_size=8, formal_ga=ga, global_batch=64,
            limitation='Four-rank mixed benchmark transferred to eight ranks; smoke and live throughput checked separately. Micro16 cannot retain global64 on eight ranks.'))
        deadline = time.monotonic() + 1800
        while True:
            usage = subprocess.check_output(['nvidia-smi', '--id=' + args.gpus,
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if all(int(x) < 500 for x in usage.splitlines()):
                break
            status('waiting_eight_rank_gpu_release', memory_mib=usage)
            if time.monotonic() > deadline:
                raise TimeoutError('Eight-rank training GPUs did not become idle')
            time.sleep(10)
        status('ddp_smoke')
        run('ddp-smoke', launch + common + ['--mode', 'train-worker', '--name', 'diagnostic-qwen3vl-ddp',
            '--micro', str(micro), '--ga', '1', '--max-steps', '2'])
        name = f'sft-qwen3vl-spar234k-hound64k-seed{args.seed}'
        status('training', run=name, micro=micro, ga=ga)
        run('sft', launch + common + ['--mode', 'train-worker', '--name', name,
            '--micro', str(micro), '--ga', str(ga)])
        status('paired_evaluation')
        results = {}
        def evaluate(benchmark, tag, model, smoke=0):
            options = [str(Path(__file__).resolve()), '--mode', 'eval-worker', '--root', str(root),
                '--source-root', str(args.source_root), '--model', str(model), '--name', tag,
                '--benchmark', benchmark, '--answer-format', 'tagged', '--max-new-tokens', '512',
                '--smoke-per-type', str(smoke)]
            if not run(tag, launch + options, required=False):
                return None
            if not run(tag + '-merge', [py, *options, '--merge'], required=False):
                return None
            return json.loads((root / 'runs' / tag / 'metrics.json').read_text())
        for benchmark in ('revsi', 'vsibench'):
            if not (root / 'manifests' / f'{benchmark}32.test.jsonl').exists():
                results[benchmark] = dict(status='blocked_missing_manifest'); continue
            checkpoints = {'baseline': args.model, 'sft': root / 'runs' / name / 'final'}
            accepted = {}
            for label, checkpoint in checkpoints.items():
                measured = evaluate(benchmark, f'{label}-{benchmark}-format-gate-v1', checkpoint, 1)
                accepted[label] = bool(measured and measured['parse_rate'] >= .9 and measured['truncation_rate'] <= .05)
            if not all(accepted.values()):
                results[benchmark] = dict(status='blocked_format_gate', accepted=accepted)
                dump(root / 'runs/comparison.json', results); continue
            pair = {label: evaluate(benchmark, f'{label}-{benchmark}-video-final-v1', checkpoint)
                    for label, checkpoint in checkpoints.items()}
            if all(pair.values()):
                contracts = [{k: v for k, v in value['contract'].items() if k not in ('model', 'model_files')} for value in pair.values()]
                if contracts[0] != contracts[1]:
                    raise ValueError('Unmatched paired evaluation contracts')
                results[benchmark] = dict(status='complete', **pair,
                    delta_extracted_points=pair['sft']['extracted']['overall_score'] - pair['baseline']['extracted']['overall_score'])
            else:
                results[benchmark] = dict(status='failed', **pair)
            dump(root / 'runs/comparison.json', results)
        status('complete' if all(r['status'] == 'complete' for r in results.values()) else 'training_complete_evaluation_blocked',
            gpu_work_finished=True, training_completion=str(root / 'runs' / name / 'completion.json'), results=results)
    except Exception as exc:
        status('failed', error=str(exc)); raise
    finally:
        stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['gate', 'fp32-parity', 'pipeline', 'train-worker', 'profile-worker', 'eval-worker'], required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--seed', type=int, default=3408)
    p.add_argument('--name'); p.add_argument('--micro', type=int, default=1)
    p.add_argument('--ga', type=int, default=8); p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--gate', type=Path); p.add_argument('--python'); p.add_argument('--detach', action='store_true')
    p.add_argument('--profile-seed', type=int, default=3408)
    p.add_argument('--adopt-profile-pid', type=int)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    args, extra = p.parse_known_args()
    if extra and args.mode != 'eval-worker':
        p.error('Unrecognized arguments: ' + ' '.join(extra))
    sys.path.insert(0, str(REPO))
    if args.mode in ('gate', 'fp32-parity'):
        interface_gate(args)
    elif args.mode == 'train-worker':
        train_worker(args)
    elif args.mode == 'profile-worker':
        profile_worker(args)
    elif args.mode == 'eval-worker':
        eval_worker(args, extra)
    else:
        pipeline(args)


if __name__ == '__main__':
    main()
