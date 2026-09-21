"""Queued matched full-language SFT with frozen online VGGT and 64 extra tokens.

This is an explicit adaptation, not the paper's LoRA/alignment/70:30 recipe.
No full dense cache is silently allocated. The preceding baseline must release
its GPUs; all gates fail closed and each training run has independent outputs.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, indent=2)); temp.replace(path)


def install(args):
    import transformers
    from spatial_intelligence import qwen35
    from spatial_intelligence.qwen3vl_geometry import load_geometry_model, GeometryCollator
    def loader(path, training=False, freeze_vision=True):
        model = load_geometry_model(path, training, freeze_vision)
        model.config.geometry_source = args.source
        model.config.geometry_weights = args.weights
        if not training:
            from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
            install_video_rope_compat(model)
        return model
    qwen35.load_model = loader; qwen35.Collator = GeometryCollator
    original = transformers.TrainingArguments.__init__
    def arguments(self, *pos, **kw):
        kw.update(seed=3407, data_seed=3407)
        original(self, *pos, **kw)
    transformers.TrainingArguments.__init__ = arguments
    transformers.set_seed(3407)


def null_geometry_features(frame_count, input_dim=2048, patches=1024):
    """Shape-faithful finite placeholder for an explicit forced-null evaluation.

    GeometryTokenAdapter replaces every output token with learned null_tokens
    when force_null=True. Native images, token insertion, and mRoPE are unchanged.
    This is not used for stochastic training dropout or normal-geometry scoring.
    """
    import torch
    if frame_count < 1:
        raise ValueError('Forced-null evaluation still requires real input frames')
    return torch.zeros((frame_count, patches, input_dim), dtype=torch.bfloat16)


def evaluation(args, extra):
    import torch
    import transformers
    from spatial_intelligence.qwen3vl_geometry import add_geometry_slots
    from spatial_intelligence.vggt_features import FrozenVGGT
    install(args)
    original = transformers.AutoProcessor.from_pretrained
    def processor(cls, path, *pos, **kw):
        return original(args.model, *pos, **kw)
    transformers.AutoProcessor.from_pretrained = classmethod(processor)
    evaluator = module('geometry_shared_evaluation', REPO/'scripts/evaluate-spatial.py')
    native = evaluator.native_video_inputs
    extractor = None
    def geometry_inputs(processor, row, answer_format):
        nonlocal extractor
        if args.null_geometry:
            features = null_geometry_features(len(row['media']))
        else:
            if extractor is None:
                extractor = FrozenVGGT(args.source, args.weights, f"cuda:{os.getenv('LOCAL_RANK', '0')}")
            features, _ = extractor.extract(row['media'])
        result = add_geometry_slots(native(processor,row,answer_format), processor,
            features[None], torch.zeros(features.shape[:2],dtype=torch.bool)[None], force_null=args.null_geometry)
        return result
    evaluator.native_video_inputs = geometry_inputs
    sys.argv = ['evaluate-spatial.py', '--root', str(args.root), '--model', str(args.model), '--name', args.name, *extra]
    evaluator.main()


def queue(args):
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    for folder in ('state','logs','runs'): (root/folder).mkdir(exist_ok=True)
    if args.detach:
        import shutil
        snapshot = root/'code'
        if not snapshot.exists():
            snapshot.mkdir()
            for folder in ('scripts','spatial_intelligence','configs','catalog'):
                shutil.copytree(REPO/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        frozen_script=snapshot/'scripts'/Path(__file__).name
        with (root/'logs/geometry-queue.log').open('ab') as log:
            child = subprocess.Popen([sys.executable,str(frozen_script),*[x for x in sys.argv[1:] if x!='--detach']],
                cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pid':child.pid,'root':str(root)})); return
    lock = (root/'state/geometry-queue.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(phase, **kw): dump(root/'state/geometry-queue.json',dict(status=phase,pid=os.getpid(),updated=time.time(),**kw))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,PYTHONPATH=str(REPO),
               OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1')
    devices=args.gpus.split(',')
    if len(devices)!=8 or len(set(devices))!=8 or not all(x.isdigit() for x in devices): raise ValueError('Eight distinct GPUs required')
    def run(command, name, timeout=172800):
        state(name, command=command)
        with (root/'logs'/f'{name}.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if code:raise subprocess.CalledProcessError(code,command)
    def distributed(mode, name, model, extra=None):
        return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',__file__,
            '--mode',mode,'--root',str(root),'--model',str(model),'--source',args.source,'--weights',args.weights,
            '--name',name,*(extra or [])]
    try:
        state('waiting_for_baseline', baseline_state=str(args.baseline_state))
        deadline=time.monotonic()+7*86400
        while True:
            value=json.loads(args.baseline_state.read_text()) if args.baseline_state.exists() else {}
            if value.get('status') in ('complete','training_complete_evaluation_blocked') and value.get('gpu_work_finished'):
                break
            if value.get('status')=='failed': raise RuntimeError('Baseline failed; not launching expensive fusion automatically')
            if time.monotonic()>deadline: raise TimeoutError('Baseline wait exceeded seven days')
            time.sleep(30)
        deadline=time.monotonic()+7200
        while True:
            used=subprocess.check_output(['nvidia-smi','--id='+args.gpus,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            measurements=used.splitlines()
            if len(measurements)!=8: raise ValueError('GPU inventory must contain eight memory measurements')
            if all(int(x)<500 for x in measurements):break
            if time.monotonic()>deadline:raise TimeoutError('Allocated GPU idle wait exceeded two hours')
            time.sleep(30)
        gate=json.loads(args.gate.read_text())
        if gate.get('status')!='accepted': raise ValueError('Real geometry forward/update/reload gate missing')
        # Inherit baseline snapshots, not mutable source manifests.
        import shutil
        baseline=args.baseline_state.parent.parent
        for directory in ('manifests','receipts'):
            (root/directory).mkdir(exist_ok=True)
            for path in (baseline/directory).glob('*.json*'):
                target=root/directory/path.name
                if target.exists() and target.read_bytes()!=path.read_bytes(): raise ValueError('Snapshot changed')
                if not target.exists():shutil.copy2(path,target)
        contract={'base':args.model,'seed':3407,'epochs':1,'global_batch':64,'world':8,'micro':1,'ga':8,
            'language':'full-parameter','native_visual':'frozen','vggt':'frozen-online-final-patches',
            'source_revision':'a288dd0f14786c93483e45524328726ab7b1b4ce',
            'weights_revision':'860abec7937da0a4c03c41d3c269c366e82abdf9',
            'geometry_slots':64,'geometry_dropout':.2,'initialization':'released base, not baseline SFT',
            'sampling':'one pass original mixture, not paper 70:30',
            'alignment_stage':False,'scope':'matched architecture adaptation, not full paper reproduction'}
        saved_contract=root/'state/experiment-contract.json'
        if saved_contract.exists() and json.loads(saved_contract.read_text())!=contract:
            raise ValueError('Experiment contract changed; use a new run')
        dump(saved_contract,contract)
        smoke='diagnostic-geometry-ddp'
        smoke_receipt=root/'runs'/smoke/'completion.json'
        if not smoke_receipt.exists():
            run(distributed('train-worker',smoke,args.model,['--max-steps','2','--ga','1']), 'geometry-ddp-smoke')
        receipt=json.loads((root/'runs'/smoke/'completion.json').read_text())
        if receipt.get('status')!='complete' or receipt['steps']!=2:raise ValueError('DDP smoke failed')
        completion=root/'runs'/args.name/'completion.json'
        if not completion.exists() or json.loads(completion.read_text()).get('status')!='complete':
            run(distributed('train-worker',args.name,args.model), 'geometry-sft')
        final=root/'runs'/args.name/'final'
        for benchmark in ('revsi','vsibench'):
            if not (root/'manifests'/f'{benchmark}32.test.jsonl').exists():
                state('training_complete_evaluation_blocked',reason=f'{benchmark} manifest missing',gpu_work_finished=True);return
            # Same output-format gate as the RGB baseline; never use accuracy to choose protocol.
            name=f'{args.name}-{benchmark}-format-smoke'
            extra=['--benchmark',benchmark,'--smoke-per-type','1']
            run(distributed('eval-worker',name,final,extra),name)
            merge=[sys.executable,__file__,'--mode','eval-worker','--root',str(root),'--model',str(final),
                   '--source',args.source,'--weights',args.weights,'--name',name,*extra,'--merge']
            run(merge,name+'-merge')
            metrics=json.loads((root/'runs'/name/'metrics.json').read_text())
            # Explicit named fields verified against the common summarizer below.
            if metrics['parse_rate']<.9 or metrics['truncation_rate']>.05:
                state('training_complete_evaluation_blocked',reason='format smoke failed',metrics=metrics,gpu_work_finished=True);return
            for null in (False,True):
                suffix='null' if null else 'normal'; name=f'{args.name}-{benchmark}-{suffix}'
                extra=['--benchmark',benchmark]+(['--null-geometry'] if null else [])
                run(distributed('eval-worker',name,final,extra),name)
                run([sys.executable,__file__,'--mode','eval-worker','--root',str(root),'--model',str(final),
                     '--source',args.source,'--weights',args.weights,'--name',name,*extra,'--merge'],name+'-merge')
        state('complete',gpu_work_finished=True)
    except Exception as exc:
        state('failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['queue','train-worker','eval-worker'],default='queue')
    p.add_argument('--root',required=True,type=Path);p.add_argument('--model',required=True)
    p.add_argument('--source',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--name',default='sft-qwen3vl-vggt-seed3407')
    p.add_argument('--baseline-state',type=Path);p.add_argument('--gate',type=Path)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--ga',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=-1);p.add_argument('--detach',action='store_true')
    p.add_argument('--null-geometry',action='store_true');a,extra=p.parse_known_args()
    if a.mode=='queue':
        if extra or a.baseline_state is None or a.gate is None:raise ValueError('Queue requires explicit gate/baseline state')
        queue(a)
    elif a.mode=='train-worker':
        if extra:raise ValueError(extra)
        install(a)
        from spatial_intelligence.study import train
        train(a.root,a.model,a.name,1,a.ga,a.max_steps)
    else:evaluation(a,extra)


if __name__=='__main__':main()
