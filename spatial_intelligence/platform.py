"""Workspace-aware shortcuts; resolves to the same reviewable training/evaluation config."""
import argparse
import copy
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from .workspace import resource_path,settings,catalog,initialize
from .io import apply_overrides,load_config,write_json

COMMANDS={'init','catalog','download','source','build-data','split-data','doctor','run','suite','cache-geometry','benchmark','libero-eval','clean-data','paired'}


def model_path(name,local):
    p=Path(name).expanduser()
    if p.exists():return str(p.resolve())
    assets=catalog('assets')
    if name not in assets:raise ValueError('Use a catalog model alias or an existing local model directory')
    p=Path(local['models'])/name
    if not p.is_dir():raise FileNotFoundError(f'Model absent: spatial download {name}')
    if assets[name].get('integration')!='hf_native':raise ValueError('Model requires an external adapter/environment; see catalog and docs/COMPATIBILITY.md')
    return str(p)


def manifest_path(name,local):
    p=Path(name).expanduser()
    if not p.exists():p=Path(local['manifests'])/(name if name.endswith('.jsonl') else name+'.jsonl')
    if not p.is_file():raise FileNotFoundError(p)
    return str(p.resolve())


def run_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value.lower() == 'plans':
        raise ValueError('Run names must be one alphanumeric/dot/dash/underscore component; plans is reserved')
    return value


def mixture_spec(spec):
    # A Windows drive colon belongs to the path, not the optional numeric weight.
    name, sep, suffix = spec.rpartition(':')
    if sep:
        try:
            return name, float(suffix)
        except ValueError:
            if len(name) == 1 and name.isalpha():
                return spec, 1.0
            raise ValueError('Training mixture syntax: manifest[:numeric_weight]') from None
    return spec, 1.0


def resolved_run(args):
    local=settings();mode=args.mode
    run_name(args.name)
    if args.gpus < 1:raise ValueError('--gpus must be positive')
    devices=[d.strip() for d in args.devices.split(',')] if args.devices else None
    if devices and (len(devices)!=args.gpus or len(set(devices))!=len(devices) or any(not d for d in devices)):
        raise ValueError('--devices must list one distinct device per process')
    path=resource_path(Path('configs')/(mode+'.yaml'))
    cfg=load_config(path)
    cfg['model']['path']=model_path(args.model,local)
    cfg['model']['adapter']=args.adapter
    cfg['output']=str(Path(local['runs'])/args.name)
    cfg['data']['train']=[]
    for spec in args.train:
        name,weight=mixture_spec(spec)
        cfg['data']['train'].append({'path':manifest_path(name,local),'weight':weight})
    cfg['data']['heldout']=[manifest_path(p,local) for p in args.heldout]
    cfg['data']['eval']=manifest_path(args.eval,local) if args.eval else (cfg['data']['heldout'][0] if cfg['data']['heldout'] else None)
    if mode=='inference' and not args.eval:raise ValueError('Inference requires --eval')
    if mode!='inference' and (not args.train or not args.heldout):raise ValueError('Training requires --train and --heldout')
    if args.teacher:cfg['teacher']['path']=model_path(args.teacher,local)
    if mode=='opd' and not args.teacher:raise ValueError('OPD requires --teacher with the same tokenizer; heterogeneous teacher uses teacher-export + offline_kd')
    if args.geometry_index:
        cfg['model']['backend']='spatial_intelligence.backends.fusion:FusionBackend'
        cfg['model']['options']={'cache_index':str(Path(args.geometry_index).resolve()),'checkpoint':args.fusion_checkpoint,'heads':4}
    elif args.fusion_checkpoint:
        raise ValueError('--fusion-checkpoint requires --geometry-index')
    # Paths and mode have dedicated CLI flags; never silently override either form.
    for override in args.set:
        key=override.partition('=')[0]
        if key in {'output','train','model','teacher','train.mode','model.path','teacher.path','data'} or key.startswith('data.'):
            raise ValueError(f'{key} is owned by run flags; use train/infer --config for a fully custom config')
    apply_overrides(cfg,args.set)
    if args.geometry_index and cfg['model']['backend']!='spatial_intelligence.backends.fusion:FusionBackend':
        raise ValueError('--geometry-index requires FusionBackend')
    from .config import validate
    validate(cfg)
    target=Path(local['runs'])/'plans'/(args.name+'.json')
    if target.exists():raise FileExistsError('Run name already has a plan; choose a new name')
    write_json(target,cfg)
    if args.dry_run:return {'config':str(target),'status':'planned','processes':args.gpus}
    if args.gpus>1 or devices:
        env=dict(os.environ)
        if devices:env['CUDA_VISIBLE_DEVICES']=','.join(devices)
        argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(args.gpus),'-m','spatial_intelligence','infer' if mode=='inference' else 'train','--config',str(target)]
        subprocess.run(argv,env=env,check=True)
        return {'config':str(target),'output':cfg['output'],'processes':args.gpus,'status':'completed'}
    if mode=='inference':
        from .runner import infer
        return infer(cfg)
    from .training import train
    return {'checkpoint':train(cfg),'config':str(target)}


def suite(config,execute=False):
    cfg=load_config(config);local=settings();base=load_config(cfg.get('base','configs/inference.yaml'))
    out=Path(local['runs'])/run_name(cfg['name']);plans=[];results=[]
    names=[run_name(m['name'])+'--'+run_name(b['name']) for m in cfg['models'] for b in cfg['benchmarks']]
    if not names or len(names)!=len(set(names)):raise ValueError('Suite needs nonempty, unique run names')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Suite directory is nonempty; choose a new name')
    for model in cfg['models']:
        for benchmark in cfg['benchmarks']:
            run=copy.deepcopy(base);name=model['name']+'--'+benchmark['name']
            run['model'].update(model.get('overrides',{}));run['model']['path']=model_path(model['model'],local)
            run['data']={'train':[],'heldout':[],'eval':manifest_path(benchmark['manifest'],local)}
            run['protocol'].update(cfg.get('protocol',{}));run['score']['official_scorer']=benchmark.get('official_scorer')
            run['protocol'].update(benchmark.get('protocol',{}))
            run['output']=str(out/name)
            from .config import validate
            validate(run);path=out/'plans'/(name+'.json');write_json(path,run);plans.append(str(path))
    # Finish resolving the whole matrix before loading any model.
    if execute:
        from .runner import infer
        for path in plans:
            run=load_config(path)
            results.append({'run':Path(run['output']).name,'report':infer(run)})
    write_json(out/'suite.json',{'plans':plans,'results':results,'status':'completed' if execute else 'planned',
        'policy':'Compare models within each benchmark using spatial compare. Do not average unlike official metrics.'})
    return {'suite':str(out/'suite.json'),'runs':len(plans),'executed':execute}


def benchmark(args):
    local=settings();entry=catalog('benchmarks')[args.name]
    args.output=str(Path(args.output).resolve())
    root=Path(args.engine_root).resolve();python=str(Path(args.python).resolve())
    if not Path(python).is_file() or not root.is_dir():raise FileNotFoundError('Supply existing external engine root and its environment Python')
    engine=entry['engine'];task=entry['task']
    if engine=='easi':
        script=root/'scripts/submissions/run_easi_eval.py'
        if not script.is_file():raise FileNotFoundError(script)
        argv=[python,str(script),'--backend',args.backend,'--model',args.model,'--model-args',args.model_args,'--nproc',str(args.nproc),'--benchmarks',task,'--output-dir',args.output,'--dataset-dir',local['datasets'],'--no-judge','--no-rich']
    elif engine=='lmms':
        matches=list((root/'lmms_eval'/'tasks').rglob('*.yaml'))
        if not any(task in p.read_text(errors='replace') for p in matches):raise ValueError(f'Engine checkout lacks task {task}; use official ReVSI-supported lmms-eval version')
        argv=[python,'-m','accelerate.commands.launch','--num_processes',str(args.nproc),'-m','lmms_eval','--model',args.model,'--model_args',args.model_args,'--tasks',task,'--batch_size','1','--output_path',args.output]
    else:
        if not (root/'run.py').exists():raise FileNotFoundError(root/'run.py')
        if not any(task in p.read_text(errors='replace') for p in (root/'vlmeval').rglob('*.py')):raise ValueError(f'Engine lacks {task}')
        if args.model_args:raise ValueError('VLMEvalKit model_args require its own model config registration')
        argv=[python,str(root/'run.py'),'--data',task,'--model',args.model,'--work-dir',args.output]
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('External benchmark output is nonempty')
    try:commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError:raise ValueError('External engine must be a git checkout for provenance')
    plan={'argv':argv,'cwd':str(root),'engine_commit':commit,'benchmark':args.name,'task':task,'status':'planned','official_scores_verified':False}
    if args.execute:
        out.mkdir(parents=True);write_json(out/'launch.json',plan)
        env=dict(os.environ);env['HF_ENDPOINT']=local['hf_endpoint'];env['HF_HOME']=str(Path(local['root'])/'hf-cache')
        result=subprocess.run(argv,cwd=root,env=env)
        plan['status']='process_completed' if result.returncode==0 else 'failed';plan['returncode']=result.returncode
        write_json(out/'launch.json',plan)
        if result.returncode:raise RuntimeError(f'External evaluation failed; see {out}')
    return plan


def doctor():
    import platform
    result={'python':platform.python_version(),'packages':{},'cuda':False}
    for package in ['torch','torchvision','transformers','peft','accelerate','yaml','PIL','cv2','huggingface_hub']:
        try:
            import importlib
            module=importlib.import_module(package);result['packages'][package]=getattr(module,'__version__','available')
            if package=='torch':result.update(cuda=module.cuda.is_available(),cuda_devices=module.cuda.device_count())
        except Exception as e:result['packages'][package]={'error':str(e)}
    try:
        local=settings();result['paths']={n:Path(local[n]).is_dir() for n in ['models','datasets','manifests','geometry','runs','external']}
    except FileNotFoundError:result['workspace']='not initialized'
    result['status']='diagnostic_only_not_a_GPU_training_certification'
    return result


def main(argv):
    p=argparse.ArgumentParser(prog='spatial');sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('init');c.add_argument('--root',required=True);c.add_argument('--output');c.add_argument('--hf-endpoint',default=os.environ.get('HF_ENDPOINT','https://huggingface.co'))
    c=sub.add_parser('catalog');c.add_argument('kind',choices=['assets','sources','benchmarks'],nargs='?',default='assets')
    for name in ['download','source']:
        c=sub.add_parser(name);c.add_argument('name');c.add_argument('--dry-run',action='store_true')
        if name=='download':c.add_argument('--include',nargs='+');c.add_argument('--extract',action='store_true')
    c=sub.add_parser('build-data')
    for name in ['input','dataset','split','media-root','output']:c.add_argument('--'+name,required=True)
    c.add_argument('--max-frames',type=int,default=8);c.add_argument('--marker-source');c.add_argument('--limit',type=int)
    c=sub.add_parser('split-data');c.add_argument('--input',required=True);c.add_argument('--output',required=True);c.add_argument('--seed',type=int,default=3407)
    c=sub.add_parser('doctor')
    c=sub.add_parser('run');c.add_argument('mode',choices=['sft','rl','opd','opsd','offline_kd','inference']);c.add_argument('--model',required=True);c.add_argument('--name',required=True)
    c.add_argument('--gpus',type=int,default=1);c.add_argument('--devices');c.add_argument('--train',nargs='+',default=[]);c.add_argument('--heldout',nargs='+',default=[]);c.add_argument('--eval');c.add_argument('--teacher');c.add_argument('--adapter');c.add_argument('--geometry-index');c.add_argument('--fusion-checkpoint');c.add_argument('--set',action='append',default=[]);c.add_argument('--dry-run',action='store_true')
    c=sub.add_parser('suite');c.add_argument('--config',required=True);c.add_argument('--execute',action='store_true')
    c=sub.add_parser('cache-geometry');c.add_argument('--manifest',required=True);c.add_argument('--config',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('benchmark');c.add_argument('name',choices=list(catalog('benchmarks')));c.add_argument('--engine-root',required=True);c.add_argument('--python',required=True);c.add_argument('--model',required=True);c.add_argument('--model-args',default='');c.add_argument('--backend',default='lmms-eval',choices=['lmms-eval','vlmevalkit']);c.add_argument('--nproc',type=int,default=1);c.add_argument('--output',required=True);c.add_argument('--execute',action='store_true')
    c=sub.add_parser('libero-eval');c.add_argument('--config',required=True)
    c=sub.add_parser('clean-data');c.add_argument('--input',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('paired');c.add_argument('--a',required=True);c.add_argument('--b',required=True);c.add_argument('--output',required=True);c.add_argument('--replicates',type=int,default=5000);c.add_argument('--seed',type=int,default=3407)
    args=p.parse_args(argv)
    if args.command=='init':result=initialize(args.root,args.output,args.hf_endpoint)
    elif args.command=='catalog':result=catalog(args.kind)
    elif args.command=='download':
        from .assets import download
        result=download(args.name,include=args.include,extract=args.extract,dry_run=args.dry_run)
    elif args.command=='source':
        from .assets import fetch_source
        result=fetch_source(args.name,dry_run=args.dry_run)
    elif args.command=='build-data':
        from .build_data import build_annotations
        local=settings();result=build_annotations(args.input,args.output,args.dataset,args.split,args.media_root,local['frames'],max_frames=args.max_frames,marker_source=args.marker_source,limit=args.limit)
    elif args.command=='split-data':
        from .build_data import partition
        result=partition(args.input,args.output,args.seed)
    elif args.command=='doctor':result=doctor()
    elif args.command=='run':result=resolved_run(args)
    elif args.command=='suite':result=suite(args.config,args.execute)
    elif args.command=='benchmark':result=benchmark(args)
    elif args.command=='cache-geometry':
        from .geometry.cache import extract
        result=extract(args.manifest,args.output,load_config(args.config))
    elif args.command=='libero-eval':
        from .transfer.libero import evaluate
        result=evaluate(load_config(args.config))
    elif args.command=='paired':
        from .statistics import paired
        result=paired(args.a,args.b,args.output,args.replicates,args.seed)
    elif args.command=='clean-data':
        from .quality import clean
        result=clean(args.input,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
