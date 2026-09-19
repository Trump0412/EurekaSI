"""Method-specific real-image caches and two-step stage-1 diagnostic weights.

This is NOT full method reproduction or a source of production stage weights.
No fabricated timestamps, correspondences, or zero-geometry fallbacks accepted.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def bridge_cache(root,source,out,rows):
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0,str(source/'src'))
    from qwen_vl.model.geometry_bank import VGGTBankExtractor
    from qwen_vl.train.build_vggt_feature_cache import load_geometry_inputs
    processor=AutoProcessor.from_pretrained(root/'models/Qwen3-VL-2B-Instruct').image_processor
    extractor=VGGTBankExtractor(model_path=None,freeze_encoder=True).eval()
    weights=torch.load(root/'models/VGGT-1B/model.pt',map_location='cpu',weights_only=True)
    weights=weights.get('model',weights)
    expected=extractor.vggt.state_dict()
    if any(k not in weights or weights[k].shape!=v.shape for k,v in expected.items()):
        raise ValueError('Missing or mismatched VGGT backbone weights')
    extractor.vggt.load_state_dict({k:weights[k] for k in expected},strict=True)
    del weights,expected
    extractor=extractor.cuda()
    records=[]
    for i,row in enumerate(rows):
        entry={'group_id':f'hound-diagnostic-{i}','source_dataset':'hound','source_sample_id':row['id'],
            'frame_paths':row['media'],'sampled_frame_indices':list(range(len(row['media']))),
            'valid_frame_mask':[True]*len(row['media']),'cache_path':str(out/f'features-{i}.pt'),
            'window_id':'window_0','cache_window_mode':'fixed8','question_type':'caption_diagnostic'}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            features=extractor.extract(load_geometry_inputs(row['media'],processor).cuda())
        payload={**entry,**{k:v.cpu().to(torch.bfloat16) for k,v in features.layer_tokens.items()},
            'token_counts':features.frame_layout.token_counts,'frame_shapes':features.frame_layout.frame_shapes,
            'patch_grid':features.patch_grid,'merged_grid':features.merged_grid}
        if not all(torch.isfinite(v).all() for v in payload.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite geometry features')
        torch.save(payload,entry['cache_path']);records.append(entry)
    (out/'cache-manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--method',choices=['geowire','geopsro','geobridge'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--detach',action='store_true')
    p.add_argument('--bridge-worker',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if a.detach:
        with (root/'logs'/('geometry-'+a.method+'.log')).open('ab') as stream:
            child=subprocess.Popen([sys.executable,__file__,*[v for v in sys.argv[1:] if v!='--detach']],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'method':a.method,'output':str(out)}));return
    if a.bridge_worker:
        source=Path(json.loads((root/'receipts/extension-source-geobridge.json').read_text())['path'])
        bridge_cache(root,source,out,json.loads((out/'samples.json').read_text()));return
    lock=(root/'state'/('geometry-'+a.method+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True,exist_ok=False)
    def record(status,**kw):
        (out/'status.json').write_text(json.dumps({'status':status,'method':a.method,'diagnostic_only':True,
            'updated':time.time(),**kw},indent=2))
    deadline=time.time()+14400
    dependencies=[root/'state/extension-env-legacy.json',root/'receipts/geometry-vggt.json',
                  root/'receipts/geometry-qwen3vl2b.json',root/'receipts'/f'extension-source-{a.method}.json']
    while True:
        statuses=[json.loads(f.read_text()).get('status') if f.exists() else 'missing' for f in dependencies]
        if 'failed' in statuses:record('blocked_dependency',dependencies=statuses);return
        if all(s=='complete' for s in statuses):break
        record('waiting_for_dependencies',dependencies=statuses)
        if time.time()>deadline:record('dependency_timeout');return
        time.sleep(15)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not gpu or ',' in gpu:raise ValueError('Allocate exactly one GPU for each geometry diagnostic')
    gpu_lock=(root/'state'/f'geometry-gpu-{gpu}.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    memory=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(memory.strip())>500:record('blocked_gpu_busy');return
    source=Path(json.loads(dependencies[-1].read_text())['path'])
    py=str(root/'envs/legacy-geo/bin/python')
    env=dict(os.environ,OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1')
    rows=[json.loads(s) for s in (root/'manifests/sft-probe.train.jsonl').read_text().splitlines()][:2]
    (out/'samples.json').write_text(json.dumps(rows,indent=2))
    (out/'protocol.json').write_text(json.dumps({'source':'Hound training diagnostic subset','frames':8,
        'frame_order':'preserved source manifest; no physical time claim','timestamps':'unknown, omitted',
        'steps':2,'purpose':'cache interface and stage-1 update only, not trained research weights'},indent=2))
    commands=[]
    def run(cmd,cwd=source):
        commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2))
        subprocess.run(cmd,cwd=cwd,env=env,check=True,timeout=3600)
    record('building_real_cache')
    try:
        if a.method=='geobridge':
            env['PYTHONPATH']=str(source/'src')
            run([py,__file__,'--root',str(root),'--method',a.method,'--output',str(out),'--bridge-worker'])
            record('cache_complete_stage_weights_pending',reason='FCP requires correspondence graph and its initializer; do not substitute a different stage-1 objective')
            return
        # Other two methods use official VGGT source and their original cache builders.
        run([py,str(REPO/'scripts/prepare-extension-sources.py'),'--root',str(root),'vggt'],REPO)
        vggt=Path(json.loads((root/'receipts/extension-source-vggt.json').read_text())['path'])
        manifest=out/'manifest.jsonl';cache=out/'cache';stage=out/'stage1'
        model=str(root/'models/Qwen3-VL-2B-Instruct');weights=str(root/'models/VGGT-1B')
        if a.method=='geopsro':
            adapted=[{**r,'media_paths':r['media'],'sample_id':r['id']} for r in rows]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in adapted));env['PYTHONPATH']=str(source)
            run([py,'-m','geopsro4d.geometry.vggt_extractor','--dataset_json',str(manifest),
                 '--cache_root',str(cache),'--num_frames','8','--model_name_or_path',weights,'--source_path',str(vggt)])
            logs=[json.loads(s) for s in (cache/'vggt_cache_log.jsonl').read_text().splitlines()]
            if len(logs)!=len(rows) or not all(r['geometry_valid'] and r['nan_count']==0 for r in logs):
                raise ValueError('Native cache contains invalid/zero fallback; refusing training')
            record('cache_verified_stage1_running')
            run([py,'-m','geopsro4d.train.train_stage1_align','--model-path',model,'--train-jsonl',str(manifest),
                 '--cache-root',str(cache),'--output',str(stage),'--steps','2','--batch-size','1','--log-every','1','--save-every','1'])
        else:
            source=source/'geowire';env['PYTHONPATH']=str(source)
            adapted=[{'clip_id':f'hound-diag-{i}','scene_id':r['id'],'source_dataset':'hound',
                'frame_paths':r['media'],'frame_indices':list(range(8)),'timestamps_s':[],
                'split':'train','question':r['question'],'answer':r['answer']} for i,r in enumerate(rows)]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in adapted))
            run([py,'scripts/cache_vggt.py','--manifest',str(manifest),'--cache-root',str(cache),'--backend','real',
                 '--qwen-checkpoint',model,'--vggt-checkpoint',weights,'--vggt-source',str(vggt)],source)
            run([py,'scripts/build_graphs.py','--manifest',str(manifest),'--cache-root',str(cache),'--output',str(out/'graph')],source)
            record('cache_complete_stage1_running')
            run([py,'scripts/train_tip.py','--manifest',str(manifest),'--cache-root',str(cache),'--output',str(stage),
                 '--steps','2','--save-every','1','--device','cuda','--tip-feature-mode','cached','--log-every','1'],source)
        record('stage1_process_finished_needs_artifact_audit',stage1_files=[str(p.relative_to(out)) for p in stage.rglob('*') if p.is_file()])
    except Exception as exc:
        record('failed',error=str(exc));raise


if __name__=='__main__':main()
