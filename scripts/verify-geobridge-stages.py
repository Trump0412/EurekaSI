"""Two-step diagnostic initializer -> learned-feature graph -> two-step FCP.

No claim of released/trained research weights. Reuses verified raw feature cache.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--cache-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(a.root);out=Path(a.output);cache=Path(a.cache_run)
    if a.detach:
        with (root/'logs/geobridge-stages.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[s for s in sys.argv[1:] if s!='--detach']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'output':str(out)}));return
    out.mkdir(parents=True,exist_ok=False)
    def record(status,**kw):
        (out/'status.json').write_text(json.dumps({'status':status,'diagnostic_only':True,'updated':time.time(),**kw},indent=2))
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not gpu or ',' in gpu:raise ValueError('Exactly one GPU required')
    with (root/'state'/f'geometry-gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=Path(json.loads((root/'receipts/extension-source-geobridge.json').read_text())['path'])
        py=str(root/'envs/legacy-geo/bin/python')
        env=dict(os.environ,PYTHONPATH=str(source/'src'),OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
        model=str(root/'models/Qwen3-VL-2B-Instruct');vggt=str(root/'models/VGGT-1B')
        commands=[]
        def run(module,args):
            cmd=[py,'-m','qwen_vl.train.'+module,*args]
            commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2))
            subprocess.run(cmd,cwd=source,env=env,check=True,timeout=1800)
        common=['--model_name_or_path',model,'--geometry_encoder_path',vggt]
        train=['--max_steps','2','--save_steps','1','--logging_steps','1',
               '--per_device_train_batch_size','1','--num_workers','0','--geometry_cache_required','True','--online_fallback','False']
        try:
            record('initializer_running')
            run('train_stage1_continuity',common+train+['--geometry_cache_manifest',str(cache/'cache-manifest.jsonl'),
                '--output_dir',str(out/'initializer'),'--log_dir',str(out/'initializer-logs')])
            record('correspondence_graph_running')
            run('build_stage1_corr_graph_cache',common+['--base_manifest_path',str(cache/'cache-manifest.jsonl'),
                '--output_manifest_path',str(out/'corr-manifest.jsonl'),'--corr_cache_dir',str(out/'corr'),
                '--projector_checkpoint_path',str(out/'initializer/latest.pt'),'--method','feature_knn'])
            record('fcp_running')
            run('train_stage1_continuity_v2',common+train+['--geometry_cache_manifest',str(out/'corr-manifest.jsonl'),
                '--output_dir',str(out/'fcp'),'--log_dir',str(out/'fcp-logs'),
                '--init_checkpoint_path',str(out/'initializer/latest.pt'),
                '--use_continuity_selector','True','--use_activated_corr_graph','True','--freeze_geo_projector','True',
                '--corr_nce_weight','0.30','--attn_weight','0.00','--lov_global_weight','0.00','--var_weight','0.01',
                '--cus_loss_weight','0.20','--phase0_steps','0','--phase1_steps','0','--phase3_start_step','0'])
            record('stage_weights_created_needs_reload_audit',initializer=str(out/'initializer/latest.pt'),
                   fcp=str(out/'fcp/latest.pt'),note='Compressed schedule exercises selector in two steps; not the formal training schedule')
        except Exception as exc:record('failed',error=str(exc));raise


if __name__=='__main__':main()
