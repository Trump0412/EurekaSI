"""Disposable full-language SFT benchmark by actual frame-count strata.

Weights and optimizer never enter formal training. ETA is an indicative range,
not a convergence prediction or the full-dataset batch-selection gate.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(a.root);out=Path(a.output)
    if a.detach:
        with (root/'logs/sft-eta-benchmark.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,*[v for v in sys.argv[1:] if v!='--detach']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'output':str(out)}));return
    out.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.benchmarks import conversation_qa
    from spatial_intelligence.qwen35 import Collator,load_model,completion_loss
    from spatial_intelligence.marker_cache import materialize
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('renderer',root/'sources/draw_marker.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    heldout={json.loads(s)['scene_id'] for s in (root/'manifests/revsi32.test.jsonl').read_text().splitlines()}
    heldout|={s.split('/')[-1] for s in heldout}
    counts=Counter();examples={}
    for ds,annotation in [('spar',root/'datasets/vg-llm/train/spar_234k.json'),
                          ('hound',root/'datasets/geothinker/train/llava_hound_64k.json')]:
        for row in json.loads(annotation.read_text()):
            parts=Path(row['images'][0]).parts
            scene=parts[parts.index('images')+1] if ds=='spar' else Path(row['images'][0]).parent.name
            if scene in heldout:continue
            group=f'{ds}-{len(row["images"])}';counts[group]+=1;examples.setdefault(group,row)
    processor=AutoProcessor.from_pretrained(root/'models/Qwen3.5-2B')
    model=load_model(root/'models/Qwen3.5-2B',training=True).cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,fused=True)
    results=[]
    for group,row in sorted(examples.items(),key=lambda pair:len(pair[1]['images'])):
        paths=[str(root/'media'/s) for s in row['images']]
        if row.get('spar_info'):
            info=json.loads(row['spar_info']) if isinstance(row['spar_info'],str) else row['spar_info']
            paths=materialize(paths,info,renderer.DRAW_FUNCTIONS[info['type']],out/'marked'/group)
        qa=conversation_qa(row)
        sample={'question':qa['question'],'answer':str(qa['answer']),'media':paths,'instruction':''}
        batch={k:v.cuda() for k,v in Collator(processor)([sample]).items()}
        times=[];losses=[];torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(5):
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
                loss,output=completion_loss(model,batch);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite benchmark')
                optimizer.step();torch.cuda.synchronize()
                if step>=2:times.append(time.monotonic()-start)
                losses.append(float(loss.detach()));del loss,output
            result={'group':group,'count':counts[group],'sample_id':row['id'],'seconds_per_example':sum(times)/len(times),
                    'peak_gib':torch.cuda.max_memory_allocated()/2**30,'input_tokens':batch['input_ids'].shape[1],
                    'times':times,'losses':losses,'status':'complete'}
        except torch.OutOfMemoryError:
            result={'group':group,'status':'oom','count':counts[group]}
            results.append(result);(out/'strata.json').write_text(json.dumps(results,indent=2));raise
        results.append(result);(out/'strata.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
        del batch
    ideal=sum(r['count']*r['seconds_per_example'] for r in results)/4/3600
    estimate={'status':'complete','rows':sum(counts.values()),'single_gpu_micro_batch':1,'gpus_assumed':4,
              'ideal_compute_hours':ideal,'indicative_training_hours':[ideal,ideal*1.5],
              'scope':'One representative per frame-count stratum; excludes preparation/evaluation, not a worst-case bound',
              'limitations':'Variable text/image shapes, DDP, data loading and gradient accumulation require steady-state correction',
              'strata':results}
    (out/'eta.json').write_text(json.dumps(estimate,indent=2));print(json.dumps(estimate),flush=True)


if __name__=='__main__':main()
