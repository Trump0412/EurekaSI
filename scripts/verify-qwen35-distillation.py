"""Real Hound images -> sampled IDs -> aligned teacher KL -> LoRA update -> reload.

Diagnostic only. OPD uses the prior three-step diagnostic teacher, not a claim
that this teacher is stronger. OPSD uses a frozen base with training-only gold.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from spatial_intelligence.backends.qwen35 import Qwen35Backend
from spatial_intelligence.objectives import kl_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--mode',choices=['opd','opsd'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if a.detach:
        log=root/'logs'/('qwen35-'+a.mode+'-diagnostic.log')
        with log.open('ab') as stream:
            child=subprocess.Popen([sys.executable,__file__,*[v for v in sys.argv[1:] if v!='--detach']],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':child.pid,'log':str(log),'output':str(out)}));return
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(3407)
    rows=[json.loads(s) for s in (root/'manifests/sft-probe.train.jsonl').read_text().splitlines()]
    row=rows[0]
    (out/'sample.json').write_text(json.dumps(row,indent=2))
    base={'path':str(root/'models/Qwen3.5-2B'),'device':'cuda:0','max_context':8192,
          'lora':{'enabled':True,'rank':8,'alpha':16,'target_modules':['q_proj','v_proj']}}
    teacher_cfg={**base,'device':'cuda:1','lora':{'enabled':False}}
    if a.mode=='opd':teacher_cfg['path']=str(root/'runs/diagnostic-ddp/final')
    (out/'config.json').write_text(json.dumps({'student':base,'teacher':teacher_cfg,'mode':a.mode,
        'diagnostic_only':True,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':3407},indent=2))
    student=Qwen35Backend(base,training=True);teacher=Qwen35Backend(teacher_cfg)
    assert student.tokenizer_fingerprint()==teacher.tokenizer_fingerprint()
    protocol={'max_frames':8,'image_max_side':448,'instruction':'',
              'generation':{'max_new_tokens':16,'do_sample':True,'top_p':1.0,'top_k':0,'temperature':1.0}}
    inp={k:v for k,v in row.items() if k!='answer'}
    batch=student.encode(inp,protocol)
    teacher_batch=teacher.encode(inp,protocol,privileged=row['answer'] if a.mode=='opsd' else None)
    assert batch['image_grid_thw'].shape[0]==len(row['media'])==8
    assert torch.equal(batch['image_grid_thw'].cpu(),teacher_batch['image_grid_thw'].cpu())
    assert torch.equal(batch['pixel_values'].cpu(),teacher_batch['pixel_values'].cpu())
    assert not any(p.requires_grad for p in teacher.model.parameters())
    ids=student.sample_ids(batch,protocol['generation'],3407)
    assert ids.shape[1]>0
    params=[p for p in student.model.parameters() if p.requires_grad]
    before=[p.detach().cpu().clone() for p in params]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)
    with torch.no_grad():
        target=teacher.response_logits(teacher_batch,ids)
        # Short response for full/selected causal-position equivalence.
        full_inp,pos=teacher.response_inputs(teacher_batch,ids[:,:2])
        full=teacher.model(**full_inp,use_cache=False).logits[:,pos,:].float()
        selected=teacher.response_logits(teacher_batch,ids[:,:2]).float()
        parity=float((full-selected).abs().max())
        assert torch.allclose(full,selected,rtol=.02,atol=.04),parity
        del full,selected
        blank={**teacher_batch,'pixel_values':torch.zeros_like(teacher_batch['pixel_values'])}
        delta=float((teacher.response_logits(blank,ids).float()-target.float()).abs().mean())
        assert delta>0,'Image conditioning appears inactive'
    logits=student.response_logits(batch,ids)
    loss=kl_loss(logits,target,direction='reverse')
    assert torch.isfinite(loss) and loss>0
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(params,1.0)
    assert torch.isfinite(norm) and norm>0
    optimizer.step()
    change=sum(float((old-p.detach().cpu()).abs().sum()) for old,p in zip(before,params))
    assert change>0
    student.save(out/'checkpoint')
    torch.save({'optimizer':optimizer.state_dict(),'step':1,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},out/'optimizer.pt')
    record={'status':'updated','mode':a.mode,'loss':float(loss.detach()),'grad_norm':float(norm),
        'parameter_l1_change':change,'response_ids':ids.cpu().tolist(),
        'response':student.tokenizer.decode(ids[0],skip_special_tokens=True),'image_count':8,
        'teacher_prompt_tokens':teacher_batch['input_ids'].shape[1],
        'student_prompt_tokens':batch['input_ids'].shape[1],
        'selected_full_max_abs':parity,'image_ablation_logit_mean_abs':delta,
        'teacher_frozen':True,'diagnostic_only':True,'time':time.time(),
        'peak_allocated_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]}
    (out/'metrics.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    with torch.no_grad():expected=student.response_logits(batch,ids).cpu()
    del logits,target,loss,optimizer,params,student,teacher
    torch.cuda.empty_cache()
    restored=Qwen35Backend({**base,'lora':{'enabled':False},'adapter':str(out/'checkpoint')})
    with torch.no_grad():actual=restored.response_logits(batch,ids).cpu()
    assert torch.allclose(expected,actual,atol=.04,rtol=.02)
    record.update(status='optimizer_checkpoint_reload_verified',reload_max_abs=float((expected-actual).abs().max()))
    (out/'completion.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
