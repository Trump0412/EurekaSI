"""Real-image full-model loss/gradient checks before enabling throughput changes."""
import argparse
import gc
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.qwen35 import Collator,load_model,completion_loss
    from spatial_intelligence.throughput import delta_backend,kernel_inventory
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    rows=[json.loads(x) for x in a.samples.read_text().splitlines()]
    choices=[]
    for n in (1,2,32):
        candidates=[r for r in rows if len(r['media'])==n]
        if candidates:choices.append(candidates[0])
    if len(choices)<2:raise ValueError('Need diverse real image counts')
    processor=AutoProcessor.from_pretrained(a.root/'models/Qwen3.5-2B');collator=Collator(processor)
    baseline_grads=None;baseline_loss=None;results=[]
    for backend,micro in [('reference',1),('reference',2),('fla',1)]:
        torch.manual_seed(3407)
        with delta_backend(backend):model=load_model(str(a.checkpoint),training=True).cuda()
        loss_value=0.
        for start in range(0,len(choices),micro):
            selected=choices[start:start+micro]
            batch={k:v.cuda() for k,v in collator(selected).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,outputs=completion_loss(model,batch,reduction='sample_mean')
                weighted=loss*len(selected)/len(choices)
            weighted.backward();loss_value+=float(weighted.detach())
            del outputs,loss,weighted,batch
        numerator=0.;denominator=0.;finite=True;current={};names=set()
        for name,param in model.named_parameters():
            if param.grad is None:continue
            names.add(name)
            grad=param.grad.detach().float().cpu()
            finite=finite and bool(torch.isfinite(grad).all())
            if baseline_grads is None:current[name]=grad
            else:
                ref=baseline_grads[name];numerator+=float((grad-ref).square().sum());denominator+=float(ref.square().sum())
        if baseline_grads is None:baseline_grads=current;baseline_loss=loss_value
        elif names!=set(baseline_grads):raise ValueError('Gradient parameter coverage changed')
        rel_loss=abs(loss_value-baseline_loss)/max(abs(baseline_loss),1e-8)
        rel_grad=(numerator/max(denominator,1e-16))**.5
        results.append({'backend':backend,'micro_batch':micro,'loss':loss_value,'loss_relative_error':rel_loss,
            'gradient_relative_l2':rel_grad,'finite':finite,'passed':finite and rel_loss<.01 and rel_grad<.05,
            'kernels':kernel_inventory(model)})
        print(json.dumps({k:v for k,v in results[-1].items() if k!='kernels'}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    result={'status':'complete' if all(r['passed'] for r in results) else 'failed','cases':results,
            'sample_ids':[r['id'] for r in choices],'thresholds':{'loss_relative_error':.01,'gradient_relative_l2':.05},
            'scope':'real-image bf16 loss and all non-null parameter gradients; not convergence equivalence'}
    a.output.write_text(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='complete' else 1)


if __name__=='__main__':main()
