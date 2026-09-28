"""Real short-sample alignment recomputation diagnostic; no optimizer or save."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser()
    for key in ('model','source','weights','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args()
    import torch
    from transformers import AutoProcessor,set_seed
    from spatial_intelligence.qwen3vl_geometry_matrix import load_matrix_model,MatrixCollator
    from spatial_intelligence.qwen35 import completion_loss
    torch.set_num_threads(4);torch.cuda.set_device(0);set_seed(3407)
    if torch.cuda.mem_get_info()[0]<18*2**30:raise RuntimeError('Insufficient free memory')
    torch.cuda.set_per_process_memory_fraction(.30)
    model=load_matrix_model(a.model,a.source,a.weights,'downsample','align',False).cuda()
    collator=MatrixCollator(AutoProcessor.from_pretrained(a.model),a.source,'downsample')
    selected=[];batches=[]
    for line in open(a.manifest):
        row=json.loads(line)
        if len(row['media'])>2:continue
        batch=collator([row])
        if batch['input_ids'].shape[1]>1024:continue
        selected.append(row['id']);batches.append({k:[v.cuda() for v in x] if isinstance(x,list) else x.cuda() for k,x in batch.items()})
        if len(batches)==2:break
    if len(batches)!=2:raise ValueError('Need two real short samples')
    reference=None;results=[]
    for checkpointing in (True,False):
        if checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:model.gradient_checkpointing_disable()
        times=[];losses=[];torch.cuda.reset_peak_memory_stats()
        for iteration in range(3):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();total=0.
            for batch in batches:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=completion_loss(model,batch,reduction='sample_mean')
                (loss/len(batches)).backward();total+=float(loss.detach())/len(batches)
            torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(total)
        grads={n:p.grad.float().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        if reference is None:reference=grads
        numerator=sum(float((grads[n]-v).double().square().sum()) for n,v in reference.items())
        denom=sum(float(v.double().square().sum()) for v in reference.values())
        results.append(dict(checkpointing=checkpointing,seconds=times,loss=losses,gradient_relative_l2=(numerator/max(denom,1e-30))**.5,
            sequences_per_second=4/sum(times[1:]),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    value=dict(status='diagnostic_complete',ids=selected,tokens=[b['input_ids'].shape[1] for b in batches],results=results,
        scope='Two short real samples micro1; shared GPU timings; no optimizer, long-sample or distributed acceptance')
    Path(a.output).write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)

if __name__=='__main__':main()
