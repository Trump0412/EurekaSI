"""Bounded GPU output/gradient parity for DeltaNet only, not end-to-end validation."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=0);a=p.parse_args()
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fast
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as reference
    torch.cuda.set_device(a.device);torch.cuda.set_per_process_memory_fraction(.05,a.device)
    torch.set_num_threads(2);torch.manual_seed(3407)
    cases=[]
    for length in (127,256,513):
        shape=(1,length,4,128)
        base=[torch.randn(shape,device='cuda',dtype=torch.bfloat16)*.1 for _ in range(3)]
        base += [-torch.rand(shape[:3],device='cuda',dtype=torch.float32),torch.rand(shape[:3],device='cuda',dtype=torch.bfloat16)]
        upstream=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
        results=[]
        for fn in (reference,fast):
            x=[v.detach().clone().requires_grad_() for v in base]
            started=time.monotonic()
            out,_=fn(*x,output_final_state=False,use_qk_l2norm_in_kernel=True)
            grads=torch.autograd.grad((out.float()*upstream.float()).sum(),x)
            torch.cuda.synchronize()
            results.append((out.detach(),[g.detach() for g in grads],time.monotonic()-started))
        def relative(x,y):return float((x.float()-y.float()).norm()/x.float().norm().clamp_min(1e-8))
        forward=relative(results[0][0],results[1][0])
        backward=[relative(x,y) for x,y in zip(results[0][1],results[1][1])]
        finite=all(bool(torch.isfinite(x).all()) for result in results for x in [result[0],*result[1]])
        cases.append({'length':length,'forward_relative_l2':forward,'gradient_relative_l2':backward,
                      'finite':finite,'passed':finite and forward<.02 and max(backward)<.05,
                      'seconds_including_first_compile':[r[2] for r in results]})
    result={'status':'complete' if all(c['passed'] for c in cases) else 'failed','cases':cases,
            'thresholds':{'forward_relative_l2':.02,'gradient_relative_l2':.05},
            'scope':'bf16 operator parity; compilation-inclusive timings are NOT throughput evidence'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result));raise SystemExit(0 if result['status']=='complete' else 1)


if __name__=='__main__':main()
