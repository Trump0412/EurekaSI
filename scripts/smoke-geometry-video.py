"""One real native-video prefill/decode with a reloaded geometry checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.qwen3vl_geometry import load_geometry_model,add_geometry_slots
    from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
    from spatial_intelligence.spatial_eval import native_video_inputs
    from spatial_intelligence.study import dump
    torch.set_num_threads(4)
    with open(a.manifest) as stream:row=json.loads(next(stream))
    processor=AutoProcessor.from_pretrained(a.model)
    model=load_geometry_model(a.model).cuda();install_video_rope_compat(model)
    batch=add_geometry_slots(native_video_inputs(processor,row,'tagged'),processor,None,None)
    batch={k:v.cuda() for k,v in batch.items()};batch['geometry_media']=[row['media']]
    calls=[]
    handle=model.geometry_adapter.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad():
        output=model.generate(**batch,max_new_tokens=4,min_new_tokens=4,do_sample=False,use_cache=True)
    handle.remove()
    if len(calls)!=1:raise ValueError('Geometry must be consumed once at prefill')
    dump(Path(a.output),{'status':'accepted','id':row['id'],'frames':len(row['media']),
        'geometry_calls':len(calls),'generated_tokens':output.shape[1]-batch['input_ids'].shape[1],
        'native_grid':batch['video_grid_thw'].tolist(),'diagnostic_only':True})


if __name__=='__main__':main()
