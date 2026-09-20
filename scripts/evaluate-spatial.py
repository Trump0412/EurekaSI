"""Four-rank resumable ReVSI/VSI inference; explicit contract, raw tokens and dual scores."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from spatial_intelligence.study import read_rows,dump,jsonl
from spatial_intelligence.spatial_eval import native_video_inputs,prompt_text,score_prediction,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--benchmark',choices=['revsi','vsibench'],required=True)
    p.add_argument('--answer-format',choices=['official','tagged'],default='tagged')
    p.add_argument('--input-mode',choices=['video','images'],default='video')
    p.add_argument('--max-new-tokens',type=int,default=512);p.add_argument('--smoke-per-type',type=int,default=0)
    p.add_argument('--memory-fraction',type=float,default=1.0,help='Optional allocator limit for bounded diagnostics')
    p.add_argument('--merge',action='store_true');a=p.parse_args()
    manifest=a.root/'manifests'/f'{a.benchmark}32.test.jsonl';rows=read_rows(manifest)
    if a.smoke_per_type:
        counts=Counter();selected=[]
        for row in rows:
            if counts[row['question_type']]<a.smoke_per_type:selected.append(row);counts[row['question_type']]+=1
        rows=selected
    out=a.root/'runs'/a.name;out.mkdir(parents=True,exist_ok=True)
    if a.merge:
        contracts=list(out.glob('contract.rank*.json'))
        if not contracts:raise ValueError('No contracts')
        contract=json.loads(contracts[0].read_text());world=contract['world'];predictions=[]
        for rank in range(world):
            if json.loads((out/f'contract.rank{rank}.json').read_text())!=contract:raise ValueError('Contract mismatch')
            if not (out/f'complete.rank{rank}.json').exists():raise ValueError('Rank incomplete')
            predictions+=read_rows(out/f'predictions.rank{rank}.jsonl')
        ids=[x['id'] for x in predictions]
        if len(set(ids))!=len(ids) or set(ids)!={x['id'] for x in rows}:raise ValueError('Coverage mismatch')
        report=summarize(predictions,a.benchmark);report['contract']=contract;report['status']='complete'
        if a.benchmark=='vsibench':
            for key,subset in [('debiased_v1',[x for x in predictions if not x['pruned']]),
                               ('no_training_scene_overlap',[x for x in predictions if not x['training_scene_overlap']])]:
                if subset:report[key]=summarize(subset,a.benchmark)
        dump(out/'metrics.json',report);print(json.dumps(report),flush=True);return
    import torch
    from transformers import AutoProcessor
    from spatial_intelligence.qwen35 import load_model,Collator
    rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    torch.cuda.set_device(local);torch.set_num_threads(4)
    if not 0<a.memory_fraction<=1:raise ValueError('Invalid memory fraction')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,local)
    contract={'benchmark':a.benchmark,'model':str(a.model),'world':world,'ids':[x['id'] for x in rows],
        'input_mode':a.input_mode,'max_side':448,'frames':32,'answer_format':a.answer_format,
        'generation':{'max_new_tokens':a.max_new_tokens,'do_sample':False,'enable_thinking':False},
        'manifest':str(manifest),'batch_size':1,'extractor':'final-answer-v1',
        'model_files':[{ 'name':f.name,'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in sorted(a.model.glob('*.safetensors'))]}
    path=out/f'contract.rank{rank}.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Changed contract requires new run name')
    dump(path,contract)
    assigned=rows[rank::world];path=out/f'predictions.rank{rank}.jsonl'
    previous=read_rows(path) if path.exists() else [];done={x['id'] for x in previous}
    if len(done)!=len(previous) or not done.issubset({x['id'] for x in assigned}):raise ValueError('Foreign/repeated predictions')
    processor=AutoProcessor.from_pretrained(a.root/'models/Qwen3.5-2B')
    model=load_model(str(a.model)).cuda();collator=Collator(processor,training=False)
    if a.input_mode=='video':
        from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
        fix=install_video_rope_compat(model)
        dump(out/f'video-compat.rank{rank}.json',{'implementation':fix})
    with path.open('a') as stream:
        for row in assigned:
            if row['id'] in done:continue
            start=time.monotonic()
            if a.input_mode=='video':batch=native_video_inputs(processor,row,a.answer_format)
            else:
                adjusted=dict(row,question=prompt_text(row,a.answer_format),choices=None,instruction='')
                batch=collator([adjusted])
            batch={k:v.cuda() for k,v in batch.items()};length=batch['input_ids'].shape[1]
            with torch.inference_mode():
                output=model.generate(**batch,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True,
                                      pad_token_id=processor.tokenizer.pad_token_id)
            tokens=output[0,length:].tolist();eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else eos or []
            truncated=len(tokens)>=a.max_new_tokens and (not tokens or tokens[-1] not in eos)
            response=processor.decode(tokens,skip_special_tokens=True)
            record=score_prediction(row,response,a.benchmark,truncated)
            record.update(raw_response=processor.decode(tokens,skip_special_tokens=False),generated_token_ids=tokens,
                generated_tokens=len(tokens),input_tokens=length,seconds=time.monotonic()-start,
                grid=batch['video_grid_thw' if a.input_mode=='video' else 'image_grid_thw'].tolist())
            stream.write(json.dumps(record)+'\n');stream.flush()
            print(json.dumps({'rank':rank,'id':row['id'],'parse':record['extraction']['status'],'tokens':len(tokens)}),flush=True)
    dump(out/f'complete.rank{rank}.json',{'count':len(assigned)})


if __name__=='__main__':main()
