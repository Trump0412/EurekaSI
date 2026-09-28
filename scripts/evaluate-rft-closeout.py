"""Evaluate the exact geometry SFT/RFT loaders on paired fixed benchmark rows."""
import argparse,json,os,re,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.dsr_sft_eval import read,write,rows,file_identity,validate_records
from spatial_intelligence.rft_closeout import PROTOCOL,validate_manifest,score,summary,inference_instruction

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--benchmark',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--merge',action='store_true');a=p.parse_args()
    plan=read(a.plan);name=a.benchmark;ready=read(Path(plan['prepared_root'])/(name+'.receipt.json'))
    if ready.get('status')!='ready' or not ready.get('media_verified'):raise ValueError('Data not accepted')
    full=validate_manifest(rows(ready['manifest']));selected=full
    if a.smoke:
        selected=[];seen=set()
        for r in full:
            kind=(r['question_type'],r['answer_type'])
            if kind not in seen:selected.append(r);seen.add(kind)
        longest=max(full,key=lambda r:len(r['media']))
        if longest['id'] not in {r['id'] for r in selected}:selected.append(longest)
    world=len(plan['gpus']);out=Path(plan['root'])/'runs'/name/('smoke' if a.smoke else 'full');out.mkdir(parents=True,exist_ok=True)
    sft=read(plan['sft_receipt'])
    if sft.get('status')!='complete' or not sft.get('reload_verified') or sft['checkpoint']!=plan['model_checkpoint']:raise ValueError('SFT lineage not verified')
    checkpoint=None
    if plan.get('rft_receipt'):
        rft=read(plan['rft_receipt'])
        if rft.get('status')!='complete' or not rft.get('reload_verified') or rft['initial_checkpoint']!=plan['model_checkpoint']:raise ValueError('RFT lineage not verified')
        checkpoint=rft['checkpoint']
    contract=dict(protocol=PROTOCOL,model=plan['model_role'],sft=plan['model_checkpoint'],rft=checkpoint,
                  manifest=file_identity(ready['manifest']),ids=[r['id'] for r in selected],world=world,
                  max_new_tokens=512,max_context=16384,max_side=448,seed=3407,do_sample=False)
    if a.merge:
        records=[]
        for rank in range(world):
            if read(out/f'contract.rank{rank}.json')!=contract:raise ValueError('Changed inference contract')
            part=rows(out/f'predictions.rank{rank}.jsonl');validate_records(part,selected[rank::world],complete=True)
            if read(out/f'complete.rank{rank}.json')['count']!=len(part):raise ValueError('Missing shard receipt')
            records.extend(part)
        validate_records(records,selected,complete=True)
        result=summary([r['score'] for r in records],name)
        accepted=not a.smoke or (result['parse_rate']>=.9 and result['truncation_rate']<=.05)
        write(out/'completion.json',dict(result,status='complete' if accepted else 'blocked_format',accepted=accepted,contract=contract))
        print(json.dumps(result),flush=True)
        if not accepted:raise ValueError('Format gate failed; not an accuracy gate')
        return
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from spatial_intelligence.geometry_rft import load_policy,prompt_inputs,to_device,cached_geometry,rft_autocast,STRUCTURED_INSTRUCTION
    from spatial_intelligence.qwen3vl_geometry_matrix import insert_slots,preprocess_geometry
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if int(os.environ.get('WORLD_SIZE',1))!=world:raise ValueError('World mismatch')
    torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.set_device(local)
    model=load_policy(dict(plan,policy_scope='language_full_geometry',use_lora=False),checkpoint,trainable=False).cuda().eval()
    processor=AutoProcessor.from_pretrained(plan['processor'])
    saved=out/f'contract.rank{rank}.json'
    if saved.exists() and read(saved)!=contract:raise ValueError('Changed resume contract')
    write(saved,contract)
    part=selected[rank::world];path=out/f'predictions.rank{rank}.jsonl'
    done=validate_records(rows(path) if path.exists() else [],part)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
    with path.open('a',encoding='utf-8') as stream:
        for row in part:
            if row['id'] in done:continue
            started=time.monotonic()
            instruction=inference_instruction(row)
            # Strict input whitelist: gold answers/explanations never reach model.
            request={k:row[k] for k in ('question','media','choices','input_mode','frame_indices','fps','total_num_frames') if k in row}
            if row.get('choices_in_question'):request['choices']=None
            if row.get('interleaved_image_refs'):
                content=[];images=[];index=0
                for piece in re.split(r'(<image \d+>)',row['question']):
                    if re.fullmatch(r'<image \d+>',piece):
                        with Image.open(row['media'][index]) as im:
                            im=im.convert('RGB');im.thumbnail((448,448));images.append(im.copy())
                        content.append({'type':'image'});index+=1
                    elif piece:content.append({'type':'text','text':piece})
                text=''
                if row.get('choices') and not row.get('choices_in_question'):text+='\nOptions:\n'+'\n'.join(k+'. '+v for k,v in row['choices'].items())
                content.append({'type':'text','text':text+'\n'+(STRUCTURED_INSTRUCTION if instruction is None else instruction)})
                prefix=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                batch=dict(processor(text=[prefix],images=images,return_tensors='pt',padding=True))
                batch=insert_slots(batch,processor,[len(images)],'downsample',max_context=16384)
                batch['geometry_images']=[preprocess_geometry(row['media'],plan['vggt_source'])]
            else:batch=prompt_inputs(processor,request,plan['vggt_source'],'downsample',instruction=instruction)
            batch=to_device(batch,torch.device('cuda',local))
            with cached_geometry(model,batch),torch.inference_mode(),rft_autocast():
                output=model.generate(**batch,do_sample=False,max_new_tokens=512,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id)
            tokens=output[0,batch['input_ids'].shape[1]:].tolist();truncated=len(tokens)>=512 and tokens[-1] not in eos
            text=processor.decode(tokens,skip_special_tokens=True)
            result=dict(id=row['id'],response=text,raw_response=processor.decode(tokens,skip_special_tokens=False),tokens=tokens,
                        frames=len(row['media']),seconds=time.monotonic()-started,score=score(row,text,name,truncated))
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            print(json.dumps(dict(id=row['id'],seconds=result['seconds'])),flush=True)
    validate_records(rows(path),part,complete=True);write(out/f'complete.rank{rank}.json',dict(count=len(part),status='complete'))

if __name__=='__main__':main()
