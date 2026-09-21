"""Shared-frame preprocessing and immutable, teacher-only correspondence cache.

The square letterbox is an explicit protocol supplement. Qwen patch16 and VGGT
patch14 do not have identical grids: track coordinates are queried and quantized
in the shared pixel canvas, never by equating their patch indices.
"""
import hashlib
import json
from pathlib import Path


def canvas(path,side=448):
    from PIL import Image,ImageOps
    with Image.open(path) as source:
        source=source.convert('RGB'); width,height=source.size
        scale=side/max(width,height)
        size=(max(1,round(width*scale)),max(1,round(height*scale)))
        resized=source.resize(size,Image.Resampling.LANCZOS)
        left=(side-size[0])//2; top=(side-size[1])//2
        result=Image.new('RGB',(side,side));result.paste(resized,(left,top))
    return result,dict(source_size=[width,height],resized_size=list(size),offset=[left,top],
        valid_box=[left,top,left+size[0],top+size[1]],side=side)


class RouteCollator:
    def __init__(self,processor,training=True,side=448,max_context=16384):
        self.processor=processor;self.training=training;self.side=side;self.max_context=max_context
        processor.tokenizer.padding_side='right' if training else 'left'

    def __call__(self,rows):
        import torch
        from .qwen35 import messages
        images=[];texts=[];prefixes=[];transforms=[]
        for row in rows:
            if not 1<=len(row['media'])<=32: raise ValueError('Require 1..32 fixed actual images; no implicit resampling')
            for path in row['media']:
                image,transform=canvas(path,self.side);images.append(image);transforms.append(transform)
            prefix=self.processor.apply_chat_template(messages(row,row.get('instruction','')),
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            text=self.processor.apply_chat_template(messages(row,row.get('instruction',''),row['answer']),
                tokenize=False,add_generation_prompt=False,enable_thinking=False) if self.training else prefix
            if not text.startswith(prefix): raise ValueError('Template prefix mismatch')
            prefixes.append(prefix);texts.append(text)
        batch=dict(self.processor(text=texts,images=images,padding=True,return_tensors='pt',do_resize=False))
        expected=self.side//16
        if any(tuple(grid)!=(1,expected,expected) for grid in batch['image_grid_thw'].tolist()):
            raise ValueError('Processor changed locked per-frame grid')
        if batch['input_ids'].shape[1]>self.max_context: raise ValueError('Context overflow; do not drop frames or labels')
        if self.training:
            prompts=self.processor(text=prefixes,images=images,padding=True,return_tensors='pt',do_resize=False)
            labels=batch['input_ids'].clone();labels[batch['attention_mask']==0]=-100
            for index,length in enumerate(prompts['attention_mask'].sum(-1).tolist()):
                if not torch.equal(batch['input_ids'][index,:length],prompts['input_ids'][index,:length]):
                    raise ValueError('Prompt/label boundary mismatch')
                labels[index,:length]=-100
            if not ((labels!=-100).any(-1)).all(): raise ValueError('Missing completion')
            batch['labels']=labels
        batch['_route_rows']=rows
        return batch


class GraphCache:
    def __init__(self,root,teacher,identity,config,device):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.teacher=teacher;self.identity=identity;self.config=dict(config);self.device=device

    def get(self,row):
        import torch
        from .georoute import RouteGraph
        media=[dict(path=str(Path(path).resolve()),size=Path(path).stat().st_size,
                    mtime_ns=Path(path).stat().st_mtime_ns) for path in row['media']]
        contract=dict(version=1,media=media,teacher=self.identity,config=self.config)
        # Filename identity only, not a replacement for the full stored contract.
        key=hashlib.blake2b(json.dumps(contract,sort_keys=True).encode(),digest_size=20).hexdigest()
        path=self.root/(key+'.pt')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['contract']!=contract: raise ValueError('Graph cache identity mismatch')
            return RouteGraph(**saved['graph']).validate()
        graph=self.build(row)
        import os,uuid
        temp=path.with_suffix('.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
        torch.save(dict(contract=contract,graph=vars(graph)),temp)
        # Equivalent concurrent builders have identical source contracts; no hot
        # mutation of checkpoints or media is permitted while this cache is used.
        os.replace(temp,path)
        return graph

    def build(self,row):
        import numpy as np
        import torch
        from .georoute import patch_layout,tracks_to_graph,build_graph
        side=int(self.config.get('side',448));grid=side//16
        processed=[canvas(path,side) for path in row['media']]
        images=torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
                            for image,_ in processed]).to(self.device)
        layout=patch_layout([(grid,grid)]*len(processed))
        if len(processed)==1:
            return build_graph(layout['frame_ids'],[],[],[])
        src=[];frames=[];coordinates=[];visibility=[];confidence=[]
        query_chunk=int(self.config.get('query_chunk',128))
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            for source in range(len(processed)):
                # Native tracking queries originate in the FIRST view. Re-run
                # ordered aggregation for each anchor rather than falsely using
                # another frame's coordinates with first-view descriptors.
                order=[source]+[i for i in range(len(processed)) if i!=source]
                ordered=images[order].unsqueeze(0)
                features,start=self.teacher.aggregator(ordered)
                nodes=(layout['frame_ids']==source).nonzero().flatten()
                centers=layout['centers'][nodes]
                box=processed[source][1]['valid_box']
                keep=(centers[:,0]>=box[0])&(centers[:,0]<box[2])&(centers[:,1]>=box[1])&(centers[:,1]<box[3])
                nodes=nodes[keep]
                for first in range(0,len(nodes),query_chunk):
                    selected=nodes[first:first+query_chunk]
                    queries=layout['centers'][selected].to(self.device).unsqueeze(0)
                    tracks,vis,conf=self.teacher.track_head(features,images=ordered,patch_start_idx=start,query_points=queries)
                    xy=tracks[-1][0].float().cpu();vis=vis[0].float().cpu();conf=conf[0].float().cpu()
                    for position,target in enumerate(order):
                        if target==source: continue
                        b=processed[target][1]['valid_box'];p=xy[position]
                        valid=(p[:,0]>=b[0])&(p[:,0]<b[2])&(p[:,1]>=b[1])&(p[:,1]<b[3])
                        quantized=(p.div(16,rounding_mode='floor')+.5)*16
                        valid &= (quantized[:,0]>=b[0])&(quantized[:,0]<b[2])&(quantized[:,1]>=b[1])&(quantized[:,1]<b[3])
                        src.extend(selected.tolist());frames.extend([target]*len(selected))
                        coordinates.append(p);visibility.append(vis[position]*valid);confidence.append(conf[position])
                del features
        if not src: return build_graph(layout['frame_ids'],[],[],[])
        return tracks_to_graph(src,frames,torch.cat(coordinates),torch.cat(visibility),torch.cat(confidence),
            [(grid,grid)]*len(processed),visibility_min=self.config.get('visibility_min',.5),
            confidence_min=self.config.get('confidence_min',.5),topk=self.config.get('topk',8))


def load_teacher(source,checkpoint,device):
    import sys,torch
    sys.path.insert(0,str(source))
    from vggt.models.vggt import VGGT
    model=VGGT()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).requires_grad_(False).eval()
