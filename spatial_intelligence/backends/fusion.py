"""Experimental shared geometry cross-attention baseline. Not paper-specific SGF/HGB/PSRO."""
import json
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from .hf import HFBackend
from ..geometry.cache import cached_tokens
from ..io import file_digest,write_json


class GeometryFusion(nn.Module):
    def __init__(self,hidden,heads=4):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.norm=nn.LayerNorm(hidden)
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.gate=nn.Parameter(torch.zeros(()))
    def forward(self,text,geometry):
        geo=self.project(geometry.to(text))
        value,_=self.attention(self.norm(text),geo,geo,need_weights=False)
        return text+self.gate.tanh()*value


class FusionBackend(HFBackend):
    def __init__(self,config,training=False):
        options=config.get('options',{})
        unknown=set(options)-{'cache_index','checkpoint','heads'}
        if unknown:raise ValueError(f'Unknown fusion settings: {unknown}')
        super().__init__(config,training)
        self.cache=json.loads(Path(options['cache_index']).read_text(encoding="utf-8"))
        hidden=self.model.get_input_embeddings().weight.shape[-1]
        module=GeometryFusion(hidden,options.get('heads',4)).to(device=self.device,dtype=self.model.get_input_embeddings().weight.dtype)
        self.model.add_module('spatial_fusion',module)
        if options.get('checkpoint'):
            module.load_state_dict(torch.load(options['checkpoint'],map_location=self.device,weights_only=True),strict=True)
        elif not training:raise ValueError('Fusion inference requires a trained fusion checkpoint; random geometry is not a result')
        module.requires_grad_(training);module.eval()
    def identity(self):
        return {**super().identity(),'fusion_cache_index_sha256':file_digest(self.config['options']['cache_index']),
                'fusion_checkpoint_sha256':file_digest(self.config['options']['checkpoint']) if self.config['options'].get('checkpoint') else None,
                'fusion_architecture':'embedding_cross_attention_v1'}
    def encode(self,sample,protocol,privileged=None):
        batch=super().encode(sample,protocol,privileged)
        condition=protocol['geometry_condition']
        if condition not in {'normal','zero','reverse_frames'}:raise ValueError('Fusion conditions: normal/zero/reverse_frames')
        value=torch.tensor(cached_tokens(self.cache,sample),device=self.device)[None]
        if condition=='reverse_frames':
            value=value.clone();value[...,4]=1-value[...,4]
        batch['_geometry']=value;batch['_geometry_enabled']=condition!='zero'
        return batch
    @contextmanager
    def context(self,batch):
        inp={k:v for k,v in batch.items() if not k.startswith('_geometry')}
        if not batch['_geometry_enabled']:
            yield inp;return
        def hook(module,args,output):return self.model.spatial_fusion(output,batch['_geometry'].expand(output.shape[0],-1,-1))
        handle=self.model.get_input_embeddings().register_forward_hook(hook)
        try:yield inp
        finally:handle.remove()
    def response_logits(self,batch,response):
        with self.context(batch) as inp:return super().response_logits(inp,response)
    def sample_ids(self,batch,generation,seed):
        with self.context(batch) as inp:return super().sample_ids(inp,generation,seed)
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        module=self.model.spatial_fusion
        torch.save(module.state_dict(),path/'spatial_fusion.pt')
        write_json(path/'spatial_fusion.json',{'architecture':'embedding_cross_attention_v1','heads':module.attention.num_heads,'input_features':7})
        # Keep HF checkpoints free of unexpected custom keys. Always save fusion separately.
        del self.model.spatial_fusion
        try:super().save(path)
        finally:self.model.add_module('spatial_fusion',module)
