"""Pre-merger sparse semantic routing, distinct from the legacy GeoWire backend.

Only integer correspondences and confidence cross the geometry interface. No
teacher features are added to language inputs. Defaults not fixed by the method
description (bottleneck, graph thresholds, TIP MSE/margin) are engineering choices.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3VLForConditionalGeneration


@dataclass(frozen=True)
class RouteGraph:
    num_nodes: int
    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    frame_ids: torch.Tensor
    sample_ids: torch.Tensor

    def validate(self):
        if self.num_nodes < 1 or self.frame_ids.shape != (self.num_nodes,) or self.sample_ids.shape != (self.num_nodes,):
            raise ValueError('Graph node/layout mismatch')
        if self.src.ndim != 1 or self.dst.shape != self.src.shape or self.weight.shape != self.src.shape:
            raise ValueError('Graph edges must be matching vectors')
        if self.src.dtype != torch.long or self.dst.dtype != torch.long:
            raise ValueError('Graph indices must be int64')
        if self.weight.requires_grad:
            raise ValueError('Discrete geometry teacher weights must be detached')
        if not torch.isfinite(self.weight).all() or (self.weight <= 0).any():
            raise ValueError('Edges must have finite positive weights')
        if self.src.numel():
            if (self.src < 0).any() or (self.dst < 0).any() or (self.src >= self.num_nodes).any() or (self.dst >= self.num_nodes).any():
                raise ValueError('Graph endpoint out of bounds')
            if (self.frame_ids[self.src] == self.frame_ids[self.dst]).any():
                raise ValueError('Same-view/self-loop edges are not permitted')
            if (self.sample_ids[self.src] != self.sample_ids[self.dst]).any():
                raise ValueError('Cross-sample edge leakage')
            pairs=self.dst*self.num_nodes+self.src
            if pairs.unique().numel()!=pairs.numel(): raise ValueError('Duplicate edges must be merged')
            sums=torch.zeros(self.num_nodes,device=self.weight.device).index_add_(0,self.dst,self.weight.float())
            if not torch.allclose(sums[sums>0],torch.ones_like(sums[sums>0]),atol=1e-5):
                raise ValueError('Incoming edge weights must sum to one')
        return self

    def to(self, device):
        return RouteGraph(self.num_nodes,*(x.to(device) for x in
            (self.src,self.dst,self.weight,self.frame_ids,self.sample_ids)))


def build_graph(frame_ids, src, dst, confidence, *, sample_ids=None, topk=8):
    """CPU graph build: deduplicate by max confidence, top-k, incoming normalize.

    Same-view candidates are omitted. Empty support is valid (exact identity).
    Max rather than summed duplicate confidence is an explicit design choice.
    """
    frames=torch.as_tensor(frame_ids,dtype=torch.long).cpu()
    samples=torch.zeros_like(frames) if sample_ids is None else torch.as_tensor(sample_ids,dtype=torch.long).cpu()
    if frames.ndim!=1 or samples.shape!=frames.shape or topk<1: raise ValueError('Invalid graph layout/topk')
    src=torch.as_tensor(src,dtype=torch.long).cpu(); dst=torch.as_tensor(dst,dtype=torch.long).cpu()
    confidence=torch.as_tensor(confidence).detach().float().cpu()
    if src.ndim!=1 or dst.shape!=src.shape or confidence.shape!=src.shape: raise ValueError('Candidate shape mismatch')
    merged={}
    for s,d,w in zip(src.tolist(),dst.tolist(),confidence.tolist()):
        if not 0<=s<len(frames) or not 0<=d<len(frames): raise ValueError('Candidate endpoint out of bounds')
        if samples[s]!=samples[d]: raise ValueError('Cross-sample candidate')
        if not math.isfinite(w) or w<0: raise ValueError('Invalid confidence')
        if frames[s]==frames[d] or w==0: continue
        merged[(d,s)]=max(merged.get((d,s),0.),w)
    by_dst={}
    for (d,s),w in merged.items(): by_dst.setdefault(d,[]).append((s,w))
    edges=[]
    for d,neighbors in sorted(by_dst.items()):
        neighbors=sorted(neighbors,key=lambda x:(-x[1],x[0]))[:topk]
        denom=sum(w for _,w in neighbors)
        edges.extend((s,d,w/denom) for s,w in neighbors)
    return RouteGraph(len(frames),torch.tensor([e[0] for e in edges],dtype=torch.long),
        torch.tensor([e[1] for e in edges],dtype=torch.long),torch.tensor([e[2] for e in edges],dtype=torch.float32),frames,samples).validate()


def batch_graphs(graphs):
    if not graphs: raise ValueError('Empty graph batch')
    src=[];dst=[];weights=[];frames=[];samples=[];offset=0;frame_offset=0;sample_offset=0
    for i,g in enumerate(graphs):
        g.validate(); src.append(g.src.cpu()+offset); dst.append(g.dst.cpu()+offset);weights.append(g.weight.cpu())
        frames.append(g.frame_ids.cpu()+frame_offset);samples.append(g.sample_ids.cpu()+sample_offset)
        offset+=g.num_nodes;frame_offset+=int(g.frame_ids.max())+1;sample_offset+=int(g.sample_ids.max())+1
    return RouteGraph(offset,torch.cat(src),torch.cat(dst),torch.cat(weights),torch.cat(frames),torch.cat(samples)).validate()


def patch_layout(grid_hw, patch_size=16, spatial_merge_size=2):
    """Qwen processor MERGE-GROUP order, not a naive row-major patch list."""
    centers=[];frames=[];maps=[];offset=0;m=spatial_merge_size
    for frame,(h,w) in enumerate(grid_hw):
        if h%m or w%m or min(h,w)<1: raise ValueError('Grid not divisible by native spatial merge')
        raster=torch.arange(h*w).reshape(h//m,m,w//m,m).permute(0,2,1,3).reshape(-1)
        inverse=torch.empty_like(raster);inverse[raster]=torch.arange(h*w)+offset
        centers.append(torch.stack(((raster%w+.5)*patch_size,(raster//w+.5)*patch_size),-1))
        frames.append(torch.full((h*w,),frame,dtype=torch.long)); maps.append(inverse.reshape(h,w))
        offset+=h*w
    return dict(centers=torch.cat(centers),frame_ids=torch.cat(frames),raster_to_node=maps)


def tracks_to_graph(source_nodes,target_frames,target_xy_qwen,visibility,confidence,grid_hw,
                    *,patch_size=16,spatial_merge_size=2,visibility_min=.5,confidence_min=.5,topk=8):
    """Quantize already affine-mapped VGGT tracks into pre-merger Qwen nodes.

    All candidate arrays have one row per directed track observation. Coordinates
    MUST be in each target Qwen processed image's pixel coordinates, not raw or
    VGGT pixels. The caller owns and records crop/resize transforms and queries
    VGGT at ``patch_layout()['centers']`` after applying the inverse transforms.
    """
    layout=patch_layout(grid_hw,patch_size,spatial_merge_size)
    source_nodes=torch.as_tensor(source_nodes,dtype=torch.long).cpu();target_frames=torch.as_tensor(target_frames,dtype=torch.long).cpu()
    xy=torch.as_tensor(target_xy_qwen).detach().cpu();vis=torch.as_tensor(visibility).detach().cpu();conf=torch.as_tensor(confidence).detach().cpu()
    n=source_nodes.numel()
    if source_nodes.shape!=(n,) or target_frames.shape!=(n,) or xy.shape!=(n,2) or vis.shape!=(n,) or conf.shape!=(n,): raise ValueError('Track shape mismatch')
    sources=[];destinations=[];weights=[]
    for s,f,p,v,c in zip(source_nodes.tolist(),target_frames.tolist(),xy,vis.tolist(),conf.tolist()):
        if not 0<=s<len(layout['frame_ids']) or not 0<=f<len(grid_hw): raise ValueError('Track source/frame out of bounds')
        if not torch.isfinite(p).all() or not torch.isfinite(torch.tensor([v,c])).all() or v<visibility_min or c<confidence_min: continue
        h,w=grid_hw[f];x,y=p.tolist()
        if not (0<=x<w*patch_size and 0<=y<h*patch_size): continue
        sources.append(s);destinations.append(int(layout['raster_to_node'][f][int(y//patch_size),int(x//patch_size)]));weights.append(c)
    return build_graph(layout['frame_ids'],sources,destinations,weights,topk=topk)


def make_tip_intervention(graph,mask_fraction=.15,generator=None):
    """CPU deterministic support-aware masks and genuinely invalid source swaps.

    Single-view/fully connected tiny graphs may have no valid counterfactual:
    reject such TIP examples explicitly instead of relabeling identity as TIP.
    """
    graph=graph.to('cpu').validate()
    if not 0<mask_fraction<1: raise ValueError('Mask fraction must be between zero and one')
    candidates=graph.dst.unique()
    if not len(candidates): raise ValueError('No supported TIP destinations')
    candidates=candidates[torch.randperm(len(candidates),generator=generator)]
    desired=max(1,int(len(candidates)*mask_fraction))
    mask=torch.zeros(graph.num_nodes,dtype=torch.bool);bad=graph.src.clone()
    for destination in candidates.tolist():
        edges=(graph.dst==destination).nonzero().flatten()
        neighbors=set(graph.src[edges].tolist());chosen={};updates={}
        # Never hide every valid source of this destination.
        if all(mask[s] or s==destination for s in neighbors): continue
        for edge in edges.tolist():
            source=int(graph.src[edge]);frame=int(graph.frame_ids[source])
            options=((graph.frame_ids==frame)&(graph.sample_ids==graph.sample_ids[source])&~mask).nonzero().flatten().tolist()
            options=[n for n in options if n not in neighbors and n!=destination and n not in chosen.get(frame,set())]
            if not options: break
            replacement=options[int(torch.randint(len(options),(1,),generator=generator))]
            chosen.setdefault(frame,set()).add(replacement);updates[edge]=replacement
        else:
            # Do not mask a node previously chosen as a counterfactual source.
            if destination in bad[mask[graph.dst]].tolist(): continue
            mask[destination]=True
            for edge,replacement in updates.items(): bad[edge]=replacement
            if int(mask.sum())>=desired: break
    if not mask.any(): raise ValueError('No valid non-neighbor TIP substitution')
    # A later mask may remove support from earlier destinations; fail closed.
    for destination in mask.nonzero().flatten().tolist():
        if mask[graph.src[graph.dst==destination]].all(): raise ValueError('Mask removed all valid incoming source evidence')
    substituted=RouteGraph(graph.num_nodes,bad,graph.dst.clone(),graph.weight.clone(),graph.frame_ids.clone(),graph.sample_ids.clone()).validate()
    return mask,substituted


class SparseTransportBlock(nn.Module):
    def __init__(self,width,bottleneck=256,alpha_init=0.):
        super().__init__()
        self.norm=nn.RMSNorm(width,eps=1e-6);self.down=nn.Linear(width,bottleneck,bias=False)
        self.up=nn.Linear(bottleneck,width,bias=False);self.alpha=nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self,hidden,graph,*,strength=1.,telemetry=None):
        if hidden.ndim!=2 or len(hidden)!=graph.num_nodes: raise ValueError('Transport graph/layout mismatch')
        _validate_routing_diagnostics(strength,telemetry)
        z=self.down(self.norm(hidden));m=torch.zeros_like(z)
        m.index_add_(0,graph.dst,z[graph.src]*graph.weight.to(z.dtype)[:,None])
        connected=torch.zeros(graph.num_nodes,dtype=torch.bool,device=hidden.device)
        connected[graph.dst]=True
        update=self.alpha*self.up(F.silu(m))
        # Preserve the original arithmetic exactly for the default operating
        # point. Strength is an explicit diagnostic intervention, not a learned
        # parameter or a prescription to amplify the training residual.
        if strength!=1.: update=update*strength
        output=torch.where(connected[:,None],hidden+update,hidden)
        if telemetry is not None:
            # Only detached Python scalars leave this scope. Measure the actual
            # applied delta (including dtype rounding), not the unscaled branch.
            # Sampling on GPU synchronizes; keep disabled during timed training.
            with torch.no_grad():
                h=hidden.detach()[connected].float()
                delta=output.detach()[connected].float()-h
                h_norm=float(torch.linalg.vector_norm(h))
                delta_norm=float(torch.linalg.vector_norm(delta))
                per_node_h=torch.linalg.vector_norm(h,dim=-1)
                per_node_delta=torch.linalg.vector_norm(delta,dim=-1)
                valid=per_node_h>0
                ratios=per_node_delta[valid]/per_node_h[valid]
                telemetry(dict(alpha=float(self.alpha.detach()),strength=float(strength),
                    connected_nodes=int(connected.sum()),total_nodes=graph.num_nodes,
                    hidden_norm=h_norm,applied_delta_norm=delta_norm,
                    applied_delta_relative_l2=delta_norm/h_norm if h_norm>0 else None,
                    nonzero_hidden_nodes=int(valid.sum()),
                    mean_node_delta_ratio=float(ratios.mean()) if ratios.numel() else None,
                    max_node_delta_ratio=float(ratios.max()) if ratios.numel() else None))
        return output


def _validate_routing_diagnostics(strength,telemetry):
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not math.isfinite(strength) or strength<0:
        raise ValueError('Routing strength must be a finite nonnegative scalar')
    if telemetry is not None and not callable(telemetry):
        raise ValueError('Routing telemetry must be a callable scalar-record sink')


class GeoRoute(nn.Module):
    def __init__(self,width,config):
        super().__init__();self.settings=dict(config)
        self.active_exits=list(config.get('active_exits',[0,1,2,3]))
        depth=config.get('blocks_per_exit',2)
        if depth not in (1,2) or not self.active_exits or any(i not in range(4) for i in self.active_exits): raise ValueError('Invalid exit/depth ablation')
        self.routes=nn.ModuleList([nn.ModuleList([SparseTransportBlock(width,config.get('bottleneck',256),config.get('alpha_init',0.)) for _ in range(depth)]) if i in self.active_exits else nn.ModuleList() for i in range(4)])
        self._graph=None;self._post_graph=None;self._capture=None;self._handles=[]
        self._diagnostic_strength=1.;self._diagnostic_telemetry=None

    def route(self,exit_index,hidden,graph):
        for block_index,block in enumerate(self.routes[exit_index]):
            sink=self._diagnostic_telemetry
            def emit(record,index=block_index,callback=sink):
                callback(dict(record,exit_index=exit_index,block_index=index))
            hidden=block(hidden,graph,strength=self._diagnostic_strength,
                         telemetry=emit if sink is not None else None)
        return hidden

    @contextmanager
    def routing_diagnostics(self,strength=1.,telemetry=None):
        """Opt-in g=0/.5/1/2 sweeps with unchanged weights and input graphs.

        Larger strength is not assumed better. Evaluate paired held-out outputs
        and residual norms; do not select a strength on the final benchmark.
        Context must cover forward AND backward if checkpointing is enabled.
        Telemetry records individual block invocations, including recomputation
        and TIP good/bad branches; the caller owns grouping/aggregation. No hooks,
        tensors, state_dict entries or checkpoint configuration are retained.
        """
        _validate_routing_diagnostics(strength,telemetry)
        previous_strength,previous_sink=self._diagnostic_strength,self._diagnostic_telemetry
        self._diagnostic_strength,self._diagnostic_telemetry=strength,telemetry
        try: yield
        finally:
            self._diagnostic_strength,self._diagnostic_telemetry=previous_strength,previous_sink

    def set_graph(self,graph):
        self._graph=None if graph is None else graph.validate()
        self._post_graph=None
        if graph is not None and self.settings.get('placement','pre')=='post':
            group=self.settings.get('spatial_merge_size',2)**2
            g=graph.to('cpu')
            if g.num_nodes%group or not torch.equal(g.frame_ids.reshape(-1,group),g.frame_ids[::group,None].expand(-1,group)):
                raise ValueError('Post-merger coarsening crosses frame boundaries')
            self._post_graph=build_graph(g.frame_ids[::group],g.src//group,g.dst//group,g.weight,
                sample_ids=g.sample_ids[::group],topk=self.settings.get('topk',8))

    @contextmanager
    def graph_context(self,graph,capture=False):
        previous=self._graph;previous_post=self._post_graph;previous_capture=self._capture
        self.set_graph(graph);self._capture={} if capture else None
        try: yield self._capture
        finally: self._graph=previous;self._post_graph=previous_post;self._capture=previous_capture

    def tip_loss(self,clean_states,graph,mask,substituted_graph,lambda_sub=.1,lambda_keep=.1,margin=.1):
        """Explicit MSE reconstruction + wrong-support margin + unmasked MSE.

        Both graphs use identical masked destinations and incoming weights. Wrong
        endpoints must come from the same source frame/sample, outside all valid
        incoming neighbors, and not be themselves masked. The exact loss metric
        and margin are engineering choices, not claimed official coefficients.
        """
        if len(clean_states)!=4: raise ValueError('TIP requires all four clean exits')
        device=clean_states[0].device
        graph=graph.to(device).validate();substituted_graph=substituted_graph.to(device).validate();mask=mask.to(device)
        if mask.shape!=(graph.num_nodes,) or mask.dtype!=torch.bool or not mask.any(): raise ValueError('Nonempty boolean TIP mask required')
        supported=torch.zeros_like(mask);supported[graph.dst]=True
        if (mask & ~supported).any(): raise ValueError('Masked destination lacks valid incoming support')
        if not torch.equal(graph.dst,substituted_graph.dst) or not torch.equal(graph.weight,substituted_graph.weight): raise ValueError('Substitution changed destination/confidence')
        for s,d,bad in zip(graph.src.tolist(),graph.dst.tolist(),substituted_graph.src.tolist()):
            if mask[d]:
                valid=graph.src[graph.dst==d]
                if bad in valid.tolist() or graph.frame_ids[bad]!=graph.frame_ids[s] or mask[bad]: raise ValueError('Substitute must be an unmasked non-neighbor in the same source frame')
        terms=[]
        for i in self.active_exits:
            clean=clean_states[i]
            target=clean.detach();masked=target.masked_fill(mask[:,None],0.)
            good=self.route(i,masked,graph);bad=self.route(i,masked,substituted_graph)
            rec=F.mse_loss(good[mask].float(),target[mask].float())
            wrong=F.mse_loss(bad[mask].float(),target[mask].float())
            sub=F.relu(margin+rec-wrong)
            keep=F.mse_loss(good[~mask].float(),target[~mask].float()) if (~mask).any() else rec*0
            terms.append((rec,sub,keep))
        rec,sub,keep=[torch.stack([t[i] for t in terms]).mean() for i in range(3)]
        return rec+lambda_sub*sub+lambda_keep*keep,dict(reconstruction=rec.detach(),substitution=sub.detach(),preservation=keep.detach())


def install_georoute(model,config=None):
    if hasattr(model,'georoute'): raise ValueError('GeoRoute already installed')
    config=dict(config or {});visual=model.model.visual
    exits=list(config.get('exits',[5,11,17,23]))
    expected=list(visual.config.deepstack_visual_indexes)+[len(visual.blocks)-1]
    if len(exits)!=4 or exits!=expected or len(visual.deepstack_merger_list)!=3:
        raise ValueError(f'Four native exits required; configured {exits}, native {expected}')
    config['exits']=exits;config.setdefault('bottleneck',256);config.setdefault('alpha_init',0.)
    model.config.georoute=config
    placement=config.get('placement','pre')
    if placement not in ('pre','post'): raise ValueError('Unknown routing placement')
    config['spatial_merge_size']=visual.config.spatial_merge_size
    route=GeoRoute(visual.config.hidden_size if placement=='pre' else visual.config.out_hidden_size,config)
    route.to(device=next(visual.parameters()).device,dtype=next(visual.parameters()).dtype)
    model.add_module('georoute',route)
    mergers=list(visual.deepstack_merger_list)+[visual.merger]
    for i,merger in enumerate(mergers):
        def prehook(module,args,index=i):
            if route._graph is None: raise ValueError('GeoRoute forward requires an explicit graph context')
            hidden=args[0];graph=route._graph.to(hidden.device)
            if route._capture is not None: route._capture[index]=hidden.detach().clone()
            return (route.route(index,hidden,graph),*args[1:])
        def posthook(module,args,output,index=i):
            if route._graph is None: raise ValueError('GeoRoute forward requires an explicit graph context')
            if route._capture is not None: route._capture[index]=output.detach().clone()
            return route.route(index,output,route._post_graph.to(output.device))
        route._handles.append(merger.register_forward_pre_hook(prehook) if placement=='pre' else merger.register_forward_hook(posthook))
    return route


def configure_stage(model,stage,teacher=None):
    if stage not in ('tip','sft','eval'): raise ValueError('Unknown GeoRoute stage')
    if any('lora_' in name for name,_ in model.named_parameters()): raise ValueError('This full-parameter variant must not contain LoRA')
    model.requires_grad_(stage=='sft');model.georoute.requires_grad_(stage!='eval')
    model.train(stage!='eval')
    if stage=='tip': model.model.visual.eval();model.model.language_model.eval()
    if teacher is not None: teacher.requires_grad_(False);teacher.eval()
    return model


class GeoRouteQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Saved config reconstructs all routing weights before HF state loading."""
    def __init__(self,config):
        super().__init__(config)
        if hasattr(config,'georoute'): install_georoute(self,config.georoute)

    def _apply(self,fn,recurse=True):
        # Keep nonpersistent native RoPE constants equal before/after BF16 save
        # and reconstruction, as in the independently validated geometry model.
        rotary=[]
        for module in self.modules():
            value=getattr(module,'inv_freq',None)
            if isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.device.type!='meta':
                original=getattr(module,'original_inv_freq',None)
                rotary.append((module,value.clone(),original.clone() if isinstance(original,torch.Tensor) and original.device.type!='meta' else None))
        result=super()._apply(fn,recurse=recurse)
        for module,value,original in rotary:
            module.inv_freq=value.to(device=module.inv_freq.device,dtype=torch.float32)
            if original is not None: module.original_inv_freq=original.to(device=module.inv_freq.device,dtype=torch.float32)
        return result

    def forward(self,input_ids=None,attention_mask=None,position_ids=None,past_key_values=None,
                inputs_embeds=None,labels=None,pixel_values=None,pixel_values_videos=None,
                image_grid_thw=None,video_grid_thw=None,mm_token_type_ids=None,cache_position=None,
                logits_to_keep=0,use_cache=None,route_tip=False,tip_mask=None,tip_substituted_graph=None,**kwargs):
        if not route_tip:
            return super().forward(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids,
                past_key_values=past_key_values,inputs_embeds=inputs_embeds,labels=labels,pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,image_grid_thw=image_grid_thw,video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,cache_position=cache_position,logits_to_keep=logits_to_keep,
                use_cache=use_cache,**kwargs)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        if self.georoute._graph is None: raise ValueError('TIP requires graph context')
        if pixel_values_videos is not None: raise ValueError('GeoRoute TIP uses separate-view image items, not native video grids')
        with self.georoute.graph_context(self.georoute._graph,capture=True) as states,torch.no_grad():
            self.model.visual(pixel_values,grid_thw=image_grid_thw)
        graph=self.georoute._post_graph if self.georoute.settings.get('placement','pre')=='post' else self.georoute._graph
        loss,_=self.georoute.tip_loss([states[i] for i in range(4)],graph,tip_mask,tip_substituted_graph,
            lambda_sub=self.georoute.settings.get('lambda_sub',.1),lambda_keep=self.georoute.settings.get('lambda_keep',.1),
            margin=self.georoute.settings.get('margin',.1))
        return CausalLMOutputWithPast(loss=loss,logits=None)


def load_georoute_model(checkpoint,route_config=None,**kwargs):
    model=GeoRouteQwen3VLForConditionalGeneration.from_pretrained(checkpoint,**kwargs)
    if not hasattr(model,'georoute'):
        if route_config is None: raise ValueError('Base checkpoint requires an explicit new routing config')
        install_georoute(model,route_config)
    elif route_config is not None and dict(route_config)!=model.config.georoute:
        raise ValueError('Saved routing architecture cannot be changed on reload')
    return model
