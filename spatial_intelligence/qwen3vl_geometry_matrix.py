"""Isolated two-stage geometry study; does not alter the original fusion run.

Qwen/VGGT and SPAR/Hound are deliberate RoboRefer adaptations, not an exact
NVILA/RefSpatial reproduction. Geometry backbone weights are registered and
included in every checkpoint, including when frozen.
"""
import sys
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .geometry_tokens import GeometryTokenAdapter
from .qwen35 import Collator


def insert_slots(batch, processor, frame_counts, adapter, max_context=16384):
    """Preserve native visual IDs/types; add variable geometry with masked labels.

Downsample tokens follow each RGB image, or each native temporal video group.
Query tokens remain a global summary after the last visual group. No frame is
dropped to accommodate context. -1 padded geometry positions are never used.
"""
    result = dict(batch)
    if 'position_ids' in result:
        raise ValueError('Positions must be recomputed after geometry insertion')
    ids, mask = result['input_ids'], result['attention_mask']
    types = result['mm_token_type_ids']
    labels = result.get('labels')
    end = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pad = processor.tokenizer.pad_token_id
    prepared = []
    for b, frames in enumerate(frame_counts):
        valid = mask[b].bool()
        row, typ = ids[b, valid], types[b, valid]
        lab = labels[b, valid] if labels is not None else None
        ends = (row == end).nonzero().flatten().tolist()
        if not ends or frames < 1:
            raise ValueError('Missing media boundary or frames')
        if adapter == 'query64':
            after = {ends[-1]: 64}
        elif adapter == 'downsample':
            if frames % len(ends):
                raise ValueError('Native temporal groups cannot be paired with geometry frames')
            after = {i: 121 * (frames // len(ends)) for i in ends}
        else:
            raise ValueError(adapter)
        pieces, type_pieces, label_pieces, geo = [], [], [], []
        start, offset = 0, 0
        for boundary, count in after.items():
            width = boundary + 1 - start
            pieces.extend((row[start:boundary+1], row.new_full((count,), pad)))
            type_pieces.extend((typ[start:boundary+1], typ.new_zeros(count)))
            if lab is not None:
                label_pieces.extend((lab[start:boundary+1], lab.new_full((count,), -100)))
            geo.extend(range(offset + width, offset + width + count))
            offset += width + count
            start = boundary + 1
        pieces.append(row[start:]); type_pieces.append(typ[start:])
        if lab is not None: label_pieces.append(lab[start:])
        prepared.append((torch.cat(pieces), torch.cat(type_pieces),
                         torch.cat(label_pieces) if lab is not None else None, geo))
    width = max(len(x[0]) for x in prepared)
    if width > max_context:
        raise ValueError(f'Geometry context {width} exceeds {max_context}; no silent truncation')
    result['input_ids'] = ids.new_full((len(prepared), width), pad)
    result['attention_mask'] = mask.new_zeros((len(prepared), width))
    result['mm_token_type_ids'] = types.new_zeros((len(prepared), width))
    if labels is not None: result['labels'] = labels.new_full((len(prepared), width), -100)
    positions = ids.new_full((len(prepared), max(len(x[3]) for x in prepared)), -1)
    for b, (row, typ, lab, geo) in enumerate(prepared):
        result['input_ids'][b, :len(row)] = row
        result['attention_mask'][b, :len(row)] = 1
        result['mm_token_type_ids'][b, :len(row)] = typ
        if lab is not None: result['labels'][b, :len(row)] = lab
        positions[b, :len(geo)] = ids.new_tensor(geo)
    result['geometry_positions'] = positions
    return result


def preprocess_geometry(paths, source):
    if source not in sys.path: sys.path.insert(0, source)
    from vggt.utils.load_fn import load_and_preprocess_images_square
    images, _ = load_and_preprocess_images_square(paths, target_size=448)
    return images


class MatrixCollator(Collator):
    def __init__(self, processor, source, adapter, training=True):
        super().__init__(processor, max_side=448, max_context=16384, training=training)
        self.source, self.adapter = source, adapter

    def __call__(self, rows):
        result = insert_slots(super().__call__(rows), self.processor,
                              [len(r['media']) for r in rows], self.adapter)
        result['geometry_images'] = [preprocess_geometry(r['media'], self.source) for r in rows]
        return result


class Qwen3VLGeometryMatrix(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        spec = getattr(config, 'geometry_matrix', None)
        if spec:
            self.attach_geometry(spec)

    def _apply(self, fn, recurse=True):
        # Qwen constructs nonpersistent rotary frequencies in float32. Casting
        # the whole model (including DeepSpeed's bf16 setup) would round them,
        # while from_pretrained reconstructs full-precision frequencies. That
        # changes positions/logits after reload despite identical saved weights.
        rotary=[]
        for module in self.modules():
            value=getattr(module,'inv_freq',None)
            if isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.device.type!='meta':
                original=getattr(module,'original_inv_freq',None)
                rotary.append((module,value.clone(),original.clone() if isinstance(original,torch.Tensor) and original.device.type!='meta' else None))
        result=super()._apply(fn,recurse=recurse)
        for module,value,original in rotary:
            module.inv_freq=value.to(device=module.inv_freq.device,dtype=torch.float32)
            if original is not None:
                module.original_inv_freq=original.to(device=module.inv_freq.device,dtype=torch.float32)
        return result

    def attach_geometry(self, spec, weights=None):
        from .geometry_backbone import RegisteredVGGT
        from .geometry_downsample import DownsampleGeometryAdapter
        self.config.geometry_matrix = dict(spec)
        hidden = self.config.text_config.hidden_size
        if spec['adapter'] == 'query64':
            self.geometry_adapter = GeometryTokenAdapter(output_dim=hidden, dropout=0.)
            # There is no stochastic/null training branch in these new arms.
            self.geometry_adapter.null_tokens.requires_grad_(False)
        elif spec['adapter'] == 'downsample':
            self.geometry_adapter = DownsampleGeometryAdapter(input_dim=2048, output_dim=hidden)
        else:
            raise ValueError(spec['adapter'])
        self.geometry_backbone = RegisteredVGGT(spec['source'], weights=weights, trainable=False)

    def configure_stage(self, stage, train_vggt=False):
        if stage not in ('align', 'sft', 'eval'):
            raise ValueError(stage)
        if stage == 'align' and train_vggt:
            raise ValueError('Alignment freezes the geometry backbone in all arms')
        self.requires_grad_(stage == 'sft')
        self.geometry_adapter.requires_grad_(stage != 'eval')
        if hasattr(self.geometry_adapter, 'null_tokens'):
            self.geometry_adapter.null_tokens.requires_grad_(False)
        self.geometry_backbone.set_trainable(stage == 'sft' and train_vggt)
        self._matrix_stage = stage
        self.train(stage != 'eval')
        if stage == 'align': self.model.visual.eval()
        self.config.use_cache = stage == 'eval'
        if stage != 'eval':
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, '_matrix_stage', None) == 'align':
            # Keep the frozen language stack in training mode so activation
            # checkpointing remains enabled for gradients into geometry inputs.
            # Qwen's attention dropout is zero in the locked base configuration.
            self.model.visual.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, mm_token_type_ids=None, cache_position=None, labels=None,
                logits_to_keep=0, use_cache=None, geometry_images=None, geometry_positions=None,
                geometry_force_null=None, **kwargs):
        threshold=getattr(self,'alignment_checkpoint_threshold',0)
        if threshold and self.training:
            from .alignment_checkpoint_policy import apply_alignment_checkpoint_policy
            width=input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            apply_alignment_checkpoint_policy(self,width,threshold)
        prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        handle = None
        if prefill:
            if geometry_images is None or geometry_positions is None:
                raise ValueError('Geometry checkpoint requires real images and explicit slots')
            images = [geometry_images] if isinstance(geometry_images, torch.Tensor) and geometry_images.ndim == 4 else geometry_images
            geometries = []
            encoder_batch=getattr(self,'geometry_encoder_batch_size',1)
            features=self.geometry_backbone.forward_batch(images,max_batch=encoder_batch)
            for feat in features:
                geometries.append(self.geometry_adapter(feat.unsqueeze(0))[0])
            if len(geometries) != geometry_positions.shape[0]:
                raise ValueError('Geometry batch mismatch')
            def replace(module, args, embeddings):
                output = embeddings.clone()
                for b, geo in enumerate(geometries):
                    pos = geometry_positions[b]; pos = pos[pos >= 0]
                    if len(pos) != len(geo): raise ValueError('Geometry slot count mismatch')
                    if geometry_force_null is not None and bool(geometry_force_null[b]):
                        geo = geo * 0
                    output[b, pos] = geo.to(output.dtype)
                return output
            handle = self.get_input_embeddings().register_forward_hook(replace)
        try:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position, labels=labels, logits_to_keep=logits_to_keep,
                use_cache=use_cache, **kwargs)
        finally:
            if handle is not None: handle.remove()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        geometry = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith('geometry_')}
        result = super().prepare_inputs_for_generation(input_ids, **kwargs)
        result.update(geometry)
        return result


def load_matrix_model(path, source, weights=None, adapter=None, stage='eval', train_vggt=False):
    config = AutoConfig.from_pretrained(path)
    existing = getattr(config, 'geometry_matrix', None)
    if existing:
        if adapter and adapter != existing['adapter']: raise ValueError('Checkpoint adapter mismatch')
        config.geometry_matrix['source'] = str(source)
    elif stage != 'align':
        raise ValueError('SFT/evaluation require a geometry checkpoint, not a vanilla model')
    model = Qwen3VLGeometryMatrix.from_pretrained(path, config=config,
                        dtype=torch.bfloat16, attn_implementation='sdpa')
    if not existing:
        if weights is None or adapter is None: raise ValueError('Alignment initialization needs weights and adapter')
        model.attach_geometry({'source': str(source), 'adapter': adapter, 'version': 1}, weights=weights)
        model.geometry_adapter.to(dtype=torch.bfloat16)
        model.geometry_backbone.to(dtype=torch.bfloat16)
    # HF low-memory construction does not restore VGGT's nonpersistent image
    # normalization buffers from a state_dict. Never reinitialize learned weights.
    model.geometry_backbone.restore_preprocessing_buffers()
    model.configure_stage(stage, train_vggt)
    return model
