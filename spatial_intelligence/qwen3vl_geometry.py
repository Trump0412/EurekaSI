"""Qwen3-VL with extra geometry tokens; native visual/DeepStack path is retained.

Use this class explicitly when reloading its checkpoints. A vanilla AutoModel
loader is not a valid geometry inference path.
"""
import torch
from transformers import Qwen3VLForConditionalGeneration
from .geometry_tokens import GeometryTokenAdapter, insert_geometry_tokens, replace_geometry_embeddings
from .qwen35 import Collator
from .vggt_features import load_features


class Qwen3VLGeometry(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(config, 'geometry_interface'):
            raise ValueError('Explicit geometry_interface configuration required')
        self.geometry_adapter = GeometryTokenAdapter(**config.geometry_interface)

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GeometryTokenAdapter):
            torch.nn.init.normal_(module.queries, std=.02)
            torch.nn.init.normal_(module.null_tokens, std=.02)
        elif isinstance(module, torch.nn.MultiheadAttention):
            module._reset_parameters()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, mm_token_type_ids=None, cache_position=None, labels=None,
                logits_to_keep=0, use_cache=None, geometry_features=None, geometry_padding_mask=None,
                geometry_positions=None, geometry_force_null=None, geometry_media=None, **kwargs):
        handle = None
        # During generation geometry is consumed only during the prompt prefill;
        # cached keys/values carry its influence through subsequent decode steps.
        prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        if prefill:
            if geometry_features is None and geometry_media is not None:
                if not hasattr(self, '_frozen_extractor'):
                    from .vggt_features import FrozenVGGT
                    self._frozen_extractor = FrozenVGGT(self.config.geometry_source,
                        self.config.geometry_weights, str(self.device))
                features = [self._frozen_extractor.extract(paths)[0] for paths in geometry_media]
                longest = max(value.shape[0] for value in features)
                geometry_features = torch.zeros((len(features), longest, 1024, 2048),
                                                dtype=torch.bfloat16, device=self.device)
                geometry_padding_mask = torch.ones(geometry_features.shape[:3], dtype=torch.bool, device=self.device)
                for i, value in enumerate(features):
                    geometry_features[i, :len(value)] = value.to(self.device)
                    geometry_padding_mask[i, :len(value)] = False
            if geometry_features is None or geometry_positions is None:
                raise ValueError('Geometry checkpoint requires explicit feature/slot inputs')
            geo = self.geometry_adapter(geometry_features, geometry_padding_mask, force_null=geometry_force_null)
            def replace(module, args, output):
                return replace_geometry_embeddings(output, geo, geometry_positions)
            handle = self.get_input_embeddings().register_forward_hook(replace)
        try:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids, cache_position=cache_position, labels=labels,
                logits_to_keep=logits_to_keep, use_cache=use_cache, **kwargs)
        finally:
            if handle is not None: handle.remove()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        custom = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith('geometry_')}
        result = super().prepare_inputs_for_generation(input_ids, **kwargs)
        result.update(custom)
        return result


class GeometryCollator(Collator):
    def __init__(self, processor, cache_root=None, max_side=448, max_context=16384, training=True):
        super().__init__(processor, max_side, max_context, training)
        self.cache_root = cache_root

    def __call__(self, rows):
        batch = super().__call__(rows)
        if self.cache_root is None:
            batch = add_geometry_slots(batch, self.processor, None, None, self.max_context)
            batch['geometry_media'] = [list(row['media']) for row in rows]
            return batch
        features = [load_features(self.cache_root, row['media']) for row in rows]
        longest = max(value.shape[0] for value in features)
        values = features[0].new_zeros((len(rows), longest, 1024, 2048))
        padding = torch.ones((len(rows), longest, 1024), dtype=torch.bool)
        for i, value in enumerate(features):
            values[i, :len(value)] = value; padding[i, :len(value)] = False
        return add_geometry_slots(batch, self.processor, values, padding, self.max_context)


def add_geometry_slots(batch, processor, features, padding, max_context=16384, force_null=False):
    batch = dict(batch)
    if 'position_ids' in batch:
        raise ValueError('Regenerate native Qwen3-VL positions after insertion, not stale position_ids')
    token = processor.tokenizer
    insertion = insert_geometry_tokens(batch['input_ids'], batch['attention_mask'], batch.pop('labels', None),
        vision_end_token_id=token.convert_tokens_to_ids('<|vision_end|>'),
        placeholder_token_id=token.pad_token_id, pad_token_id=token.pad_token_id,
        mm_token_type_ids=batch.get('mm_token_type_ids'))
    if insertion.input_ids.shape[1] > max_context:
        raise ValueError('Geometry tokens exceed context budget; no silent truncation')
    batch.update(input_ids=insertion.input_ids, attention_mask=insertion.attention_mask,
                 geometry_positions=insertion.geometry_positions, geometry_features=features,
                 geometry_padding_mask=padding,
                 geometry_force_null=torch.full((insertion.input_ids.shape[0],), force_null, dtype=torch.bool))
    if features is None:
        batch.pop('geometry_features'); batch.pop('geometry_padding_mask')
    if insertion.labels is not None: batch['labels'] = insertion.labels
    if insertion.mm_token_type_ids is None:
        raise ValueError('Native Qwen3-VL mRoPE requires processor mm_token_type_ids')
    batch['mm_token_type_ids'] = insertion.mm_token_type_ids
    return batch


def load_geometry_model(path, training=False, freeze_vision=True):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path)
    if not hasattr(config, 'geometry_interface'):
        if not training: raise ValueError('Refuse random geometry adapter for evaluation')
        config.geometry_interface = dict(input_dim=2048, latent_dim=256,
            output_dim=config.text_config.hidden_size, num_tokens=64, num_heads=8, dropout=.2)
        config.geometry_representation = 'vggt-final-patches-square448-bf16-v1'
    model = Qwen3VLGeometry.from_pretrained(path, config=config, dtype=torch.bfloat16, attn_implementation='sdpa')
    if training:
        model.config.use_cache = False
        if freeze_vision:
            model.model.visual.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model
