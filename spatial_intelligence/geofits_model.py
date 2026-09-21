"""Registered Qwen3-VL GeoFits model with answer-blind, prefill-only fusion.

External frozen teachers are deliberately not children of the trainable model.
The six raw patch grids are supplied through feature_context, which MUST span
forward and backward when activation checkpointing is enabled. This is a local
adaptation, not an assertion of original-paper implementation parity.
"""
from contextlib import contextmanager
from dataclasses import asdict

import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from .geofits import GeoFitsConfig, GeoFitsFusion


def parse_config(settings):
    settings = dict(settings)
    for key in ('vggt_levels', 'pi3_levels', 'fusion_layers'):
        if key in settings:
            settings[key] = tuple(settings[key])
    return GeoFitsConfig(**settings)


class GeoFitsQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        settings = getattr(config, 'geofits_config', None)
        if settings is None:
            raise ValueError('GeoFits architecture must be explicit before construction')
        cfg = parse_config(settings)
        if cfg.hidden_size != config.text_config.hidden_size:
            raise ValueError('GeoFits/native language hidden-size mismatch')
        self.geofits = GeoFitsFusion(cfg)
        self.config.geofits_config = asdict(cfg)
        self._features = None
        self._contexts = None
        self._cached_decode = False
        self._geometry_handles = [self.model.language_model.register_forward_pre_hook(
            self._capture_original, with_kwargs=True)]
        self._geometry_handles.extend(self.model.language_model.layers[i].register_forward_hook(
            self._fusion_hook(i+1)) for i in range(3))

    def _apply(self, fn, recurse=True):
        # Native rotary frequencies are nonpersistent: do not quantize constants
        # during bf16 conversion, otherwise fresh reload changes the computation.
        constants = [(m, n, value.float().clone()) for m in self.modules()
                     for n, value in m.named_buffers(recurse=False) if n.endswith('inv_freq')]
        result = super()._apply(fn, recurse=recurse)
        for module, name, value in constants:
            module._buffers[name] = value.to(device=module._buffers[name].device)
        return result

    @contextmanager
    def feature_context(self, features):
        """Each sample: vggt/pi3 grids, native_grid, visual_indices, prefix_length.

        Optional timestamps follow the declared time encoding. Labels must be
        supplied during SFT, including when a custom loss removes model labels.
        Only raw frozen teacher grids may be reused; never cache native features.
        """
        if self._features is not None:
            raise RuntimeError('Nested/concurrent geometry contexts unsupported')
        self._features = features
        try:
            yield self
        finally:
            self._features = self._contexts = None
            self._cached_decode = False

    def _capture_original(self, module, args, kwargs):
        if self._cached_decode:
            return
        hidden = kwargs.get('inputs_embeds')
        if hidden is None or self._features is None:
            raise ValueError('GeoFits requires explicit native embeddings and teacher features')
        if len(self._features) != hidden.shape[0]:
            raise ValueError('One aligned teacher feature entry per sample required')
        contexts = []
        for i, feature in enumerate(self._features):
            indices = feature['visual_indices'].to(device=hidden.device, dtype=torch.long).reshape(1, -1)
            prefix = int(feature['prefix_length'])
            if not 0 < prefix <= hidden.shape[1] or not indices.numel():
                raise ValueError('Invalid question prefix or empty visual layout')
            if (indices < 0).any() or (indices >= prefix-1).any():
                raise ValueError('Geometry indices must precede terminal prompt token')
            def grids(name):
                return {int(k): value.to(device=hidden.device, dtype=hidden.dtype)
                        for k, value in feature[name].items()}
            times = feature.get('timestamps')
            if times is not None:
                times = times.to(hidden.device)
            bank = self.geofits.bank(grids('vggt'), grids('pi3'), times,
                                     native_grid=tuple(feature['native_grid']))
            if bank.shape[1] != indices.numel():
                raise ValueError('Teacher bank/native image token count mismatch')
            labels = feature.get('labels')
            if labels is not None:
                labels = labels.to(hidden.device).reshape(1, -1)
            contexts.append(dict(original_visual=hidden[i:i+1, indices[0]], bank=bank,
                visual_indices=indices, question_indices=torch.tensor([prefix-1], device=hidden.device),
                prefix_lengths=torch.tensor([prefix], device=hidden.device), labels=labels))
        self._contexts = contexts

    def _fusion_hook(self, layer):
        def hook(module, args, output):
            if self._cached_decode:
                return output
            if self._contexts is None:
                raise ValueError('Missing answer-blind prefill context')
            hidden = output[0] if isinstance(output, tuple) else output
            pieces = [self.geofits(hidden[i:i+1], layer=layer, **context)[0]
                      for i, context in enumerate(self._contexts)]
            fused = torch.cat(pieces, dim=0)
            return (fused, *output[1:]) if isinstance(output, tuple) else fused
        return hook

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, cache_position=None,
                logits_to_keep=0, use_cache=None, **kwargs):
        self._cached_decode = bool(past_key_values is not None and past_key_values.get_seq_length() > 0)
        if not self._cached_decode and self._features is None:
            raise ValueError('Use feature_context for every GeoFits prefill/training forward')
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids, cache_position=cache_position, logits_to_keep=logits_to_keep,
            use_cache=use_cache, **kwargs)


def load_geofits_model(path, *, architecture=None, **kwargs):
    config = AutoConfig.from_pretrained(path)
    saved = getattr(config, 'geofits_config', None)
    if architecture is not None:
        settings = asdict(parse_config(architecture))
        if saved is not None and asdict(parse_config(saved)) != settings:
            raise ValueError('Saved GeoFits architecture differs from requested recipe')
        config.geofits_config = settings
    if not getattr(config, 'geofits_config', None):
        raise ValueError('Base checkpoint needs an explicit GeoFits architecture')
    return GeoFitsQwen3VLForConditionalGeneration.from_pretrained(path, config=config, **kwargs)
