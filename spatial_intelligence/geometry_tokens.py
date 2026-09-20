"""Generic fixed-budget geometry tokens and sequence insertion primitives.

The extractor is external and must be frozen by the caller. These utilities do
not implement a model-family-specific rotary-position or DeepStack adapter.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


def _sinusoidal(length, width, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    rates = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                      * (-math.log(10000.0)/width))
    angles = position*rates
    result = torch.zeros(length, width, device=device, dtype=torch.float32)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :width//2].cos()
    return result.to(dtype)


class GeometryTokenAdapter(nn.Module):
    """[B,T,N,D] cached features -> [B,K,H], with explicit frame/patch positions.

    ``key_padding_mask`` is True for invalid geometry, unlike attention_mask.
    Learned-null replacement covers all K tokens of a sample. Random replacement
    applies only in training; force_null is an explicit train/eval intervention.
    Fully masked samples also take the null branch (never all-masked attention).
    The returned tokens require model-specific insertion/position handling.
    """

    def __init__(self, input_dim=2048, latent_dim=256, output_dim=2048,
                 num_tokens=64, num_heads=8, dropout=0.2):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in
               (input_dim, latent_dim, output_dim, num_tokens, num_heads)):
            raise ValueError('All dimensions must be positive integers')
        if latent_dim % num_heads or latent_dim < 4:
            raise ValueError('latent_dim must be >=4 and divisible by num_heads')
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be in [0,1]')
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.output_dim, self.num_tokens = output_dim, num_tokens
        self.sample_dropout = float(dropout)
        self.input_projection = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.queries = nn.Parameter(torch.randn(num_tokens, latent_dim)*0.02)
        self.resampler = nn.MultiheadAttention(latent_dim, num_heads, dropout=0., batch_first=True)
        self.projector = nn.Sequential(nn.Linear(latent_dim, output_dim), nn.GELU(),
                                       nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim))
        self.null_tokens = nn.Parameter(torch.randn(num_tokens, output_dim)*0.02)

    def forward(self, features, key_padding_mask=None, *, force_null=None):
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError('features must be [batch,frames,patches,input_dim]')
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError('Geometry features must be finite floating point')
        batch, frames, patches, _ = features.shape
        if min(batch, frames, patches) < 1:
            raise ValueError('Empty batch/frame/patch dimensions are invalid')
        if features.device != self.queries.device:
            raise ValueError('Features and adapter must be on the same device')
        if key_padding_mask is None:
            key_padding_mask = torch.zeros((batch, frames, patches), dtype=torch.bool, device=features.device)
        if (key_padding_mask.shape != features.shape[:3] or key_padding_mask.dtype != torch.bool
                or key_padding_mask.device != features.device):
            raise ValueError('key_padding_mask must be boolean [B,T,N] on feature device')
        if force_null is None:
            force_null = torch.zeros(batch, dtype=torch.bool, device=features.device)
        if force_null.shape != (batch,) or force_null.dtype != torch.bool or force_null.device != features.device:
            raise ValueError('force_null must be boolean [B] on feature device')
        dropped = force_null | key_padding_mask.flatten(1).all(1)
        if self.training and self.sample_dropout:
            dropped = dropped | (torch.rand(batch, device=features.device) < self.sample_dropout)
        # Always execute both branches: micro-batch=1 may drop every sample on
        # one DDP rank. Skipping its adapter would leave unused parameters with
        # find_unused_parameters=False. where supplies zero, non-None gradients.
        projected = self.input_projection(features.to(self.queries.dtype))
        # Distinct channel bands encode frame and patch indices. Flattening
        # alone would leave the cross-attention invariant to permutations.
        frame_width = self.latent_dim//2
        frame_pos = _sinusoidal(frames, frame_width, features.device, projected.dtype)
        patch_pos = _sinusoidal(patches, self.latent_dim-frame_width, features.device, projected.dtype)
        positions = torch.cat((frame_pos[:, None, :].expand(-1, patches, -1),
                               patch_pos[None, :, :].expand(frames, -1, -1)), dim=-1)
        tokens = (projected+positions).flatten(1, 2)
        safe_mask = key_padding_mask.flatten(1).clone()
        all_invalid = safe_mask.all(1)
        # Fully invalid rows are replaced by learned nulls after attention, but
        # their computed branch must still be finite for zero-gradient backward.
        safe_mask[all_invalid, 0] = False
        tokens = tokens.masked_fill(all_invalid[:, None, None], 0.)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        resampled, _ = self.resampler(queries, tokens, tokens,
                                     key_padding_mask=safe_mask, need_weights=False)
        output = self.projector(resampled).to(self.null_tokens.dtype)
        nulls = self.null_tokens.unsqueeze(0).expand(batch, -1, -1)
        return torch.where(dropped[:, None, None], nulls, output)


@dataclass
class GeometryInsertion:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    geometry_positions: torch.Tensor
    original_positions: torch.Tensor
    geometry_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor | None


def insert_geometry_tokens(input_ids, attention_mask, labels=None, *, vision_end_token_id,
                           placeholder_token_id, num_tokens=64, pad_token_id=0,
                           mm_token_type_ids=None):
    """Insert geometry after each sample's last *valid* native vision-end token.

    Valid original tokens are preserved in order and right-padded in the output.
    original_positions[B,L] maps old valid positions to new positions, -1 for
    padding. Use it to remap model-specific token types/visual masks; regenerate
    model-specific rotary positions rather than copying stale position_ids.
    Geometry labels and all padding labels are -100. Native vision IDs are never
    replaced, so their ordered DeepStack correspondence is preserved.
    """
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError('input_ids must be int64 [B,L]')
    if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
        raise ValueError('attention_mask must match input_ids shape/device')
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError('attention_mask must contain only zero/one')
    if labels is not None and (labels.shape != input_ids.shape or labels.dtype != torch.long
                               or labels.device != input_ids.device):
        raise ValueError('labels must be int64 matching input_ids')
    if mm_token_type_ids is not None and (mm_token_type_ids.shape != input_ids.shape
            or mm_token_type_ids.device != input_ids.device or mm_token_type_ids.dtype != torch.long):
        raise ValueError('mm_token_type_ids must be int64 matching input_ids')
    if type(num_tokens) is not int or num_tokens < 1:
        raise ValueError('num_tokens must be positive')
    if any(type(v) is not int or v < 0 for v in (vision_end_token_id, placeholder_token_id, pad_token_id)):
        raise ValueError('Token IDs must be nonnegative integers')
    if placeholder_token_id == vision_end_token_id:
        raise ValueError('Geometry placeholder must not be a vision boundary')
    batch, _ = input_ids.shape
    if batch < 1:
        raise ValueError('Empty batch')
    valid = attention_mask.bool()
    lengths = valid.sum(1)
    width = int(lengths.max())+num_tokens
    ids = input_ids.new_full((batch, width), pad_token_id)
    mask = attention_mask.new_zeros((batch, width))
    target = labels.new_full((batch, width), -100) if labels is not None else None
    geometry_positions = input_ids.new_empty((batch, num_tokens))
    original_positions = input_ids.new_full(input_ids.shape, -1)
    geometry_mask = torch.zeros((batch, width), dtype=torch.bool, device=input_ids.device)
    types = mm_token_type_ids.new_zeros((batch, width)) if mm_token_type_ids is not None else None
    for sample in range(batch):
        old = torch.nonzero(valid[sample], as_tuple=False).flatten()
        tokens = input_ids[sample, old]
        boundaries = torch.nonzero(tokens == vision_end_token_id, as_tuple=False).flatten()
        if not boundaries.numel():
            raise ValueError(f'Sample {sample} has no valid vision_end token')
        insertion = int(boundaries[-1])+1
        mapped = torch.arange(tokens.numel(), device=input_ids.device)
        mapped[insertion:] += num_tokens
        geo = torch.arange(insertion, insertion+num_tokens, device=input_ids.device)
        ids[sample, mapped] = tokens
        ids[sample, geo] = placeholder_token_id
        mask[sample, :tokens.numel()+num_tokens] = 1
        original_positions[sample, old] = mapped
        geometry_positions[sample] = geo
        geometry_mask[sample, geo] = True
        if types is not None:
            types[sample, mapped] = mm_token_type_ids[sample, old]
        if target is not None:
            target[sample, mapped] = labels[sample, old]
    return GeometryInsertion(ids, mask, target, geometry_positions, original_positions, geometry_mask, types)


def replace_geometry_embeddings(token_embeddings, geometry_embeddings, geometry_positions):
    """Differentiable replacement at explicit slots; never searches token IDs."""
    if token_embeddings.ndim != 3 or geometry_embeddings.ndim != 3:
        raise ValueError('Embeddings must be [B,L,H] and [B,K,H]')
    batch, length, hidden = token_embeddings.shape
    if geometry_embeddings.shape[0] != batch or geometry_embeddings.shape[2] != hidden:
        raise ValueError('Geometry embedding batch/hidden dimensions differ')
    if (geometry_positions.shape != geometry_embeddings.shape[:2]
            or geometry_positions.dtype != torch.long):
        raise ValueError('geometry_positions must be int64 [B,K]')
    if geometry_embeddings.device != token_embeddings.device or geometry_positions.device != token_embeddings.device:
        raise ValueError('All embeddings and positions must share a device')
    if ((geometry_positions < 0) | (geometry_positions >= length)).any():
        raise ValueError('Geometry positions out of bounds')
    if any(row.unique().numel() != row.numel() for row in geometry_positions):
        raise ValueError('Geometry positions must be unique per sample')
    indices = geometry_positions.unsqueeze(-1).expand(-1, -1, hidden)
    return token_embeddings.scatter(1, indices, geometry_embeddings.to(token_embeddings.dtype))
