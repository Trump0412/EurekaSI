"""Region-aligned six-entry geometry retrieval, independent of HF model hooks.

This is an explicit adaptation core, not a claim of paper/runtime reproduction.
Teacher outputs must be real aligned patch grids. Caller owns preprocessing,
native video temporal grouping, decoder post-layer hooks and prefill KV caches.
No LoRA: language/native RGB/fusion are full-parameter trainable; teachers frozen.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GeoFitsConfig:
    hidden_size: int
    vggt_width: int
    pi3_width: int
    temporal_bottleneck: int
    # Underspecified in the paper: caller must explicitly select these.
    temporal_group_size: int
    temporal_group_reduction: str
    timestamp_encoding: str
    pooling: str
    retrieval_width: int
    vggt_levels: tuple = (11, 17, 23)
    pi3_levels: tuple = (17, 26, 35)
    fusion_layers: tuple = (1, 2, 3)  # one-based, post decoder block
    top_k: int = 2
    gate_bias: float = 0.0

    def __post_init__(self):
        if min(self.hidden_size, self.vggt_width, self.pi3_width,
               self.temporal_bottleneck, self.retrieval_width) < 1:
            raise ValueError('All feature widths must be positive')
        if self.temporal_group_size not in (1, 2):
            raise ValueError('Explicit native temporal grouping must be 1 or 2')
        if self.temporal_group_reduction != 'mean' or self.pooling != 'average_2x2':
            raise ValueError('Only declared mean temporal / average spatial adaptation supported')
        if self.timestamp_encoding not in ('normalized_linear_sincos', 'order_only_sincos'):
            raise ValueError('Explicit timestamp encoding adaptation required')
        if self.vggt_levels != (11, 17, 23) or self.pi3_levels != (17, 26, 35):
            raise ValueError('Six-entry bank requires the declared teacher levels')
        if self.top_k != 2 or self.fusion_layers != (1, 2, 3):
            raise ValueError('This configuration implements three-stage TopK2 only')


def relative_time(timestamps):
    """Normalize real, ordered timestamps per clip; no inferred FPS/seconds."""
    if timestamps.ndim != 2 or timestamps.shape[1] < 1:
        raise ValueError('timestamps must have shape [batch, frames]')
    if not torch.isfinite(timestamps).all() or (timestamps[:, 1:] < timestamps[:, :-1]).any():
        raise ValueError('Finite nondecreasing actual timestamps required')
    time = timestamps.float() - timestamps[:, :1].float()
    span = time[:, -1:]
    if timestamps.shape[1] > 1 and (span <= 0).any():
        raise ValueError('A multi-frame clip needs a positive measured temporal span')
    time = time / span.clamp_min(torch.finfo(torch.float32).eps)
    return torch.stack((time, torch.sin(math.pi * time), torch.cos(math.pi * time)), -1)


class FrozenGeometryTeacher(nn.Module):
    """Boundary wrapper; feature adapters must live OUTSIDE this no-grad trunk."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.backbone(*args, **kwargs)


class TemporalAdapter(nn.Module):
    """Explicit residual bottleneck adaptation; not a recovered pi3 release."""
    def __init__(self, width, bottleneck):
        super().__init__()
        self.down = nn.Linear(width + 3, bottleneck)
        self.up = nn.Linear(bottleneck, width)

    def forward(self, grid, encoded_time):
        time = encoded_time[:, :, None, None, :].expand(*grid.shape[:-1], 3)
        return grid + self.up(F.gelu(self.down(torch.cat((grid, time.to(grid)), -1))))


class GeometryBank(nn.Module):
    """B,T,H,W,C -> B,N,6,D, with strict space/time layout checking."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projectors = nn.ModuleList([nn.Linear(width, config.hidden_size)
            for width in [config.vggt_width] * 3 + [config.pi3_width] * 3])
        self.temporal = nn.ModuleList([TemporalAdapter(config.pi3_width, config.temporal_bottleneck)
                                       for _ in range(3)])

    def forward(self, vggt, pi3, timestamps, *, native_grid):
        cfg = self.config
        if set(vggt) != set(cfg.vggt_levels) or set(pi3) != set(cfg.pi3_levels):
            raise ValueError('Missing or extra geometry bank levels')
        if timestamps is None:
            if cfg.timestamp_encoding != 'order_only_sincos':
                raise ValueError('Measured timestamp mode cannot synthesize timestamps')
            first = vggt[cfg.vggt_levels[0]]
            timestamps = torch.arange(first.shape[1], device=first.device).expand(first.shape[0], -1)
        elif cfg.timestamp_encoding == 'order_only_sincos':
            raise ValueError('Order-only mode accepts no timestamps; provenance stays explicit')
        encoded = relative_time(timestamps)
        entries = []
        shape = None
        for index, (mapping, level) in enumerate(
                [(vggt, n) for n in cfg.vggt_levels] + [(pi3, n) for n in cfg.pi3_levels]):
            grid = mapping[level]
            width = cfg.vggt_width if index < 3 else cfg.pi3_width
            if grid.ndim != 5 or grid.shape[-1] != width:
                raise ValueError('Teacher feature must be restored [B,T,H,W,C] patch grid')
            b, t, h, w, c = grid.shape
            if shape is None: shape = (b, t, h, w)
            if shape != (b, t, h, w) or timestamps.shape != (b, t):
                raise ValueError('Teacher space/time grids and timestamps must align exactly')
            if h % 2 or w % 2 or t % cfg.temporal_group_size:
                raise ValueError('No silent padding/cropping of geometry grids or temporal groups')
            if tuple(native_grid) != (t // cfg.temporal_group_size, h // 2, w // 2):
                raise ValueError('Native visual merger grid mismatch; explicit alignment required')
            if not torch.isfinite(grid).all(): raise ValueError('Nonfinite teacher features')
            if grid.requires_grad:
                raise ValueError('External teacher features must be frozen before trainable adapters')
            if index >= 3: grid = self.temporal[index - 3](grid, encoded)
            # Frame-major, then row-major matches the declared merged-grid contract.
            grid = grid.reshape(b, t, h // 2, 2, w // 2, 2, c).mean((3, 5))
            grid = grid.reshape(b, t // cfg.temporal_group_size, cfg.temporal_group_size,
                                h // 2, w // 2, c).mean(2)
            entries.append(self.projectors[index](grid.flatten(1, 3)))
        return torch.stack(entries, dim=2)


class RegionRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, r = config.hidden_size, config.retrieval_width
        self.norm_x = nn.LayerNorm(d); self.norm_c = nn.LayerNorm(d); self.norm_h = nn.LayerNorm(d)
        self.query = nn.Linear(3 * d, r)
        self.key = nn.Linear(d, r); self.value = nn.Linear(d, d)
        self.output = nn.Linear(d, d)
        self.gate = nn.Sequential(nn.Linear(3 * d, max(1, d // 4)), nn.GELU(), nn.Linear(max(1, d // 4), 1))
        nn.init.constant_(self.gate[-1].bias, config.gate_bias)
        self.retrieval_width = r

    def forward(self, current, original, context, bank):
        if current.shape != original.shape or bank.shape != (*current.shape[:2], 6, current.shape[-1]):
            raise ValueError('Region, original visual tokens and six-entry bank must align')
        context = self.norm_c(context)[:, None, :].expand_as(current)
        normalized = self.norm_x(current)
        query = self.query(torch.cat((normalized, original, context), -1))
        scores = (query.unsqueeze(2).float() * self.key(bank).float()).sum(-1) / math.sqrt(self.retrieval_width)
        selected, indices = scores.topk(2, dim=-1)
        weights = torch.zeros_like(scores).scatter(-1, indices, selected.softmax(-1))
        retrieved = (weights.to(bank).unsqueeze(-1) * self.value(bank)).sum(2)
        h = self.norm_h(retrieved)
        gate = self.gate(torch.cat((normalized, context, h), -1)).sigmoid()
        return current + gate * self.output(h), {'weights': weights, 'indices': indices, 'gate': gate}


class GeoFitsFusion(nn.Module):
    """Caller invokes after decoder blocks 1/2/3 on PREFILL/full SFT only.

    Prefix-internal question->visual feedback is intentional, so this is not
    ordinary triangular attention inside the prompt. Only answer-blind prefix
    states may condition feedback. Never use the full sequence's last token.
    Incremental decoding reuses prefill KV state, not this fusion again.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bank = GeometryBank(config)
        self.layers = nn.ModuleList([RegionRetrieval(config) for _ in range(3)])

    def forward(self, hidden, original_visual, bank, *, visual_indices,
                question_indices, prefix_lengths, layer, labels=None):
        b, s, d = hidden.shape
        if layer not in self.config.fusion_layers: raise ValueError('Only post-layers 1/2/3 supported')
        if visual_indices.ndim != 2 or visual_indices.shape[0] != b:
            raise ValueError('Visual indices require [B,N] shape')
        if question_indices.shape != (b,) or prefix_lengths.shape != (b,):
            raise ValueError('Per-sample question index and prefix length required')
        for value in (visual_indices, question_indices, prefix_lengths):
            if value.dtype != torch.long: raise ValueError('Indices must use int64')
        if (prefix_lengths < 1).any() or (prefix_lengths > s).any(): raise ValueError('Invalid prompt prefix')
        if (question_indices < 0).any() or (question_indices >= prefix_lengths).any():
            raise ValueError('Question index must precede every answer token')
        if (visual_indices < 0).any() or (visual_indices >= question_indices[:, None]).any():
            raise ValueError('Visual regions must precede terminal question token')
        if any(row.unique().numel() != row.numel() for row in visual_indices):
            raise ValueError('Visual indices cannot repeat')
        if labels is not None:
            if labels.shape != (b, s): raise ValueError('Labels shape mismatch')
            prefix = torch.arange(s, device=hidden.device)[None, :] < prefix_lengths[:, None]
            if (labels[prefix] != -100).any(): raise ValueError('Answer supervision leaked into question prefix')
        row = torch.arange(b, device=hidden.device)
        current = hidden[row[:, None], visual_indices]
        fused, diagnostics = self.layers[layer - 1](current, original_visual,
                                                    hidden[row, question_indices], bank)
        output = hidden.clone()
        output[row[:, None], visual_indices] = fused
        return output, diagnostics


def enable_full_parameter_training(language, native_rgb, fusion, teachers):
    """Explicit new-experiment policy; does not act on any existing global model."""
    for module in (language, native_rgb, fusion):
        if any('lora_' in name for name, _ in module.named_parameters()):
            raise ValueError('Full-parameter recipe must not contain LoRA adapters')
        module.requires_grad_(True)
    for teacher in teachers:
        if not isinstance(teacher, FrozenGeometryTeacher):
            raise ValueError('Teacher requires enforced frozen/eval boundary')
        teacher.requires_grad_(False).eval()


@dataclass
class FusionContext:
    original_visual: torch.Tensor
    bank: torch.Tensor
    visual_indices: torch.Tensor
    question_indices: torch.Tensor
    prefix_lengths: torch.Tensor
    labels: torch.Tensor | None = None


class InstalledFusionHooks:
    """Persistent post-block hooks; context must span forward AND backward.

    Keeping hooks/context through backward makes activation-checkpoint recompute
    equivalent to the original forward. Do not cache fused states across calls.
    Bind the same question-only context for full teacher forcing and prefill.
    Incremental decode runs with no context: prefill KV already contains fusion.
    Qwen native DeepStack updates happen in the text-model loop AFTER the block
    hook, so ordering is decoder -> geometry -> native DeepStack (additive).
    """
    def __init__(self, decoder_layers, fusion):
        if len(decoder_layers) < 3: raise ValueError('Three decoder layers required')
        self.fusion = fusion
        self.current = None
        self.handles = [decoder_layers[index].register_forward_hook(self._hook(index + 1))
                        for index in range(3)]

    def _hook(self, layer):
        def callback(module, args, output):
            if self.current is None: return output
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor): raise TypeError('Unsupported decoder block output')
            context = self.current
            if hidden.shape[1] < int(context.prefix_lengths.max()):
                raise ValueError('Do not reinject prefill geometry during incremental decoding')
            fused, _ = self.fusion(hidden, context.original_visual, context.bank,
                visual_indices=context.visual_indices, question_indices=context.question_indices,
                prefix_lengths=context.prefix_lengths, labels=context.labels, layer=layer)
            return (fused, *output[1:]) if isinstance(output, tuple) else fused
        return callback

    @contextmanager
    def context(self, context):
        if self.current is not None: raise RuntimeError('Nested or concurrent fusion contexts unsupported')
        self.current = context
        try: yield self
        finally: self.current = None

    def remove(self):
        if self.current is not None: raise RuntimeError('Do not remove hooks during active backward context')
        for handle in self.handles: handle.remove()
        self.handles = []


def install_geofits(decoder_layers, fusion):
    """Install on actual decoder blocks; call handle.context(FusionContext(...))."""
    return InstalledFusionHooks(decoder_layers, fusion)
