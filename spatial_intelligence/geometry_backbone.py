"""Registered VGGT aggregator for frozen and end-to-end geometry SFT.

Uses the separately installed, pinned VGGT source under its upstream license.
No prediction heads are executed or saved; all aggregator parameters are part
of the parent model state_dict, including when frozen.
"""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch import nn


def synchronized_sequence_chunks(rank_shapes, max_batch):
    """Contiguous chunks legal on every rank; identical module call counts.

    ZeRO-3's parameter collectives must follow the same module execution order.
    Different local length buckets must never independently choose call counts.
    No sample, frame, or optimizer-step order is changed by this plan.
    """
    if not rank_shapes or len({len(row) for row in rank_shapes}) != 1:
        raise ValueError('All ranks must have equal geometry microbatch counts')
    if max_batch < 1:
        raise ValueError('Positive encoder batch required')
    result = []
    start = 0
    while start < len(rank_shapes[0]):
        stop = start + 1
        while stop < min(start + max_batch, len(rank_shapes[0])):
            if any(row[stop] != row[start] for row in rank_shapes):
                break
            stop += 1
        result.append(list(range(start, stop)))
        start = stop
    return result


class RegisteredVGGT(nn.Module):
    def __init__(self, source, weights=None, trainable=False):
        super().__init__()
        self.source = str(Path(source).resolve())
        if self.source not in sys.path:
            sys.path.insert(0, self.source)
        from vggt.models.aggregator import Aggregator
        # HF from_pretrained constructs children under a meta-device context.
        # Upstream DINO builds a stochastic-depth schedule using Tensor.item(),
        # which cannot run on meta tensors. Explicit CPU construction is needed
        # before HF replaces registered tensors with saved checkpoint weights.
        with torch.device('cpu'):
            self.aggregator = Aggregator()
        # Only the final layer is consumed. This changes retention, not values.
        if hasattr(self.aggregator, 'cached_layer_indices'):
            self.aggregator.cached_layer_indices = {self.aggregator.depth - 1}
        if weights is not None:
            self.load_released(weights)
        self.set_trainable(trainable)

    def load_released(self, weights):
        """Strictly validate the complete released model before retaining its trunk."""
        from vggt.models.vggt import VGGT
        path = Path(weights)
        if path.is_dir():
            path = path / 'model.pt'
        full = VGGT()
        state = torch.load(path, map_location='cpu', weights_only=True)
        full.load_state_dict(state, strict=True)
        self.aggregator.load_state_dict(full.aggregator.state_dict(), strict=True)
        del state, full

    def set_trainable(self, trainable):
        self.trainable = bool(trainable)
        self.aggregator.requires_grad_(self.trainable)
        # Upstream's mask token is never consumed by unmasked RGB inference.
        patch_embed = getattr(self.aggregator, 'patch_embed', None)
        if patch_embed is not None and hasattr(patch_embed, 'mask_token'):
            patch_embed.mask_token.requires_grad_(False)
        self.aggregator.train(self.training and self.trainable)
        return self

    def restore_preprocessing_buffers(self):
        """Restore upstream constants discarded by HF meta checkpoint loading.

        These are nonpersistent normalization buffers, not learned parameters.
        Calling this after from_pretrained is necessary: HF can materialize
        nonpersistent buffers with empty storage rather than constructor values.
        """
        for name, values in (('_resnet_mean', (0.485, 0.456, 0.406)),
                             ('_resnet_std', (0.229, 0.224, 0.225))):
            old = getattr(self.aggregator, name)
            device = old.device if old.device.type != 'meta' else torch.device('cpu')
            # Upstream buffers are float32. Preserve normalization arithmetic.
            setattr(self.aggregator, name, torch.tensor(values, device=device,
                    dtype=torch.float32).reshape(1, 1, 3, 1, 1))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Trainer/DeepSpeed may cast the whole module after initial loading.
        # Keep RGB normalization identical to the upstream float32 constants;
        # casting constants to bf16 changes the inputs, not just compute dtype.
        if hasattr(self, 'aggregator') and hasattr(self.aggregator, '_resnet_mean'):
            if self.aggregator._resnet_mean.device.type != 'meta':
                self.restore_preprocessing_buffers()
        return result

    def train(self, mode=True):
        super().train(mode)
        self.aggregator.train(mode and self.trainable)
        return self

    def forward(self, images):
        """[T,3,448,448] -> [T,1024,2048], preserving training gradients."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 448, 448):
            raise ValueError(f'Expected [T,3,448,448], got {tuple(images.shape)}')
        context = nullcontext() if self.trainable else torch.no_grad()
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if images.device.type == 'cuda' else nullcontext()
        with context, precision:
            layers, start = self.aggregator(images.unsqueeze(0))
            result = layers[-1][0, :, start:]
        if tuple(result.shape) != (images.shape[0], 1024, 2048):
            raise ValueError(f'Invalid VGGT final patch shape {tuple(result.shape)}')
        return result

    def forward_batch(self, sequences, max_batch=1):
        """Batch equal-length independent sequences along B, never along time.

        Trainable batching requires an explicit opt-in and a deterministic
        trunk. Its shared cross-rank plan preserves ZeRO collective order.
        """
        if not isinstance(max_batch,int) or isinstance(max_batch,bool) or max_batch<1:
            raise ValueError('max_batch must be a positive integer')
        trainable_batch = self.trainable and getattr(self, 'trainable_batching', False)
        if max_batch==1 or (self.trainable and not trainable_batch):
            return [self(image) for image in sequences]
        groups={}; output=[None]*len(sequences)
        for index,image in enumerate(sequences):
            if image.ndim!=4 or tuple(image.shape[1:])!=(3,448,448) or image.shape[0]<1:
                raise ValueError('Expected nonempty [T,3,448,448] sequences')
            key=(tuple(image.shape),image.dtype,image.device)
            groups.setdefault(key,[]).append(index)
        if trainable_batch:
            for module in self.aggregator.modules():
                if ((isinstance(module, nn.Dropout) and module.p > 0)
                    or float(getattr(module, 'sample_drop_ratio', 0)) > 0
                    or float(getattr(module, 'drop_prob', 0) or 0) > 0):
                    raise ValueError('Trainable batching requires zero stochastic dropout/drop-path')
        # Frozen trunks can also carry ZeRO parameter-gather hooks. Therefore
        # distributed frozen batching requires the same consensus as trainable
        # batching, despite not recording geometry gradients.
        if trainable_batch or torch.distributed.is_initialized():
            shapes = [(tuple(x.shape), str(x.dtype)) for x in sequences]
            rank_shapes = [shapes]
            if torch.distributed.is_initialized():
                rank_shapes = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(rank_shapes, shapes)
            chunks = synchronized_sequence_chunks(rank_shapes, max_batch)
        else:
            chunks = [indices[offset:offset+max_batch]
                      for indices in groups.values()
                      for offset in range(0, len(indices), max_batch)]
        for selected in chunks:
            images=torch.stack([sequences[i] for i in selected])
            precision=torch.autocast('cuda',dtype=torch.bfloat16) if images.device.type=='cuda' else nullcontext()
            context = nullcontext() if self.trainable else torch.no_grad()
            with context,precision:
                layers,start=self.aggregator(images)
                features=layers[-1][:,:,start:]
            if tuple(features.shape)!=(len(selected),images.shape[1],1024,2048):
                raise ValueError('Invalid batched VGGT final patch shape')
            for position,index in enumerate(selected): output[index]=features[position]
        return output

    def preprocess(self, paths):
        from vggt.utils.load_fn import load_and_preprocess_images_square
        return load_and_preprocess_images_square(paths, target_size=448)
