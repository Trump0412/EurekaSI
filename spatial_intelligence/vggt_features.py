"""Frozen final-layer VGGT patch features, not predicted XYZ/depth tokens.

Exact ordered, marked RGB media are preserved. Cache identity uses file metadata
and pinned source/weight revision, not an additional SHA256 inventory.
"""
import hashlib
import json
from pathlib import Path
import os
import sys

SCHEMA = 'vggt-final-patches-square448-bf16-v1'


def media_identity(paths):
    result = []
    for value in paths:
        path = Path(value).resolve(strict=True)
        stat = path.stat()
        result.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    if not result:
        raise ValueError('Geometry requires nonempty ordered RGB media')
    return result


def cache_key(paths):
    return hashlib.blake2b(json.dumps({'schema': SCHEMA, 'media': media_identity(paths)},
                                    sort_keys=True).encode(), digest_size=20).hexdigest()


def cache_path(root, paths):
    key = cache_key(paths)
    return Path(root) / key[:2] / (key + '.pt')


def load_features(root, paths):
    import torch
    value = torch.load(cache_path(root, paths), map_location='cpu', weights_only=True)
    if value['schema'] != SCHEMA or value['media'] != media_identity(paths):
        raise ValueError('Geometry cache input identity mismatch')
    features = value['features']
    if features.ndim != 3 or features.shape != (len(paths), 1024, 2048):
        raise ValueError(f'Unexpected final patch feature shape: {features.shape}')
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite cached VGGT features')
    return features


class FrozenVGGT:
    def __init__(self, source, weights, device='cuda:0'):
        import torch
        sys.path.insert(0, str(Path(source).resolve()))
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        self.preprocess = load_and_preprocess_images_square
        self.device = device
        # Load all original weights strictly; only aggregator is executed.
        self.model = VGGT()
        state = torch.load(Path(weights) / 'model.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        for name in ('camera_head', 'depth_head', 'point_head', 'track_head'):
            setattr(self.model, name, None)
        self.model.to(device).eval().requires_grad_(False)

    def extract(self, paths):
        import torch
        # Upstream aspect-preserving square padding, no crop / no frame dropping.
        images, coords = self.preprocess(paths, target_size=448)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            layers, start = self.model.aggregator(images[None].to(self.device))
            features = layers[-1][0, :, start:].to(dtype=torch.bfloat16).cpu().contiguous()
        if features.shape != (len(paths), 1024, 2048) or not torch.isfinite(features).all():
            raise ValueError('Invalid VGGT feature extraction')
        return features, coords

    def save(self, root, paths):
        import torch
        target = cache_path(root, paths)
        if target.exists():
            load_features(root, paths)
            return target
        before = media_identity(paths)
        features, coords = self.extract(paths)
        if media_identity(paths) != before:
            raise ValueError('Input media changed during extraction')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        torch.save({'schema': SCHEMA, 'media': before, 'features': features,
                    'original_coords': coords, 'patch_start_excluded': True}, temp)
        temp.replace(target)
        return target
