# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. SPDX-License-Identifier: Apache-2.0
"""RoboRefer 3x3 geometry interface, adapted to batched multi-frame VGGT.

Algorithm/MLP attribution: Zhoues/RoboRefer, commit
d97a995ad28376720a4c8beb64915c58ed16c844,
llava/model/multimodal_projector/base_projector.py. This adapter adds shape
validation and explicit padding/null masks; it does not implement query pooling.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def downsample_token_count(patches=1024):
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError('3x3 downsampling requires square patch grids')
    return ((side + 2) // 3) ** 2


def space_to_depth_3x3(features):
    """Exact upstream neighborhood/channel ordering and zero edge padding."""
    n, patches, channels = features.shape
    side = math.isqrt(patches)
    downsample_token_count(patches)
    grid = features.reshape(n, side, side, channels)
    pad = (-side) % 3
    grid = F.pad(grid, (0, 0, 0, pad, 0, pad))
    width = side + pad
    grid = grid.reshape(n, width, width // 3, channels * 3)
    grid = grid.permute(0, 2, 1, 3).contiguous()
    grid = grid.reshape(n, width // 3, width // 3, channels * 9)
    return grid.permute(0, 2, 1, 3).reshape(n, -1, channels * 9)


class DownsampleGeometryAdapter(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim * 9),
            nn.Linear(input_dim * 9, input_dim * 3), nn.GELU(),
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, output_dim), nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features, padding=None, force_null=False):
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError('Expected geometry features [B,T,N,input_dim]')
        batch, frames, patches, _ = features.shape
        if padding is not None:
            if tuple(padding.shape) != (batch, frames, patches):
                raise ValueError('Padding must be [B,T,N], True means invalid')
            features = features.masked_fill(padding[..., None], 0)
        values = space_to_depth_3x3(features.reshape(batch * frames, patches, -1))
        output = self.layers(values).reshape(batch, frames * values.shape[1], -1)
        if padding is not None:
            # The caller must also exclude padded slots from LM attention.
            valid = space_to_depth_3x3((~padding).to(features.dtype).reshape(batch * frames, patches, 1))
            invalid = ~valid.bool().any(-1).reshape(batch, -1)
            output = output.masked_fill(invalid[..., None], 0)
        if isinstance(force_null, torch.Tensor):
            output = torch.where(force_null.to(output.device).bool().reshape(batch, 1, 1), torch.zeros_like(output), output)
        elif force_null:
            output = output * 0
        return output
