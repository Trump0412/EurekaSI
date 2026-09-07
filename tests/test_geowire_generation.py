import sys
from pathlib import Path

import pytest
import torch
from torch import nn

source=Path(__file__).resolve().parents[1]/"legacy_sources/GeoWire"
if not source.exists():
    from spatial_intelligence.workspace import settings
    try:source=Path(settings()["external"])/"geowire"
    except FileNotFoundError:pytest.skip("Historical source absent: spatial init, then spatial source geowire",allow_module_level=True)
if not source.exists():pytest.skip("Fetch pinned source: spatial source geowire",allow_module_level=True)
sys.path.insert(0,str(source/'geowire'))
from geowire.geometry.graph_builder import self_loop_graph
from geowire.models.geowire import GeoWireTransport
from geowire.models.qwen3vl_bridge import Qwen3VLGeoWireForConditionalGeneration


class Inner(nn.Module):
    def get_image_features(self, pixel_values, image_grid_thw=None):
        return (pixel_values,), ()


class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.model=Inner(); self.config=object()
    def forward(self, pixel_values=None, **kwargs):
        return self.model.get_image_features(pixel_values)[0][0]
    def generate(self, **kwargs):
        return self(**kwargs)


def test_generate_no_recursion_and_zero_gate_parity():
    base=Base(); transport=GeoWireTransport(8,2)
    bridge=Qwen3VLGeoWireForConditionalGeneration(base,transport,None)
    x=torch.randn(6,8); original=base.model.get_image_features
    out=bridge.generate(graph=self_loop_graph(6),pixel_values=x)
    assert torch.equal(out,x)
    assert base.model.get_image_features == original


def test_generate_restores_after_failure():
    base=Base(); bridge=Qwen3VLGeoWireForConditionalGeneration(base,GeoWireTransport(8,2),None)
    original=base.model.get_image_features
    with pytest.raises(ValueError):
        bridge.generate(graph=self_loop_graph(8),pixel_values=torch.randn(6,8))
    assert base.model.get_image_features == original
