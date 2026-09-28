import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from spatial_intelligence.vggt_tracking_execution import cached_tracking_positions


def test_exact_cache_and_restoration_on_error(monkeypatch):
    module=ModuleType('fake_frozen_tracker')
    calls=[]
    def original(embed_dim,grid_size,return_grid=False):
        calls.append((embed_dim,grid_size))
        return torch.arange(embed_dim,dtype=torch.float32)
    module.get_2d_sincos_pos_embed=original
    monkeypatch.setitem(sys.modules,module.__name__,module)
    cls=type('Tracker',(),{'__module__':module.__name__})
    teacher=SimpleNamespace(training=False,parameters=lambda:[],track_head=SimpleNamespace(tracker=cls()))
    with pytest.raises(RuntimeError,match='probe'):
        with cached_tracking_positions(teacher,'cpu'):
            a=module.get_2d_sincos_pos_embed(4,(2,2))
            b=module.get_2d_sincos_pos_embed(4,(2,2))
            assert a is b
            assert torch.equal(a,torch.arange(4,dtype=torch.float32))
            assert len(calls)==1
            raise RuntimeError('probe')
    assert module.get_2d_sincos_pos_embed is original


def test_trainable_teacher_rejected():
    teacher=SimpleNamespace(training=True)
    with pytest.raises(ValueError,match='frozen eval'):
        with cached_tracking_positions(teacher,'cpu'):
            pass
