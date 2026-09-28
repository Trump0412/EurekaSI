"""Opt-in frozen-teacher execution caches; no query/frame/iteration changes."""
from contextlib import contextmanager
from functools import lru_cache
import importlib
import threading

_LOCK = threading.RLock()


@contextmanager
def cached_tracking_positions(teacher, device):
    """Memoize deterministic CPU-built position grids on the teacher device.

    Scoped to one build and restored even on error. Only pinned VGGT's tracker
    binding is changed, not its utility module or checkpoint. The original
    function still computes the values on CPU before the identical device copy.
    Callers must use an eval/frozen teacher and one device per process.
    """
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Tracking execution cache requires a frozen eval teacher')
    module = importlib.import_module(type(teacher.track_head.tracker).__module__)
    with _LOCK:
        original = module.get_2d_sincos_pos_embed

        @lru_cache(maxsize=2)
        def cached(embed_dim, grid_size, return_grid=False):
            value = original(embed_dim, grid_size, return_grid=return_grid)
            if return_grid:
                return tuple(v.to(device) for v in value)
            return value.to(device)

        module.get_2d_sincos_pos_embed = cached
        try:
            yield
        finally:
            module.get_2d_sincos_pos_embed = original
            cached.cache_clear()
