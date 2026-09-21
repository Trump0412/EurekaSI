import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace())
    path = Path(__file__).resolve().parents[1]/'scripts/recover-rft-download.py'
    spec = importlib.util.spec_from_file_location('rft_recovery_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_display_size_is_rounded_binary_units_but_zero_never_passes(recovery):
    assert recovery.size_matches(16388830, '15.63MB')
    assert not recovery.size_matches(16300000, '15.63MB')
    assert not recovery.size_matches(0, '0B')
    assert not recovery.size_matches(0, '15.63MB')
    with pytest.raises(ValueError):
        recovery.size_matches(42, 'unknown')


def test_source_empty_is_distinct_from_transient_download_failure(recovery):
    assert recovery.upstream_empty_files({'a.mp4': '0B', 'b.mp4': '4.53MB',
                                          'c.mp4': '0.00KB'}) == ['a.mp4', 'c.mp4']
