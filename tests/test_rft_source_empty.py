import importlib.util
import json
from pathlib import Path
import sys

import pytest

from spatial_intelligence.rft_recovery import (
    audit_source_inventory, display_size_matches, exclude_source_empty_after_split, reusable_cache,
)


def test_exact_byte_sizes_and_rounded_binary_units():
    assert display_size_matches(1, '1B')
    assert not display_size_matches(0, '1B')
    assert not display_size_matches(2, '1B')
    assert display_size_matches(16388830, '15.63MB')
    assert not display_size_matches(16000000, '15.63MB')


def test_inventory_requires_all_nonempty_files_and_both_zero_sources(tmp_path):
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'count': 2, 'files': {'ok.mp4': '3B', 'empty.mp4': '0B'}}))
    (tmp_path/'ok.mp4').write_bytes(b'abc')
    (tmp_path/'empty.mp4').touch()
    assert audit_source_inventory(inventory, tmp_path)['upstream_empty_files'] == ['empty.mp4']
    (tmp_path/'ok.mp4').write_bytes(b'ab')
    with pytest.raises(ValueError, match='size mismatch'):
        audit_source_inventory(inventory, tmp_path)
    (tmp_path/'ok.mp4').write_bytes(b'abc')
    (tmp_path/'empty.mp4').write_bytes(b'replacement')
    with pytest.raises(ValueError, match='changed locally'):
        audit_source_inventory(inventory, tmp_path)


def test_filter_keeps_existing_split_order_and_precise_exclusion_identity():
    rows = [dict(id=f'4drl:{i}', source='4drl', source_group=f'group{i}', scene_id=f'scene{i}',
                 split='validation' if i % 2 else 'train', video_path=f'/media/{i}.mp4') for i in range(5)]
    keep, excluded = exclude_source_empty_after_split(rows, ['1.mp4', '2.mp4'])
    assert keep == [rows[0], rows[3], rows[4]]
    assert [row['id'] for row in excluded] == ['4drl:1', '4drl:2']
    assert [row['original_split'] for row in excluded] == ['validation', 'train']
    assert all(row['reason'] == 'upstream_empty_source_media' for row in excluded)
    assert all(row['remote_bytes'] == row['local_bytes'] == 0 for row in excluded)


def test_rounded_zero_is_not_upstream_empty_evidence(tmp_path):
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'count': 1, 'files': {'empty.mp4': '0.00KB'}}))
    (tmp_path/'empty.mp4').touch()
    with pytest.raises(ValueError, match='size mismatch'):
        audit_source_inventory(inventory, tmp_path)


def test_cache_reuse_identity_requires_size_mtime_and_sampling_count(tmp_path):
    source = tmp_path/'video.mp4'
    source.write_bytes(b'abc')
    cache = tmp_path/'cache'; cache.mkdir()
    stat = source.stat()
    (cache/'receipt.json').write_text(json.dumps({'identity': {
        'size': 3, 'mtime_ns': stat.st_mtime_ns, 'count': 32}}))
    assert reusable_cache(source, cache, 32)
    assert not reusable_cache(source, cache, 8)
    source.write_bytes(b'abcd')
    assert not reusable_cache(source, cache, 32)


def test_preparer_exclusion_flag_is_opt_in(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/prepare-geometry-rft-data.py'
    spec = importlib.util.spec_from_file_location('rft_prepare_optin_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    captured = []
    monkeypatch.setattr(module, 'prepare', captured.append)
    command = [str(path)]
    for option in ('four-d-annotations', 'four-d-media', 'dsr-annotations', 'dsr-media',
                   'spatial-annotations', 'spatial-media', 'output', 'benchmark-manifest'):
        command.extend(['--'+option, '/example/'+option])
    monkeypatch.setattr(sys, 'argv', command)
    module.main()
    assert captured[-1].exclude_upstream_empty is False
    monkeypatch.setattr(sys, 'argv', command+['--exclude-upstream-empty'])
    module.main()
    assert captured[-1].exclude_upstream_empty is True
