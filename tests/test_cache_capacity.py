import concurrent.futures
from collections import namedtuple
import sqlite3

import pytest
from spatial_intelligence.cache_capacity import CacheCapacityError, publish_cache


def process_write(arguments):
    root,index=arguments
    try:
        return publish_cache(root,f'process-{index}.pt',b'1234',max_bytes=8,min_free_bytes=0)
    except CacheCapacityError:
        return False


def test_cap_preserves_prior_files_and_reuses_same_key(tmp_path):
    assert publish_cache(tmp_path,'one.pt',b'abcd',max_bytes=5,min_free_bytes=0)
    assert not publish_cache(tmp_path,'one.pt',b'other',max_bytes=0,min_free_bytes=0)
    with pytest.raises(CacheCapacityError,match='capacity blocked'):
        publish_cache(tmp_path,'two.pt',b'xy',max_bytes=5,min_free_bytes=0)
    assert (tmp_path/'one.pt').read_bytes()==b'abcd'
    assert not (tmp_path/'two.pt').exists()


def test_existing_cache_counted_once(tmp_path,monkeypatch):
    (tmp_path/'old.pt').write_bytes(b'12345')
    assert publish_cache(tmp_path,'new.pt',b'x',max_bytes=6,min_free_bytes=0)
    from pathlib import Path
    monkeypatch.setattr(Path,'glob',lambda *args: (_ for _ in ()).throw(AssertionError('rescanned')))
    with pytest.raises(CacheCapacityError):
        publish_cache(tmp_path,'extra.pt',b'x',max_bytes=6,min_free_bytes=0)


def test_minimum_free_space_blocks_even_with_capacity(tmp_path,monkeypatch):
    import spatial_intelligence.cache_capacity as module
    Usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root: Usage(100,90,10))
    with pytest.raises(CacheCapacityError):
        publish_cache(tmp_path,'one.pt',b'abcd',max_bytes=100,min_free_bytes=7)
    assert not (tmp_path/'one.pt').exists()


def test_concurrent_writers_cannot_exceed_budget(tmp_path):
    def write(index):
        try:
            return publish_cache(tmp_path,f'{index}.pt',b'1234',max_bytes=8,min_free_bytes=0)
        except CacheCapacityError:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(write,range(4)))==2
    assert sum(path.stat().st_size for path in tmp_path.glob('*.pt'))==8
    with sqlite3.connect(tmp_path/'.capacity.sqlite3') as db:
        assert db.execute('SELECT bytes FROM totals').fetchone()[0]==8


def test_independent_process_writers_share_lock_and_budget(tmp_path):
    import multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        assert sum(pool.map(process_write,[(str(tmp_path),i) for i in range(5)]))==2
    assert sum(path.stat().st_size for path in tmp_path.glob('*.pt'))==8


def test_crash_reservation_is_conservative_and_reusable(tmp_path,monkeypatch):
    import spatial_intelligence.cache_capacity as module
    original=module.os.replace
    monkeypatch.setattr(module.os,'replace',lambda *args: (_ for _ in ()).throw(OSError('simulated crash')))
    with pytest.raises(OSError):
        publish_cache(tmp_path,'one.pt',b'abcd',max_bytes=4,min_free_bytes=0)
    with pytest.raises(CacheCapacityError):
        publish_cache(tmp_path,'two.pt',b'x',max_bytes=4,min_free_bytes=0)
    monkeypatch.setattr(module.os,'replace',original)
    with pytest.raises(CacheCapacityError,match='operator recovery'):
        publish_cache(tmp_path,'one.pt',b'abcd',max_bytes=4,min_free_bytes=0)
    # Simulate explicit operator recovery, preserving the one complete payload.
    temporary=next(tmp_path.glob('.one.pt.*.tmp'))
    original(temporary,tmp_path/'one.pt')
    assert not publish_cache(tmp_path,'one.pt',b'abcd',max_bytes=4,min_free_bytes=0)
    assert (tmp_path/'one.pt').read_bytes()==b'abcd'


def test_filename_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        publish_cache(tmp_path,'../bad.pt',b'x',min_free_bytes=0)
