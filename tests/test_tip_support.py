import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('support',Path(__file__).resolve().parents[1]/'scripts/prepare-tip-support.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)

def test_audit_shards_are_disjoint_and_cover_every_row():
    for world in (1,4,8):
        shards=[{i for i in range(131) if support.owns_row(i,rank,world)} for rank in range(world)]
        assert set.union(*shards)==set(range(131))
        assert sum(map(len,shards))==131
        for rank,shard in enumerate(shards):
            assert shard==set(range(rank,131,world))
