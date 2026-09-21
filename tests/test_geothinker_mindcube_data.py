import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('audit_geo_data', Path(__file__).parents[1] / 'scripts/audit-geothinker-mindcube-data.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_question_normalization_preserves_frame_reference():
    row = {'conversations': [{'value': '<image>\n<image>\n Where is frame 14 of 32?'}]}
    assert m.norm_question(row) == 'Where is frame 14 of 32?'


def test_mindcube_overlap_checks_media_group_not_only_id():
    a = [{'id': 'a', 'images': ['among/scene1/one.jpg']}]
    b = [{'id': 'b', 'images': ['among/scene1/two.jpg']}]
    assert m.mind_overlap(a, b) == dict(ids=0, media_rows=0, group_rows=1)
    assert not any(m.mind_overlap(a, [{'id': 'c', 'images': ['among/scene2/one.jpg']}]).values())


def test_final_gate_never_accepts_partial_component(tmp_path):
    spec = importlib.util.spec_from_file_location('watch_six', Path(__file__).parents[1] / 'scripts/watch-six-source-readiness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.aggregate({'component_ready': True}, {}, {})['ready_for_training']
    merged = dict.fromkeys(['ready_for_training', 'media_verified', 'leakage_checked',
                            'global_scene_split_verified', 'six_sources_verified'], True)
    assert not module.aggregate({}, {}, merged)['ready_for_training']
    for key in ['train_manifest', 'validation_manifest']:
        path = tmp_path / key
        path.write_text('{}\n')
        merged[key] = str(path)
    assert module.aggregate({}, {}, merged)['ready_for_training']
    merged['global_scene_split_verified'] = False
    assert not module.aggregate({}, {}, merged)['ready_for_training']


def test_waiver_is_bounded_and_not_reported_clean(tmp_path):
    spec = importlib.util.spec_from_file_location('watch_six_waiver', Path(__file__).parents[1] / 'scripts/watch-six-source-readiness.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p = tmp_path/'train.jsonl';p.write_text('{}\n')
    merged = dict(ready_for_training=True, media_verified=True, six_sources_verified=True,
                  known_sources_leakage_checked=True, leakage_checked=False,
                  train_manifest=str(p), validation_manifest=str(p),
                  contamination_waiver=dict(authorized=True, scope='openspatial_only', scene_overlap='unknown',
                      validation_policy='all OpenSpatial train-only', reason='explicit user permission'))
    value = module.aggregate({}, {}, merged)
    assert value['ready_for_training'] and not value['leakage_checked']
    merged['contamination_waiver']['scope'] = 'all_sources'
    assert not module.aggregate({}, {}, merged)['ready_for_training']


def test_tar_extract_is_selective_and_preserves_existing(tmp_path):
    import tarfile
    import io
    import pytest
    spec = importlib.util.spec_from_file_location('prepare_six_waiver', Path(__file__).parents[1] / 'scripts/prepare-six-source-waiver.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    archive=tmp_path/'source.tar.gz';dest=tmp_path/'dest';dest.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for name in ['data/good.jpg', 'data/other.jpg']:
            item=tarfile.TarInfo(name);item.size=3;t.addfile(item,io.BytesIO(b'abc'))
    assert module.extract_archive(archive,dest,{'data/good.jpg'})==1
    assert not (dest/'data/other.jpg').exists()
    assert module.extract_archive(archive,dest,{'data/good.jpg'})==0
    (dest/'data/good.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='size mismatch'):
        module.extract_archive(archive,dest,{'data/good.jpg'})
