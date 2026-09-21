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


def test_reuse_symlink_requires_explicit_root_and_is_read_only(tmp_path):
    import tarfile, io, pytest
    spec=importlib.util.spec_from_file_location('six_reuse',Path(__file__).parents[1]/'scripts/prepare-six-source-waiver.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=tmp_path/'old';old.mkdir();original=old/'image.jpg';original.write_bytes(b'abc')
    dest=tmp_path/'new';dest.mkdir();link=dest/'image.jpg'
    try:link.symlink_to(original)
    except OSError:pytest.skip('Symlinks unavailable for current Windows permissions')
    archive=tmp_path/'input.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        item=tarfile.TarInfo('image.jpg');item.size=3;t.addfile(item,io.BytesIO(b'xyz'))
    with pytest.raises(ValueError,match='Unapproved reuse symlink'):
        module.extract_archive(archive,dest,{'image.jpg'})
    assert module.extract_archive(archive,dest,{'image.jpg'},[old])==0
    assert original.read_bytes()==b'abc'
    with tarfile.open(archive,'w:gz') as t:
        item=tarfile.TarInfo('../escape');item.size=3;t.addfile(item,io.BytesIO(b'bad'))
    with pytest.raises(ValueError,match='Unsafe archive member'):
        module.extract_archive(archive,dest,{'../escape'},[old])


def test_recovery_video_uses_video_reader_not_pil(tmp_path,monkeypatch):
    import sys, types
    spec=importlib.util.spec_from_file_location('recover_media_test',Path(__file__).parents[1]/'scripts/recover-geothinker-media.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Capture:
        def __init__(self):self.n=0;self.closed=False
        def isOpened(self):return True
        def get(self,key):return 2
        def read(self):self.n+=1;return (self.n<=2,object())
        def release(self):self.closed=True
    cap=Capture()
    monkeypatch.setitem(sys.modules,'cv2',types.SimpleNamespace(VideoCapture=lambda p:cap,CAP_PROP_FRAME_COUNT=7))
    assert module.validate_media(tmp_path/'video.mp4')==dict(type='video',decoded_frames=2)
    assert cap.closed


def test_only_two_known_invalid_media_rows_are_quarantined():
    import pytest
    spec=importlib.util.spec_from_file_location('recover_quarantine',Path(__file__).parents[1]/'scripts/recover-geothinker-media.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for sid in module.KNOWN_INVALID_IDS:
        row=dict(id='vlm3r::'+sid,source_id=sid,source_file='vlm3r_vsi_205k_32frames.json',
                 media=['/source/scene0290_00/video_color/scene0290_00_video.mp4'])
        assert module.quarantine_reason(row).startswith('invalid_media_type')
    row['source_id']='unexpected'
    with pytest.raises(ValueError,match='Unexpected video'):
        module.quarantine_reason(row)
    row['media']=['/missing/file.jpg']
    assert module.quarantine_reason(row) is None  # Missing images are not silently filtered.
