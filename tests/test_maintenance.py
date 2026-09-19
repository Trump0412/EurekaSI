"""Regression cases from the static open-source readiness review.

These cases use temporary files only. They do not download or load model weights.
"""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from spatial_intelligence.assets import download
from spatial_intelligence.config import validate
from spatial_intelligence.data import normalize
from spatial_intelligence.io import digest, file_digest, load_config, write_json, write_jsonl
from spatial_intelligence.platform import main, mixture_spec
from spatial_intelligence.quality import clean
from spatial_intelligence.workspace import initialize, settings


def test_json_plan_preserves_scientific_notation(tmp_path):
    cfg=load_config('configs/smoke.yaml')
    path=tmp_path/'plan.json';write_json(path,cfg)
    restored=load_config(path)
    assert restored==cfg
    assert type(restored['train']['learning_rate']) is float
    validate(restored)


def test_marked_evidence_is_not_a_conflicting_label(tmp_path):
    original = tmp_path / 'original.png'
    Image.new('RGB', (4, 4), 'white').save(original)
    rows = []
    for i, color in enumerate(('red', 'blue')):
        marked = tmp_path / f'{color}.png'
        Image.new('RGB', (4, 4), color).save(marked)
        rows.append(normalize({'dataset': 'spar', 'id': i, 'split': 'train',
                               'question': 'Which marked object is closer?', 'answer': str(i),
                               'media': [str(marked)], 'geometry_media': [str(original)]}))
    manifest = tmp_path / 'train.jsonl'
    write_jsonl(manifest, rows)
    result = clean(manifest, tmp_path / 'clean')
    assert result['kept'] == 2 and result['excluded'] == 0


@pytest.mark.parametrize('spec, expected', [
    ('spar.train:0.8', ('spar.train', 0.8)),
    ('a:0.5', ('a', 0.5)),
    (r'C:\data\train.jsonl', (r'C:\data\train.jsonl', 1.0)),
    (r'C:\data\train.jsonl:0.2', (r'C:\data\train.jsonl', 0.2)),
])
def test_mixture_paths(spec, expected):
    assert mixture_spec(spec) == expected


def test_init_honors_custom_config(tmp_path, monkeypatch):
    config = tmp_path / 'nested' / 'local.yaml'
    monkeypatch.setenv('SPATIAL_CONFIG', str(config))
    initialize(tmp_path / 'workspace')
    assert settings()['config_path'] == str(config.resolve())


def workspace(tmp_path, monkeypatch):
    config = tmp_path / 'local.yaml'
    monkeypatch.setenv('SPATIAL_CONFIG', str(config))
    local = initialize(tmp_path / 'workspace')
    model = tmp_path / 'model'
    model.mkdir()
    manifest = Path(local['manifests']) / 'example.test.jsonl'
    write_jsonl(manifest, [normalize({'id': 'x', 'dataset': 'd', 'question': 'q', 'answer': 'a', 'split': 'test'})])
    return local, model


def test_final_plan_keeps_model_overrides(tmp_path, monkeypatch):
    local, model = workspace(tmp_path, monkeypatch)
    main(['run', 'inference', '--model', str(model), '--name', 'override',
          '--eval', 'example.test', '--set', 'model.adapter=/my/trained/adapter', '--dry-run'])
    plan = load_config(Path(local['runs']) / 'plans' / 'override.json')
    assert plan['model']['adapter'] == '/my/trained/adapter'


@pytest.mark.parametrize('flags', [
    ['--gpus', '0'], ['--gpus', '2', '--devices', '0,0'],
    ['--set', 'train.mode=rl'], ['--fusion-checkpoint', '/unused.pt'],
    ['--name', '../escape'],
])
def test_invalid_run_does_not_reserve_plan(tmp_path, monkeypatch, flags):
    local, model = workspace(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        main(['run', 'inference', '--model', str(model), '--name', 'invalid',
              '--eval', 'example.test', '--dry-run', *flags])
    assert not (Path(local['runs']) / 'plans' / 'invalid.json').exists()


@pytest.mark.parametrize('key, value', [
    ('steps', 0), ('batch_size', True), ('learning_rate', float('nan')),
    ('weight_decay', float('inf')), ('max_grad_norm', -1),
])
def test_bad_training_numbers_rejected_before_loading(key, value):
    cfg = load_config('configs/sft.yaml')
    cfg['train'][key] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_download_dry_run_reports_receipt_revision(tmp_path, monkeypatch):
    local, _ = workspace(tmp_path, monkeypatch)
    from spatial_intelligence.workspace import catalog
    item = catalog('assets')['qwen3-vl-2b']
    frozen = 'a' * 40
    write_json(Path(local['receipts']) / 'qwen3-vl-2b.json',
               {'repo_id': item['repo_id'], 'repo_type': 'model', 'resolved_revision': frozen})
    assert download('qwen3-vl-2b', dry_run=True)['resolved_revision'] == frozen


def test_release_excludes_workspace_assets_and_preserves_geometry_source(tmp_path):
    spec = importlib.util.spec_from_file_location('package_release', 'scripts/package_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('README.md', 'spatial_intelligence/geometry/cache.py',
                 'datasets/private.json', 'models/config.json', 'frames/private.png',
                 '_archives/original.zip', '.env.production', 'spatial.local.yaml'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    selected = {p.relative_to(tmp_path).as_posix() for p in module.release_files(tmp_path)}
    assert selected == {'README.md', 'spatial_intelligence/geometry/cache.py'}


def test_changed_geometry_is_not_registered_again(tmp_path, monkeypatch):
    from spatial_intelligence.geometry import cache
    weights = tmp_path / 'weights'
    weights.mkdir()
    write_json(weights / 'config.json', {'fixture': True})
    cfg = {'name': 'vggt', 'weights': str(weights), 'source_commit': 'c' * 40, 'grid': 2}
    descriptor = {k: v for k, v in cfg.items() if k != 'weights'}
    descriptor['adapter_sha256'] = file_digest(cache.__file__)
    descriptor['weights_sha256'] = {'config.json': file_digest(weights / 'config.json')}
    image = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image)
    row = normalize({'id': 'x', 'dataset': 'd', 'question': 'q', 'split': 'test', 'media': [str(image)]})
    manifest = tmp_path / 'test.jsonl'
    write_jsonl(manifest, [row])
    out = tmp_path / 'geometry'
    out.mkdir()
    key = cache.cache_key(row, descriptor)
    blob = out / (key + '.npz')
    blob.write_bytes(b'original')
    write_json(out / 'index.json', {
        'schema': 'spatial-points-v1', 'extractor': descriptor,
        'entries': {digest(cache.media_hashes(row)): {
            'path': str(blob), 'sha256': file_digest(blob), 'cache_key': key}},
    })
    blob.write_bytes(b'changed')
    monkeypatch.setattr(cache, 'load_extractor', lambda _: pytest.fail('Must reject before loading a model'))
    with pytest.raises(ValueError, match='cache changed'):
        cache.extract(manifest, out, cfg)


def test_revsi_cannot_accept_fewer_views_than_declared(tmp_path):
    from spatial_intelligence.build_data import build_annotations
    annotation = tmp_path / 'revsi.json'
    write_json(annotation, [{'id': 'x', 'question': 'How many?', 'answer': 1,
                             'num_frames': 4, 'images': ['one.png', 'two.png']}])
    with pytest.raises(ValueError, match='prescribed num_frames'):
        build_annotations(annotation, tmp_path / 'test.jsonl', 'revsi', 'test',
                          tmp_path, tmp_path / 'frames', max_frames=4)
