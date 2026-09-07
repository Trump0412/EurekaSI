"""Resource resolution contracts without building/installing a package or running models."""
import pytest

from spatial_intelligence import workspace
from spatial_intelligence.io import load_config


def target_install(tmp_path, monkeypatch):
    target = tmp_path / 'installed'
    resources = target / 'share' / 'spatial-intelligence'
    (resources / 'catalog').mkdir(parents=True)
    (resources / 'configs').mkdir()
    (resources / 'sources.lock.json').write_text('{}', encoding='utf-8')
    (resources / 'catalog' / 'assets.yaml').write_text('sample: {}\n', encoding='utf-8')
    monkeypatch.setattr(workspace, 'REPO', target)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config-home'))
    monkeypatch.delenv('SPATIAL_CONFIG', raising=False)
    return target, resources


def test_target_wheel_resources_and_user_config(tmp_path, monkeypatch):
    target, resources = target_install(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert workspace.resource_root() == resources
    assert workspace.catalog('assets') == {'sample': {}}
    workspace.initialize(tmp_path / 'data')
    assert workspace.settings()['root'] == str(tmp_path / 'data')
    assert not (target / 'spatial.local.yaml').exists()


def test_explicit_config_path_survives_wheel_install(tmp_path, monkeypatch):
    target_install(tmp_path, monkeypatch)
    config = tmp_path / 'custom.yaml'
    monkeypatch.setenv('SPATIAL_CONFIG', str(config))
    workspace.initialize(tmp_path / 'data')
    assert workspace.settings()['config_path'] == str(config.resolve())


def test_config_fallback_preserves_local_override(tmp_path, monkeypatch):
    _, resources = target_install(tmp_path, monkeypatch)
    (resources / 'configs' / 'inference.yaml').write_text('origin: packaged\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert load_config('configs/inference.yaml')['origin'] == 'packaged'
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs' / 'inference.yaml').write_text('origin: local\n', encoding='utf-8')
    assert load_config('configs/inference.yaml')['origin'] == 'local'
    with pytest.raises(ValueError):
        workspace.resource_path('../outside')
