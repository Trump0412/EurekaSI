"""Release contracts adapted from the handoff; no network or model execution."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def release_module(name='package_release'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_is_deterministic_and_has_integrity_manifest(tmp_path):
    module = release_module()
    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'README.md').write_text('hello\n', encoding='utf-8')
    (source / 'scripts' / 'run.sh').write_text('#!/bin/sh\n', encoding='utf-8')
    # Input transfer materials must never be included, even without Git metadata.
    (source / 'trans').mkdir()
    (source / 'trans' / 'private.md').write_text('local handoff', encoding='utf-8')
    a, b = tmp_path / 'a.zip', tmp_path / 'b.zip'
    assert module.package(source, a) == module.package(source, b) == 2
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as archive:
        assert len(archive.namelist()) == 3
        for line in archive.read('spatial-intelligence/MANIFEST.sha256').decode().splitlines():
            sha, name = line.split('  ', 1)
            assert hashlib.sha256(archive.read('spatial-intelligence/' + name)).hexdigest() == sha
        assert archive.getinfo('spatial-intelligence/scripts/run.sh').external_attr >> 16 == 0o100755
        assert archive.getinfo('spatial-intelligence/README.md').date_time == (1980, 1, 1, 0, 0, 0)
    with pytest.raises(FileExistsError):
        module.package(source, a)


def test_metadata_reader_does_not_execute_source(tmp_path):
    module = release_module('release_check')
    path = tmp_path / 'init.py'
    path.write_text('raise RuntimeError("must not execute")\n__version__ = "0.2.1"\n', encoding='utf-8')
    assert module.literal_assignment(path, '__version__') == '0.2.1'


def test_public_release_metadata_and_resources():
    assert release_module('release_check').check(ROOT)['errors'] == []
