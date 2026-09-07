"""Package public source roots without walking local data or experiment directories."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile


ROOT_FILES = {
    '.editorconfig', '.gitattributes', '.gitignore', 'AGENTS.md',
    'CONTRIBUTING.md', 'LICENSE', 'NOTICE.md', 'README.md', 'README.en.md',
    'pyproject.toml', 'sources.lock.json', 'CHANGELOG.md', 'CITATION.cff', 'SECURITY.md',
}
SOURCE_DIRS = {'.github', 'catalog', 'configs', 'docs', 'examples', 'patches',
               'projects', 'requirements', 'scripts', 'spatial_intelligence', 'tests'}
SOURCE_SUFFIXES = {'.py', '.sh', '.md', '.yaml', '.yml', '.json', '.jsonl',
                   '.toml', '.txt', '.xml', '.patch', '.cff'}


def release_files(root, include_legacy=False):
    root = Path(root).resolve()
    for name in sorted(ROOT_FILES):
        path = root / name
        if path.is_file() and not path.is_symlink():
            yield path
    roots = set(SOURCE_DIRS)
    if include_legacy:
        if not (root / 'legacy_sources').is_dir():
            raise FileNotFoundError('--include-legacy requires a reviewed legacy_sources directory')
        roots.add('legacy_sources')
    for name in sorted(roots):
        base = root / name
        if not base.is_dir() or base.is_symlink():
            continue
        pending = [base]
        while pending:
            directory = pending.pop()
            for path in sorted(directory.iterdir()):
                if path.is_symlink():
                    continue
                if path.is_dir():
                    if (path.name in {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'build', 'dist'}
                            or path.name.startswith('.venv') or path.name.endswith('.egg-info')):
                        continue
                    pending.append(path)
                elif (path.name != 'spatial.local.yaml' and not path.name.startswith('.env')
                      and (path.suffix in SOURCE_SUFFIXES or path.name in {'LICENSE', 'NOTICE'})):
                    yield path


def zip_timestamp(epoch):
    value = datetime.fromtimestamp(epoch, tz=timezone.utc)
    if not 1980 <= value.year <= 2107:
        raise ValueError('SOURCE_DATE_EPOCH must be within the ZIP timestamp range (1980-2107)')
    return (value.year, value.month, value.day, value.hour, value.minute, value.second // 2 * 2)


def package(root, output, include_legacy=False, *, source_date_epoch=315532800):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Release output already exists; use a new filename')
    timestamp = zip_timestamp(source_date_epoch)
    files = sorted(release_files(root, include_legacy))
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip.tmp', delete=False) as f:
        temporary = Path(f.name)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                data = path.read_bytes()
                checks.append(hashlib.sha256(data).hexdigest() + '  ' + relative)
                _entry(archive, relative, data, timestamp, executable=path.suffix == '.sh')
            _entry(archive, 'MANIFEST.sha256', ('\n'.join(checks) + '\n').encode('utf-8'), timestamp)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(files)


def _entry(archive, relative, data, timestamp, executable=False):
    entry = zipfile.ZipInfo('spatial-intelligence/' + relative, date_time=timestamp)
    entry.create_system = 3
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = (0o100755 if executable else 0o100644) << 16
    archive.writestr(entry, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-legacy', action='store_true')
    parser.add_argument('--source-date-epoch', type=int, default=int(os.environ.get('SOURCE_DATE_EPOCH', '315532800')))
    args = parser.parse_args()
    count = package(Path(__file__).resolve().parents[1], args.output, args.include_legacy, source_date_epoch=args.source_date_epoch)
    print(f'{args.output}: {count} source files')


if __name__ == '__main__':
    main()
