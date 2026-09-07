"""Read and parse repository files without importing the application or running tests."""
import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ('spatial_intelligence', 'scripts', 'tests', 'configs', 'catalog',
                'docs', 'examples', 'projects', '.github', 'requirements')


def main():
    files = [p for name in SOURCE_ROOTS for p in (ROOT / name).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += list(ROOT.glob('*.md'))
    counts = {'python': 0, 'json': 0, 'jsonl': 0, 'yaml': 0, 'toml': 0, 'local_links': 0}
    errors = []
    try:
        import yaml
    except ImportError:
        raise SystemExit('Static YAML parsing requires PyYAML; no packages were installed.')
    for path in sorted(files):
        try:
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                counts['python'] += 1
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
                counts['json'] += 1
            elif path.suffix == '.jsonl':
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        json.loads(line)
                counts['jsonl'] += 1
            elif path.suffix in {'.yaml', '.yml'}:
                # Parse nodes only; vendored lmms task YAML uses an inert !function tag.
                yaml.compose(path.read_text(encoding='utf-8'))
                counts['yaml'] += 1
            elif path.suffix == '.md':
                for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                    target = link.split('#')[0]
                    if target and not re.match(r'[a-zA-Z]+:', target):
                        if not (path.parent / target).exists():
                            raise ValueError(f'Broken local link: {target}')
                        counts['local_links'] += 1
        except (SyntaxError, ValueError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        counts['toml'] = 1
    except (ValueError, ImportError) as exc:
        errors.append(f'pyproject.toml: {exc}')
    print(json.dumps({'status': 'failed' if errors else 'static_checks_passed',
                      'counts': counts, 'errors': errors,
                      'scope': 'parsing/local links only; no application imports, tests, downloads, or model execution'},
                     ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
