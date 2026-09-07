"""Inspect the exact source release selection without executing application code."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'README.md', 'README.en.md', 'LICENSE', 'NOTICE.md', 'CONTRIBUTING.md',
            'CHANGELOG.md', 'CITATION.cff', 'SECURITY.md', 'AGENTS.md', 'pyproject.toml',
            'sources.lock.json', '.github/pull_request_template.md', 'docs/MAINTENANCE.md'}


def literal_assignment(path, name):
    """Read version constants via AST, never exec an __init__.py."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'Missing literal {name} in {path.name}')


def selected_files(root):
    spec = importlib.util.spec_from_file_location('spatial_release_selection', ROOT / 'scripts/package_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted(module.release_files(root))


def check(root=ROOT):
    root = Path(root).resolve()
    files = selected_files(root)
    relative = {p.relative_to(root).as_posix() for p in files}
    errors = [f'Missing public release file: {name}' for name in sorted(REQUIRED - relative)]
    if errors:
        return {'status': 'failed', 'errors': errors, 'public_files': len(files)}
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    version = project['project']['version']
    if project['project']['name'] != 'eurekasi':
        errors.append('Public distribution must be named eurekasi')
    scripts = project['project']['scripts']
    if scripts.get('eurekasi') != 'spatial_intelligence.cli:main' or scripts.get('spatial') != scripts.get('eurekasi'):
        errors.append('EurekaSI and compatible spatial CLI entry points disagree')
    repository = 'https://github.com/Trump0412/EurekaSI'
    if project['project'].get('urls', {}).get('Repository') != repository:
        errors.append('Project repository URL is inconsistent')
    if literal_assignment(root / 'spatial_intelligence/__init__.py', '__version__') != version:
        errors.append('Package and project versions disagree')
    citation = yaml.safe_load((root / 'CITATION.cff').read_text(encoding='utf-8'))
    if citation.get('title') != 'EurekaSI' or citation.get('repository-code') != repository:
        errors.append('Citation project identity is inconsistent')
    if str(citation.get('version')) != version:
        errors.append('Citation and project versions disagree')
    if f'## {version}' not in (root / 'CHANGELOG.md').read_text(encoding='utf-8'):
        errors.append('Changelog is missing the project version')
    for name in ('README.md', 'README.en.md'):
        if f'v{version}' not in (root / name).read_text(encoding='utf-8'):
            errors.append(f'{name} does not identify the current version')
    core = {line.strip() for line in (root / 'requirements/core.txt').read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')}
    if set(project['project']['dependencies']) != core:
        errors.append('Core requirements and project dependencies disagree')
    native = (root / 'requirements/native.txt').read_text(encoding='utf-8').splitlines()
    extras = project['project']['optional-dependencies']
    pins = {s for s in [*native, *extras['train'], *extras['test']] if s.startswith('transformers==')}
    if len(pins) != 1:
        errors.append('Transformers pins disagree')
    resources = project['tool']['setuptools'].get('data-files', {})
    declared = set()
    for destination, patterns in resources.items():
        if not destination.startswith('share/spatial-intelligence'):
            errors.append(f'Unexpected wheel data destination: {destination}')
        for pattern in patterns:
            matches = list(root.glob(pattern))
            if not matches:
                errors.append(f'Wheel data pattern has no files: {pattern}')
            for path in matches:
                name = path.relative_to(root).as_posix()
                declared.add(name)
                if name not in relative:
                    errors.append(f'Wheel resource excluded from source archive: {name}')
    expected_resources = {name for name in relative if name.startswith(('catalog/', 'configs/', 'patches/'))}
    expected_resources.add('sources.lock.json')
    if expected_resources - declared:
        errors.append('Wheel declarations omit runtime resources: ' + ', '.join(sorted(expected_resources - declared)))
    sources = yaml.safe_load((root / 'catalog/sources.yaml').read_text(encoding='utf-8'))
    lock = json.loads((root / 'sources.lock.json').read_text(encoding='utf-8'))
    for name, entry in sources.items():
        if not re.fullmatch(r'[0-9a-f]{40}', str(entry.get('commit', ''))):
            errors.append(f'Unpinned source: {name}')
        for patch in entry.get('patches', []):
            if patch not in relative:
                errors.append(f'Missing source patch: {patch}')
    for name in ('GeoWire', 'GeoBridge', 'GeoPSRO'):
        a, b = lock.get(name, {}), sources.get(name.lower(), {})
        if (not a or not b or a.get('url') != b.get('url') or a.get('commit') != b.get('commit')
                or a.get('patches_applied', []) != b.get('patches', [])):
            errors.append(f'Lock/catalog mismatch: {name}')
    vendor = lock['VSI-official']
    if hashlib.sha256((root / vendor['local_file']).read_bytes()).hexdigest() != vendor['sha256']:
        errors.append('Vendored VSI source differs from its recorded hash')
    patterns = [r'\bhf_[A-Za-z0-9]{20,}\b', r'\bgh[pousr]_[A-Za-z0-9]{36,}\b',
                r'\bsk-[A-Za-z0-9_-]{20,}\b', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    for path in files:
        text = path.read_text(encoding='utf-8')
        name = path.relative_to(root).as_posix()
        if any(re.search(pattern, text) for pattern in patterns):
            errors.append(f'Potential credential in {name}')  # never print the matching value
        if path.suffix in {'.yaml', '.yml', '.cff'}:
            yaml.compose(text)  # inert !function declarations stay inert
        if path.suffix == '.md':
            for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                target = raw.strip().strip('<>').split(maxsplit=1)[0]
                if urlsplit(target).scheme or target.startswith('#'):
                    continue
                target = unquote(target.split('#', 1)[0])
                if target:
                    resolved = (path.parent / target).resolve()
                    if not resolved.is_relative_to(root):
                        errors.append(f'Link escapes release: {name}')
                    elif resolved.is_file() and resolved.relative_to(root).as_posix() not in relative:
                        errors.append(f'Link targets an excluded release file: {name} -> {target}')
                    elif not resolved.exists():
                        errors.append(f'Broken local link: {name} -> {target}')
    return {'status': 'failed' if errors else 'release_static_checks_passed', 'errors': errors,
            'public_files': len(files), 'runtime_resources': len(declared), 'version': version,
            'scope': 'local source selection and metadata only; no build, install, app imports or tests'}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['errors']))
