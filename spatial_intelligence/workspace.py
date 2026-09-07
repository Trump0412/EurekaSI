"""One local configuration for machine paths; never commit machine-specific paths."""
import os
from pathlib import Path
import sysconfig
import yaml


REPO = Path(__file__).resolve().parents[1]


def resource_root():
    """Locate checkout assets or wheel data installed in this Python environment."""
    if (REPO / 'pyproject.toml').is_file() and (REPO / 'catalog').is_dir():
        return REPO
    # pip --target installs data under the target; ordinary/user installs use
    # the interpreter's data scheme. Never search the invocation directory.
    candidates = [REPO / 'share' / 'spatial-intelligence',
                  Path(sysconfig.get_path('data')) / 'share' / 'spatial-intelligence']
    user_scheme = 'nt_user' if os.name == 'nt' else 'posix_user'
    if user_scheme in sysconfig.get_scheme_names():
        user_data = Path(sysconfig.get_path('data', scheme=user_scheme)) / 'share' / 'spatial-intelligence'
        if REPO == Path(sysconfig.get_path('purelib', scheme=user_scheme)):
            candidates.insert(0, user_data)
        else:
            candidates.append(user_data)
    for candidate in candidates:
        if all((candidate / name).exists() for name in ('catalog', 'configs', 'sources.lock.json')):
            return candidate
    raise FileNotFoundError('Runtime assets missing; install a complete wheel or use a source checkout')


def resource_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Runtime resource must be a relative path without parent traversal')
    path = resource_root() / relative
    if not path.exists():
        raise FileNotFoundError(f'Runtime resource missing: {relative}')
    return path


def default_settings_path():
    if resource_root() == REPO:
        return REPO / 'spatial.local.yaml'
    config_home = os.environ.get('XDG_CONFIG_HOME')
    if not config_home and os.name == 'nt':
        config_home = os.environ.get('APPDATA')
    return Path(config_home or Path.home() / '.config').expanduser() / 'spatial-intelligence' / 'config.yaml'


def settings(path=None):
    path = Path(path or os.environ.get("SPATIAL_CONFIG") or default_settings_path()).expanduser()
    if not path.exists():
        raise FileNotFoundError("Run `spatial init --root /your/data/spatial` first")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["config_path"] = str(path.resolve())
    return cfg


def initialize(root, output=None, endpoint="https://huggingface.co"):
    root = Path(root).expanduser().resolve()
    path = Path(output or os.environ.get("SPATIAL_CONFIG") or default_settings_path()).expanduser()
    if path.exists():
        raise FileExistsError(f"Existing local settings preserved: {path}")
    cfg = {"root":str(root),"hf_endpoint":endpoint}
    for name in ["models", "datasets", "manifests", "frames", "geometry", "runs", "external", "receipts"]:
        dest = root/name
        dest.mkdir(parents=True,exist_ok=True)
        cfg[name] = str(dest)
    cfg["python"] = os.sys.executable
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False), encoding="utf-8")
    return cfg


def catalog(kind):
    if kind not in {'assets', 'sources', 'benchmarks'}:
        raise ValueError('Unknown catalog kind')
    return yaml.safe_load(resource_path(Path('catalog') / (kind+'.yaml')).read_text(encoding="utf-8"))
