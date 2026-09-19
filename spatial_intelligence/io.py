import hashlib
import importlib
import json
import os
import platform
import subprocess
from pathlib import Path

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def apply_overrides(cfg, overrides=()):
    """Apply strict overrides to an already resolved config, without dropping CLI values."""
    for value in overrides:
        key, sep, raw = value.partition("=")
        if not sep:
            raise ValueError("Override must be key.path=value")
        keys = key.split(".")
        parent = cfg
        for k in keys[:-1]:
            if k not in parent or not isinstance(parent[k], dict):
                raise ValueError(f"Unknown override path: {key}")
            parent = parent[k]
        if keys[-1] not in parent:
            raise ValueError(f"Unknown override key: {key}")
        parent[keys[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path, overrides=()):
    # All paths are relative to the invocation directory, not the YAML directory.
    path = Path(path).expanduser()
    if not path.exists() and not path.is_absolute() and path.parts[:1] == ('configs',):
        from .workspace import resource_path
        path = resource_path(path)
    text = os.path.expandvars(path.read_text(encoding="utf-8"))
    if "${" in text:
        raise ValueError("Unresolved environment variable in config")
    # YAML 1.1 parses JSON scientific notation such as 1e-05 as a string.
    # Plans written by write_json must round-trip without changing numeric types.
    cfg = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return apply_overrides(cfg, overrides)


def symbol(spec):
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError("Plugin must be module:symbol")
    return getattr(importlib.import_module(module), name)


def environment():
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ["torch", "transformers", "peft", "accelerate", "PyYAML"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "packages": packages, "git_commit": commit, "git_dirty": dirty}
