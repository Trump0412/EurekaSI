"""Run original experiment entrypoints in their original environment."""
import os
import subprocess
from pathlib import Path

from .io import file_digest


ENTRIES = {
    "geowire_tip": ("GeoWire/geowire", "scripts/train_tip.py", "python"),
    "geowire_sft": ("GeoWire/geowire", "scripts/train_sft.py", "python"),
    "geopsro_align": ("GeoPSRO", "geopsro4d.train.train_stage1_align", "module"),
    "geopsro_sft": ("GeoPSRO", "geopsro4d.train.train_stage2_sft", "module"),
    "geopsro_prepare_rl": ("GeoPSRO", "geopsro4d.train.train_stage3_rft", "module"),
    "geobridge_fcp": ("GeoBridge", "scripts/train/train_stage1_geobridge_fcp_g11.sh", "bash"),
    "geobridge_sft_qwen3": ("GeoBridge", "scripts/train/train_stage2_qwen3vl_2b_spatialfit_hgb.sh", "bash"),
    "geobridge_sft_qwen25": ("GeoBridge", "scripts/train/train_stage2_qwen25vl_7b_spatialfit_hgb.sh", "bash"),
}


def launch(cfg, execute=False):
    if cfg["entry"] not in ENTRIES:
        raise ValueError(f"Available legacy entries: {list(ENTRIES)}")
    subdir, target, kind = ENTRIES[cfg["entry"]]
    if cfg.get("source_root") == "workspace":
        from .workspace import settings
        parts=Path(subdir).parts
        cwd=Path(settings()["external"])/parts[0].lower()
        for part in parts[1:]:cwd=cwd/part
    else:
        root = Path(cfg.get("source_root", "legacy_sources")).resolve()
        cwd = root / subdir
    if not cwd.is_dir():
        raise FileNotFoundError(cwd)
    if kind == "bash":
        argv = ["bash", target]
    else:
        argv = [cfg.get("python", "python")]
        if cfg.get("nproc", 1) > 1:
            argv += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(cfg["nproc"])]
        argv += ["-m", target] if kind == "module" else [target]
    argv += [str(a) for a in cfg.get("args", [])]
    target_file = cwd / (target.replace(".", "/")+".py" if kind == "module" else target)
    plan = {"cwd": str(cwd), "argv": argv, "env_overrides": cfg.get("env", {}),
            "entry_sha256": file_digest(target_file), "executed": execute,
            "operation": "prepare_only" if cfg["entry"] == "geopsro_prepare_rl" else "legacy_entry"}
    if execute:
        env = os.environ.copy()
        env.update({k:str(v) for k,v in cfg.get("env", {}).items()})
        env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
        # subprocess without shell interpolation; original launcher owns its flags.
        subprocess.run(argv, cwd=cwd, env=env, check=True)
    return plan
