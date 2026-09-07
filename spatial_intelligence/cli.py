import argparse
import itertools
import json
from pathlib import Path

from .data import leakage, load_samples, normalize
from .io import load_config, read_jsonl, symbol, write_json, write_jsonl


def main():
    import sys
    from .platform import COMMANDS, main as platform_main
    if len(sys.argv)>1 and sys.argv[1] in COMMANDS:
        return platform_main(sys.argv[1:])
    p = argparse.ArgumentParser(prog="spatial", epilog="Workspace commands: " + ", ".join(sorted(COMMANDS)) + ". Use spatial COMMAND --help.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["infer", "train", "validate"]:
        c = sub.add_parser(name)
        c.add_argument("--config", required=True)
        c.add_argument("--set", action="append", default=[])
    c = sub.add_parser("score")
    c.add_argument("--config", required=True)
    c.add_argument("--predictions", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("reports", nargs="+")
    c.add_argument("--output", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--split", required=True, choices=["train", "val", "test"])
    c.add_argument("--media-root", default=".")
    c.add_argument("--adapter", help="Optional raw-row converter module:function, returns one canonical-compatible row")
    c = sub.add_parser("audit")
    c.add_argument("--train", nargs="+", required=True)
    c.add_argument("--test", nargs="+", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("legacy")
    c.add_argument("--config", required=True)
    c.add_argument("--execute", action="store_true")
    c = sub.add_parser("sweep")
    c.add_argument("--config", required=True)
    c.add_argument("--grid", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--execute", choices=["train", "infer"])
    c = sub.add_parser("frames")
    c.add_argument("--video", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--count", type=int, default=8)
    c = sub.add_parser("teacher-export")
    c.add_argument("--config", required=True)
    c.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command in {"train", "infer", "validate", "score", "teacher-export"}:
        cfg = load_config(args.config, getattr(args, "set", []))
        from .config import validate
        validate(cfg)
    if args.command == "train":
        from .training import train
        print(train(cfg))
    elif args.command == "infer":
        from .runner import infer
        print(json.dumps(infer(cfg), ensure_ascii=False, indent=2))
    elif args.command == "validate":
        result = {"config": "valid", "eval_samples": len(load_samples(cfg["data"]["eval"])) if cfg["data"]["eval"] else 0}
        for entry in cfg["data"]["train"]:
            result[entry["path"]] = len(load_samples(entry["path"], training=True))
        print(json.dumps(result, indent=2))
    elif args.command == "score":
        from .runner import evaluate
        print(json.dumps(evaluate(cfg, args.predictions, args.output), indent=2))
    elif args.command == "compare":
        from .evaluation import compare
        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
        if any("mock_not_model_result" in r.get("backend_status", []) for r in reports):
            raise ValueError("Mock runs cannot be used for model comparisons")
        write_json(args.output, compare(reports))
    elif args.command == "convert":
        raw = json.loads(Path(args.input).read_text(encoding="utf-8")) if Path(args.input).suffix == ".json" else read_jsonl(args.input)
        if not isinstance(raw, list):
            raise ValueError("Expected JSON array or JSONL; use explicit adapter for nested source data")
        adapter = symbol(args.adapter) if args.adapter else lambda x: x
        rows = [normalize(adapter(r), args.dataset, args.media_root, args.split) for r in raw]
        write_jsonl(args.output, rows)
        load_samples(args.output, training=args.split == "train")
    elif args.command == "audit":
        train_rows = [r for path in args.train for r in load_samples(path, training=True)]
        test_rows = [r for path in args.test for r in load_samples(path)]
        overlaps = leakage(train_rows, test_rows)
        write_json(args.output, overlaps)
        if any(overlaps.values()):
            raise SystemExit("Overlap detected; see audit file")
    elif args.command == "legacy":
        from .legacy import launch
        print(json.dumps(launch(load_config(args.config), args.execute), indent=2, ensure_ascii=False))
    elif args.command == "sweep":
        from .config import validate
        grid = load_config(args.grid)
        out = Path(args.output)
        if out.exists() and any(out.iterdir()):
            raise ValueError("Sweep output is nonempty")
        keys = list(grid)
        if not keys or any(not isinstance(v, list) or not v for v in grid.values()):
            raise ValueError("Grid must map config key paths to nonempty lists")
        paths = []
        for i, values in enumerate(itertools.product(*(grid[k] for k in keys))):
            # Reuse strict override handling, including typo detection.
            overrides = [f"{k}={json.dumps(v)}" for k,v in zip(keys, values)]
            cfg = load_config(args.config, overrides)
            cfg["output"] = str(out / f"run-{i:04d}")
            validate(cfg)
            path = out / f"config-{i:04d}.json"
            write_json(path, cfg)
            paths.append(str(path))
            if args.execute:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "spatial_intelligence", args.execute, "--config", str(path)], check=True)
        print(json.dumps(paths, indent=2))
    elif args.command == "frames":
        from .video import freeze_frames
        print(json.dumps(freeze_frames(args.video, args.output, args.count), indent=2))
    elif args.command == "teacher-export":
        from .backends import create
        from .data import model_input
        from .io import environment, file_digest
        if Path(args.output).exists() or Path(args.output).with_suffix('.teacher.json').exists():
            raise FileExistsError('Teacher export already exists; preserve the previous distillation target')
        rows = load_samples(cfg["data"]["eval"], training=True)
        if cfg["model"]["backend"] == "mock":
            raise ValueError("Mock is not a teacher")
        backend = create(cfg["model"])
        identity = backend.identity() if hasattr(backend, 'identity') else {'status': 'plugin_identity_not_implemented'}
        for i, row in enumerate(rows):
            result = backend.generate(model_input(row), cfg["protocol"], cfg["protocol"]["seed"]+i)
            row["metadata"]["teacher_response"] = result["response"]
            row["metadata"]["teacher"] = cfg["model"]
            row["metadata"]["teacher_protocol"] = cfg["protocol"]
        write_jsonl(args.output, rows)
        write_json(Path(args.output).with_suffix('.teacher.json'), {
            'config': cfg, 'model_identity': identity, 'environment': environment(),
            'input_manifest_sha256': file_digest(cfg['data']['eval']),
            'output_manifest_sha256': file_digest(args.output), 'rows': len(rows),
            'status': 'teacher_export_completed',
        })


if __name__ == "__main__":
    main()
