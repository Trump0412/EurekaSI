import time
from pathlib import Path

from .backends import create
from .data import content_fingerprint, key, load_samples, model_input
from .evaluation import score
from .io import digest, environment, file_digest, read_jsonl, write_json, write_jsonl


def protocol_hash(protocol):
    return digest({"implementation": "spatial_frames_v1", "protocol": protocol,
                   "runner_sha256":file_digest(__file__),
                   "hf_input_sha256":file_digest(Path(__file__).parent/"backends/hf.py")})


def infer(cfg):
    from .distributed import Distributed
    context=Distributed(cfg)
    try:return _infer(context.cfg,context)
    finally:context.close()


def _infer(cfg,context):
    rows = load_samples(cfg["data"]["eval"])
    out = Path(cfg["output"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output is nonempty; use a new directory to preserve run provenance")
    if cfg["model"].get("adapter") and cfg["model"]["backend"] == "mock":
        raise ValueError("Mock does not load checkpoints")
    ds_hash = content_fingerprint(rows)
    ph = protocol_hash(cfg["protocol"])
    context.barrier()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": cfg, "dataset_hash": ds_hash, "protocol_hash": ph,
                "environment": environment(), "status": "running"}
    if context.main:write_json(out / "run.json", manifest)
    backend = create(cfg["model"])
    manifest["model_identity"] = backend.identity() if hasattr(backend, "identity") else {"status":"plugin_identity_not_implemented"}
    if context.main:write_json(out / "run.json", manifest)
    predictions = []
    pred_path = out / ("predictions.jsonl" if context.world==1 else f"predictions.rank{context.rank}.jsonl")
    # Stream output so long inference retains completed examples after interruption.
    import json
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows[context.rank::context.world]:
            start = time.perf_counter()
            seed = cfg["protocol"]["seed"] + int(digest(key(row))[:8], 16)
            result = backend.generate(model_input(row), cfg["protocol"], seed)
            pred = {**result, "id": key(row), "seconds": time.perf_counter()-start,
                    "frame_indices": row["frame_indices"], "protocol_hash": ph, "dataset_hash": ds_hash}
            f.write(json.dumps(pred, ensure_ascii=False, allow_nan=False)+"\n")
            f.flush()
            predictions.append(pred)
    context.barrier()
    if context.world>1:
        if context.main:
            predictions=[p for rank in range(context.world) for p in read_jsonl(out/f"predictions.rank{rank}.jsonl")]
            order={key(r):i for i,r in enumerate(rows)}
            predictions.sort(key=lambda p:order[p["id"]])
            write_jsonl(out/"predictions.jsonl",predictions)
        pred_path=out/"predictions.jsonl"
        context.barrier()
    if not context.main:
        context.barrier()
        return {"status":"rank_completed","rank":context.rank}
    manifest["world_size"]=context.world
    manifest["status"] = "completed"
    manifest["predictions_sha256"] = file_digest(pred_path)
    if context.main:write_json(out / "run.json", manifest)
    if all(r["answer"] is not None for r in rows):
        result=evaluate(cfg, pred_path, out / "metrics.json")
    else:result=manifest
    context.barrier()
    return result


def evaluate(cfg, predictions_path, output):
    rows = load_samples(cfg["data"]["eval"])
    preds = read_jsonl(predictions_path)
    ds_hash = content_fingerprint(rows)
    ph = protocol_hash(cfg["protocol"])
    if not preds or any(p.get("protocol_hash") != ph or p.get("dataset_hash") != ds_hash for p in preds):
        raise ValueError("Prediction provenance is absent or differs from evaluation config/data")
    run_path = Path(predictions_path).parent / "run.json"
    import json
    if not run_path.exists():
        raise ValueError("Missing inference run.json; do not relabel imported predictions as verified runs")
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if manifest.get("predictions_sha256") != file_digest(predictions_path):
        raise ValueError("Prediction file changed since inference completed")
    if manifest["config"]["model"] != cfg["model"]:
        raise ValueError("Scoring model config differs from prediction provenance")
    report, details = score(rows, preds, cfg["score"])
    report.update({"model": cfg["model"], "protocol_hash": ph, "dataset_hash": ds_hash,
                   "model_identity": manifest.get("model_identity"),
                   "backend_status": sorted({p["backend_status"] for p in preds})})
    write_json(output, report)
    write_jsonl(Path(output).with_suffix(".details.jsonl"), details)
    return report
