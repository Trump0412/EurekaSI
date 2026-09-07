"""Canonical, frame-explicit datasets. Labels never enter inference inputs."""
import random
import re
import math
from pathlib import Path

from .io import digest, file_digest, read_jsonl


def first(row, keys, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def normalize(row, dataset=None, media_root=".", split=None):
    sid = first(row, ["id", "sample_id", "qid", "clip_id"])
    ds = first(row, ["dataset", "source_dataset"], dataset)
    question = first(row, ["question", "prompt"])
    answer = first(row, ["answer", "target", "label"])
    if sid is None or not ds or not isinstance(question, str) or not question.strip():
        raise ValueError("Each row needs explicit dataset, id and nonempty question")
    media = first(row, ["media", "media_paths", "frame_paths", "images"], [])
    if row.get("video") and not media:
        raise ValueError("Decode video into ordered frames first: spatial frames --help")
    if isinstance(media, str):
        media = [media]
    paths = [str((Path(media_root) / p).resolve()) for p in media]
    if any(Path(p).suffix.lower() in {".mp4", ".avi", ".mov", ".webm"} for p in paths):
        raise ValueError("Raw video unsupported here; freeze frames with spatial frames")
    choices = row.get("choices")
    if isinstance(choices, list):
        if len(choices) > 26:
            raise ValueError("At most 26 choices")
        choices = {chr(65+i): re.sub(r"^[A-Z][.):]\s*", "", str(v)) for i, v in enumerate(choices)}
    if choices is not None:
        if not isinstance(choices, dict) or not choices or any(not re.fullmatch("[A-Z]", str(k)) for k in choices):
            raise ValueError("choices must map uppercase letters to text")
        choices = {k: str(v) for k, v in choices.items()}
    meta = dict(row.get("metadata") or {})
    actual_split = first(row, ["split"], meta.get("split", split))
    if actual_split not in {"train", "val", "test"}:
        raise ValueError("Explicit split must be train/val/test")
    out = {"id": str(sid), "dataset": str(ds), "question": question.strip(),
           "answer": None if answer is None else str(answer), "media": paths,
           "choices": choices, "split": actual_split,
           "scene_id": first(row, ["scene_id"], meta.get("scene_id")),
           "task": first(row, ["task", "task_type"], "unspecified"),
           "metric": row.get("metric", "choice" if choices else "exact"),
           "frame_indices": first(row, ["frame_indices"], meta.get("frame_indices", [])),
           "timestamps_s": first(row, ["timestamps_s"], meta.get("timestamps_s", [])),
           "metadata": meta}
    if "geometry_media" in row:
        out["geometry_media"] = [str((Path(media_root)/p).resolve()) for p in row["geometry_media"]]
        if len(out["geometry_media"]) != len(paths):
            raise ValueError("Clean geometry frames must align with rendered VLM frames")
    if out["scene_id"] is not None:
        out["scene_id"] = str(out["scene_id"]).strip() or None
    for key in ("frame_indices", "timestamps_s"):
        if out[key] and len(out[key]) != len(paths):
            raise ValueError(f"{key} and frame count differ: {sid}")
    if out["metric"] not in {"choice", "exact", "numeric"}:
        raise ValueError("Built-in metrics: choice/exact/numeric; official scoring uses a plugin")
    if out["metric"] == "choice" and (not choices or (answer is not None and str(answer) not in choices)):
        raise ValueError("Choice gold must be an explicit option label")
    return out


def key(row):
    return row["dataset"] + "::" + row["id"]


def load_samples(path, *, training=False):
    rows = [normalize(r) for r in read_jsonl(path)]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids = [key(r) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate dataset::id")
    if training and any(r["split"] != "train" or r["answer"] is None or not r["answer"].strip() for r in rows):
        raise ValueError("Training requires split=train and nonempty answers")
    return rows


def model_input(row):
    # Narrow allowlist: metadata/labels are not forwarded to model backends.
    return {k: row[k] for k in ("id", "dataset", "question", "media", "geometry_media", "choices", "frame_indices", "timestamps_s") if k in row}


def content_fingerprint(rows):
    cached = {}
    result = []
    for row in rows:
        hashes = []
        for path in row["media"]:
            if path not in cached:
                cached[path] = file_digest(path)
            hashes.append(cached[path])
        value = {k: v for k, v in row.items() if k not in {"media", "geometry_media", "metadata"}}
        value["media_sha256"] = hashes
        value["geometry_media_sha256"] = [file_digest(p) for p in row.get("geometry_media", [])]
        result.append(value)
    return digest(result)


def leakage(train, test):
    """Exact IDs, scenes, media bytes and media+question duplicates, across aliases."""
    def index(rows):
        out = {"id": set(), "scene": set(), "media": set(), "question_media": set()}
        cache = {}
        for row in rows:
            out["id"].add(key(row))
            if row["scene_id"] is not None:
                out["scene"].add(str(row["scene_id"]))
            hs = []
            for p in row.get("geometry_media", row["media"]):
                if p not in cache:
                    cache[p] = file_digest(p)
                hs.append(cache[p])
            out["media"].update(hs)
            out["question_media"].add(digest([row["question"].lower(), hs]))
        return out
    a, b = index(train), index(test)
    return {k: sorted(a[k] & b[k]) for k in a}


class Mixture:
    def __init__(self, entries, seed):
        self.rng = random.Random(seed)
        self.datasets = [load_samples(e["path"], training=True) for e in entries]
        self.weights = [float(e["weight"]) for e in entries]
        if not self.weights or any(not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Mixture weights must be positive")
        ids = [key(r) for rows in self.datasets for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample across training manifests")

    def sample(self):
        i = self.rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        return self.rng.choice(self.datasets[i])
