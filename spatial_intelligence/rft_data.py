"""Auditable RFT-only adapters. No training or GPU imports.

Released frame order is authoritative. Source-group matching is conservative
identifier matching, NOT a claim of cross-dataset perceptual deduplication.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choices_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    result = {}
    for option in value:
        match = re.fullmatch(r"\s*([A-Z])[.)]\s*(.*)", str(option), re.S)
        if not match or match[1] in result:
            raise ValueError("Ambiguous option labels")
        result[match[1]] = match[2]
    return result


def source_group(value):
    """Collapse ScanNet rescans and DSR clips; preserve other identifiers."""
    stem = Path(str(value)).stem
    scan = re.fullmatch(r"(scene\d+)_\d+", stem)
    if scan:
        return "scannet:" + scan[1]
    clip = re.fullmatch(r"([A-Za-z0-9_-]{11})_\d+", stem)
    if clip:
        return "video:" + clip[1]
    return "id:" + stem


def _canonical(source, index, scene, question, answer, choices, question_type):
    if not str(question).strip() or answer is None:
        raise ValueError("Missing question or answer")
    answer_type = "mcq" if choices else "numeric" if re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(answer).strip()) else "text"
    raw_answer = answer
    if choices:
        letter = str(answer).strip().upper()
        matches = [k for k,v in choices.items() if str(v).strip() == str(answer).strip()]
        if letter in choices: answer = letter
        elif len(matches) == 1: answer = matches[0]
        else: raise ValueError("Ambiguous MCQ answer: neither label nor unique exact option text")
    return dict(id=f"{source}:{index}", source=source, source_id=str(index),
                scene_id=scene, source_group=source_group(scene), question=str(question),
                answer=answer, raw_answer=raw_answer, answer_type=answer_type,
                choices=choices, question_type=question_type)


def normalize_4drl(row, index, media_root):
    video = Path(row["video_path"]).name
    choices = {k: row[k] for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if k in row}
    choices = {k: re.sub(r"^"+k+r"[.)]\s*", "", str(v)) for k,v in choices.items()}
    out = _canonical("4drl", index, Path(video).stem, row["Question"], row["Correct"],
                     choices, row.get("Type", "unspecified_released_annotation"))
    out.update(video_path=str(Path(media_root) / video), input_mode="video",
               raw_options={k:row[k] for k in choices}, original_video_path=row["video_path"])
    # CoT annotations are provenance only; never silently used as the target.
    out["raw_annotation"] = row
    return out


def normalize_dsr(row, index, media_root):
    out = _canonical("dsr", index, row["videoID"], row["question"], row["answer"],
                     choices_dict(row["options"]), row["type"])
    out.update(video_path=str(Path(media_root) / (row["videoID"] + ".mp4")),
               input_mode="video", raw_options=list(row["options"]), split="test")
    return out


def normalize_spatialladder(row, media_root):
    refs = row["image"]
    if len(refs) <= 1:
        raise ValueError("SpatialLadder RFT selection requires multiple frames")
    if any(Path(x).is_absolute() or ".." in Path(x).parts or "\\" in x for x in refs):
        raise ValueError("Unsafe image path")
    scenes = {Path(x).parts[0] for x in refs}
    if len(scenes) != 1:
        raise ValueError("Ambiguous source scene")
    out = _canonical("spatialladder", row["question_id"], next(iter(scenes)),
                     row["question"], row["answer"], choices_dict(row["options"]),
                     row["question_type"])
    indices = [int(Path(x).stem) if Path(x).stem.isdigit() else None for x in refs]
    out.update(media=[str(Path(media_root) / x) for x in refs], input_mode="images",
               original_frame_names=list(refs), original_frame_indices=indices,
               frame_order="released_annotation_order", fps=None,
               temporal_order_verified=False if row["data_type"] == "video" else None,
               data_type=row["data_type"], raw_options=row["options"],
               nonmonotonic_frame_order=all(x is not None for x in indices)
               and indices != sorted(indices))
    return out


def uniform_indices(total, count):
    if total < count or count < 2:
        raise ValueError(f"Need at least {count} distinct frames, got {total}")
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def group_split(rows, seed=3407, validation_fraction=0.02):
    """Fixed groups, not QA rows; the same source video never crosses splits."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups = sorted({row["source_group"] for row in rows})
    if len(groups) < 2:
        raise ValueError("At least two independent groups required")
    random.Random(seed).shuffle(groups)
    validation = set(groups[:max(1, min(len(groups)-1, round(len(groups)*validation_fraction)))])
    return [dict(row, split="validation" if row["source_group"] in validation else "train") for row in rows]


def benchmark_groups(rows):
    return {source_group(row["scene_id"]) for row in rows}


def filter_leakage(rows, forbidden):
    kept, excluded = [], []
    for row in rows:
        if row["source_group"] in forbidden:
            excluded.append(dict(id=row["id"], source=row["source"],
                                 scene_id=row["scene_id"], reason="benchmark_source_group_overlap"))
        else:
            kept.append(row)
    return kept, excluded


def extract_video(path, destination, count):
    """CPU/OpenCV sequential decoding, exact sampled frame indices; resumable.

    Decode every frame once: trusting CAP_PROP_FRAME_COUNT alone can silently
    accept a truncated download. Cache is tied to size+mtime and sampling count.
    """
    import cv2
    from PIL import Image
    cv2.setNumThreads(1)
    path, destination = Path(path), Path(destination)
    stat = path.stat()
    identity = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, count=count)
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        cached = json.loads(receipt_path.read_text())
        if cached.get("identity") == identity:
            for frame in cached["media"]:
                with Image.open(frame) as image:
                    image.load()
            return cached
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid actual video FPS")
        indices = uniform_indices(total, count)
        wanted = set(indices)
        destination.mkdir(parents=True, exist_ok=True)
        media, decoded = [], 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if decoded in wanted:
                target = destination / f"{decoded:08d}.png"
                temp = target.with_suffix(".part.png")
                if not cv2.imwrite(str(temp), frame):
                    raise ValueError("Frame write failed")
                temp.replace(target)
                media.append(str(target))
            decoded += 1
        if decoded != total or len(media) != count:
            raise ValueError(f"Incomplete decode: actual={decoded}, advertised={total}, selected={len(media)}")
        result = dict(identity=identity, media=media, fps=fps,
                      total_num_frames=total, frame_indices=indices,
                      timestamps_seconds=[i / fps for i in indices],
                      sampling="uniform_inclusive_full_clip", decoded_frames=decoded)
        temp = receipt_path.with_suffix(".tmp")
        temp.write_text(json.dumps(result), encoding="utf-8")
        temp.replace(receipt_path)
        return result
    finally:
        cap.release()
