from pathlib import Path

from .io import file_digest, write_json


def freeze_frames(video, output, count):
    import cv2
    if count < 1:
        raise ValueError("count must be positive")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Frame output is nonempty")
    cap = cv2.VideoCapture(str(video))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    import math
    if not cap.isOpened() or total < 1 or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Cannot decode video metadata")
    n = min(count, total)
    indices = [round(i*(total-1)/(n-1)) for i in range(n)] if n > 1 else [total//2]
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at frame {idx}")
            path = out / f"frame-{idx:08d}.png"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Cannot write {path}")
            paths.append(str(path.resolve()))
    finally:
        cap.release()
    result = {"media": paths, "frame_indices": indices, "timestamps_s": [i/fps for i in indices],
              "media_sha256": [file_digest(p) for p in paths],
              "source_sha256": file_digest(video), "fps": fps, "total_frames": total,
              "sampling": "uniform_endpoints_v1", "timestamp_status": "index/fps; use official timestamps for VFR videos"}
    write_json(out / "frames.json", result)
    return result
