# SpatialLadder multi-image subset

Source: [official dataset](https://huggingface.co/datasets/hongxingli/SpatialLadder-26k)
and [official code](https://github.com/ZJU-REAL/SpatialLadder).
Pinned dataset revision: `41d36f87a67ff676d438417b5a09079add8de866`.

```bash
"$NODE_ROOT/envs/qwen35/bin/python" scripts/prepare-spatialladder.py \
  --root "$NODE_ROOT/datasets/SpatialLadder-26k" --detach
```

Requires Linux, curl and Pillow. Downloads run on the server via HF mirror,
with a process lock, retries, resumable `.part` files and a detached log. Do not
reuse this output root for a different revision. No GPU, pip installation,
credential transfer or training queue change is involved.

## Selection verified from the released annotations

The spatial annotation file contains 20,680 rows. Select `len(image) > 1`,
preserving question IDs, original annotation order and image order:

| data_type | Selected QA rows |
|---|---:|
| multi_view | 5,752 |
| video (released frame sequences) | 9,000 |
| total | 14,752 |

Thus the approximately 15K multi-image selection includes both modalities;
the strictly multi-view category alone is only 5,752. No random 15K resampling,
frame thinning or single-image/grounding data is added. The release stores
image paths even for `video`; these are frame sequences, not downloaded MP4s.

Outputs:

- `spld_spatial_data.jsonl`: untouched official spatial annotations.
- `multi-image.jsonl`: selected rows with untouched field semantics.
- `images.zip`: full official ~2.62GB media archive (not separately sharded).
- `media/<original image path>`: only images referenced by selected rows.
- `README.md`: official dataset card/license metadata.
- `receipt.json`, `prepare.log`: version, counts, stage and errors.

The worker extracts selected paths with traversal protection, checks ZIP CRC
while reading and uses Pillow `verify` on every selected image. Only a
`receipt.json` status of `ready` claims this preparation completed. A running
download or existing manifest is not ready media. `verify` is a file-structure
check, not proof of scene/annotation correctness or a full downstream loader test.

Before training, adapt `image` paths against the `media` root explicitly and
retain task/answer types: some questions are numeric/open-ended rather than
multiple-choice. Audit source-scene overlap with the chosen held-out benchmarks;
this preparation does not establish evaluation independence. No train/validation
split or training-format conversion is made here.
