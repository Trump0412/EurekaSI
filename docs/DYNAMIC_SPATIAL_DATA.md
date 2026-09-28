# Dynamic spatial annotations and video pairing

Official sources:

- [4DThinker](https://github.com/zhangquanchen/4DThinker), [training data](https://huggingface.co/datasets/jankin123/4DThinker-Training-Data).
- [DSR Suite](https://github.com/TencentARC/DSR_Suite), [annotations](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data).

This downloads annotations, dataset cards and the available DSR license, not model
weights or the large DIFT processed-frame archives. It does not install upstream
Transformers forks, modify environments, or enqueue training. HF downloads happen
on the server through `hf-mirror.com` at pinned dataset revisions. Raw annotations
are retained unchanged; receipts record repository, revision, filename and size.

```bash
"$NODE_ROOT/envs/qwen35/bin/python" scripts/download-dynamic-annotations.py \
  --root "$NODE_ROOT/datasets/dynamic-annotations" --detach

# Once download.log ends with ANNOTATIONS_DOWNLOADED:
"$NODE_ROOT/envs/qwen35/bin/python" scripts/audit-dynamic-annotations.py \
  --node-root "$NODE_ROOT"
```

Audit requires pandas/pyarrow and the inventories written by
`download-baidu-folders.py` for `/4drl_clips` and `/dsr-bench`.
Do not run multiple downloaders against the same root. The runner uses a lock,
bounded curl retries and temporary files. A rerun downloads the small annotation
set again; it does not resume partially downloaded annotation files.

## Verified annotation inventory (2026-09-20)

| Source/file | Actual QA rows | Referenced videos | Matching cloud-folder names |
|---|---:|---:|---:|
| 4DThinker `4drl_data_filtered.jsonl` | 36,710 | 6,618 | 6,125 in `4drl_clips`; covers 34,003 QA rows |
| 4DThinker `dift_data.jsonl` | 38,033 | frame/mask paths, not MP4 | not covered by this MP4 audit |
| DSR `benchmark.parquet` | 1,450 | 563 | all 563 in `dsr-bench` |
| DSR `train_qa_pairs.parquet` | 55,182 | 9,966 | 6,125 in `4drl_clips`; covers 34,003 QA rows |

The DSR JSON is a mapping keyed by video ID, not a flat QA list; the reported
training count is measured from the parquet. Do not count both formats as
independent training data. Likewise, 4DRL is derived partly from DSR: do not
concatenate their training annotations without QA-level deduplication.

Pinned revisions:

- 4DThinker: `37b53b800f07f2363364dd6c0f9b394d37138bd0`.
- DSR Suite: `414132f02d03cecc583cc6ae77c9cfdc661f87a7`.

Current DSR benchmark file has **1,450**, not the headline 1,484 questions.
Use the actual pinned file and denominator in reports; the reason for the
difference has not been established.

## Readiness boundaries

The audit matches exact video basenames (`videoID + '.mp4'` for DSR). It checks
the remote inventory, **not completed downloads or video decoding**. At initial
audit the media transfers were still running. Wait for completion and perform
decode validation before generating runnable training/evaluation manifests.

- The supplied 6,142-file clip directory misses 493 referenced 4DRL videos;
  17 supplied names are not referenced by that annotation version. Do not silently
  drop the associated 2,707 questions or call a subset full-data reproduction.
- Full DSR training needs 3,841 additional video names beyond that inventory.
- The two supplied directories have zero exact filename overlap. This alone does
  not prove independence of original source videos/scenes or absence of duplicate
  content. Keep DSR benchmark labels out of training and audit source IDs before use.
- DIFT requires `processed_data/.../frames` and mask overlays in `image_output`.
  These are not equivalent to the 4DRL MP4 files; its large archives are deliberately
  not downloaded by this annotation-only task.
- Preserve original paths. Future adapters should explicitly map `video_path`
  basenames to the train clip root, and DSR `videoID` to the benchmark video root.
  No symlinks/path rewrites or active training changes are performed here.

`coverage-audit.json` records schemas and coverage; `*-missing-media.json` records
missing basenames. These runtime outputs stay outside public source. Credential
configuration, signed URLs and deployment mappings must never be committed.
