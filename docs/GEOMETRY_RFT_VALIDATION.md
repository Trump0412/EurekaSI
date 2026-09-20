# Geometry RFT acceptance record

This records implementation and data-readiness evidence, not research scores.
The [RFT runbook](GEOMETRY_RFT_RUNBOOK.md) defines the paired study.

## Checks actually executed

- Isolated Linux CPU, CUDA hidden, four CPU threads: 30 tests passed. These
  comprise 14 real tiny-Qwen/PEFT integration tests, nine reward tests and seven
  GSPO objective/gradient tests. VGGT is a registered test double in these tiny
  tests, not a claim of full VGGT GPU acceptance.
- Integration checks cover language-only LoRA targeting, trainable geometry
  ownership, unchanged frozen RGB/VGGT, exact fixed-reference restoration,
  image group8 with generation micro1/2/4/8, native-video group8 with micro2,
  prompt-local geometry caching, causal response log probabilities and reload.
- Optimizer/scheduler and Torch/Python RNG states restore exactly in the CPU
  continuation test. The next optimizer update agrees within 1e-7 parameter
  tolerance (observed maximum approximately 5.96e-8); this is not a promise of
  bitwise CUDA or cross-hardware continuation.
- Linux CPU queue tests: 25 passed, including real file locking and detection
  of changed manifest content even when sample IDs remain the same.
- Data preparation tests: seven passed, including an actual encoded/decoded
  synthetic video and frame-cache validation.
- Two lightweight sampling tests confirm deterministic resume/world-size
  allocation and exact 7:3 source allocation within full ten-draw blocks.
- Static parsing and documentation-link checks passed. Static checks do not
  validate model execution. Test suites listed here have distinct files;
  repeated runs must not be added again as new evidence.

## Correctness finding: PEFT checkpoint precision

PEFT 0.18.1's generic adapter reload copied `modules_to_save` from a BF16 base
and rounded saved FP32 interface weights during loading. Converting the module
to FP32 afterward did not recover lost updates. The new loader allocates FP32
policy copies first, loads serialized adapter weights, and checks exact key,
dtype and value equality. A 1e-7 update below BF16 resolution is preserved by
the regression test; the original fixed-reference interface remains unchanged.
PEFT's in-place checkpoint-key remapping is isolated using a copied dictionary.

## Data audit before full media readiness

The initially matched subset is not automatically a ready training dataset.
Source-video grouping revealed 616 training QA overlapping benchmark original
video groups even though clip filenames were disjoint. These are excluded.
Another 48 ambiguous answer records are quarantined and 2,699 records lack
media in the supplied subset. Exclusions and original labels are retained.

Candidate source-group-disjoint splits:

| Source | Train | Validation |
|---|---:|---:|
| 4DRL | 32,656 | 691 |
| SpatialLadder | 14,517 | 235 |

All 26,284 referenced unique SpatialLadder images passed actual image decoding.
DSR video extraction and the ongoing 4DRL media download/decode remain separate
readiness gates. Only the final `ready` receipt establishes runnable manifests.
The currently pinned DSR benchmark contains 1,450 QA across 563 videos.

SpatialLadder supplies 5,752 eight-image and 9,000 sixteen-image examples. Its
official loader treats the latter video-labelled records as image lists, with
no timestamps and mostly nonmonotonic filename indices. We preserve that
released image-list order and do not claim verified temporal ordering. The
native-video 4DRL branch uses eight ordered frames and measured source timing.
Identifier grouping does not establish complete perceptual deduplication or
exclude unknown base-model pretraining exposure.

## Remaining mandatory GPU gates

Initial queue deployment correctly blocked when upstream SFT failed before its
first update. The cause was telemetry, not a model loss: FP32 `linspace` rounded
the final probe index of a 113,246,208-element projector matrix to its size,
causing a CUDA out-of-bounds assertion. Exact int64 spacing replaces it in both
SFT and RFT update probes; a regression reproduces the old error. Failed run
records are retained and replacement runs use new immutable versions.

Legacy ReVSI/VSI manifests are normalized without changing frame identities or
inventing timestamps; their existing benchmark scorers remain in use, rather
than the SpatialLadder training reward. Both before/after models receive the
same structured prompt (an adapted protocol, not leaderboard-equivalent).
Internal source-group-held-out validation manifests are also snapshotted into
the paired evaluation suite; they are not added to training.

The after-SFT queues must independently prove full-model group8 sampling,
within-group reward differences, finite nonzero actual updates, frozen reference
and VGGT identity, checkpoint reload and safe memory. They run only after the
declared SFT pipelines finish and the assigned GPUs are idle. CPU acceptance
does not waive these gates or authorize a group-size/input-budget fallback.

No formal RFT loss, DSR score or RFT ETA is claimed by this record. The candidate
47,173 training rows imply 5,897 updates, 94,352 prompt draws and 754,816 rollout
responses **per arm** under the shared budget rule. This is sampling with
replacement, not guaranteed two-epoch coverage. Actual time needs complete
rollout/log-probability/update timing on the selected model and hardware.
