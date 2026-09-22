# SpatialFit / GeoFits implementation and acceptance

Engineering name: GeoFits; manuscript name: SpatialFit. The current recipe is
an explicit full-parameter adaptation inspired by active geometry integration,
not a claim to reproduce either manuscript scores or GeoThinker's full recipe.

## Executable matrix

| Variant | Active bank | Retrieval | Gate | Decoder layers |
| --- | --- | --- | --- | --- |
| full | VGGT + Pi3, six entries | TopK2 | learned sigmoid | 1,2,3 |
| 3d_only | VGGT, three entries | TopK2 | learned sigmoid | 1,2,3 |
| 4d_only | Pi3 + temporal adapters, three entries | TopK2 | learned sigmoid | 1,2,3 |
| dense | same six entries | softmax over all six | learned sigmoid | 1,2,3 |
| no_gate | same six entries | TopK2 | fixed one, no gate parameters | 1,2,3 |
| single_layer | same six entries | TopK2 | learned sigmoid | 3 only |

All use the same released initialization, six-source split, one epoch, global64,
LR1e-5 and seed3407. Language/native RGB/active fusion modules are fully trained,
without LoRA; geometry teachers are frozen. Native RGB freezing in the manuscript
is intentionally overridden. Ordered images are not assigned invented timestamps.
Dense softmax, teacher grid resizing and order-only time encoding are explicit
implementation choices, not recovered undocumented author settings.

## Interfaces

- `scripts/build-geometry-followup-plans.py`: generates independent per-role
  runtime, training, reload and benchmark stages; does **not** launch them.
- `scripts/verify-geofits-runtime.py`: actual-model single/multiple/32-frame
  diagnostics. It requires an idle allocation and successful predecessors.
- `scripts/train-geofits-stage.py`: full-parameter worker and separate reload.
- `scripts/evaluate-geofits.py`: real teacher inference and deterministic shards;
  fixed source/model contract, exact ID coverage, and score recomputation at merge.
- `scripts/prepare-followup-benchmarks.py`: converts existing media, never trains.

New plans/code must get new immutable roots. Existing armed plans are not upgraded
by editing this repository. A generated plan is not an installed queue or a
successful GPU run. In particular, prior full-only GeoFits plans remain historical
and must not run alongside a replacement on the same allocation.

## Fusion visibility

Opt-in `model.fusion_diagnostics()` records JSON-only statistics per decoder
layer: gate min/mean/max, bank selection frequency and weight, intended and actual
residual relative norms, changed element/region fractions, and nonzero residuals
swallowed by low-precision addition. It is eval/no-grad only, disabled in training
by default, and skipped during cached decode. No activations are persisted.

These statistics establish influence, not accuracy. Neither a large residual nor
an attractive heatmap is a success criterion. Compare downstream scores on fixed
held-out IDs; do not increase gate strength using test answers or visual contrast.

## Current evidence and boundaries

Core six-variant, real tiny-Qwen, telemetry and worker CPU tests: 35 passed in an
isolated CPU-only environment. They cover backward/GC, answer-blind conditioning,
save/reload, KV equivalence, actual generation, inactive branches and BF16 residual
rounding. A broader integration suite also checks data gates, queue ownership,
disk-cache bounds and effect measurements. These are **not** released-model GPU
memory/throughput validation or formal benchmark results.

MMSI1000, MindCube-Tiny1050, ViewSpatial5712 and CV-Bench2638 have explicit
adapters. Strict choice extraction is ground-truth-blind; invalid/truncated outputs
count wrong. CV-Bench uses 0.25 ADE20K + 0.25 COCO + 0.5 Omni3D. No pooled accuracy
is silently substituted. Source/scene exposure and OpenSpatial's authorized
unknown scene overlap remain report limitations.

Pending before scientific completion: each variant's real multi-GPU gate, formal
one-epoch training, independent reload and complete benchmark outputs. Upstream
GeoRoute paper completion still includes its own SITE/correspondence requirements;
missing evidence is not bypassed just to make GeoFits start.
