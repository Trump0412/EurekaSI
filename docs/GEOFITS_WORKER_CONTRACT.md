# GeoFits SFT worker: executable contract and acceptance boundaries

This worker is a local full-parameter adaptation. It is not a declaration that
the private paper has been reproduced or that released-model GPU training passed.
It leaves the prior paper's training and RFT snapshots unchanged.

## Current implementation update

All six variants now have distinct executable architectures: `full`, `3d_only`,
`4d_only`, `dense`, `no_gate`, and `single_layer` (decoder layer **3**, not 1).
Use `architecture_for_variant` in `spatial_intelligence/geofits_recipe.py`;
inactive teachers/modules are not loaded or trained. Dense uses the same regional
query with softmax over all entries, an explicit choice where the manuscript
does not uniquely specify its formula. Language, native RGB and active fusion
parameters train fully; the external teachers remain frozen. This is active
geometry retrieval, not annotation acquisition/sample-selection active learning.

`verify-geofits-runtime.py` runs separate real single-image, multi-image and
32-frame update/save/reload gates before each variant's formal training.
`evaluate-geofits.py` keeps the custom model and real teacher path intact, saves
per-example answers, and emits eval-only gate/bank/residual telemetry. ReVSI,
VSI, MMSI, MindCube-Tiny, ViewSpatial and CV-Bench are supported through explicit
scorers. CV-Bench is source-weighted, not a pooled average. The strict final-letter
parser and image-input protocol are declared adaptations, not exact official
inference parity. SITE is a separate GeoRoute requirement, not one of the four
complementary benchmarks in the SpatialFit manuscript.

See [implementation acceptance](GEOFITS_IMPLEMENTATION_ACCEPTANCE.md) for current
tests, deployment boundaries and remaining real-GPU acceptance.

## Entry points

```bash
torchrun --nproc_per_node=8 scripts/train-geofits-stage.py --plan "$PLAN" --micro 1 --diagnostic-steps 2
python scripts/train-geofits-stage.py --plan "$PLAN" --micro 1 --diagnostic-steps 2 --verify-saved
# Only after independent runtime acceptance:
torchrun --nproc_per_node=8 scripts/train-geofits-stage.py --plan "$PLAN" --micro 1
python scripts/train-geofits-stage.py --plan "$PLAN" --micro 1 --verify-saved
```

Private plan keys: `root`, `model`, `processor`, `data_receipt`,
`runtime_acceptance`, `use_lora:false`, `variant:"full"`, `architecture`,
`vggt_source`, `vggt_weights`, `pi3_source`, `pi3_weights`, `deepspeed`, `seed`.
No infrastructure credentials or real node paths belong in a public plan.

Architecture fields: `hidden_size` equal to the native language width,
`vggt_width:2048`, `pi3_width:1024`, `temporal_bottleneck:256`,
`temporal_group_size:1`, `temporal_group_reduction:"mean"`,
`timestamp_encoding:"order_only_sincos"`, `pooling:"average_2x2"`,
and explicitly chosen `retrieval_width`, plus variant fields from the recipe helper.
Selected teacher layers are VGGT 11/17/23 and Pi3 17/26/35. Fusion follows
decoder blocks 1/2/3, with six entries, TopK2 and sigmoid gate in the full model.
Ablations alter only their declared source/retrieval/gate/layer factor.

Outputs: `$ROOT/diagnostic/micro1/` or `$ROOT/formal/micro1/`, including locked
`contract.json`, `live-eta.json`, resumable Trainer checkpoints, `final/`,
`reload-evidence.pt`, and `completion.json`. A diagnostic checkpoint must never
initialize a formal experiment.

## Semantics and resource policy

- Actual 1–32 input frames are preserved as independently ordered images.
  A shared RGB letterbox448 is resized to392 only for frozen geometry teachers;
  both produce 28x28 grids, pooled to14x14 native image regions. This is not the
  native-video evaluation protocol of the previous paper.
- Language, RGB encoder/mergers, retrieval, projectors and temporal adapters are
  full-parameter trainable. Neither teacher has trainable parameters or LoRA.
- Teacher features are computed from images only. No native learned features
  are persisted or substituted by zero tensors. Current extraction is per row;
  throughput optimization requires a separate numerical/resource gate.
- Geometry context carries absolute image-token positions and a terminal
  **prompt** index, never the last answer token. Training labels audit the
  prefix. Native visual embeddings are captured on the real language-model
  path, retaining gradients. Variable frame counts are handled per sample.
- Geometry modifies the first prefill/full-SFT pass, not cached autoregressive
  decode. Context remains bound through backward for activation recomputation.
  Native DeepStack runs after the decoder post-hooks, an explicit ordering.
- One deterministic shuffled instruction epoch, global batch64, seed3407,
  LR1e-5, cosine, warmup3%, weight decay0, sample-mean completion loss.
  Tail padding is counted explicitly. GA is64/(world×micro), not a hidden change
  in effective training batch. Safe memory/throughput must be measured.

## Stage gates and remaining scope

The data receipt must declare ready, verified media, and point to an audited
train manifest. It must pass the shared leakage policy: checked isolation or
the explicit OpenSpatial-only unknown-scene authorization, never a blanket waiver.
Merely downloading media does not satisfy acceptance. Formal execution additionally
requires independent `runtime_acceptance` with ready/full_model_verified and
the exact architecture; the worker does not fabricate that receipt.

Training records numerical changes in language, native visual, geometry
projection, temporal and retrieval parameters. A separate process must reload
the custom registered model, compare logits and generate real continuation
tokens before the stage receipt becomes accepted. Always load through
`load_geofits_model`: a vanilla Qwen loader would discard the geometry model.
The verifier checks optimizer/RNG checkpoint files but does **not** establish
exact resumed-trajectory parity. Full distributed real-model validation and
actual benchmark execution remain additional gates; CPU tests are not substitutes.

Tests: pure scheduling/gate tests and tiny real-Qwen integration are separate
from a released-model/multi-GPU acceptance. Never turn passing CPU tests into
a formal training or benchmark result.
