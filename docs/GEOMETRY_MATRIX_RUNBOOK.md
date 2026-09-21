# Geometry fusion matrix: two-stage controlled comparison

This matrix adapts RoboRefer's **training schedule and spatial projector** to
Qwen3-VL-2B + VGGT and the existing SPAR/Hound manifests. It is not an exact
RoboRefer reproduction: NVILA, SigLIP depth inputs, RefSpatial, and the original
distributed hardware are deliberately replaced. Actual host mappings and paths
belong in ignored private deployment configuration, never this repository.

## Questions and experiment contract

1. With the same downsample interface and two-stage recipe, does updating VGGT
   improve held-out spatial QA relative to keeping it frozen?
2. With trainable VGGT and the same recipe, does spatial downsampling outperform
   the global 64-query interface, and at what token/compute cost?
3. How do these adapted recipes compare with the preserved original direct-SFT
   baseline? This third comparison changes multiple factors and is not causal
   evidence for one component.

| Arm | Interface | VGGT during SFT | Alignment | Native RGB during SFT |
|---|---|---|---|---|
| `downsample-trainable` | 3x3 spatial merge + MLP | trainable | one epoch | trainable |
| `query64-trainable` | 256-dimensional bottleneck + 64 learned queries + MLP | trainable | one epoch | trainable |
| `downsample-frozen` | same 3x3 merge + MLP | frozen | one epoch | trainable |
| preserved direct baseline | original 64-query interface | frozen | none | frozen |

The three new arms use the same released Qwen3-VL-2B-Instruct, initial VGGT,
seed/data seed 3407, exact ordered manifests, full frame sequences, and fixed
evaluation manifests. Both stages use all 297899 training rows (SPAR 234149,
Hound 63750), one pass each. Do not silently resample to 70:30, drop frames,
truncate labels, or add test data. The prior baseline uses one pass total;
new arms have two passes total. Base-model pretraining exposure remains unknown.

The two downsample arms may share the **same completed alignment checkpoint**:
both freeze VGGT during alignment, so repeating that identical stage wastes
compute and introduces unnecessary initialization variation. The frozen SFT arm
must wait for a validated alignment receipt and read its checkpoint read-only.
Charge one alignment pass to each model's scientific training budget, but count
the alignment GPU-hours only once in actual infrastructure expenditure. The
query64 arm needs its own alignment because its interface is different.

## Official settings and deliberate differences

2026-09-21 user-authorized revision: new formal SFT arms use **global batch64**
via explicit private-plan `sft_global_batch: 64`. The table below preserves the
original reference settings, not the revised SFT deployment. Completed alignment
at global448 is retained read-only. SFT starts afresh from that alignment, not a
global384 optimizer checkpoint. Historical diagnostics remain in their original
roots; new profiles/selection receipts use global64 and cannot reuse global384
throughput receipts as acceptance. Data, one epoch, LR2e-5, warmup ratio0.03 and
cosine scheduling remain unchanged; total optimizer updates rise to about4655.
The scheduler is constructed for the new total, not copied from an old run.
For 8 ranks, micro1/2/4 use GA8/4/2; for 4 ranks GA16/8/4.
The runner's missing-option default remains384 for historical compatibility;
new deployments must explicitly set64. RFT prompt batch16 and G8 are unchanged.

Snapshot migration fix: when a previous study is the input root, copy formal
manifests but exclude generated `diagnostic-*` manifests and receipts. A six-step
global384 profile has2304 rows, whereas global64 has384; inheriting the former
correctly triggers `Diagnostic sample identity changed` and prevents training.
For a failed-before-training destination, preserve the failure evidence and move
inherited diagnostics into a recovery archive before regeneration and an explicit
retry. Do not relax identity checks, delete formal data, or import old profile
results as acceptance of the new batch. Keep active worker snapshots immutable.

Pinned official source: `Zhoues/RoboRefer@d97a995ad28376720a4c8beb64915c58ed16c844`.

| Setting | Alignment | SFT |
|---|---:|---:|
| Epochs | 1 | 1 |
| Global batch | 448 | 384 |
| Learning rate | 1e-3 | 2e-5 |
| Weight decay | 0 | 0 |
| Warmup fraction | 0.03 | 0.03 |
| Scheduler | cosine | cosine |
| Context limit | 16384 | 16384 |
| Trainable parameters | geometry interface only | language, native RGB and interface; VGGT per arm |

Both official default launch scripts use the same RGB/RGB-D RefSpatial mixture;
stage distinction is not a required change of dataset. Our SPAR/Hound substitution
is explicit. The official scripts use ZeRO-3, bf16, gradient checkpointing,
16 data-loader workers and checkpoint intervals of 1000 steps. Adapted runtime
saves every 100 steps and retains two checkpoints for recoverability: the full
stages have fewer than 1000 optimizer steps, so the official interval would not
provide an intermediate recovery point. The adapted worker uses four data-loader
workers per rank instead of sixteen to avoid oversubscribing shared CPU resources.
Record actual sharding, workers,
microbatch and checkpoint cadence rather than claiming bitwise configuration
identity. Qwen chat template, DeepStack/mRoPE and visual processor are preserved,
not replaced by RoboRefer's Qwen2 template or SigLIP processor.

Runtime adaptation: alignment uses DDP, while SFT uses ZeRO-3. Real diagnostics
found a ZeRO-3 partition/activation-checkpoint metadata incompatibility when the
language backbone was frozen for alignment. The alignment stage fits the tested
GPU memory envelope with DDP; switching its distributed runtime does not change
the data, learning rate, global batch or trainable modules. Waiting queue v1
snapshots are retained as superseded-before-training; corrected queues use fresh
v2 roots. Do not interpret a diagnostic runtime failure as a method result.

Sources: [alignment script](https://github.com/Zhoues/RoboRefer/blob/d97a995ad28376720a4c8beb64915c58ed16c844/scripts/RoboRefer/depth_align_2B.sh),
[SFT script](https://github.com/Zhoues/RoboRefer/blob/d97a995ad28376720a4c8beb64915c58ed16c844/scripts/RoboRefer/depth_sft_2B.sh),
[projector implementation](https://github.com/Zhoues/RoboRefer/blob/d97a995ad28376720a4c8beb64915c58ed16c844/llava/model/multimodal_projector/base_projector.py).

The spatial projector packs 3x3 adjacent patches into channels before its MLP;
it is not average pooling. At 448-square VGGT input, a 32x32 patch grid becomes
11x11 tokens with right/bottom padding. Thus 8 frames produce 968 geometry tokens
and 32 frames produce 3872, versus 64 global tokens for the query arm. Token
budgets, interface parameter counts, FLOPs, latency and memory are not matched.
Report them; accuracy differences alone do not establish efficiency superiority.

All three new arms disable geometry dropout, unlike the preserved direct baseline
(20% learned-null). Downsample tokens pair with their RGB images/native temporal
groups; query tokens remain a global summary. Thus the interface comparison tests
the complete compression-and-layout choice, not the projector MLP in isolation.
No learned-null training is claimed in new arms. Zeroed-geometry evaluation, if
provided, is only a distribution-shift diagnostic.

## Queues, resources and throughput

- Install DeepSpeed only in an independent copied Conda environment; never alter
  an environment used by active training. Clone with `conda create --copy --clone`
  and a new prefix, using configured mirror channels. Keep Torch/torchvision,
  Transformers and NumPy constrained to the validated source versions. Verify
  `pip check`, imports and the actual ZeRO-3 update/save/reload before acceptance;
  installation alone is not a training gate. Copying tens of thousands of files
  on shared storage can take several minutes and is setup time, not GPU training.
  The pinned additions/core versions are in
  [geometry-matrix requirements](../requirements/geometry-matrix.txt); the older
  `geometry-constraints.txt` is for standalone legacy encoders, not this matrix.
- Preserve active jobs, their code snapshots, optimizer checkpoints and evaluation.
- A dependent queue starts only after its predecessor explicitly releases GPU
  work and the allocated GPUs are idle. A state file alone is not process proof.
- Each arm has independent root, code snapshot, locks, state, logs and checkpoints.
  Shared model/media assets are read-only. Do not sync changes over a running
  checkout used by workers or pending evaluations.
- The frozen downsample arm waits for the original direct-fusion train/eval queue,
  not merely the preceding RGB baseline. Separate hosts can run their arms in
  parallel. Never launch filler work to inflate utilization.
- Benchmark micro 1/2/4 with identical sample sets and effective global batch;
  GA is `global_batch / (world_size * micro)`. For eight ranks, alignment GA is
  56/28/14 and SFT GA is 48/24/12. For four ranks these values double.
- Reuse effective-batch balancing from [throughput optimization](THROUGHPUT_OPTIMIZATION.md),
  but revalidate on Qwen3-VL and trainable VGGT. The previous Qwen3.5 result does
  not validate this model; its DeltaNet optimization does not apply to Qwen3-VL.
- Choose the fastest numerically accepted candidate with a long-sample forward,
  backward and optimizer step. The default memory gate is under 92% capacity;
  explicit private-plan `memory_limit_fraction: 1.0` accepts measured peaks
  within physical capacity when the user accepts reduced memory headroom.
  This does not guarantee absence of future OOM, nor waive actual OOM,
  finite-update, numerical-parity, long-sample or reload checks.
  Larger microbatch is not automatically faster. OOM candidates terminate only
  their own process groups and leave official checkpoints untouched.
- Use sample-mean completion loss consistently when comparing microbatch sizes;
  record this as an explicit adapted objective, not verified RoboRefer parity.

## Required gates before full runs

1. Confirm input row IDs, order, markers, frame counts, masks and labels against
   the locked manifest. Geometry sees the same permitted RGB inputs, no labels.
2. Test true real-image forward/backward for both interfaces and both VGGT modes.
   Interface gradients must be finite/nonzero; VGGT gradients and parameter
   updates must be nonzero only in trainable SFT. Native RGB must update in SFT
   and remain frozen in alignment. Do not use no-grad features for trainable VGGT.
3. Save/reload and compare outputs; verify saved VGGT weights are the trained
   weights, not silently reloaded released weights. Check optimizer/scheduler/RNG
   restoration. Diagnostic steps are not research training.
4. Check geometry padding, variable-length multi-sample batches, native video
   mRoPE/DeepStack, generation prefill-only insertion, and explicit context overflow.
5. Pass numerical and long-sample memory gates, distributed smoke, then start
   formal alignment. Final alignment weights initialize SFT; diagnostic/profile
   weights never do. Stop on a failed semantic gate, do not silently skip it.

## Deployment entry point

The bootstrap requires an already prepared compatible Qwen training environment;
it is not a standalone installer from a bare system. It makes an independent
Conda copy, installs the pinned DeepSpeed stack, then checks dependencies and
imports. An incomplete existing target is preserved and reported for inspection.

```bash
export NODE_ROOT=/persistent/independent-node
export SOURCE_ENV=/persistent/prepared-qwen-env
export CONDA_BIN=/path/to/miniconda3/bin/conda
bash scripts/bootstrap-geometry-matrix.sh
```

Optional `GEOMETRY_CONDA_CHANNEL` and `GEOMETRY_PIP_INDEX` override the default
mirrors. The script never accepts channel terms on the user's behalf or modifies
the source prefix. A passing bootstrap still requires the real GPU gates below.

Prepare a private JSON plan with independent `root`, isolated `python`, released
`model`/`processor`, `vggt_source`/`vggt_weights`, audited `input_root`, allocated
`gpus`, predecessor `dependencies`, and `jobs` containing `name`, `adapter` and
`train_vggt`. Set `deepspeed` to the snapshotted `configs/geometry-zero3.json`.
For read-only shared alignment, add the producer's `alignment_checkpoint`,
`alignment_receipt` and `alignment_failure_state` to the consuming job. Consult
`validate_plan` in [the runner](../scripts/run-geometry-matrix.py) for validation.

```bash
"$NODE_ROOT/envs/geometry-matrix/bin/python" scripts/run-geometry-matrix.py \
  --plan "$NODE_ROOT/.private/geometry-matrix-plan.json" --detach
```

The runner freezes its code and input manifests, writes `state/matrix.json` and
per-job receipts, and selects each stage's microbatch from measured diagnostics.
A prior independent job may be accepted as terminal `failed` only by explicit
private-plan policy; allocated GPUs must still be idle. Failure of shared
alignment is different: its consumer cannot invent replacement weights and must
block. A background PID or queue receipt is not a successful optimizer update.

### Opt-in encoder batching and checkpoint continuation

Private plans may set `encoder_batch_size: 4`,
`trainable_encoder_batching: true`, and `save_steps: 20`. The runner caps the
encoder batch at the selected microbatch: micro1 retains the original serial
path. Independent sequences are batched along B, never concatenated along time.
For distributed frozen **and** trainable VGGT, all ranks agree on contiguous
equal-length chunks before executing the trunk, preserving ZeRO-3 parameter
collective order. Nonzero stochastic dropout/drop-path rejects trainable batching;
the optimization never silently disables it. Existing armed snapshots stay intact.

Current acceptance evidence: six CPU tests cover order, gradient equivalence,
AdamW state continuation, and actual two-rank Gloo scheduling; 19 queue/resume
contract tests pass. A real eight-rank ZeRO-3 diagnostic with micro2/encoder2
and two-frame inputs completed two optimizer steps, updated geometry interface,
VGGT, native vision, and language parameters, and reloaded with zero maximum
logit difference. Its maximum reserved memory was 13.38 GiB. This is an interface
gate, **not** evidence of full-mixture acceleration or 32-frame safety. Each
allocation still profiles micro1/2/4 on the locked mixture and long samples,
checks reload/numerical gates, and selects the fastest candidate within the
explicit memory policy (92% by default). Record the policy in selection receipts.

`jobs[].resume_checkpoint` passes a full Trainer checkpoint to the SFT worker.
Continuation requires optimizer and RNG states, identical manifest bytes/order,
world size, effective batch, stage, seed, training budget and DeepSpeed engine
configuration. Only microbatch/GA, encoder execution policy and checkpoint cadence
may change. These checks and tiny CPU optimizer tests do not prove exact GPU
restart parity; retain that distinction in receipts. If no optimizer checkpoint
exists, restarting from the completed alignment is a **restart**, not a resume.
When a producer root changes, migrate downstream RL/evaluation dependencies to
the new version together; never leave consumers waiting on an abandoned producer.

## Evaluation, results and ETA

Use [paired spatial evaluation protocol](BATCH_AND_EVAL_PROTOCOL.md): fixed ReVSI
and VSI manifests, native 32-frame video, maximum image side448, answer-tag/512
tokens, greedy decoding, ground-truth-blind extraction. Format smoke is selected
by fixed IDs, not accuracy. Report strict/extracted scores, parse/truncation rates,
per-task results and paired sample records. Never relabel a blocked eval as 0%.

Report stage status separately: queued, gate-running, alignment-running,
SFT-running, evaluating, complete or blocked. Trainable-VGGT ETA is unknown until
real backward and optimizer timing are available. Use warmed measured samples/s,
remaining rows, checkpoint overhead and evaluation seconds/example; separate
queue wait, setup/profile, alignment, SFT and eval. Rough optimizer counts are
665 alignment + 776 SFT at the reference batch384; revised batch64 SFT has
about4655 updates. Loader tail handling is recorded.
Do not infer ETA solely from optimizer count because batch and geometry cost differ.

Keep final per-arm config, manifest identity, source versions, selected throughput
receipt, first healthy loss, checkpoint lineage, trainable parameter inventory,
GPU-hours and scores. No fabricated results or premature architecture conclusions.
