# Geometry RFT: an after-SFT paired experiment

This study asks whether adding Spatial reasoning examples to 4D-RL training
improves held-out spatial QA compared with the same RFT budget on 4D-RL alone.
Both arms start from **the exact same predeclared formal downsample/trainable-VGGT
SFT checkpoint**. Using different SFT architectures as the two starting points
would confound the data-mixture comparison. Do not select that checkpoint using
test scores, and do not replace it with diagnostic weights.

## Reference versus adaptation

[RoboRefer appendix D.4](https://arxiv.org/html/2506.04308v4#A4.SS4) reports two
epochs, one prompt per GPU and eight sampled outputs using GRPO. Its public
[repository](https://github.com/Zhoues/RoboRefer) still describes RFT code as coming
soon at the audited revision. This does not supply a ready-made Qwen3-VL/VGGT
trainer. Our models, datasets, global prompt batch, parameter-efficient updates,
rewards and backend are explicit adaptations, not exact official reproduction.

The earlier SFT protocol follows the released launch scripts, not every paper
detail: the paper describes different dataset mixtures between alignment and
full SFT and different learning rates from those script defaults. Keep those two
sources distinct; neither makes our SPAR/Hound substitution official RefSpatial.

The runtime here is a **custom HF geometry reference backend**. Existing vanilla
Qwen VERL acceptance does not demonstrate that VERL rollout understands our VGGT
branch, inserted tokens or checkpoint. Do not claim native VERL support without
a separately implemented and validated custom model path.

## Paired scientific contract

| Factor | Mixed arm | Control arm |
|---|---|---|
| Starting model | Same exact formal SFT checkpoint | Same |
| Prompt sources | 70% 4D-RL + 30% Spatial | 100% 4D-RL |
| Logical rollout group | 8 actual responses per prompt | Same |
| Global prompt batch | 16 | Same |
| Sampled prompt budget | `ceil(2*N/16)*16` | Same |
| Optimizer updates | `ceil(2*N/16)` | Same |
| Seed and evaluation IDs | Fixed shared contract | Same |

`N` is the audited count of all accepted training rows in the combined data
receipt. Both arms receive the same derived budget, even though the control has
fewer unique eligible examples. Thus the control may revisit its source more
often; do not label each arm as exactly two unique-data epochs. Record rounded
tail draws and actual source counts. Deterministic 7:3 allocation in blocks of
ten avoids stochastic mixture drift; the final partial block is reported. Within
each source, deterministic sampling is **with replacement**. Therefore this is
a two-pass-equivalent prompt budget, not guaranteed two-epoch unique coverage.

Logical group8 is independent of generation microbatch. Serial generation with
micro1/2/4/8 must still produce all eight responses before normalizing the
group's advantages. OOM must not silently change group8 to group4. Keep prompt
batch, update count, response cap, sampling parameters, reward definitions,
reference policy, KL/clipping, trainable scope and checkpoint schedule identical.
These runtime choices must be locked in the worker contract before acceptance.

The selected adapted worker recipe uses LR1e-6 with a constant schedule, KL0.02,
asymmetric clipping0.0003/0.0004, response cap512, context16384, temperature0.7,
top-p0.9 and checkpoint every25 updates. Language LoRA is rank64/alpha128 with
dropout0; geometry interface modules are trainable and saved in FP32. Native RGB
and VGGT are frozen at their **trained SFT weights**, not reset to released VGGT.
Actor/reference share the frozen SFT base; disabling adapters must also restore
the original SFT geometry interface. This requires a real policy/reference
parity gate, not an assumption based on PEFT installation.

Sampling the same number of prompts does not guarantee equal compute: task
lengths and generated response lengths differ. Report generated/valid tokens,
rollout and optimizer seconds, GPU-hours and per-source reward distributions.
One seed and a single source checkpoint support a bounded data-mixture
comparison, not a universal ranking of RL algorithms or geometry architectures.

## Data and runtime gates

The data receipt must declare `status=ready` (or `complete`), `ready_for_training=true`,
`accepted_train_rows`, `train_test_overlap=0`, named training `manifests`, and a
fixed `eval_manifest`, with `leakage_checked=true` and `media_verified=true`.
Missing annotations are a blocker, not permission to
invent process rewards. Keep ground truth out of policy inputs and reward-free
generation; only the reward function consumes the required training labels.

Before formal optimization, each arm must demonstrate real images/geometry,
initial policy-reference parity, eight actual sampled responses per prompt,
nonconstant within-group rewards, finite loss and a nonzero update, plus saved
policy reload. Adapter exports must preserve the declared geometry modules and
the fixed SFT base pointer. Optimizer/scheduler/RNG recovery is separate from
merely loading LoRA weights. Preserve native visual tokens, DeepStack/mRoPE and
the float32 rotary/preprocessing buffers validated in the SFT matrix.

The two arms must evaluate the same fixed IDs, frame sequences, prompts, decoding
budget and parser. Include the shared SFT initialization as a baseline. Report
per-task and aggregate accuracy, format validity, truncation, raw predictions
and paired outcomes. A training reward improvement alone does not prove held-out
reasoning improvement. A null/incorrect geometry diagnostic is distribution
shift evidence, not a causal proof of geometry reasoning by itself.

Keep metric semantics separate: DSR uses the declared exact-answer/per-task
accuracy; Spatial numeric tasks may use a graded answer reward. A mean graded
reward is not an exact accuracy or an official benchmark score. Existing
ReVSI/VSI strict/extracted metrics apply only when those benchmarks are actually
included with their established scorer, not as names for arbitrary QA rewards.
Source-held-out manifests are also snapshotted as `validation_4drl` and
`validation_spatialladder`, separately from external benchmarks. Their IDs and
contents are locked in the paired contract; they do not increase the train budget.

## Durable queue and versioning

Use an independent root/environment allocation; never interrupt existing SFT or
its evaluation. The queue requires all declared predecessor pipelines to report
terminal status **and** `gpu_work_finished=true`, then separately checks allocated
GPU memory is idle. Both arms may be queued on separate nodes only after those
nodes' existing work ends; their shared SFT producer may be elsewhere.

A private plan contains `root`, `python`, `model_checkpoint`, `processor`,
`vggt_source`, `sft_receipt`,
`data_receipt`, `scientific_config`, `pair_id`, `gpus`, `dependencies` and `jobs`.
Jobs are a subset of `mixed` / `four_d_only`; they cannot override initialization
or budgets. Real server mappings never enter public source. Example entry:

Both deployed plans also name the same `pair_contract_path` on shared storage.
Under a file lock, the first ready queue records the scientific recipe, source
data receipt, ordered train/evaluation IDs, source checkpoint identity and budget;
the peer must match exactly. Local output roots, Python paths, allocated GPU IDs
and copied-manifest paths are excluded from that equality check. Source manifest
counts are checked against the accepted data count before budget resolution.

```bash
"$NODE_ROOT/envs/geometry-matrix/bin/python" scripts/run-geometry-rft-queue.py \
  --plan "$NODE_ROOT/.private/geometry-rft-plan.json" --detach
```

The worker must already exist before arming the queue. Code and the scientific
recipe are snapshotted immediately; accepted data manifests and source evidence
are snapshotted when ready. `runtime-plan.json` stores the resolved budget and
exact input paths. Changes require a new versioned queue, not a hot edit.
The source checkpoint remains read-only, with file size/mtime identity checks;
no new weight checksum requirement is imposed.

Worker interface:

```text
train-geometry-rft.py --plan runtime-plan.json --arm mixed --mode gate
train-geometry-rft.py --plan runtime-plan.json --arm mixed --mode train
train-geometry-rft.py --plan runtime-plan.json --arm mixed --mode evaluate
```

Each mode writes `runs/ARM/MODE/completion.json`. Gate failure prevents expensive
formal work for that arm, while a separately allocated peer remains independent.
The queue preserves failed artifacts, uses a process lock, and refuses duplicate
live children. A queued supervisor PID is not a rollout/update acceptance.

## ETA and reporting

SFT samples/second are not RFT throughput. First measure actual generated tokens,
eight-response rollout time, reference/old-policy log-probability time, optimizer
time, checkpoint time and reload/evaluation overhead on real mixed examples.
Derive remaining time from warmed complete RL updates and the fixed prompt
budget; report the generation-length distribution and uncertainty. Separate
waiting-for-SFT, data preparation, runtime gates, formal RFT and evaluation.
Until those measurements exist, RFT ETA is **unmeasured**, not inherited from SFT.

Scientific settings: [geometry-rft.json](../configs/geometry-rft.json).
Executed checks and remaining GPU gates: [acceptance record](GEOMETRY_RFT_VALIDATION.md).
Validated environment constraints: [geometry-rft.txt](../requirements/geometry-rft.txt).
Prior SFT contract: [geometry matrix](GEOMETRY_MATRIX_RUNBOOK.md).
Paired spatial evaluation: [protocol](BATCH_AND_EVAL_PROTOCOL.md).
