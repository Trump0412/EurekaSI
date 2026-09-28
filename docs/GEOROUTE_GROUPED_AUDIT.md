# Unique-sequence TIP audit

`scripts/prepare-tip-support-grouped.py` replaces static QA-row sharding for
**preparation only**. It does not change SFT examples, epoch/batch settings,
geometry rules, source anchors, query chunks or VGGT weights.

```bash
python scripts/prepare-tip-support-grouped.py \
  --plan "$PLAN" --output "$AUDIT_OUTPUT" \
  --previous-audit "$PRIOR_AUDIT_OUTPUT" --gpus 8 --workers-per-gpu 3
```

- A single parent indexes all question IDs and ordered media tuples in SQLite.
  Different questions over identical ordered media share a geometry result;
  reversed frame order is a different group. Every QA row is retained.
- Single-frame rows receive the original `single_frame` TIP exclusion without
  GPU work. This never excludes those rows from instruction SFT.
- Previous per-rank audit records can be imported after checking semantic
  contracts. Conflicting results or unknown/duplicate IDs fail closed.
- Only the parent writes the database and assigns each pending group once.
  Workers own frozen VGGT instances and use the position-grid execution cache.
  Persisted completed groups survive restart; interrupted pending groups retry.
- A runtime CUDA OOM retires that worker and requeues the intact group, reducing
  concurrency on its GPU. The last worker can be replaced; repeated group OOM
  fails visibly rather than reducing frames or dropping samples. Initialization
  failures also fail visibly; this is not a guarantee of recovery from every fault.
- Group results expand to the original question IDs at completion. No completion
  receipt is published before all groups finish and child workers release GPUs.
- Parent progress is `grouped-progress.json`, per-worker phase/memory is
  `worker*.json`, timings are `grouped-metrics.jsonl`, and OOM retirements are
  `oom-recovery.jsonl`. Report unique groups/s, not QA rows/s as teacher throughput.

One eight-GPU deployment indexes 1,321,623 QA rows into 426,948 total media
groups, of which 207,912 are multi-frame. Completed graph cache reuse already
captures part of this reduction; do not claim an additional 4.5x without timing.
Three workers/GPU is a memory-envelope trial, not an asserted optimum. Measure
32-frame concurrency and sustained throughput before claiming a speedup.

Keep active source/plan snapshots immutable. Preparation can continue writing
the prior accepted output location under a new exclusive controller after the
old owner and its workers stop. Never run two audit owners over the same output.
Graph caches and prior JSONL evidence are retained, not deleted or rewritten.

Deployment acceptance (2026-09-28): 24 ready workers on eight 40-GiB GPUs,
58 newly committed unique groups, no OOM retirements. A 20-second sample measured
99.4--100% mean GPU utilization and 33.05--33.92 GiB resident memory/card.
These are resource/first-output checks, not stable full-dataset throughput or a
formal TIP/SFT result. Four simultaneous 32-frame workers are not validated and
their individual measured memory peaks would exceed the physical budget.
