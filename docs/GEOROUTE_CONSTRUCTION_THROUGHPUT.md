# Frozen VGGT graph construction throughput

## Measured bottleneck (2026-09-28)

Short sustained sampling on an eight-GPU allocation found mean GPU utilization
of 13--47%, despite live graph workers. An isolated real-input probe identified
the pinned VGGT tracker's repeated CPU sinusoidal position-grid construction and
device transfers inside every refinement iteration. These grids depend on shape,
not query coordinates, answers, or model updates. Repeating the track head's DPT
feature extraction was measurable but contributed little to the hot runtime.

The opt-in `GraphCache(..., cache_tracking_positions=True)` uses a scoped,
bounded cache of the original CPU-generated values on the frozen teacher device.
It preserves graph keys, original arithmetic, query chunks, every source anchor,
frame order, thresholds and refinement iterations. It restores the tracker module
binding even on exceptions and rejects trainable/training-mode teachers.
No upstream source or weight file is modified. Do not apply it to trainable VGGT.

`prepare-tip-support.py --cache-tracking-positions` records execution policy
separately from semantic shard contracts, permitting continuation of the same
rank/world shards without discarding audited rows. Existing graph cache remains
valid. Active snapshots must never be hot edited; deploy a new controller/source.

## Evidence and limits

- Two real two-frame examples: combined feature/position reuse reduced hot graph
  construction to 1.25--1.58 seconds versus 10.9--12.8 seconds. Edge endpoints,
  weights, frame IDs and sample IDs were exactly equal, including a zero-edge case.
- The production position-only helper on one real eight-frame example reduced
  57.49 seconds to 6.78 seconds; all 2724 edges and weights were exactly equal.
- One actual 32-frame example completed optimized graph construction in 103.97
  seconds with 11.07 GiB peak reserved and 9467 valid edges. This is a memory and
  graph-validity probe, not a paired 32-frame exact-parity measurement.
- Original implementation, same two two-frame samples and four graphs/worker:
  one/two/three same-GPU instances yielded 0.0998/0.1965/0.2655 graphs/second.
  Concurrent graph outputs exactly matched the single-instance graphs.
- Each worker's short-input allocator peak was 5.22 GiB. This is not a long-input
  capacity guarantee. A suspended production worker retained its allocation on
  the profiling GPU; other GPUs continued processing. These are bounded probes,
  not full-dataset throughput or an end-to-end training speedup claim.
- CPU regression: 13 passed, 1 skipped (the optional real-processor test lacked
  its environment inputs). GPU graph probes provide separate evidence.

Do not multiply the position-cache and multi-instance speedups without measuring
them together: removing CPU overhead can make GPU compute the limiting resource.
Likewise eight GPUs were already in use; an eightfold node multiplier cannot be
counted again. Validate longest sequences and warmed dataset-wide throughput
before increasing workers/GPU. Report preparation, TIP, SFT and evaluation ETA
separately. Geometry-only deduplication may reuse identical ordered media, but
must never remove QA rows from instruction SFT or alter frame sampling.

A read-only full-manifest inventory counted 1,321,623 QA rows, including 935,262
multi-frame rows but only 207,912 unique ordered multi-frame media groups.
748,751 rows have 32 frames. Consequently short-input probes cannot predict the
full preparation ETA. Group-level scheduling and preventing simultaneous cold
cache misses for the same group are promising next optimizations; existing
completed-cache reuse already captures part of this opportunity.
