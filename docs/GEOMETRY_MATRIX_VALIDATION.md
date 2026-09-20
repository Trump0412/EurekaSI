# Geometry matrix validation record

This is an engineering acceptance record, not benchmark results. Formal arms
and their comparison limits are defined in [the matrix runbook](GEOMETRY_MATRIX_RUNBOOK.md).

## Checks performed during implementation

- Static Python/configuration/link checks passed. These do not prove GPU execution.
- The isolated CPU environment passed 39 focused tests covering interfaces,
  spatial merging, variable slots, parameter ownership, gradients, generation,
  checkpoint round trips and queue behavior. After adding native rotary-buffer
  protection, the six integration tests passed again, including the new dtype
  invariant regression. The queue's 15 tests also passed in a second isolated
  Linux environment. These counts overlap; do not add them as unique tests.
- Real short-image alignment produced finite losses, interface updates and a
  saved full-model checkpoint for both interface families.
- A real four-rank A100 40 GiB, 32-frame, ZeRO-3 SFT diagnostic completed two
  updates with finite loss and intended component updates. Diagnostic memory
  observations were approximately 29–32 GiB per rank. These were not a warmed,
  representative throughput benchmark and do not establish a whole-run ETA.

## Reload correctness findings

1. VGGT's upstream constructor uses operations incompatible with a meta device.
   The registered aggregator is constructed on CPU before loading its weights.
2. VGGT's nonpersistent RGB normalization buffers were not restored by the
   low-memory checkpoint loading path. This caused invalid features despite
   finite learned weights. Only the fixed upstream constants are restored;
   learned parameters are never reset to released weights during reload.
3. DeepSpeed input preparation can cast RGB pixels before VGGT normalization.
   The matrix trainer now preserves geometry pixels as float32, with explicit
   bf16 autocast for the VGGT projection/attention path.
4. Casting the entire model to bf16 also rounded Qwen's nonpersistent rotary
   frequencies, while checkpoint reload reconstructed float32 frequencies.
   A controlled diagnostic held learned weights fixed: matching only this
   precision condition changed maximum logit difference from 0.375 to 0.
   The new implementation preserves native float32 frequencies across device
   and dtype transformations instead of reproducing the rounded-frequency path.
5. ZeRO-3 with the frozen language model failed activation recomputation when
   a saved tensor was represented by an empty parameter partition. Alignment
   now uses DDP; trainable joint SFT retains ZeRO-3. This is an explicit runtime
   adaptation, not a change to data, global batch, learning rate or epoch count.

A fresh four-GPU DDP alignment diagnostic on 128 fixed real samples (1,093
frames) completed two updates, with finite losses 3.3893 and 3.4393 and interface
updates. Its independently reloaded checkpoint had maximum logit difference 0.
This validates this diagnostic's round trip, not all future checkpoints.

The earlier diagnostic checkpoints are not research initialization checkpoints.
Formal alignment starts again from the released base. New reload evidence records
pixel dtype, normalization values and rotary buffers and checks them explicitly.
Every queued stage still requires its own real save/reload acceptance; this
document is not a substitute for the immutable run's receipts.

## Throughput and interpretation

The portable balanced sampler is reused, not Qwen3.5-specific DeltaNet kernels.
Microbatch 1/2/4 candidates require actual mixed-data timing, longest-sample
pressure checks and, for larger batches, loss/full-gradient numerical comparison.
Report queue wait, installation/compilation, profiling, alignment, SFT and
evaluation separately. No final QA score or general speedup is claimed here.

An initial alignment-only extrapolation on four A100 40 GiB GPUs at microbatch 1
was about 15 hours for 297,899 rows: the second update processed 64 samples in
11.576 seconds (5.529 samples/s), with approximately 9.90 GiB peak reserved
memory. This uses only one post-cold-start update and is a rough diagnostic
estimate, not a full two-stage ETA or the final selected throughput. Larger
microbatches still require the declared gates.

A subsequent four-GPU trainable-VGGT ZeRO-3 SFT diagnostic on that mixed sample
set completed two optimizer updates with all four intended components updated.
Its second update took 79.7438 seconds for 64 samples (0.80264 samples/s), with
34.379 GiB maximum reserved memory. A microbatch-1 extrapolation is about 103
hours for SFT alone, or 118 hours including the rough alignment estimate, before
evaluation and overhead. This is one warmed update, not a selected final ETA.
The much slower joint stage must not inherit the alignment-only estimate.
Larger microbatches and alternative distributed runtimes require their own
timing, long-sample memory and checkpoint acceptance before adoption.

Private deployment mappings, machine paths and raw logs remain outside Git.
