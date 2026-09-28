# Recovering a distributed rollout OOM

A successful initial rollout profile does not prove that every later mixed-data
prompt fits. Sequence length, visual grids, generation length and resident gradient
buffers can differ. GPU utilization can remain high while healthy ranks wait for an
OOM rank in NCCL; verify new optimizer metrics, not just utilization or process IDs.

Runtime-only recovery options in the private RFT plan:

- `rollout_micro_cap: 4`: profile only micro1/2/4, preserving the logical group of8.
  The training worker refuses a gate receipt selecting a larger microbatch.
- `rollout_probe_indices: [...]`: include deterministic training-sampler indices
  implicated in a failure. Compare their processed prompt length with the existing
  long-example probe; record IDs and selected prompt length in the gate receipt.
  These are training inputs, not benchmark examples or silently excluded samples.

The group size, global prompt batch, source mixture, answer/reward policy, learning
rate and update budget stay unchanged. Sampling in different microbatches can
consume randomness differently; do not claim bitwise continuation equivalence.
Without a complete model/optimizer/RNG checkpoint, restart from the original SFT
model in a fresh study root and explicitly report discarded optimizer steps.
Preserve failed logs and manifests, and update dependent reward-ablation waiters
to reference the replacement run without restarting a healthy comparison arm.

The supervisor monitors newly appended training logs for CUDA OOM and NCCL
watchdog failure. It terminates only that worker's process group, records failure,
and does not label the run complete. This prevents a known fatal GPU error from
leaving peers spinning until the long overall training timeout. Gate failures and
scientific validation checks are not bypassed.

Acceptance has three separate levels: CPU regression checks; real gate updates
and checkpoint reload; formal training progressing past the original failing
sampler indices. Report each level only after observing its evidence. A passed
finite set of long examples still does not guarantee no later OOM.
