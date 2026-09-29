# GSPO dynamic sampling: prepared, not armed

## Historical full-run audit (2026-09-29)

Each run contains 94,160 groups, G=8, 753,280 responses and 5,885 updates.
Temperature=0.7, top_p=0.9, top_k=0, response cap=512, context cap=16,384.
No explicit minimum response length was set. Counts include generated EOS.

| Run | Constant-total-reward groups | Actual response tokens | Mean length | Truncated | Positive structure / words |
|---|---:|---:|---:|---:|---:|
| Mixed | 78,144 (82.99%) | 1,648,723 | 2.189 | 2 | 0 / 0 |
| 4D-only | 85,073 (90.35%) | 1,751,848 | 2.326 | 325 | 0 / 0 |

The theoretical response ceiling was 385,679,360 tokens per run, not actual
compute usage and not total input/visual/logprob training tokens. Raising that
ceiling alone cannot fix predominantly two-token direct answers. Temperature
may improve exploration; it is not evidence of learning structured reasoning.

## Mechanism and limitations

The worker optionally rejects groups with max(total reward)-min(total reward)
<=1e-8 BEFORE old/reference log probabilities and backward. Rejected prompts
remain in the source pool. Replacements are sampled with replacement from the
same source as the accepted slot, maintaining the original 7:3 source schedule.
Future steps generate new responses with the updated actor; no stale responses
are reused. Each update still contains 16 accepted prompts and eight responses
per prompt. A bounded retry failure stops all ranks before optimizer.step;
it never silently trains a partial batch. Candidate logs retain rejected outputs.

Accepted rollouts remain in rollouts.rank*.jsonl; all attempts are separately in
candidates.rank*.jsonl. Metrics/checkpoints retain cumulative candidate groups,
discarded groups and candidate generated tokens. The ordinary generated_tokens
field counts accepted responses only. Resume restores cumulative counters at
complete checkpoints; attempt rows beyond the last checkpoint may repeat after
recovery and must be deduplicated by (step,prompt_index,attempt), keeping the
latest completed attempt. No claim of exactly-once logging is made.

The original sequence-level GSPO objective and KL remain. Discarded groups lose
their KL contribution too, even though their reward advantage was already zero.
This borrows DAPO dynamic sampling, NOT its token-level loss, clipping settings,
KL removal or overlong reward shaping. At historical acceptance rates, filling
a batch would require roughly 5.9x/10.4x candidate groups under a stationary
approximation; extra sampling is not free. Accepted-update matching is not
token-budget matching and no longer means two passes through the data.

## Next gate

Apply configs/rft-dynamic-sampling-pilot.json as a training override in a NEW,
unarmed plan after the separately authorized annotation/cold-start workflow.
It proposes T=1, top_p=.95, cap=1024 and minimum=0. Keep source split, prompt,
reward and G=8 fixed. On a fixed training-only diagnostic set, first compare
T=.7 vs 1 at identical cap/top_p; then separately test the length cap. A minimum
of 64 can be a diagnostic arm, not an assumed improvement: it prevents EOS,
not filler, and must not become a reward for verbosity. Compare correctness,
format/words hit rates, component-induced advantage, diversity, truncation,
accepted groups/sec, candidate tokens, peak memory and GPU-hours.

The 64-attempt limit is a safety stop, not a guarantee of quota completion.
Accepted prompts currently re-encode geometry after candidate selection; this
is correct but adds overhead. Input timing includes dynamic selection costs.
Real four-rank generation/update/reload and long-response memory acceptance
remain required. Existing SFT, legacy recipes and immutable run snapshots are
not changed. Annotation is not authorized by this preparation.

Validation on 2026-09-29: static parser/link checks passed; isolated CPU tests
for dynamic sampling, worker sampling, full-policy, queue, reward activity and
cold-start returned 55 passed. This is not GPU/distributed runtime acceptance.

References: [GSPO](https://arxiv.org/abs/2507.18071),
[DAPO dynamic sampling](https://arxiv.org/html/2503.14476v2#S3.SS2).
