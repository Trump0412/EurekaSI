# Full-language RFT reference precision

Full-language RFT keeps trainable language/interface parameters in FP32. A
separately loaded BF16 SFT reference is not arithmetically equivalent, even
under BF16 autocast: normalization and residual paths can retain different
dtypes. This caused the initial policy/reference gate to fail before updates.

`match_reference_precision` promotes the fixed reference parameters to the
actor's corresponding dtypes without copying actor values. The reference stays
frozen, independent of trainable actor parameters, and initialized from the
same accepted SFT checkpoint. Shared native RGB/VGGT trunks remain frozen.
Rounding to lower precision is rejected. The existing tolerance is unchanged.

Real-image isolation evidence: maximum response log-probability discrepancy
was 0.097757 before dtype matching and 0 afterwards on the same prompt/tokens.
The repaired eight-rank mixed-arm initialization also measured zero discrepancy
on every rank. These checks establish initial parity only; grouped rollouts,
nonzero updates, reload and formal training require their own receipts.

Each rank now writes `initial-parity-rankN.json`, including measured difference,
resume status and tolerances. A failed initialization must not be reported as
RL training. Fresh repairs use versioned queues and retain previous failures.

When bringing up paired nodes on shared storage, finish publishing and checking
the first queue's paired manifests before starting the peer. A simultaneous
initialization observed a transient partial-copy mismatch on the deployment
filesystem. All eight completed manifest copies subsequently matched byte for
byte; the failed peer was retained and retried only after this check. Do not
disable byte-content validation or assume cross-node file locks suffice on
every shared-storage implementation.

## No-grad rollout and activation-checkpoint recomputation

After fixing initial parity, the real first backward exposed a second error:
autocast weight casts created under no-grad were reused within a surrounding
autocast scope, while non-reentrant checkpoint recomputation saved a different
autograd graph. The resulting `CheckpointError` reported tensor metadata shifts.
The worker now uses `rft_autocast(cache_enabled=False)` throughout. BF16 arithmetic,
activation checkpointing, rewards and the loss are retained; metadata validation
is not disabled. A real fixed-prompt probe reproduced the error with caching and
passed without it (loss 3.939107, query-projection gradient norm 2.060631). A CPU
regression covers no-grad inference followed by checkpointed backward in one
autocast scope. Sixteen focused CPU tests passed. Distributed optimizer and
reload acceptance are still separate checks.

## User-authorized independent answer reward

The strict-v2 gate produced 256 responses per arm but no reward variation: the
SFT model generated bare letters/numbers, which were rejected for missing tags.
The newly authorized `geopsro-independent-answer-v3` accepts only an unambiguous
whole-response scalar for correctness. Malformed wrappers, prose, multiple
answers, invalid choices and truncated generations still fail. Parsing does not
see gold answers. Bare answers receive zero structure and lexical reward.
Prompts, the original lexical reward, G=8 and training budget remain unchanged.
Old versions retain their original behavior. Both arms and subsequent reward
ablations must use the same version, not mix strict-v2 with independent-v3.

DSR comparison uses the fixed 1,450 MCQ test examples, exact answer accuracy and
per-task breakdowns for the common SFT initialization and both final RFT models.
Training rollout reward is not a DSR score. At this revision no completed DSR
baseline/post-RFT comparison exists; earlier ReVSI scores do not substitute for it.

The repaired eight-rank gate subsequently passed in both arms: two effective
updates, language/interface changes, unchanged fixed reference and independent
reload with zero maximum log-probability difference. Mixed/control had 22/24
variable-reward groups out of 32 respectively. These are **training diagnostics**,
not held-out DSR improvements. Both formal queues then entered the 5,885-update
budget from the original shared SFT initialization, not from diagnostic weights.
