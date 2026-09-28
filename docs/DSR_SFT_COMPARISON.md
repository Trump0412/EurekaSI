# Four SFT checkpoints on DSR-Bench

This queue evaluates preserved direct fusion, downsample/frozen, downsample/trainable,
and query64/trainable checkpoints. It does not retrain them or interrupt current GPU
owners. Private plans hold real paths and allocations; they must not be published.

The independent supervisor waits for the entire declared predecessor queue to
release GPU work and for its assigned GPUs to be idle. A terminal predecessor
with unrelated failures may release the allocation, but each SFT checkpoint still
requires its own successful training receipt. A blocked/waiting predecessor is
**not** terminal and is reported explicitly. One arm's failure is recorded and does
not suppress other independent arms.

```bash
python scripts/run-dsr-sft-queue.py --plan "$PRIVATE_PLAN" --detach
```

The plan supplies `root`, `python`, `gpus`, `source_manifest`, immutable local
`manifest` (`$STUDY/inputs/dsr.test.jsonl`), `processor`, `vggt_source`,
`vggt_weights`, `dependencies`, and four `arms`. Each arm has `name`, `kind`
(`legacy`, `downsample`, or `query64`), `checkpoint`, and `completion`.

## Locked comparison

- One identical MCQ manifest and its native frame order/FPS; no silent subsampling.
- Same structured spatial prompt as the RFT comparison, BF16, maximum RGB side448,
  greedy decoding, 512 new tokens, context16384. This is an **adapted protocol**,
  not a claim of official leaderboard equivalence. Do not mix it with older
  answer-only prompt scores.
- Answer accuracy uses `geopsro-independent-answer-v3`'s answer component only;
  format and vocabulary bonuses are **not** benchmark points. Gold never enters
  inference inputs. Truncated/invalid answers are recorded, not guessed.
- First fixed row per task plus the longest sequence forms the smoke gate. Only
  parse rate (at least90%) and truncation (at most5%) determine gate acceptance,
  not correctness. A failed gate blocks that arm's full evaluation.
- The legacy checkpoint uses its original final-patch interface and released
  frozen VGGT. Matrix checkpoints load their own embedded VGGT state. Features
  are computed online with no large persistent DSR VGGT cache.
- The direct baseline has a different stage count/native vision freezing and
  dropout history. Its comparison is not a single-factor causal ablation.

## Evidence and recovery

`plan.json`, `code/` and `inputs/dsr.test.jsonl` preserve configuration/code/data.
Every arm writes `runs/<arm>/{smoke,full}/contract.rank*.json`,
`predictions.rank*.jsonl`, `complete.rank*.json`, `metrics.json`, and
`completion.json`. Predictions contain raw/decoded text, token IDs, gold/parsed
answers, scores, truncation, timing, and peak reserved GPU memory. Contracts contain
ordered IDs, frame identities/timestamps, prompt, model metadata and software versions.

The merger refuses missing/duplicate/foreign IDs. `comparison.json` includes per-arm
scores and paired changed-correctness IDs for all successful pairs. A full run is
complete only when all four arms have accepted full receipts, not when a PID exists.
Interrupted JSONL rows are deliberately not silently discarded: preserve evidence,
repair a partial final line explicitly before resuming. Changed inference contracts
require a fresh output root. These CPU tests do not substitute for the deferred
real-GPU smoke on each model.

Historical ReVSI/VSI results remain in their original roots. Inventory should list
actual metrics, per-rank contracts, completion receipts and unique prediction counts.
Do not equate a stored failed-run log, partial predictions, or smoke result with a
completed benchmark. Shared server storage is not an independent backup.
