# GeoRoute reload precision gate

The real ZeRO-3 two-step SFT gate exposed a nonpersistent vision RoPE mismatch:
the live visual `inv_freq` buffer was BF16, while independent reload rebuilt
native FP32 frequencies. Language RoPE buffers matched. All 633 exported weight
tensors matched the optimizer's reconstructed FP32 master weights cast to BF16,
so this was not missing or corrupted checkpoint parameters. Autocast on/off did
not remove the observed maximum logit difference of 0.3125.

Root-level `_apply` protection alone did not protect the actual ZeRO-aware
loading/submodule-casting path. `FP32VisionRotaryEmbedding` restores the original
native frequency formula in FP32 before use when its buffer has been cast.
It retains the native dimensions, theta and positional convention. It does not
load a rounded training buffer into inference or increase comparison tolerances.

New training evidence records rotary buffers and parameter dtypes. Fresh-process
verification compares rotary buffers after forward and checks logits with the
existing tolerances. The direct-submodule BF16 regression test requires exact
FP32-frequency recovery. The repair still requires actual multi-GPU training,
save/reload and generation acceptance; CPU tests alone are not sufficient.

Failed diagnostic checkpoints remain diagnostic evidence, not initializers for
formal training. This finding does not by itself invalidate unrelated benchmark
results or establish their cause; other model paths require independent checks.

The repaired real eight-rank mixed-input gate completed two TIP updates and two
ZeRO-3 SFT updates, followed by accepted independent reload and generation for
both stages. SFT diagnostic losses were 2.406 and 2.393. This validates the reload
repair, not the entire runtime gate or formal training.

The subsequent influence check exposed a separate limitation: after this very
short zero-initialized warmup, learned gates were approximately 1.4e-5 and -5.4e-6.
Pre-merger changes were measurable but disappeared at BF16 merger outputs;
next-token logits were identical with routing enabled/disabled. The check failed
closed. TIP diagnostics now allow a longer 32-update warmup before measuring
influence; the formal learning rate, initialization and budget are unchanged.
The longer mixed-input diagnostic passed: 32 TIP updates plus two SFT updates,
independent reload/generation, and nonzero BF16 influence. Learned gates were
approximately 1.62e-4 and -1.59e-4. The final merger relative delta was 7.93e-5;
the output-logit relative delta was 0.01286. The weak-effect warning remains.
These single-prompt numerical differences are not accuracy improvements and do
not establish a linear amplification ratio through the nonlinear merger/decoder.
Actual 32-frame pressure acceptance and formal preparation remain separate,
pending stages. These diagnostic weights never initialize formal training.

## 32-frame pressure correction (2026-09-26)

The later 32-frame TIP diagnostic completed 32 updates with finite loss but
failed independent logit reload comparison. Its 633 final/checkpoint weight
tensors matched exactly; this was not evidence of a corrupted weight export.
On its actual graph (25,088 nodes, 9,467 edges), a bounded CUDA operator probe
found different BF16 `index_add_` outputs on every one of 20 repeats, with a
maximum difference of 0.015625. This establishes aggregation nondeterminism,
not yet the sole cause of the full-model reload failure.

New plans explicitly select `segment_fp32_v1`: stable destination ordering,
FP32 accumulation for low-precision messages, and unique destination writes.
Legacy checkpoints retain `legacy_atomic` by default, so existing evaluations
are not silently changed. The new operator had zero changed output elements
across 20 GPU repeats, finite/nonzero gradients, and a 0.161 GiB peak allocation.
The focused CPU suite passed 25 tests. This is operator acceptance only.

Training evidence now records CUDA arithmetic flags and reload restores them;
legacy evidence without this contract must be regenerated, not retroactively
accepted. The tested language RoPE operation had zero TF32 on/off difference,
so TF32 alone is not an established cause. Logit tolerances remain unchanged
(`atol=0.02`, `rtol=0.01`); a separate comparison receipt records actual deltas.

The formal queue also now includes TIP support preparation before TIP training.
Waiting controllers use fresh immutable plans with current predecessor receipts.
Full mixed-input and 32-frame model training/reload/effect gates still must pass
before formal training. Paper-level SITE/independent correspondence evidence
remains a separate requirement; these fixes do not establish paper reproduction
or improved benchmark accuracy.
