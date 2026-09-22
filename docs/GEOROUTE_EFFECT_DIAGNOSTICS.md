# GeoRoute: measure transmission, not just a nonzero alpha

The optional audit compares **one fixed checkpoint, fixed prompt tensors and
fixed correspondence graph** at strength `g=0` and `g=1`. It does not train,
change saved alpha values, build VGGT graphs or claim higher task accuracy.

```bash
python scripts/audit-georoute-effect.py \
  --checkpoint /models/georoute-final \
  --bundle /evidence/trusted-prompt-and-graph.pt \
  --output /evidence/effect-float32.json
```

The trusted local tensor-only bundle is `{'inputs': tensor_dict, 'graph':
RouteGraph_field_dict}`. Inputs contain a prompt, image tensors/grid and masks,
not labels, cached keys or a generated answer. CPU/float32 is the default.
Explicit `--device cuda:0 --dtype bfloat16` measures the intended low-precision
runtime, but must only be run on an allocated free GPU. No inference accuracy
comparison should mix float32 and BF16 runs. Do not load untrusted pickle files.

At each of four exits the audit brackets existing routing hooks and records:

- clean input before the routing pre-hook;
- routed input to the native merger;
- merger output before any post-merger route;
- final exit output;
- next-token logit changes, cosine, relative L2, distribution KL and argmax;
- graph coverage and each block's alpha and actual applied residual norm.

These stages also distinguish pre-merger and post-merger variants. A post-merger
variant legitimately has zero pre-merger intervention; inspect `post_exit`
instead. Different input/output widths and normalizations mean the ratio of
relative deltas is **not** an information-retention estimate. A nonzero downstream
logit delta is evidence of influence, not evidence of useful influence. Follow
with paired held-out predictions before any capability claim.

Optional `--gradient-target-token N` measures route-gradient connectivity for an
explicit next-token target, without an optimizer update. It is not a real task
loss unless the target and prompt were chosen accordingly. Diagnostic outputs
contain scalar summaries, not retained activation tensors. Temporary CPU
captures are released after the report. Telemetry synchronizes and is not a
timing benchmark.

## Zero alpha and BF16

At alpha zero, the original residual is exactly zero. Alpha can receive a
gradient while down/up projections receive zero gradient on that first step;
once alpha updates, projection gradients can become nonzero. This is intentional
gated initialization, not automatically a dead branch. Cancellation, unsupported
graphs and degenerate features can nevertheless produce zero gate gradients.

In BF16, a small nonzero residual can round away when added to a much larger
hidden value. Autograd can still return a nonzero local gradient through the
addition: identical forward values alone do not prove a permanently dead branch.
The audit therefore measures **actual output differences**, gradients separately
when requested, and float32/BF16 in separate same-checkpoint runs. No production
alpha initialization has been enlarged to force a visible effect.

Synthetic tests cover a projection that annihilates a perturbation, BF16 rounded
addition with a surviving gradient, and real tiny-Qwen pre/post-merger hooks.
These validate the measuring tool, not the final trained checkpoint.

Validation on 2026-09-22: an isolated CPU-only review snapshot passed 36 combined
effect-audit, STB, tiny-Qwen integration and capacity tests in 43.41 seconds.
The capacity suite was subsequently extended to independent spawned processes
and passed 7 tests in 2.32 seconds on the actual shared storage mount. These
overlapping counts must not be added as independent tests. No production GPU
or armed training snapshot was changed by these checks.

## Graph-cache budget and limitations

Before this change GraphCache had no capacity limit, eviction or free-space gate.
Graphs use media order/path/stat plus teacher and graph configuration as identity;
questions sharing exactly the same permitted image sequence reuse a graph.
At patch16/448, `N=784*T`. With top-k 8, int64 source/destination and frame/sample
IDs and float32 edge weights, the dense upper bound is approximately `176*N`
bytes before serialization overhead. For 8/16/32 frames this is roughly
1.05/2.11/4.21 MiB per unique sequence. 100K distinct 32-frame sequences could
therefore approach **411 GiB**, not a guarantee of actual storage usage.

The new writer defaults to **512 GiB graph-payload budget** and **100 GiB minimum
filesystem free space after publication**. Configure `cache_max_bytes` and
`cache_min_free_bytes` in the graph-cache settings. Storage limits are removed
from semantic graph identity, so quota changes neither invalidate graphs nor
create duplicate keys. Existing cache hits remain readable even above a newly
lowered quota. No images, frames, correspondences or teachers are reduced.

An interprocess file lock serializes publication. SQLite indexes existing graph
and orphan temporary payloads once, then maintains a constant-time byte total.
Reservations commit before writing, conservatively counting a crashed writer.
An existing reserved temporary payload requires explicit operator recovery; no
automatic removal or duplicate rewrite occurs. Capacity, locking, readonly or
filesystem errors fail closed. No cache eviction is implemented.

The storage root must support functioning advisory locks and SQLite filesystem
semantics. Test on the actual target mount before enabling full training. The
quota does not reserve all other applications' filesystem writes and excludes
small index/lock overhead. Unmanaged external cache writers or manual deletions
invalidate index assumptions; do not mix them with the managed writer. A changed
quota is not permission to delete prior evidence. These additions are local code
preparation, not changes to an armed production snapshot.
