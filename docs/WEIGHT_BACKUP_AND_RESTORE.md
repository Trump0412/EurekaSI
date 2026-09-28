# Weight and evidence backup / restore

This repository contains source code, public recipes and tests, not model weights,
private infrastructure mappings, account cookies or raw deployment logs.
An uploaded source tree is not proof that every experiment has completed.

## What must be retained together

| Experiment family | Inference dependencies | Additional resume dependencies |
|---|---|---|
| Native RGB SFT | Complete final model, tokenizer and processor | Trainer model/optimizer/scheduler/state and per-rank RNG |
| Geometry SFT (direct, downsample frozen/trainable, query64) | Complete final model, trained interface, matching native vision/VGGT, processor, architecture configuration | Full stage checkpoint; stage, manifest/order, effective batch and optimizer schedule |
| Geometry RFT (mixed, 4D-only and reward ablations) | `policy/full_trainable.pt`, `policy_scope.json`, tokenizer/processor AND the exact SFT initializer containing frozen visual weights | Entire checkpoint directory, including optimizer, RNG, base lineage and completion record |
| Unfinished runs | Last fully saved checkpoint, clearly labeled incomplete | Frozen source snapshot, data identity, topology and resume configuration |

Keep completed alignment initializers as well as final SFT models. Diagnostic
checkpoints must remain labeled diagnostic, never substituted for formal weights.
The historical shaping-reward problem is documented in
[the reward activity audit](GAP4D_REWARD_ACTIVITY_REPAIR.md); backing up those
models does not validate their original reward-ablation interpretation.

## Evidence bundle

Retain exact training/evaluation manifests and sample IDs, source revisions,
prompt/reward/scorer versions, code snapshots, environment dependency inventories,
per-step metrics, raw generated responses, per-example predictions, summary tables,
failure receipts and checkpoint lineage. Preserve actual answers, not only scores.
Live logs and future checkpoints are not covered by a point-in-time snapshot;
create a new final snapshot when the active run finishes.

Released base weights and licensed raw datasets can be listed by repository,
revision and preparation recipe rather than duplicated in every backup. Keep
locally transformed annotations and split manifests. Regenerable feature caches
are not trained weights. Record exclusions explicitly; this is not a complete
offline mirror if raw media/base models are excluded.

## Personal cloud backup

Use a new private destination for each snapshot. Do not publish a share link
automatically. Authenticate interactively using [the account guide](BAIDU_DATA_DOWNLOAD.md);
never put cookies in commands, Git, reports, or uploaded credential bundles.

```bash
export NODE_ROOT=/persistent/your-root
bash scripts/baidu-cli.sh quota
bash scripts/baidu-cli.sh mkdir /EurekaSI_Backups/your-snapshot
bash scripts/baidu-cli.sh upload -p 4 -l 2 --retry 3 \
  "$BACKUP_STAGING" /EurekaSI_Backups/your-snapshot
bash scripts/baidu-cli.sh meta /EurekaSI_Backups/your-snapshot
```

Inspect the installed client's help before using these options. Subscription
status is not evidence of achieved upload throughput. Estimate ETA from accepted
uploaded bytes / elapsed time and the inventory's remaining bytes; include retry
and final verification overhead. Do not infer success solely from exit status:
check failure summaries, remote file paths/counts/sizes and restore a small
configuration/manifest as a round-trip test. No new SHA256 inventory is required.

Hard-link staging saves local space but is not an independent disk backup:
in-place writes affect all names. Only immutable checkpoint files may be staged
this way; copy changing text before upload. Record duplicate-file aliases if
deduplicating; restore every required path. Do not delete source/shared storage
until the remote inventory and restoration have been verified.

## Restore acceptance

1. Restore the code revision and declared environment into an independent prefix.
2. Extract evidence bundles, restore weights and aliases, and acquire missing
   released inputs at their recorded revisions.
3. Explicitly remap historical absolute paths; preserve initializer lineage.
4. Check imports and load the real model/processor; perform a fixed real-image
   inference and compare its contract with the original evidence.
5. For training, test optimizer/scheduler/RNG restoration separately. A GPU-count
   change is a topology migration, not a promise of bitwise-identical continuation.

Source static checks, unit tests, successful upload, inference reload and full
distributed resume are distinct acceptance levels and must be reported separately.
