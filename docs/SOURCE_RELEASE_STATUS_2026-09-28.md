# Source publication checks — 2026-09-28

The public source includes the geometry alignment/SFT matrix, native and geometry
RFT paths, checkpoint lineage, reward activity auditing, DSR/expanded evaluation
queues, result aggregation and recovery documentation. Private deployment plans,
credentials, raw infrastructure logs, datasets and weights are excluded.

Checks executed on a separate source snapshot, without changing active workers:

- Python/JSON/YAML parsing and local documentation links: passed.
- Source packaging metadata and public resource selection: passed.
- Infrastructure-identifier and credential-pattern scan over the selected public
  files: no flagged files. This is a scoped publication review, not a proof that
  arbitrary credentials or all historical Git objects have been exhaustively audited.
- Full tests in the prepared environment with GPUs hidden: **569 passed, 4 skipped**.
  This is not a fresh GPU training/inference acceptance for every backend.
- An initial run had one failing outdated geometry-evaluation test double. The
  test now covers both normal online VGGT and forced-null evaluation, asserting
  that forced-null does not load VGGT and uses frame-faithful bf16 placeholders.
  No production model behavior was changed to make this test pass.

Known boundaries remain: historical shaping rewards were inactive in the audited
RFT runs; the new activity gate is a safeguard, not a completed repaired experiment.
Official/adapted scoring differences and incomplete benchmark lanes are documented
in the [RFT subproject](../projects/geometry-rft/README.md). Source tests cannot
establish scientific reproduction or effect sizes.

Weight/cloud retention has separate acceptance: inventory, upload, remote byte
verification and actual restore. See [backup and restore](WEIGHT_BACKUP_AND_RESTORE.md).
Do not interpret publication of this code as completion of a running upload or
an active training experiment.
