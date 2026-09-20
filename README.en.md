# EurekaSI

Opt-in multimodal bucketing, within-step rank balancing, explicit DeltaNet backends and gated throughput experiments are documented in [THROUGHPUT_OPTIMIZATION](docs/THROUGHPUT_OPTIMIZATION.md). Four-GPU speedup remains pending measurement; active research runs are not modified automatically.

Current Qwen3.5 operation follows the [four-GPU batch and evaluation protocol](docs/BATCH_AND_EVAL_PROTOCOL.md): measured mixed-data throughput, checkpoint resume, native video, ground-truth-blind final-answer extraction and paired ReVSI/VSI-Bench scoring. Historical scores are not validated capability baselines.

For the isolated Conda Qwen3.5-2B / ReVSI baseline → SPAR+Hound SFT → paired evaluation workflow, see [SERVER_RUNBOOK](docs/SERVER_RUNBOOK.md). Start with `bash scripts/start-qwen35-study.sh /persistent/your-root`; queued stages are not completed validation.

v0.2.1 — development version; no formal release yet

[EurekaSI](https://github.com/Trump0412/EurekaSI) is shared spatial intelligence infrastructure for controlled paper experiments and downstream applications. See the [research and application workflow](docs/RESEARCH_WORKFLOW.md) for project boundaries and evidence requirements.

The distribution is named `eurekasi`; both `eurekasi` and the compatible `spatial` command use the same CLI. The Python module remains `spatial_intelligence`, preserving existing settings and cache paths.

A shared research codebase for GeoWire, GeoBridge, GeoPSRO, and controlled studies of how spatial capabilities are acquired and transferred.

The original maintenance pass was static-only. Subsequent Qwen3.5 server training, native-video smoke tests and targeted regressions are recorded in [VALIDATION](docs/VALIDATION.md); full paired benchmark results remain pending. See the [readiness review](docs/READINESS_REVIEW.md) for broader unverified capabilities. Run instructions target Linux/WSL2 and Python 3.12; native Windows training is not validated.

Features include explicit visual manifests, resumable pinned downloads, data audits, SFT, GRPO/GSPO, on-policy distillation, privileged self-distillation, multi-model evaluation, synchronous data parallel training, frozen VGGT/DA3/Pi3 caches, a shared geometry fusion baseline, and a LIBERO evaluation contract.

```bash
bash scripts/bootstrap.sh native
source .venv/bin/activate
spatial init --root /data/spatial
bash scripts/check.sh
```

Start with the [Chinese quickstart](docs/QUICKSTART.md), [GPU/batch semantics](docs/DISTRIBUTED.md), [resource catalog](docs/RESOURCES.md), [validation record](docs/VALIDATION.md), and [contribution guide](CONTRIBUTING.md).

This is research infrastructure, not a claim of reproducing every linked model. CPU/tiny-model tests do not certify real multimodal GPU performance. Geometry fusion is a separate baseline, not a replacement labeled as one of the three original paper methods. LIBERO requires an action-trained policy; a spatial QA model alone is not a robot controller. Original code snapshots preserve their provenance; new shared code uses a scoped MIT license; see [NOTICE](NOTICE.md) for third-party exclusions.

Original ZIP files remain locally under ignored `_archives/`. `python scripts/static_check.py` parses source/configuration and local links without application execution (Python 3.10–3.12, PyYAML, and tomli on Python 3.10). `python scripts/package_release.py --output dist/eurekasi-source.zip` packages public source directories and creates an archive manifest. Publishing source does not certify runtime readiness.

Handoff changes were reviewed selectively; see the [merge decisions](docs/HANDOFF_MERGE.md). Wheel runtime assets are now declared and resolved independently of the working directory; build/install verification remains pending. Run `python scripts/release_check.py` for static release metadata/resource checks. The [maintenance guide](docs/MAINTENANCE.md) describes configuration precedence and environment isolation.
