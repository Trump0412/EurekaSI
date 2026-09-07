# Repository maintenance

- The infrastructure is named EurekaSI; its canonical repository is https://github.com/Trump0412/EurekaSI. Keep the `spatial_intelligence` module and `spatial` CLI compatible; `eurekasi` is the public distribution and additional CLI name.
- Follow `docs/RESEARCH_WORKFLOW.md` for paper and downstream application work. Shared fixes belong in the common implementation, and each study must preserve its own configuration and evidence.

- This root is the public spatial intelligence infrastructure. Shared implementation belongs in `spatial_intelligence/`; method-specific settings belong in `projects/` and `configs/`.
- The original delivery is preserved locally in ignored `_archives/`. Do not edit, vendor, or publish those archives by default. Fetch legacy implementations at catalog commits when needed.
- Preserve data, checkpoints, local settings, experiment outputs and historical evidence. Do not commit machine paths, credentials, model weights or datasets.
- User requests for static review mean no environment installation, data/model download, training, inference, simulator runs or optimizer tests. `python scripts/static_check.py` only parses source/configuration and local documentation links.
- Runtime validation, when requested, uses `bash scripts/check.sh` in the prepared Linux/WSL environment. Report exactly which checks were run. Historical logs do not validate new changes.
- Keep changes to data semantics, marker rendering, sample splits, model inputs, scoring and cache identity explicit. Add a focused regression case for a substantive failure mode.
- Keep Chinese and English entry documentation accurate. `docs/READINESS_REVIEW.md` records current static findings; `docs/VALIDATION.md` separates current and historical evidence.
- Do not commit, push or publish unless the user requests it. Keep changes ready for the user's review and commit.
