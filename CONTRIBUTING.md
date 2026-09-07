# Contributing

Contribute adapters, benchmark recipes, dataset audits, reproducible results, or documentation.

1. Keep shared infrastructure in `spatial_intelligence/`; place paper-specific configurations and design notes in `projects/<project>/`.
2. Register resources in `catalog/` with authoritative URLs, immutable source revisions, license notes and honest validation status.
3. A model adapter must support `encode`, `response_logits`, `sample_ids`, `generate`, `save`, and model/tokenizer identity where applicable. Never silently discard visual inputs or missing geometry.
4. A benchmark contribution must distinguish official evaluation from the shared diagnostic protocol. Include subset, prompt, view selection, metric, coverage and scorer version.
5. Add a focused contract/integration test for real failure modes. For a static-only review, run `python scripts/static_check.py` and explicitly report runtime tests as not run. When runtime validation is in scope, run `bash scripts/check.sh` in the prepared environment; that suite includes real tiny-model optimizer steps and distributed processes.
6. Post a pull request with the concrete change and test evidence. Real model results require weights, data recipe, full config, seeds, environment, hardware and run reports; mock results are rejected.

No leaderboard scores are invented or automatically submitted. Public results should include failures and denominator, not only successful responses. Cite and preserve upstream licenses; review model/data terms separately from code.

Keep local data and original delivery ZIPs outside public source roots. The release script selects named source directories and generates a fresh archive manifest. Historical validation logs are evidence for their original version only; do not rewrite them as results for a new change.
