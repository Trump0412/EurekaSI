## Change

Describe the behavior changed and why it belongs in the shared core or a project-specific adapter.

## Validation

- Static checks run and results:
- Runtime checks run and results (or explicitly "not run"):
- [ ] `python scripts/static_check.py` and `python scripts/release_check.py`
- [ ] `bash scripts/check.sh` when runtime validation is in scope
- [ ] New or changed behavior has a focused test
- [ ] Source, model, dataset, and protocol revisions are pinned where applicable
- [ ] No credentials, local absolute paths, model weights, datasets, or generated run outputs are included
- [ ] Validation limits and unsupported hardware paths remain explicit

## Upstream impact

List affected GeoWire, GeoBridge, GeoPSRO, or third-party commits and patches. Write “none” when the change is internal to the shared framework.
