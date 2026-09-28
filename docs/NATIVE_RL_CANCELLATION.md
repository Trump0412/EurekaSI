# Native RL progress correction and cancellation

The native spatial pilot was deliberately stopped to reallocate the node to an eight-GPU SFT experiment. It completed **4 optimizer updates**, not 422 or 449. The old watchdog incorrectly matched the duration field `timing_s/step:449.513927` as an integer step counter. Its negative ETA and inflated progress were invalid, not evidence of completed training.

`watch-native-rl.py` now reads only `training/global_step`, and regression tests explicitly cover duration contamination. This corrects observation only; it does not alter training, rewards or historical rollout outputs. The corrected final health record reports `cancelled_by_user`, 4 steps and no remaining ETA.

The pilot saved checkpoints every 25 updates, so **no formal pilot checkpoint existed at cancellation** and its four in-memory updates were lost. Configuration, before-training validation, rollout records and logs were retained. The separate diagnostic gate's one-step full FSDP checkpoint and exported/reloaded model were also retained; these are diagnostic weights, not the pilot's final model.

For explicitly authorized cancellation, use the identity-checked helper:

```bash
python3 scripts/stop-native-rl.py --output "$NODE_ROOT/runs/RUN_NAME" \
  --supervisor-pid SUPERVISOR_PID --watchdog-pid WATCHDOG_PID
# Inspect the dry-run identities; only then repeat with --execute.
```

The helper validates owner command lines and `/proc` start times, records an inventory, and stops only the supervisor, watchdog and verified descendant tree. It never runs global `ray stop` or targets unrelated GPU jobs. The receipt retains the original inaccurate health record for audit. No files are deleted. A finished cancellation must be followed by an independent GPU/process check before allocating replacement work.
