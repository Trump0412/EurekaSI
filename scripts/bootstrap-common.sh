#!/usr/bin/env bash
# Sourced by installation scripts; functions do not change the caller's directory.
spatial_prepare_venv() {
  ENV_DIR=$("$PYTHON_BIN" - "$ENV_DIR" "$PROJECT_DIR" <<'PY'
from pathlib import Path
import sys
target, project = (Path(value).expanduser().resolve() for value in sys.argv[1:])
if target == project or target in project.parents or target == Path.home():
    raise SystemExit('Use a dedicated virtual environment directory, not a workspace/root/home directory')
if target.exists() and not target.is_dir():
    raise SystemExit('Virtual environment target is not a directory')
if (target.is_dir() and any(target.iterdir()) and not (target / 'pyvenv.cfg').is_file()
        and not (target / '.spatial-bootstrap-incomplete').is_file()):
    raise SystemExit('Preserving a nonempty directory that is not a virtual environment')
print(target)
PY
  )
  mkdir -p "$ENV_DIR"
  INCOMPLETE_MARKER="$ENV_DIR/.spatial-bootstrap-incomplete"
  touch "$INCOMPLETE_MARKER"
  trap 'status=$?; if [[ $status -ne 0 ]]; then echo "Bootstrap incomplete: $ENV_DIR; fix the reported error and rerun." >&2; fi' EXIT
}
