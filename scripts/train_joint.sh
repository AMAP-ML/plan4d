#!/usr/bin/env bash
set -euo pipefail

ROOT="${PLAN4D_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd -- "$ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train.py" joint \
  --num-processes "${NUM_PROCESSES:-1}" "$@"
