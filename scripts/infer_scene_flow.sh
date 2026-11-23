#!/usr/bin/env bash
set -euo pipefail

ROOT="${PLAN4D_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT:$ROOT/Depth-Anything-3:$ROOT/Depth-Anything-3/src:$ROOT/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$ROOT"
exec "$PYTHON_BIN" -m depth_anything_3.tools.infer_video_da3_anyview \
  --model_dir "${DA3_MODEL_PATH:-$ROOT/checkpoints/da3}" \
  --scene_flow_head_checkpoint_path "${SCENE_FLOW_HEAD_PATH:-$ROOT/checkpoints/scene_flow_head.bin}" \
  --output_dir "$ROOT/outputs/scene_flow" "$@"
