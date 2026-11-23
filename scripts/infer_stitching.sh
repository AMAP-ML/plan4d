#!/usr/bin/env bash
set -euo pipefail

ROOT="${PLAN4D_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT:$ROOT/Depth-Anything-3:$ROOT/Depth-Anything-3/src:$ROOT/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$ROOT"
exec "$PYTHON_BIN" "$ROOT/inference_stitched_decode.py" \
  --checkpoint_path "${STITCHED_DECODER_PATH:-$ROOT/checkpoints/stitched_decoder.pth}" \
  --da3_model_path "${DA3_MODEL_PATH:-$ROOT/checkpoints/da3}" \
  --wan_vae_path "${WAN_VAE_PATH:-$ROOT/checkpoints/longcat/vae}" \
  --da3_enable_scene_flow_head \
  --output_dir "$ROOT/outputs/stitching" "$@"
