#!/usr/bin/env bash
set -euo pipefail

ROOT="${PLAN4D_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT:$ROOT/Depth-Anything-3:$ROOT/Depth-Anything-3/src:$ROOT/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
ENTRYPOINT="LongCat-Video-stitched-joint.py"
if [[ "${1:-}" == "--batch" ]]; then
  ENTRYPOINT="LongCat-Video-stitched-joint-sequential-batch-manifest.py"
  shift
fi
cd -- "$ROOT"
exec "$PYTHON_BIN" "$ROOT/DiffSynth-Studio/examples/wanvideo/model_inference/$ENTRYPOINT" \
  --longcat_weight_root "${LONGCAT_WEIGHT_ROOT:-$ROOT/checkpoints/longcat}" \
  --longcat_vae_file "${LONGCAT_VAE_FILE:-$ROOT/checkpoints/Wan2.1_VAE.safetensors}" \
  --stitched_decoder_checkpoint_path "${STITCHED_DECODER_PATH:-$ROOT/checkpoints/stitched_decoder.pth}" \
  --da3_model_path "${DA3_MODEL_PATH:-$ROOT/checkpoints/da3}" \
  --wan_vae_path "${WAN_VAE_PATH:-$ROOT/checkpoints/longcat/vae}" \
  --da3_enable_scene_flow_head \
  --output_dir "$ROOT/outputs/pipeline" "$@"
