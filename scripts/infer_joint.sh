#!/usr/bin/env bash
set -euo pipefail

ROOT="${PLAN4D_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE_ARGS=()
if [[ "${1:-}" == "--batch" ]]; then
  MODE_ARGS=(--batch)
  shift
fi

HAS_DIT_OVERRIDE=false
for argument in "$@"; do
  case "$argument" in
    --dit_lora_path|--dit_lora_path=*|--checkpoint_path|--checkpoint_path=*|--dit_checkpoint_path|--dit_checkpoint_path=*)
      HAS_DIT_OVERRIDE=true
      ;;
  esac
done

WEIGHT_ARGS=()
if [[ "$HAS_DIT_OVERRIDE" == false ]]; then
  if [[ -n "${DIT_LORA_PATH:-}" && -n "${DIT_CHECKPOINT_PATH:-}" ]]; then
    printf '%s\n' 'Set only one of DIT_LORA_PATH and DIT_CHECKPOINT_PATH.' >&2
    exit 2
  elif [[ -n "${DIT_CHECKPOINT_PATH:-}" ]]; then
    WEIGHT_ARGS=(--dit_checkpoint_path "$DIT_CHECKPOINT_PATH")
  else
    WEIGHT_ARGS=(--dit_lora_path "${DIT_LORA_PATH:-$ROOT/checkpoints/longcat_lora.safetensors}")
  fi
fi

exec bash "$ROOT/scripts/infer_pipeline.sh" "${MODE_ARGS[@]}" \
  "${WEIGHT_ARGS[@]}" --output_dir "$ROOT/outputs/joint" "$@"
