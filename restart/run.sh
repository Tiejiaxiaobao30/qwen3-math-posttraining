#!/usr/bin/env bash
set -euo pipefail
# WORLD is the GPU count ON ONE HOST. No implicit cloud rental.
: "${MODEL_DIR:?Set MODEL_DIR to the downloaded Qwen3-1.7B-Base directory}"
: "${SFT_ADAPTER:?Set SFT_ADAPTER to your uploaded final_adapter}"
WORLD="${WORLD:-1}"
ARM="${ARM:-strict}"
SEED="${SEED:-42}"
STEPS="${STEPS:-200}"
OUT="${OUT:-outputs/restart/${ARM}-seed${SEED}-gpus${WORLD}}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
if [[ "$WORLD" != 1 && "$WORLD" != 2 && "$WORLD" != 3 ]]; then
  echo 'WORLD must be 1, 2 or 3' >&2; exit 2
fi
mkdir -p "$(dirname "$OUT")"
torchrun --standalone --nnodes=1 --nproc_per_node="$WORLD" -m restart.train \
  --model-dir "$MODEL_DIR" --sft-adapter "$SFT_ADAPTER" --arm "$ARM" \
  --seed "$SEED" --steps "$STEPS" --output "$OUT" --swanlab-mode "$SWANLAB_MODE" "$@"
