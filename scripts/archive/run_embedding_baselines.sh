#!/usr/bin/env bash
set -uo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

CKPT_DIR=${CKPT_DIR:-runs/smoke/base/ckpt}
OUT_DIR=${OUT_DIR:-runs/baseline/embedding/eval}
LOG_ROOT=${LOG_ROOT:-runs/baseline/embedding/logs}
TAG=${TAG:-baseline-embedding}

mkdir -p "$LOG_ROOT"
log_path="$LOG_ROOT/pipeline_${TAG}_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  python -m latentwire.cli.eval \
    --config configs/baseline/embedding_baselines.json \
    --override "ckpt=$CKPT_DIR" \
    --override "out_dir=$OUT_DIR" \
    --tag "$TAG"
)

echo "=== Starting ${CMD[*]} ===" | tee -a "$log_path"
"${CMD[@]}" 2>&1 | tee -a "$log_path"
status=${PIPESTATUS[0]:-1}
if [[ $status -eq 0 ]]; then
  echo "--- $TAG completed successfully ---" | tee -a "$log_path"
else
  echo "--- $TAG failed with exit code ${status} ---" | tee -a "$log_path"
fi

echo | tee -a "$log_path"
