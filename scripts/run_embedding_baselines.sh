#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3

TAGS=("baseline-embedding")
CMDS=(
  "python -m latentwire.cli.train --config configs/baseline/embedding_baselines.json --tag baseline-embedding"
)

for idx in "${!TAGS[@]}"; do
  tag="${TAGS[$idx]}"
  cmd="${CMDS[$idx]}"
  log_dir="runs/baseline/${tag}"
  mkdir -p "$log_dir"
  log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"

  echo "=== Starting $cmd ===" | tee -a "$log_path"
  if eval "$cmd" 2>&1 | tee -a "$log_path"; then
    echo "--- $tag completed successfully ---" | tee -a "$log_path"
  else
    status=${PIPESTATUS[0]:-1}
    echo "--- $tag failed with exit code ${status} ---" | tee -a "$log_path"
  fi
  echo | tee -a "$log_path"

done
