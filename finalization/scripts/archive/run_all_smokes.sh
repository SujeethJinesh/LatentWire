#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3

declare -a TAGS=(
  "smoke-base"
  "smoke-lora"
  "smoke-prefix"
  "smoke-deep-prefix"
  "smoke-latent-adapters"
  "smoke-coprocessor"
  "smoke-gist"
  "smoke-refiner"
)

declare -A CMDS=(
  ["smoke-base"]='python -m latentwire.cli.train --config configs/smoke/base.json --tag smoke-base'
  ["smoke-lora"]='python -m latentwire.cli.train --config configs/smoke/lora.json --tag smoke-lora'
  ["smoke-prefix"]='python -m latentwire.cli.train --config configs/smoke/prefix.json --tag smoke-prefix'
  ["smoke-deep-prefix"]='python -m latentwire.cli.train --config configs/smoke/deep_prefix.json --tag smoke-deep-prefix'
  ["smoke-latent-adapters"]='python -m latentwire.cli.train --config configs/smoke/latent_adapters.json --tag smoke-latent-adapters'
  ["smoke-coprocessor"]='python -m latentwire.cli.train --config configs/smoke/coprocessor.json --tag smoke-coprocessor'
  ["smoke-gist"]='python -m latentwire.cli.train --config configs/smoke/gist_head.json --tag smoke-gist'
  ["smoke-refiner"]='python -m latentwire.cli.train --config configs/smoke/refiner.json --tag smoke-refiner'
)

for tag in "${TAGS[@]}"; do
  cmd="${CMDS[$tag]}"
  log_dir="runs/smoke/${tag}"
  mkdir -p "$log_dir"
  log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"

  echo "=== Starting $cmd ===" | tee -a "$log_path"
  eval "$cmd" 2>&1 | tee -a "$log_path"
  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "--- $tag failed with exit code $status ---" | tee -a "$log_path"
  else
    echo "--- $tag completed successfully ---" | tee -a "$log_path"
  fi
  echo | tee -a "$log_path"
done
