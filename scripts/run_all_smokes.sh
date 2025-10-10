#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3

tag='smoke-base'
cmd='python -m latentwire.cli.train --config configs/smoke/base.json --tag smoke-base'
log_dir='runs/smoke/smoke-base'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-lora'
cmd='python -m latentwire.cli.train --config configs/smoke/lora.json --tag smoke-lora'
log_dir='runs/smoke/smoke-lora'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-prefix'
cmd='python -m latentwire.cli.train --config configs/smoke/prefix.json --tag smoke-prefix'
log_dir='runs/smoke/smoke-prefix'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-deep-prefix'
cmd='python -m latentwire.cli.train --config configs/smoke/deep_prefix.json --tag smoke-deep-prefix'
log_dir='runs/smoke/smoke-deep-prefix'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-latent-adapters'
cmd='python -m latentwire.cli.train --config configs/smoke/latent_adapters.json --tag smoke-latent-adapters'
log_dir='runs/smoke/smoke-latent-adapters'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-coprocessor'
cmd='python -m latentwire.cli.train --config configs/smoke/coprocessor.json --tag smoke-coprocessor'
log_dir='runs/smoke/smoke-coprocessor'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-gist'
cmd='python -m latentwire.cli.train --config configs/smoke/gist_head.json --tag smoke-gist'
log_dir='runs/smoke/smoke-gist'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

tag='smoke-refiner'
cmd='python -m latentwire.cli.train --config configs/smoke/refiner.json --tag smoke-refiner'
log_dir='runs/smoke/smoke-refiner'
mkdir -p "$log_dir"
log_path="$log_dir/pipeline_${tag}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting $cmd ===" | tee -a "$log_path"
$cmd 2>&1 | tee -a "$log_path"
echo

