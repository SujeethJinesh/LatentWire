#!/usr/bin/env bash
# run_sst2_baselines.sh
# Evaluate baselines to compare against bridge result (93.46%)

set -euo pipefail

# HPC Environment
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load gcc/13.1.0 2>/dev/null || true
    module load conda/24.3.0-0 2>/dev/null || true
    module load stockcuda/12.6.2 2>/dev/null || true
fi

export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="runs/sst2_baselines_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/baselines.log"

echo "=========================================================================="
echo " SST-2 BASELINES"
echo "=========================================================================="
echo " Output: $OUTPUT_DIR"
echo ""
echo " Purpose: Validate that 93.46% bridge accuracy is real"
echo ""
echo " Baselines to compute:"
echo "   1. Random: 50%"
echo "   2. Majority class: based on label distribution"
echo "   3. Mistral text: Mistral given full text (upper bound)"
echo "   4. Noise: Random soft tokens to Mistral"
echo "   5. Llama text: Llama given full text"
echo ""
echo " If bridge (93.46%) > noise baseline and close to text baseline,"
echo " then the bridge is genuinely transmitting information."
echo "=========================================================================="
echo ""

{
    python telepathy/eval_sst2_baselines.py \
        --num_samples 200 \
        --output_dir "$OUTPUT_DIR" \
        --bf16
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================================================="
echo " BASELINES COMPLETE"
echo "=========================================================================="
echo " Results: $OUTPUT_DIR/sst2_baselines.json"
echo " Log: $LOG_FILE"
echo "=========================================================================="
