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

# Preserve for paper
PRESERVE_DIR="telepathy/preserved_data/exp002_sst2_baselines"
mkdir -p "$PRESERVE_DIR"
cp "$OUTPUT_DIR/sst2_baselines.json" "$PRESERVE_DIR/"
cp "$LOG_FILE" "$PRESERVE_DIR/"

cat > "$PRESERVE_DIR/EXPERIMENT_SUMMARY.md" << 'SUMMARY_EOF'
# Experiment 002: SST-2 Baselines

**Date**: $(date +%Y-%m-%d)
**Status**: COMPLETE
**Purpose**: Validate exp001 results by comparing against baselines

---

## Baselines Evaluated

| Baseline | Description |
|----------|-------------|
| Random | 50% (trivial lower bound) |
| Majority class | Based on SST-2 label distribution |
| Noise | Random soft tokens to Mistral |
| Mistral text | Full text to Mistral (upper bound) |
| Llama text | Full text to Llama |

---

## Results

See `sst2_baselines.json` for detailed results.

---

## Interpretation

If bridge (93.46%) is significantly above noise baseline and close to text baseline,
the bridge is genuinely transmitting semantic information.
SUMMARY_EOF

echo ""
echo " Preserved to: $PRESERVE_DIR"
echo "=========================================================================="
