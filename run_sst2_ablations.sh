#!/usr/bin/env bash
# run_sst2_ablations.sh
# Comprehensive ablation studies for SST-2 bridge

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

# Find the trained checkpoint
CHECKPOINT="${CHECKPOINT:-runs/sst2_signal_check_20251202_212303/bridge_sst2.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please set CHECKPOINT environment variable or ensure the file exists."
    exit 1
fi

OUTPUT_DIR="runs/sst2_ablations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/ablations.log"

echo "=========================================================================="
echo " SST-2 ABLATION STUDIES"
echo "=========================================================================="
echo " Checkpoint: $CHECKPOINT"
echo " Output: $OUTPUT_DIR"
echo ""
echo " ABLATIONS TO RUN:"
echo "   1. Trained bridge (reference: 93.46%)"
echo "   2. Untrained bridge (random Perceiver) - Is training necessary?"
echo "   3. Mean pooling (no Perceiver) - Is attention necessary?"
echo "   4. Last token only - Is full sequence necessary?"
echo "   5. Linear projection - Is Perceiver architecture necessary?"
echo "   6. Token budget (32 tokens) - Fair text comparison"
echo ""
echo " These answer: What makes the bridge work?"
echo "=========================================================================="
echo ""

{
    python telepathy/eval_sst2_ablations.py \
        --trained_checkpoint "$CHECKPOINT" \
        --num_samples 200 \
        --output_dir "$OUTPUT_DIR" \
        --soft_tokens 32 \
        --depth 2 \
        --heads 8 \
        --source_layer 16 \
        --bf16
} 2>&1 | tee "$LOG_FILE"

# Preserve for paper
PRESERVE_DIR="telepathy/preserved_data/exp003_sst2_ablations"
mkdir -p "$PRESERVE_DIR"
cp "$OUTPUT_DIR/sst2_ablations.json" "$PRESERVE_DIR/"
cp "$LOG_FILE" "$PRESERVE_DIR/"

cat > "$PRESERVE_DIR/EXPERIMENT_SUMMARY.md" << 'EOF'
# Experiment 003: SST-2 Ablation Studies

**Date**: $(date +%Y-%m-%d)
**Status**: COMPLETE
**Purpose**: Validate each architecture choice in the bridge

---

## Ablations

| Ablation | Question Answered |
|----------|-------------------|
| Trained bridge | Reference (93.46%) |
| Untrained bridge | Is training necessary? |
| Mean pooling | Is Perceiver attention necessary? |
| Last token only | Is the full sequence necessary? |
| Linear projection | Is the Perceiver architecture necessary? |
| Token budget (32 tok) | Fair comparison to text |

---

## Results

See `sst2_ablations.json` for detailed results.

---

## Key Findings

[To be filled after results]
EOF

echo ""
echo "=========================================================================="
echo " ABLATIONS COMPLETE"
echo "=========================================================================="
echo " Results: $OUTPUT_DIR/sst2_ablations.json"
echo " Log: $LOG_FILE"
echo " Preserved: $PRESERVE_DIR"
echo "=========================================================================="
