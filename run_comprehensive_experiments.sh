#!/usr/bin/env bash
# run_comprehensive_experiments.sh
# Run ALL experiments for paper - complete end-to-end

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/comprehensive_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/experiment.log"

mkdir -p "$OUTPUT_DIR"

echo "=========================================================================="
echo " COMPREHENSIVE TELEPATHY BRIDGE EXPERIMENTS"
echo "=========================================================================="
echo " Output: $OUTPUT_DIR"
echo " Log: $LOG_FILE"
echo ""
echo " EXPERIMENT MATRIX:"
echo ""
echo " 1. BRIDGE ARCHITECTURES (6 types)"
echo "    - Continuous (Perceiver + RMS norm)"
echo "    - Diffusion Transformer"
echo "    - MLP (pooling + feedforward)"
echo "    - Linear projection"
echo "    - Mean pooling"
echo "    - Identity (last token)"
echo ""
echo " 2. LAYER ABLATION (5 layers)"
echo "    - Layers: 0, 8, 16, 24, 31"
echo ""
echo " 3. COMPRESSION ABLATION (5 sizes)"
echo "    - Soft tokens: 4, 8, 16, 32, 64"
echo ""
echo " 4. DEPTH ABLATION (3 depths)"
echo "    - Perceiver depth: 1, 2, 4"
echo ""
echo " 5. DIVERSITY LOSS ABLATION (3 weights)"
echo "    - Weights: 0.0, 0.1, 0.5"
echo ""
echo " 6. TRANSFER DIRECTION"
echo "    - Llama → Mistral (primary)"
echo "    - Mistral → Llama (reverse)"
echo ""
echo " 7. BASELINES"
echo "    - Llama text, Mistral text"
echo "    - Random, majority"
echo "    - Untrained versions of each bridge"
echo ""
echo " TOTAL: ~40 experiments"
echo "=========================================================================="
echo ""

# Configuration - adjust based on compute budget
TRAIN_STEPS="${TRAIN_STEPS:-500}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_TRAIN="${MAX_TRAIN:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo " Configuration:"
echo "   Train steps: $TRAIN_STEPS"
echo "   Eval samples: $EVAL_SAMPLES"
echo "   Max train samples: $MAX_TRAIN"
echo "   Batch size: $BATCH_SIZE"
echo ""
echo "=========================================================================="
echo ""

{
    python telepathy/comprehensive_experiments.py \
        --output_dir "$OUTPUT_DIR" \
        --train_steps "$TRAIN_STEPS" \
        --eval_samples "$EVAL_SAMPLES" \
        --max_train_samples "$MAX_TRAIN" \
        --batch_size "$BATCH_SIZE" \
        --bridge_types continuous diffusion mlp linear meanpool identity \
        --source_layers 0 8 16 24 31 \
        --num_latents_list 4 8 16 32 64 \
        --depths 1 2 4 \
        --diversity_weights 0.0 0.1 0.5 \
        --test_reverse
} 2>&1 | tee "$LOG_FILE"

# Preserve for paper
PRESERVE_DIR="telepathy/preserved_data/exp_comprehensive_${TIMESTAMP}"
mkdir -p "$PRESERVE_DIR"
cp "$OUTPUT_DIR/results.json" "$PRESERVE_DIR/"
cp "$LOG_FILE" "$PRESERVE_DIR/"

# Generate summary
python << 'SUMMARY_SCRIPT'
import json
import os
import sys

output_dir = os.environ.get("OUTPUT_DIR", "runs/comprehensive")
results_path = os.path.join(output_dir, "results.json")

if not os.path.exists(results_path):
    print("No results found")
    sys.exit(0)

with open(results_path) as f:
    results = json.load(f)

preserve_dir = os.environ.get("PRESERVE_DIR", "telepathy/preserved_data/exp_comprehensive")

summary = f"""# Comprehensive Experiments Summary

**Date**: {results['meta'].get('start_time', 'N/A')}
**Status**: COMPLETE

---

## Baselines

| Baseline | Accuracy |
|----------|----------|
"""

for name, res in results.get("baselines", {}).items():
    acc = res.get("accuracy", 0)
    summary += f"| {name} | {acc:.1f}% |\n"

summary += """
---

## Experiment Results

| Experiment | Accuracy |
|------------|----------|
"""

# Sort by accuracy
exps = sorted(results.get("experiments", {}).items(),
              key=lambda x: x[1].get("accuracy", 0), reverse=True)

for name, res in exps:
    acc = res.get("accuracy", 0)
    summary += f"| {name} | {acc:.1f}% |\n"

summary += """
---

## Key Findings

[To be analyzed after results review]

---

## Files

- `results.json` - Complete structured results
- `experiment.log` - Full execution log
- `*.pt` - Model checkpoints
"""

summary_path = os.path.join(preserve_dir, "EXPERIMENT_SUMMARY.md")
with open(summary_path, 'w') as f:
    f.write(summary)

print(f"Summary written to: {summary_path}")
SUMMARY_SCRIPT

echo ""
echo "=========================================================================="
echo " COMPREHENSIVE EXPERIMENTS COMPLETE"
echo "=========================================================================="
echo " Results: $OUTPUT_DIR/results.json"
echo " Preserved: $PRESERVE_DIR"
echo " Log: $LOG_FILE"
echo ""
echo " Next: Review results and update EXPERIMENT_SUMMARY.md with findings"
echo "=========================================================================="
