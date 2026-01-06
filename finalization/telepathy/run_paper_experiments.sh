#!/usr/bin/env bash
# telepathy/run_paper_experiments.sh
#
# COMPREHENSIVE EXPERIMENT SUITE FOR PAPER
# ==========================================
#
# This script runs ALL experiments needed to strengthen the paper:
#
# 1. PROMPT-TUNING BASELINE (The "Killer" Experiment)
#    - Proves whether Llama (sender) actually helps
#    - If Bridge >> Prompt-Tuning: Llama contributes meaningfully
#    - If Bridge ≈ Prompt-Tuning: Claims need revision
#
# 2. BRIDGE WITH MULTIPLE SEEDS
#    - 3 seeds for statistical rigor
#    - Reports mean ± std
#
# 3. LAYER ABLATION
#    - Which layer is best? (16, 24, 28, 31)
#
# 4. REVERSE DIRECTION
#    - Does Mistral → Llama also work?
#
# 5. RTE DATASET
#    - Additional GLUE task for broader coverage
#
# Usage (from LatentWire root):
#   PYTHONPATH=. bash telepathy/run_paper_experiments.sh
#
# Expected runtime: ~4-6 hours on single H100
# ==========================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Set PYTHONPATH to include parent directory (LatentWire root)
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs/paper_experiments_$(date +%Y%m%d_%H%M%S)}"
SEEDS="42 123 456"
DATASETS="sst2 agnews trec"
STEPS=2000

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_BASE/experiments_${TIMESTAMP}.log"

echo "=============================================="
echo "TELEPATHY PAPER EXPERIMENTS"
echo "=============================================="
echo "Output: $OUTPUT_BASE"
echo "Log: $LOG_FILE"
echo "Seeds: $SEEDS"
echo "Datasets: $DATASETS"
echo "=============================================="
echo ""

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================
# SECTION 1: PROMPT-TUNING BASELINE
# ============================================
# This is the CRITICAL experiment that proves Llama helps

echo ""
echo "=============================================="
echo "SECTION 1: PROMPT-TUNING BASELINE (CRITICAL)"
echo "=============================================="
echo "This proves whether Llama (sender) actually helps."
echo ""

for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        echo ">>> Prompt-Tuning Baseline: $DATASET (seed=$SEED)"

        OUTPUT_DIR="$OUTPUT_BASE/prompt_tuning_baseline/${DATASET}_seed${SEED}"
        mkdir -p "$OUTPUT_DIR"

        python train_prompt_tuning_baseline.py \
            --dataset "$DATASET" \
            --soft_tokens 8 \
            --steps $STEPS \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR" \
            --eval_every 200 \
            --save_every 500

        echo ">>> Done: $DATASET seed=$SEED"
        echo ""
    done
done

# ============================================
# SECTION 2: BRIDGE WITH MULTIPLE SEEDS
# ============================================
# For statistical rigor (mean ± std)

echo ""
echo "=============================================="
echo "SECTION 2: BRIDGE WITH MULTIPLE SEEDS"
echo "=============================================="
echo ""

for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        echo ">>> Bridge: $DATASET (seed=$SEED)"

        OUTPUT_DIR="$OUTPUT_BASE/bridge_multiseed/${DATASET}_seed${SEED}"
        mkdir -p "$OUTPUT_DIR"

        # Use the appropriate training script based on dataset
        if [ "$DATASET" == "sst2" ]; then
            TRAIN_SCRIPT="train_telepathy_sst2.py"
            SOFT_TOKENS=8
        elif [ "$DATASET" == "agnews" ]; then
            TRAIN_SCRIPT="train_telepathy_agnews.py"
            SOFT_TOKENS=8
        elif [ "$DATASET" == "trec" ]; then
            TRAIN_SCRIPT="train_telepathy_trec.py"
            SOFT_TOKENS=16
        fi

        python "$TRAIN_SCRIPT" \
            --soft_tokens $SOFT_TOKENS \
            --steps $STEPS \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR" \
            --save_path "$OUTPUT_DIR/bridge_${DATASET}_seed${SEED}.pt" \
            --eval_every 200 \
            --save_every 500

        echo ">>> Done: $DATASET seed=$SEED"
        echo ""
    done
done

# ============================================
# SECTION 3: LAYER ABLATION
# ============================================
# Which layer is best for extraction?

echo ""
echo "=============================================="
echo "SECTION 3: LAYER ABLATION"
echo "=============================================="
echo "Testing layers: 16, 24, 28, 31"
echo ""

LAYERS="16 24 28 31"
ABLATION_DATASET="sst2"

for LAYER in $LAYERS; do
    echo ">>> Layer Ablation: layer=$LAYER"

    OUTPUT_DIR="$OUTPUT_BASE/layer_ablation/layer${LAYER}"
    mkdir -p "$OUTPUT_DIR"

    python train_telepathy_sst2.py \
        --source_layer $LAYER \
        --soft_tokens 8 \
        --steps $STEPS \
        --seed 42 \
        --output_dir "$OUTPUT_DIR" \
        --save_path "$OUTPUT_DIR/bridge_layer${LAYER}.pt" \
        --eval_every 200 \
        --save_every 500

    echo ">>> Done: layer=$LAYER"
    echo ""
done

# ============================================
# SECTION 4: REVERSE DIRECTION (Mistral → Llama)
# ============================================
# Does it work bidirectionally?

echo ""
echo "=============================================="
echo "SECTION 4: REVERSE DIRECTION (Mistral → Llama)"
echo "=============================================="
echo ""

for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        echo ">>> Reverse Direction: $DATASET (seed=$SEED)"

        OUTPUT_DIR="$OUTPUT_BASE/reverse_direction/${DATASET}_seed${SEED}"
        mkdir -p "$OUTPUT_DIR"

        python train_telepathy_reverse.py \
            --dataset "$DATASET" \
            --soft_tokens 8 \
            --steps $STEPS \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR" \
            --eval_every 200 \
            --save_every 500

        echo ">>> Done: $DATASET seed=$SEED"
        echo ""
    done
done

# ============================================
# SECTION 5: RTE DATASET (Additional GLUE task)
# ============================================

echo ""
echo "=============================================="
echo "SECTION 5: RTE DATASET"
echo "=============================================="
echo ""

for SEED in $SEEDS; do
    echo ">>> RTE: seed=$SEED"

    # Prompt-tuning baseline for RTE
    OUTPUT_DIR="$OUTPUT_BASE/rte/prompt_tuning_seed${SEED}"
    mkdir -p "$OUTPUT_DIR"

    python train_prompt_tuning_baseline.py \
        --dataset rte \
        --soft_tokens 8 \
        --steps $STEPS \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" \
        --eval_every 200 \
        --save_every 500

    echo ">>> Done: RTE prompt-tuning seed=$SEED"
done

# ============================================
# SECTION 6: COMPILE RESULTS
# ============================================

echo ""
echo "=============================================="
echo "COMPILING RESULTS"
echo "=============================================="
echo ""

# Create a summary script
python << 'EOF'
import os
import json
import glob
from collections import defaultdict

output_base = os.environ.get('OUTPUT_BASE', 'runs/paper_experiments')

results = defaultdict(lambda: defaultdict(list))

# Collect all results
for json_file in glob.glob(f"{output_base}/**/*_results.json", recursive=True):
    try:
        with open(json_file) as f:
            data = json.load(f)

        experiment = data.get('experiment', 'unknown')
        accuracy = data.get('final_results', {}).get('accuracy', 0)

        # Parse experiment type
        if 'prompt_tuning_baseline' in experiment:
            dataset = experiment.replace('prompt_tuning_baseline_', '')
            results['prompt_tuning'][dataset].append(accuracy)
        elif 'reverse_direction' in experiment:
            dataset = experiment.replace('reverse_direction_', '')
            results['reverse'][dataset].append(accuracy)
        elif 'layer' in json_file:
            layer = json_file.split('layer')[-1].split('/')[0]
            results['layer_ablation'][f'layer_{layer}'].append(accuracy)
        else:
            # Bridge experiments
            for ds in ['sst2', 'agnews', 'trec']:
                if ds in experiment or ds in json_file:
                    results['bridge'][ds].append(accuracy)
                    break

    except Exception as e:
        print(f"Error reading {json_file}: {e}")

# Print summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

import numpy as np

for exp_type, datasets in sorted(results.items()):
    print(f"\n{exp_type.upper()}:")
    for dataset, accs in sorted(datasets.items()):
        if len(accs) > 1:
            mean = np.mean(accs)
            std = np.std(accs)
            print(f"  {dataset}: {mean:.1f}% ± {std:.1f}% (n={len(accs)})")
        else:
            print(f"  {dataset}: {accs[0]:.1f}%")

# Save summary
summary = {
    'results': {k: {k2: {'mean': np.mean(v), 'std': np.std(v), 'values': v}
                    for k2, v in v2.items()}
                for k, v2 in results.items()}
}

summary_path = f"{output_base}/summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {summary_path}")

print("\n" + "=" * 60)
EOF

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Output directory: $OUTPUT_BASE"
echo "Log file: $LOG_FILE"
echo ""
echo "Key comparisons to make:"
echo "  1. Bridge vs Prompt-Tuning: Does Llama help?"
echo "  2. Bridge std: Statistical significance?"
echo "  3. Layer ablation: Which layer is best?"
echo "  4. Reverse direction: Is it bidirectional?"
echo "=============================================="
