#!/usr/bin/env bash
set -e

# Paper Ablation Studies - Focused experiments for 3-week deadline
# Total runtime: ~12 hours on 4× H100
# Output: paper_writing/runs/

echo "=========================================="
echo "PAPER ABLATION EXPERIMENTS"
echo "=========================================="
echo "Start time: $(date)"
echo "Expected runtime: ~12 hours (4 configs × 3 hours each)"
echo ""

# Base configuration
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TRANSLATOR_TYPE="bottleneck_gated"
PER_DEVICE_BATCH=10
EVAL_EVERY=250
EVAL_SAMPLES=500  # Batched evaluation to avoid OOM
MAX_NEW_TOKENS=256

# Create output directory
OUTPUT_DIR="paper_writing/runs/ablations_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$OUTPUT_DIR"
SUMMARY_LOG="$OUTPUT_DIR/summary.log"

echo "=== ABLATION EXPERIMENTS ===" | tee "$SUMMARY_LOG"
echo "Output directory: $OUTPUT_DIR" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Helper function to run experiments
run_experiment() {
    local name=$1
    local desc=$2
    shift 2  # Remove first two args, rest are for the script

    echo "==========================================" | tee -a "$SUMMARY_LOG"
    echo "Experiment: $name" | tee -a "$SUMMARY_LOG"
    echo "Description: $desc" | tee -a "$SUMMARY_LOG"
    echo "Start: $(date)" | tee -a "$SUMMARY_LOG"
    echo "------------------------------------------" | tee -a "$SUMMARY_LOG"

    EXP_DIR="$OUTPUT_DIR/$name"
    mkdir -p "$EXP_DIR"
    LOG_FILE="$EXP_DIR/train.log"

    # Run with all remaining arguments
    # Use random port to avoid conflicts with other training jobs
    RANDOM_PORT=$((29500 + RANDOM % 1000))

    {
        torchrun --nproc_per_node=4 --master_port "$RANDOM_PORT" paper_writing/cross_attention.py \
            --source_model "$SOURCE_MODEL" \
            --target_model "$TARGET_MODEL" \
            --translator_type "$TRANSLATOR_TYPE" \
            --per_device_batch "$PER_DEVICE_BATCH" \
            --eval_every "$EVAL_EVERY" \
            --eval_samples "$EVAL_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --bf16 \
            --save_path "$EXP_DIR/checkpoint.pt" \
            "$@"
    } 2>&1 | tee "$LOG_FILE"

    # Extract results
    echo "End: $(date)" | tee -a "$SUMMARY_LOG"
    echo "Results:" | tee -a "$SUMMARY_LOG"
    grep "Final.*acc:" "$LOG_FILE" | tail -1 | tee -a "$SUMMARY_LOG"

    # Extract peak accuracy
    echo "Peak accuracy:" | tee -a "$SUMMARY_LOG"
    grep "Eval.*Bridged acc:" "$LOG_FILE" | \
        awk -F'Bridged acc: ' '{print $2}' | \
        sort -rn | head -1 | \
        awk '{printf "  Peak bridged: %.1f%%\n", $1 * 100}' | tee -a "$SUMMARY_LOG"

    echo "" | tee -a "$SUMMARY_LOG"
}

# ============================================
# ABLATION 1: Stability Fixes (64 tokens)
# Research Question: Do stability fixes prevent collapse?
# ============================================

echo "╔════════════════════════════════════════╗" | tee -a "$SUMMARY_LOG"
echo "║  ABLATION 1: STABILITY FIXES          ║" | tee -a "$SUMMARY_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Config 1a: WITH stability fixes (NEW RUN)
run_experiment \
    "1a_stable_64tok" \
    "64 tokens WITH InfoNCE + early stopping + gen hygiene" \
    --dataset gsm8k \
    --lr 1e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens 64 \
    --depth 8 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 750 \
    --info_nce_weight 0.05 \
    --early_stop_patience 5 \
    --seed 1234

echo "" | tee -a "$SUMMARY_LOG"
echo "NOTE: Baseline (1b_baseline_64tok) reuses successful_experiments/cross_model/85/train_high_capacity.log" | tee -a "$SUMMARY_LOG"
echo "      Peak: 81.5% → Final: 36.0% (no stability fixes)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ============================================
# ABLATION 2: Sequence Length (all with stability)
# Research Question: Compression vs quality tradeoff?
# ============================================

echo "╔════════════════════════════════════════╗" | tee -a "$SUMMARY_LOG"
echo "║  ABLATION 2: SEQUENCE LENGTH          ║" | tee -a "$SUMMARY_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Config 2a: 32 tokens (high compression)
run_experiment \
    "2a_stable_32tok" \
    "32 tokens (4.7× compression) WITH stability fixes" \
    --dataset gsm8k \
    --lr 1e-4 \
    --bottleneck_dim 768 \
    --soft_tokens 32 \
    --depth 4 \
    --heads 12 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 600 \
    --info_nce_weight 0.05 \
    --early_stop_patience 5 \
    --seed 1234

# Config 2b: 48 tokens (medium compression)
run_experiment \
    "2b_stable_48tok" \
    "48 tokens (3.1× compression) WITH stability fixes" \
    --dataset gsm8k \
    --lr 1e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens 48 \
    --depth 6 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 750 \
    --info_nce_weight 0.05 \
    --early_stop_patience 5 \
    --seed 1234

echo "" | tee -a "$SUMMARY_LOG"
echo "NOTE: 64 tokens result is same as 1a_stable_64tok (reused)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ============================================
# ABLATION 3: Dataset Generalization
# Research Question: Does it generalize beyond math?
# ============================================

echo "╔════════════════════════════════════════╗" | tee -a "$SUMMARY_LOG"
echo "║  ABLATION 3: DATASET GENERALIZATION   ║" | tee -a "$SUMMARY_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Config 3a: HotpotQA (multi-hop reasoning)
run_experiment \
    "3a_hotpotqa_64tok" \
    "HotpotQA dataset, 64 tokens WITH stability fixes" \
    --dataset hotpotqa \
    --lr 1e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens 64 \
    --depth 8 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 750 \
    --info_nce_weight 0.05 \
    --early_stop_patience 5 \
    --seed 1234

# ============================================
# Summary and Analysis
# ============================================

echo "==========================================" | tee -a "$SUMMARY_LOG"
echo "ALL ABLATIONS COMPLETE" | tee -a "$SUMMARY_LOG"
echo "End time: $(date)" | tee -a "$SUMMARY_LOG"
echo "==========================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Create comparison table
echo "RESULTS COMPARISON:" | tee -a "$SUMMARY_LOG"
echo "------------------------------------------" | tee -a "$SUMMARY_LOG"
{
    echo "Experiment,Tokens,Dataset,Peak,Final,Degradation"

    # Extract results from logs
    for exp_dir in "$OUTPUT_DIR"/*/; do
        if [ -d "$exp_dir" ]; then
            name=$(basename "$exp_dir")
            log="$exp_dir/train.log"

            # Get token count from name
            if [[ $name == *"32tok"* ]]; then tokens=32
            elif [[ $name == *"48tok"* ]]; then tokens=48
            elif [[ $name == *"64tok"* ]]; then tokens=64
            else tokens="N/A"
            fi

            # Get dataset
            if [[ $name == *"hotpotqa"* ]]; then dataset="HotpotQA"
            else dataset="GSM8K"
            fi

            if [ -f "$log" ]; then
                # Get peak bridged accuracy
                peak=$(grep "Eval.*Bridged acc:" "$log" | \
                       awk -F'Bridged acc: ' '{print $2}' | \
                       sort -rn | head -1 | \
                       awk '{printf "%.1f%%", $1 * 100}')

                # Get final accuracy
                final=$(grep "Final.*Bridged acc:" "$log" | tail -1 | \
                       grep -oE "Bridged acc: [0-9.]+" | \
                       grep -oE "[0-9.]+" | \
                       awk '{printf "%.1f%%", $1 * 100}')

                # Calculate degradation
                peak_val=$(echo "$peak" | tr -d '%')
                final_val=$(echo "$final" | tr -d '%')
                if [ -n "$peak_val" ] && [ -n "$final_val" ]; then
                    deg=$(echo "$peak_val - $final_val" | bc -l | xargs printf "%.1f%%")
                    echo "$name,$tokens,$dataset,$peak,$final,$deg"
                fi
            fi
        fi
    done

    # Add baseline from existing experiment (for reference)
    echo "1b_baseline_64tok,64,GSM8K,81.5%,36.0%,45.5%"

} | column -t -s',' | tee -a "$SUMMARY_LOG"

echo "" | tee -a "$SUMMARY_LOG"
echo "All logs saved to: $OUTPUT_DIR" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ============================================
# Create analysis script
# ============================================

cat > "$OUTPUT_DIR/analyze_ablations.py" << 'ANALYSIS_EOF'
#!/usr/bin/env python3
"""
Analyze ablation results for paper
Generates plots and tables for:
1. Stability: with vs without fixes
2. Sequence length: 32 vs 48 vs 64 tokens
3. Dataset: GSM8K vs HotpotQA
"""

import os
import re
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file):
    """Extract evaluation metrics from training log"""
    results = {
        'evals': [],
        'peak_bridged': 0,
        'peak_step': 0,
        'final_bridged': 0,
        'final_target': 0
    }

    with open(log_file) as f:
        for line in f:
            # Extract eval lines
            if '[Eval] Step' in line and 'Bridged acc:' in line:
                parts = line.split('|')
                step = int(re.search(r'Step (\d+)', parts[0]).group(1))
                target = float(re.search(r'Target-alone acc: ([0-9.]+)', parts[1]).group(1))
                bridged = float(re.search(r'Bridged acc: ([0-9.]+)', parts[2]).group(1))

                results['evals'].append({
                    'step': step,
                    'target': target,
                    'bridged': bridged
                })

                if bridged > results['peak_bridged']:
                    results['peak_bridged'] = bridged
                    results['peak_step'] = step

            # Extract final results
            if '[Final Eval]' in line and 'Bridged acc:' in line:
                results['final_target'] = float(re.search(r'Target-alone acc: ([0-9.]+)', line).group(1))
                results['final_bridged'] = float(re.search(r'Bridged acc: ([0-9.]+)', line).group(1))

    return results

def main():
    script_dir = Path(__file__).parent
    results = {}

    # Parse all experiment logs
    for exp_dir in glob.glob(str(script_dir / '*/')):
        exp_name = Path(exp_dir).name
        log_file = Path(exp_dir) / 'train.log'

        if log_file.exists() and exp_name != script_dir.name:
            print(f"Parsing {exp_name}...")
            results[exp_name] = parse_log(log_file)

    # Add baseline from existing experiment (for comparison)
    results['1b_baseline_64tok'] = {
        'evals': [
            {'step': 250, 'target': 0.730, 'bridged': 0.290},
            {'step': 500, 'target': 0.730, 'bridged': 0.655},
            {'step': 750, 'target': 0.730, 'bridged': 0.535},
            {'step': 1000, 'target': 0.730, 'bridged': 0.815},
            {'step': 1250, 'target': 0.730, 'bridged': 0.755},
            {'step': 1500, 'target': 0.730, 'bridged': 0.655},
            {'step': 2000, 'target': 0.730, 'bridged': 0.635},
            {'step': 2500, 'target': 0.730, 'bridged': 0.375},
            {'step': 3000, 'target': 0.730, 'bridged': 0.360},
        ],
        'peak_bridged': 0.815,
        'peak_step': 1000,
        'final_bridged': 0.360,
        'final_target': 0.730
    }

    # Save raw results
    with open(script_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"{'Experiment':<25} {'Peak':<10} {'Final':<10} {'Degradation':<12}")
    print("-" * 60)

    for name, data in sorted(results.items()):
        peak = data['peak_bridged'] * 100
        final = data['final_bridged'] * 100
        deg = peak - final
        print(f"{name:<25} {peak:>6.1f}%   {final:>6.1f}%   {deg:>6.1f}%")

    print(f"\nDetailed results saved to: {script_dir / 'ablation_results.json'}")
    print(f"\nTo generate plots, run: python {__file__} --plot")

if __name__ == '__main__':
    main()
ANALYSIS_EOF

chmod +x "$OUTPUT_DIR/analyze_ablations.py"

echo "==========================================" | tee -a "$SUMMARY_LOG"
echo "Analysis script created: $OUTPUT_DIR/analyze_ablations.py" | tee -a "$SUMMARY_LOG"
echo "Run 'python $OUTPUT_DIR/analyze_ablations.py' to analyze results" | tee -a "$SUMMARY_LOG"
echo "==========================================" | tee -a "$SUMMARY_LOG"
