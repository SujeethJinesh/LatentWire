#!/usr/bin/env bash
set -euo pipefail

# Load HPC modules if available (cluster environment)
# Skip if module command doesn't exist (local/CI environments)
if command -v module >/dev/null 2>&1; then
    module purge  # Clear any conflicting modules from environment
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75  # Latest cuDNN for CUDA 12.x
    # NOTE: cudatoolkit/12.5 and nvhpc/24.7 cause "Error 803: unsupported display driver"
    # Verified working on n23: stockcuda/12.6.2 + cudnn/cuda12/9.3.0.75
fi

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation
: ${PY_SCRIPT:=paper_writing/cross_attention.py}
SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PER_DEVICE_BATCH=2  # Lower for long soft tokens to avoid OOM
EVAL_EVERY=250
EVAL_SAMPLES=200  # Reduced for faster eval iterations (was 500)
MAX_NEW_TOKENS=256  # Shorter generations for faster eval

# Prompt-mode variants (soft_tokens only vs. soft tokens + raw prompt)
PROMPT_MODES=("soft_only" "soft_plus_text")

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
        torchrun --standalone --nproc_per_node=4 --master_port "$RANDOM_PORT" "$PY_SCRIPT" \
            --source_model "$SOURCE_MODEL" \
            --target_model "$TARGET_MODEL" \
            --per_device_batch "$PER_DEVICE_BATCH" \
            --eval_every "$EVAL_EVERY" \
            --eval_samples "$EVAL_SAMPLES" \
            --eval_batch_size 36 \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --bf16 \
            --no_compile \
            --save_path "$EXP_DIR/checkpoint.pt" \
            --log_dir "$EXP_DIR" \
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

# Helper to run both prompt variants for a given config
run_prompt_variants() {
    local base_name=$1
    local desc=$2
    shift 2

    for mode in "${PROMPT_MODES[@]}"; do
        local name="${base_name}_${mode}"
        local full_desc="${desc} (prompt=${mode})"
        run_experiment "$name" "$full_desc" --eval_prompt_mode "$mode" "$@"
    done
}

# ============================================
# ABLATION 1: DiT Bridge Architecture
# Research Question: Does iterative refinement prevent collapse?
# ============================================

echo "╔════════════════════════════════════════╗" | tee -a "$SUMMARY_LOG"
echo "║  ABLATION 1: DiT BRIDGE               ║" | tee -a "$SUMMARY_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Config 1a: DiT-2step (minimal diffusion, faster)
run_prompt_variants \
    "1a_dit_2step_64tok" \
    "DiT 64 tokens, 2 training steps (like Transfusion)" \
    --dataset gsm8k \
    --bridge dit \
    --lr 1e-4 \
    --dit_dim 512 \
    --soft_tokens -1 \
    --dit_depth 6 \
    --dit_heads 8 \
    --dit_steps_train 2 \
    --dit_steps_eval 4 \
    --dit_dropout 0.1 \
    --dit_pool mean \
    --dit_loss_weight 0.1 \
    --weight_decay 0.01 \
    --train_steps 2000 \
    --warmup_steps 200 \
    --info_nce_weight 0.05 \
    --early_stop_patience 3 \
    --seed 1234

# Config 1b: DiT-4step (more refinement)
run_prompt_variants \
    "1b_dit_4step_64tok" \
    "DiT 64 tokens, 4 training steps (more denoising)" \
    --dataset gsm8k \
    --bridge dit \
    --lr 1e-4 \
    --dit_dim 512 \
    --soft_tokens -1 \
    --dit_depth 6 \
    --dit_heads 8 \
    --dit_steps_train 4 \
    --dit_steps_eval 8 \
    --dit_dropout 0.1 \
    --dit_pool mean \
    --dit_loss_weight 0.1 \
    --weight_decay 0.01 \
    --train_steps 2000 \
    --warmup_steps 200 \
    --info_nce_weight 0.05 \
    --early_stop_patience 3 \
    --seed 1234

# Config 1c: DiT with attention pooling (richer conditioning)
run_prompt_variants \
    "1c_dit_attn_64tok" \
    "DiT 64 tokens, attention pooling for source conditioning" \
    --dataset gsm8k \
    --bridge dit \
    --lr 1e-4 \
    --dit_dim 512 \
    --soft_tokens -1 \
    --dit_depth 6 \
    --dit_heads 8 \
    --dit_steps_train 2 \
    --dit_steps_eval 4 \
    --dit_dropout 0.1 \
    --dit_pool attn \
    --dit_loss_weight 0.1 \
    --weight_decay 0.01 \
    --train_steps 2000 \
    --warmup_steps 200 \
    --info_nce_weight 0.05 \
    --early_stop_patience 3 \
    --seed 1234

# Config 1d: DiT with CFG (classifier-free guidance)
run_prompt_variants \
    "1d_dit_cfg_64tok" \
    "DiT 64 tokens with CFG for better mode coverage" \
    --dataset gsm8k \
    --bridge dit \
    --lr 1e-4 \
    --dit_dim 512 \
    --soft_tokens -1 \
    --dit_depth 6 \
    --dit_heads 8 \
    --dit_steps_train 2 \
    --dit_steps_eval 4 \
    --dit_dropout 0.1 \
    --dit_pool mean \
    --dit_cfg 1.5 \
    --dit_cfg_dropout 0.1 \
    --dit_loss_weight 0.1 \
    --weight_decay 0.01 \
    --train_steps 2000 \
    --warmup_steps 200 \
    --info_nce_weight 0.05 \
    --early_stop_patience 3 \
    --seed 1234

# Config 1e: DiT with prompt-teacher + flow warmup
run_prompt_variants \
    "1e_dit_prompt_teacher_64tok" \
    "DiT 64 tokens, teacher=prompt, flow warmup 500" \
    --dataset gsm8k \
    --bridge dit \
    --lr 1e-4 \
    --dit_dim 512 \
    --soft_tokens -1 \
    --dit_depth 6 \
    --dit_heads 8 \
    --dit_steps_train 2 \
    --dit_steps_eval 4 \
    --dit_dropout 0.1 \
    --dit_pool mean \
    --dit_teacher prompt \
    --dit_loss_weight 0.1 \
    --dit_loss_warmup 500 \
    --weight_decay 0.01 \
    --train_steps 2000 \
    --warmup_steps 200 \
    --info_nce_weight 0.05 \
    --early_stop_patience 3 \
    --seed 1234

echo "" | tee -a "$SUMMARY_LOG"
echo "NOTE: DiT experiments test if iterative refinement prevents collapse (81.5% → 36%)" | tee -a "$SUMMARY_LOG"
echo "      Key question: Does DiT maintain high final accuracy vs cross-attention?" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ============================================
# ABLATION 2: Stability Fixes (64 tokens)
# Research Question: Do stability fixes prevent collapse?
# ============================================

echo "╔════════════════════════════════════════╗" | tee -a "$SUMMARY_LOG"
echo "║  ABLATION 2: STABILITY FIXES          ║" | tee -a "$SUMMARY_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Config 2a: WITH stability fixes (NEW RUN)
run_prompt_variants \
    "2a_stable_64tok" \
    "64 tokens WITH InfoNCE + early stopping + gen hygiene" \
    --dataset gsm8k \
    --lr 1e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens -1 \
    --depth 8 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 750 \
    --info_nce_weight 0.05 \
    --early_stop_patience 2 \
    --seed 1234

echo "" | tee -a "$SUMMARY_LOG"
echo "NOTE: Baseline (2b_baseline_64tok) reuses successful_experiments/cross_model/85/train_high_capacity.log" | tee -a "$SUMMARY_LOG"
echo "      Peak: 81.5% → Final: 36.0% (no stability fixes)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ============================================
# ABLATION 3: Sequence Length (all with stability)
# Research Question: Compression vs quality tradeoff?
# ============================================

echo "╔════════════════════════════════════════╗" | tee -a "$SUMMARY_LOG"
echo "║  ABLATION 3: SEQUENCE LENGTH          ║" | tee -a "$SUMMARY_LOG"
echo "╚════════════════════════════════════════╝" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Config 3a: 32 tokens (high compression)
run_prompt_variants \
    "3a_stable_32tok" \
    "32 tokens (4.7× compression) WITH stability fixes" \
    --dataset gsm8k \
    --lr 1e-4 \
    --bottleneck_dim 768 \
    --soft_tokens -1 \
    --depth 4 \
    --heads 12 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 600 \
    --info_nce_weight 0.05 \
    --early_stop_patience 2 \
    --seed 1234

# Config 3b: 48 tokens (medium compression)
run_prompt_variants \
    "3b_stable_48tok" \
    "48 tokens (3.1× compression) WITH stability fixes" \
    --dataset gsm8k \
    --lr 1e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens -1 \
    --depth 6 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 750 \
    --info_nce_weight 0.05 \
    --early_stop_patience 2 \
    --seed 1234

echo "" | tee -a "$SUMMARY_LOG"
echo "NOTE: 64 tokens result is same as 2a_stable_64tok (reused)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

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
    echo "Experiment,Tokens,Dataset,Peak,Final,FinalSource,FinalTarget,Degradation"

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
                    final_source=$(grep "Final.*Source-alone acc:" "$log" | tail -1 | \
                           grep -oE "Source-alone acc: [0-9.]+" | \
                           grep -oE "[0-9.]+" | \
                           awk '{printf "%.1f%%", $1 * 100}')

                    final_target=$(grep "Final.*Target-alone acc:" "$log" | tail -1 | \
                           grep -oE "Target-alone acc: [0-9.]+" | \
                           grep -oE "[0-9.]+" | \
                           awk '{printf "%.1f%%", $1 * 100}')

                    deg=$(awk -v p="$peak_val" -v f="$final_val" 'BEGIN{printf "%.1f%%", (p - f)}')
                    echo "$name,$tokens,$dataset,$peak,$final,$final_source,$final_target,$deg"
                fi
            fi
        fi
    done

    # Add baseline from existing experiment (for reference)
    echo "2b_baseline_64tok,64,GSM8K,81.5%,36.0%,N/A,73.0%,45.5%"

} | column -t -s',' | tee -a "$SUMMARY_LOG"

echo "" | tee -a "$SUMMARY_LOG"
echo "STABILITY ANALYSIS:" | tee -a "$SUMMARY_LOG"
echo "Cross-attention baseline (2b): Peak 81.5% → Final 36.0% (45.5% degradation)" | tee -a "$SUMMARY_LOG"
echo "Cross-attention w/ fixes (2a): Target <10% degradation" | tee -a "$SUMMARY_LOG"
echo "DiT experiments (1a-1d): Testing if iterative refinement prevents collapse" | tee -a "$SUMMARY_LOG"
echo "  → Success criteria: Final accuracy within 10% of peak" | tee -a "$SUMMARY_LOG"
echo "  → Key: Does DiT maintain performance better than cross-attention?" | tee -a "$SUMMARY_LOG"
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
        'final_target': 0,
        'final_source': 0
    }

    with open(log_file) as f:
        for line in f:
            # Extract eval lines
            if '[Eval] Step' in line and 'Bridged acc:' in line:
                parts = line.split('|')
                step = int(re.search(r'Step (\d+)', parts[0]).group(1))
                source = float(re.search(r'Source-alone acc: ([0-9.]+)', line).group(1))
                target = float(re.search(r'Target-alone acc: ([0-9.]+)', line).group(1))
                bridged = float(re.search(r'Bridged acc: ([0-9.]+)', line).group(1))

                results['evals'].append({
                    'step': step,
                    'source': source,
                    'target': target,
                    'bridged': bridged
                })

                if bridged > results['peak_bridged']:
                    results['peak_bridged'] = bridged
                    results['peak_step'] = step

            # Extract final results
            if '[Final Eval]' in line and 'Bridged acc:' in line:
                results['final_source'] = float(re.search(r'Source-alone acc: ([0-9.]+)', line).group(1))
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
    results['2b_baseline_64tok'] = {
        'evals': [
            {'step': 250, 'source': 0.120, 'target': 0.730, 'bridged': 0.290},
            {'step': 500, 'source': 0.120, 'target': 0.730, 'bridged': 0.655},
            {'step': 750, 'source': 0.120, 'target': 0.730, 'bridged': 0.535},
            {'step': 1000, 'source': 0.120, 'target': 0.730, 'bridged': 0.815},
            {'step': 1250, 'source': 0.120, 'target': 0.730, 'bridged': 0.755},
            {'step': 1500, 'source': 0.120, 'target': 0.730, 'bridged': 0.655},
            {'step': 2000, 'source': 0.120, 'target': 0.730, 'bridged': 0.635},
            {'step': 2500, 'source': 0.120, 'target': 0.730, 'bridged': 0.375},
            {'step': 3000, 'source': 0.120, 'target': 0.730, 'bridged': 0.360},
        ],
        'peak_bridged': 0.815,
        'peak_step': 1000,
        'final_bridged': 0.360,
        'final_target': 0.730,
        'final_source': 0.120
    }

    # Save raw results
    with open(script_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"{'Experiment':<25} {'PeakBr':<10} {'FinalBr':<10} {'FinalSrc':<10} {'FinalTgt':<10} {'Degradation':<12}")
    print("-" * 60)

    for name, data in sorted(results.items()):
        peak = data['peak_bridged'] * 100
        final = data['final_bridged'] * 100
        deg = peak - final
        final_source = data.get('final_source', 0) * 100
        final_target = data.get('final_target', 0) * 100
        print(f"{name:<25} {peak:>6.1f}%   {final:>6.1f}%   {final_source:>6.1f}%   {final_target:>6.1f}%   {deg:>6.1f}%")

    print(f"\nDetailed results saved to: {script_dir / 'ablation_results.json'}")

if __name__ == '__main__':
    main()
ANALYSIS_EOF

chmod +x "$OUTPUT_DIR/analyze_ablations.py"

echo "==========================================" | tee -a "$SUMMARY_LOG"
echo "Analysis script created: $OUTPUT_DIR/analyze_ablations.py" | tee -a "$SUMMARY_LOG"
echo "Run 'python $OUTPUT_DIR/analyze_ablations.py' to analyze results" | tee -a "$SUMMARY_LOG"
echo "==========================================" | tee -a "$SUMMARY_LOG"
