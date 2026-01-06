#!/usr/bin/env bash
# =============================================================================
# Master script to run all reviewer response experiments
# =============================================================================
#
# This script runs experiments to address the 6 major reviewer critiques:
# 1. Only Classification, No Generation/Reasoning (6/10 reviewers)
# 2. Missing Fine-tuning (LoRA) Comparison (4/10 reviewers)
# 3. Weak Individual Model Baselines (Zero-shot Only) (4/10 reviewers)
# 4. Text-Relay Baseline Artificially Weak (3/10 reviewers)
# 5. No Cross-Task Transfer (Zero-shot Bridge) (2/10 reviewers)
# 6. Missing Batched Latency/Throughput (2/10 reviewers)
#
# Usage:
#   bash run_reviewer_experiments.sh              # Run all experiments
#   bash run_reviewer_experiments.sh quick        # Run quick experiments only
#   bash run_reviewer_experiments.sh gsm8k        # Run GSM8K only
#
# =============================================================================

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/reviewer_experiments}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/master_log_${TIMESTAMP}.log"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "REVIEWER RESPONSE EXPERIMENTS"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Function to run experiment with logging
run_experiment() {
    local name="$1"
    local cmd="$2"
    local output_subdir="$3"

    echo ""
    echo "=============================================="
    echo "Running: $name"
    echo "=============================================="
    echo "Started at: $(date)"

    mkdir -p "$OUTPUT_DIR/$output_subdir"

    {
        echo "=== $name ==="
        echo "Started: $(date)"
        echo "Command: $cmd"
        echo ""
        eval "$cmd"
        echo ""
        echo "Completed: $(date)"
    } 2>&1 | tee -a "$LOG_FILE" | tee "$OUTPUT_DIR/$output_subdir/${name}_${TIMESTAMP}.log"

    echo "Completed: $name"
}

# Parse command line argument
MODE="${1:-all}"

case $MODE in
    quick)
        echo "Running QUICK experiments only (Phase 1)"
        echo ""

        # Critique 3: Few-shot baselines (~30 min)
        run_experiment "fewshot_sst2" \
            "python telepathy/eval_fewshot_baselines.py --dataset sst2 --shots 5 --max_samples 200 --output_dir $OUTPUT_DIR/fewshot" \
            "fewshot"

        run_experiment "fewshot_agnews" \
            "python telepathy/eval_fewshot_baselines.py --dataset agnews --shots 5 --max_samples 200 --output_dir $OUTPUT_DIR/fewshot" \
            "fewshot"

        run_experiment "fewshot_trec" \
            "python telepathy/eval_fewshot_baselines.py --dataset trec --shots 5 --max_samples 200 --output_dir $OUTPUT_DIR/fewshot" \
            "fewshot"

        # Critique 6: Batched latency (~30 min)
        run_experiment "batched_latency" \
            "python telepathy/benchmark_batched_latency.py --batch_sizes 1 2 4 8 16 --num_samples 32 --output_dir $OUTPUT_DIR/latency" \
            "latency"
        ;;

    moderate)
        echo "Running MODERATE experiments (Phase 2)"
        echo ""

        # Critique 2: LoRA comparison (~2 hours per dataset)
        run_experiment "lora_sst2" \
            "python telepathy/train_lora_baseline.py --dataset sst2 --rank 8 --epochs 3 --max_train_samples 2000 --output_dir $OUTPUT_DIR/lora" \
            "lora"

        run_experiment "lora_agnews" \
            "python telepathy/train_lora_baseline.py --dataset agnews --rank 8 --epochs 3 --max_train_samples 2000 --output_dir $OUTPUT_DIR/lora" \
            "lora"

        # Critique 4: CoT text-relay (~1 hour per dataset)
        run_experiment "cot_sst2" \
            "python telepathy/eval_cot_relay.py --dataset sst2 --max_samples 100 --output_dir $OUTPUT_DIR/cot_relay" \
            "cot_relay"

        run_experiment "cot_agnews" \
            "python telepathy/eval_cot_relay.py --dataset agnews --max_samples 100 --output_dir $OUTPUT_DIR/cot_relay" \
            "cot_relay"
        ;;

    gsm8k)
        echo "Running GSM8K reasoning experiment (Phase 3)"
        echo ""

        # Critique 1: GSM8K reasoning (~4 hours)
        run_experiment "gsm8k_bridge" \
            "python telepathy/train_gsm8k_bridge.py --num_train_samples 2000 --num_eval_samples 200 --epochs 5 --output_dir $OUTPUT_DIR/gsm8k" \
            "gsm8k"
        ;;

    transfer)
        echo "Running transfer experiments"
        echo ""

        # Critique 5: Cross-task transfer
        # Note: Requires trained bridge checkpoint
        if [ -f "$OUTPUT_DIR/../sst2/bridge.pt" ]; then
            run_experiment "transfer_from_sst2" \
                "python telepathy/eval_transfer.py --source_task sst2 --checkpoint $OUTPUT_DIR/../sst2/bridge.pt --output_dir $OUTPUT_DIR/transfer" \
                "transfer"
        else
            echo "WARNING: No SST-2 bridge checkpoint found. Skipping transfer test."
            echo "Train a bridge first with: python telepathy/train_telepathy_sst2.py"
        fi
        ;;

    all)
        echo "Running ALL experiments"
        echo "Estimated time: 6-8 hours"
        echo ""

        # Phase 1: Quick wins
        echo ""
        echo "========== PHASE 1: QUICK EXPERIMENTS =========="

        run_experiment "fewshot_sst2" \
            "python telepathy/eval_fewshot_baselines.py --dataset sst2 --shots 5 --max_samples 200 --output_dir $OUTPUT_DIR/fewshot" \
            "fewshot"

        run_experiment "fewshot_agnews" \
            "python telepathy/eval_fewshot_baselines.py --dataset agnews --shots 5 --max_samples 200 --output_dir $OUTPUT_DIR/fewshot" \
            "fewshot"

        run_experiment "fewshot_trec" \
            "python telepathy/eval_fewshot_baselines.py --dataset trec --shots 5 --max_samples 200 --output_dir $OUTPUT_DIR/fewshot" \
            "fewshot"

        run_experiment "batched_latency" \
            "python telepathy/benchmark_batched_latency.py --batch_sizes 1 2 4 8 16 --num_samples 32 --output_dir $OUTPUT_DIR/latency" \
            "latency"

        # Phase 2: Moderate effort
        echo ""
        echo "========== PHASE 2: MODERATE EXPERIMENTS =========="

        run_experiment "lora_sst2" \
            "python telepathy/train_lora_baseline.py --dataset sst2 --rank 8 --epochs 3 --max_train_samples 2000 --output_dir $OUTPUT_DIR/lora" \
            "lora"

        run_experiment "cot_sst2" \
            "python telepathy/eval_cot_relay.py --dataset sst2 --max_samples 100 --output_dir $OUTPUT_DIR/cot_relay" \
            "cot_relay"

        # Phase 3: Major experiment
        echo ""
        echo "========== PHASE 3: MAJOR EXPERIMENT =========="

        run_experiment "gsm8k_bridge" \
            "python telepathy/train_gsm8k_bridge.py --num_train_samples 2000 --num_eval_samples 200 --epochs 5 --output_dir $OUTPUT_DIR/gsm8k" \
            "gsm8k"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [quick|moderate|gsm8k|transfer|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Master log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "1. Review results in $OUTPUT_DIR"
echo "2. Update paper with new findings"
echo "3. Update REVIEWER_RESPONSE_PLAN.md with outcomes"
echo ""

# Generate summary report
{
    echo "=============================================="
    echo "EXPERIMENT SUMMARY"
    echo "=============================================="
    echo "Completed at: $(date)"
    echo ""
    echo "Output files:"
    find "$OUTPUT_DIR" -name "*.json" -type f 2>/dev/null | head -20
    echo ""
} | tee -a "$LOG_FILE"
