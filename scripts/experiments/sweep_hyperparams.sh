#!/usr/bin/env bash
#
# Hyperparameter Sweep for LatentWire
#
# Searches over key hyperparameters to find best configuration:
# - Latent length M (compression ratio)
# - Latent dimension d_z (capacity)
# - First token CE weight (training objective balance)
# - K-token teacher forcing
#
# Usage: PYTHONPATH=. bash scripts/experiments/sweep_hyperparams.sh
#

set -e

echo "========================================================================"
echo "HYPERPARAMETER SWEEP - LatentWire"
echo "========================================================================"
echo ""
echo "Searching over:"
echo "  - Latent length M ∈ {16, 32, 48, 64}"
echo "  - Latent dim d_z ∈ {128, 256, 512}"
echo "  - First token CE weight ∈ {0.0, 0.5, 1.0}"
echo "  - K-token teacher forcing ∈ {1, 4, 8}"
echo ""
echo "This will run ~48 training jobs. Estimated time: ~6-8 hours on 4x H100"
echo ""
echo "========================================================================"
echo ""

# Configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-10000}"  # Use subset for sweep
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-runs/sweeps/hyperparam}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_LOG="$BASE_OUTPUT_DIR/sweep_${TIMESTAMP}.log"

echo "Sweep configuration:" | tee "$SWEEP_LOG"
echo "  SAMPLES: $SAMPLES" | tee -a "$SWEEP_LOG"
echo "  EPOCHS: $EPOCHS" | tee -a "$SWEEP_LOG"
echo "  BATCH_SIZE: $BATCH_SIZE" | tee -a "$SWEEP_LOG"
echo "  BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

# Hyperparameter grid
LATENT_LENS=(32 48 64)
D_ZS=(256 512)
FTCE_WEIGHTS=(0.5 1.0)
K_VALUES=(4 8)

TOTAL_RUNS=$((${#LATENT_LENS[@]} * ${#D_ZS[@]} * ${#FTCE_WEIGHTS[@]} * ${#K_VALUES[@]}))
CURRENT_RUN=0

echo "Starting sweep with $TOTAL_RUNS configurations..." | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for M in "${LATENT_LENS[@]}"; do
    for D_Z in "${D_ZS[@]}"; do
        for FTCE in "${FTCE_WEIGHTS[@]}"; do
            for K in "${K_VALUES[@]}"; do
                CURRENT_RUN=$((CURRENT_RUN + 1))

                RUN_NAME="M${M}_dz${D_Z}_ftce${FTCE}_K${K}"
                OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"

                echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $RUN_NAME" | tee -a "$SWEEP_LOG"
                echo "  M=$M, d_z=$D_Z, ftce_weight=$FTCE, K=$K" | tee -a "$SWEEP_LOG"

                # Run training
                {
                    python latentwire/train.py \
                        --llama_id "$LLAMA_ID" \
                        --qwen_id "$QWEN_ID" \
                        --samples "$SAMPLES" \
                        --epochs "$EPOCHS" \
                        --batch_size "$BATCH_SIZE" \
                        --latent_len "$M" \
                        --d_z "$D_Z" \
                        --encoder_type byte \
                        --dataset "$DATASET" \
                        --sequential_models \
                        --warm_anchor_text "Answer: " \
                        --first_token_ce_weight "$FTCE" \
                        --K "$K" \
                        --save_dir "$OUTPUT_DIR"
                } 2>&1 | tee "$OUTPUT_DIR/train.log"

                echo "  ✓ Complete: $OUTPUT_DIR" | tee -a "$SWEEP_LOG"
                echo "" | tee -a "$SWEEP_LOG"
            done
        done
    done
done

echo "========================================================================"  | tee -a "$SWEEP_LOG"
echo "SWEEP COMPLETE!"  | tee -a "$SWEEP_LOG"
echo "========================================================================"  | tee -a "$SWEEP_LOG"
echo ""  | tee -a "$SWEEP_LOG"
echo "Results saved to: $BASE_OUTPUT_DIR"  | tee -a "$SWEEP_LOG"
echo "Sweep log: $SWEEP_LOG"  | tee -a "$SWEEP_LOG"
echo ""  | tee -a "$SWEEP_LOG"

# Analyze results
echo "Analyzing sweep results..." | tee -a "$SWEEP_LOG"
python scripts/experiments/analyze_sweep.py --sweep_dir "$BASE_OUTPUT_DIR" | tee -a "$SWEEP_LOG"

echo ""
echo "To view best configurations:"
echo "  cat $BASE_OUTPUT_DIR/sweep_summary.json | python -m json.tool | less"
echo ""
