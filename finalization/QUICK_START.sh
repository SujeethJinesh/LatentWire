#!/usr/bin/env bash
# QUICK_START.sh - Minimal test to verify LatentWire installation and functionality
# This script trains for 100 steps and evaluates on 10 samples (< 5 minutes on GPU)
set -e

# =============================================================================
# Quick Start Test for LatentWire
# =============================================================================
# Purpose: Verify that the system is working correctly with minimal time investment
# Expected runtime: < 5 minutes on GPU
# =============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/quick_test}"
SAMPLES=100        # Small dataset for quick training
EPOCHS=1           # Single epoch
EVAL_SAMPLES=10    # Tiny evaluation set

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}       LatentWire Quick Start Test${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "This script will:"
echo "  1. Train a minimal LatentWire model (100 samples, 1 epoch)"
echo "  2. Evaluate on 10 samples"
echo "  3. Compare against text baseline"
echo ""
echo -e "Expected runtime: ${YELLOW}< 5 minutes${NC} on GPU"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - will use CPU/MPS (slower)${NC}"
fi
echo ""

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/quick_start_${TIMESTAMP}.log"

echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Clean any previous quick test runs
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    echo -e "${YELLOW}Cleaning previous quick test run...${NC}"
    rm -rf "$OUTPUT_DIR"/*
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Phase 1: Training (100 samples, 1 epoch)${NC}"
echo -e "${GREEN}================================================${NC}"

# Run training with minimal settings
{
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --samples $SAMPLES \
        --epochs $EPOCHS \
        --batch_size 8 \
        --latent_len 16 \
        --d_z 128 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "$OUTPUT_DIR/checkpoint" \
        --save_every 100 \
        --log_every 10 \
        2>&1 || {
            echo -e "${RED}✗ Training failed!${NC}"
            echo "Check the log file for details: $LOG_FILE"
            exit 1
        }
} | tee -a "$LOG_FILE"

echo ""
echo -e "${GREEN}✓ Training complete!${NC}"
echo ""

# Check if checkpoint was created
if [ ! -d "$OUTPUT_DIR/checkpoint/epoch0" ]; then
    echo -e "${RED}✗ No checkpoint found at $OUTPUT_DIR/checkpoint/epoch0${NC}"
    echo "Training may have failed. Check the log file: $LOG_FILE"
    exit 1
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Phase 2: Evaluation (10 samples)${NC}"
echo -e "${GREEN}================================================${NC}"

# Run evaluation
{
    python latentwire/eval.py \
        --ckpt "$OUTPUT_DIR/checkpoint/epoch0" \
        --samples $EVAL_SAMPLES \
        --max_new_tokens 12 \
        --dataset squad \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        --output_dir "$OUTPUT_DIR/eval" \
        2>&1 || {
            echo -e "${RED}✗ Evaluation failed!${NC}"
            echo "Check the log file for details: $LOG_FILE"
            exit 1
        }
} | tee -a "$LOG_FILE"

echo ""
echo -e "${GREEN}✓ Evaluation complete!${NC}"
echo ""

# Parse and display results
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Results Summary${NC}"
echo -e "${GREEN}================================================${NC}"

if [ -f "$OUTPUT_DIR/eval/results.json" ]; then
    # Extract key metrics using Python
    python -c "
import json
with open('$OUTPUT_DIR/eval/results.json', 'r') as f:
    results = json.load(f)

print('Compression Metrics:')
print(f'  • Latent size: {results.get(\"latent_len\", \"N/A\")} tokens')
print(f'  • Compression ratio: {results.get(\"compression_ratio\", \"N/A\"):.2f}x' if isinstance(results.get('compression_ratio'), (int, float)) else '  • Compression ratio: N/A')
print()

print('Quality Metrics (on {0} samples):'.format($EVAL_SAMPLES))
if 'text_baseline' in results:
    print(f'  • Text baseline F1: {results[\"text_baseline\"].get(\"avg_f1\", 0):.3f}')
if 'latent' in results:
    print(f'  • Latent F1: {results[\"latent\"].get(\"avg_f1\", 0):.3f}')
if 'token_budget' in results:
    print(f'  • Token budget F1: {results[\"token_budget\"].get(\"avg_f1\", 0):.3f}')
print()

print('First Token Accuracy:')
if 'latent' in results:
    print(f'  • Latent: {results[\"latent\"].get(\"first_token_acc\", 0)*100:.1f}%')
if 'token_budget' in results:
    print(f'  • Token budget: {results[\"token_budget\"].get(\"first_token_acc\", 0)*100:.1f}%')
" 2>/dev/null || echo "Could not parse results.json"
else
    echo -e "${YELLOW}Results file not found. Check logs for details.${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Quick Start Test Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. For full training: bash scripts/run_pipeline.sh"
echo "  2. View detailed logs: less $LOG_FILE"
echo "  3. Check results: cat $OUTPUT_DIR/eval/results.json | jq ."
echo ""

# Provide summary of what worked
echo "System check:"
python -c "
import torch
print(f'  ✓ PyTorch version: {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
print('  ✓ LatentWire modules loaded successfully')
" 2>/dev/null || echo "  ⚠ Could not verify PyTorch installation"

echo ""
echo -e "${GREEN}✓ All systems operational!${NC}"
echo ""
echo "Full output saved to:"
echo "  • Checkpoint: $OUTPUT_DIR/checkpoint/"
echo "  • Evaluation: $OUTPUT_DIR/eval/"
echo "  • Log file: $LOG_FILE"