#!/usr/bin/env bash
# Run LLMLingua baseline compression experiments for comparison with LatentWire
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/llmlingua_baseline}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-200}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/llmlingua_baseline_${TIMESTAMP}.log"

echo "=============================================================="
echo "LLMLingua Baseline Experiments"
echo "=============================================================="
echo "Dataset: $DATASET"
echo "Samples: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=============================================================="
echo ""

# Run LLMLingua baseline experiments
{
    echo "Starting LLMLingua baseline compression experiments..."
    echo ""

    # Install LLMLingua if not already installed
    echo "Checking LLMLingua installation..."
    python -c "import llmlingua" 2>/dev/null || {
        echo "LLMLingua not found. Installing..."
        pip install llmlingua
    }
    echo ""

    # Experiment 1: LLMLingua-2 at M=32 (same as LatentWire)
    echo "----------------------------------------"
    echo "Experiment 1: LLMLingua-2 (M=32)"
    echo "----------------------------------------"
    python latentwire/llmlingua_baseline.py \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --target_tokens 32 \
        --use_llmlingua2 \
        --question_aware \
        --output_dir "$OUTPUT_DIR/llmlingua2_m32"
    echo ""

    # Experiment 2: LLMLingua-2 at M=64
    echo "----------------------------------------"
    echo "Experiment 2: LLMLingua-2 (M=64)"
    echo "----------------------------------------"
    python latentwire/llmlingua_baseline.py \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --target_tokens 64 \
        --use_llmlingua2 \
        --question_aware \
        --output_dir "$OUTPUT_DIR/llmlingua2_m64"
    echo ""

    # Experiment 3: LLMLingua-2 at M=128
    echo "----------------------------------------"
    echo "Experiment 3: LLMLingua-2 (M=128)"
    echo "----------------------------------------"
    python latentwire/llmlingua_baseline.py \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --target_tokens 128 \
        --use_llmlingua2 \
        --question_aware \
        --output_dir "$OUTPUT_DIR/llmlingua2_m128"
    echo ""

    # Experiment 4: Question-agnostic compression (ablation)
    echo "----------------------------------------"
    echo "Experiment 4: Question-agnostic (M=32)"
    echo "----------------------------------------"
    python latentwire/llmlingua_baseline.py \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --target_tokens 32 \
        --use_llmlingua2 \
        --no-question_aware \
        --output_dir "$OUTPUT_DIR/llmlingua2_m32_qagnostic"
    echo ""

    # Experiment 5: Original LLMLingua (unidirectional) for comparison
    echo "----------------------------------------"
    echo "Experiment 5: LLMLingua v1 (M=32)"
    echo "----------------------------------------"
    echo "NOTE: This requires more memory. Using GPT-2 Small instead of LLaMA."
    python latentwire/llmlingua_baseline.py \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --target_tokens 32 \
        --no-use_llmlingua2 \
        --compressor_model "gpt2" \
        --question_aware \
        --output_dir "$OUTPUT_DIR/llmlingua1_m32"
    echo ""

    echo "=============================================================="
    echo "All experiments complete!"
    echo "=============================================================="
    echo ""
    echo "Results saved to:"
    echo "  - $OUTPUT_DIR/llmlingua2_m32/llmlingua_results.json"
    echo "  - $OUTPUT_DIR/llmlingua2_m64/llmlingua_results.json"
    echo "  - $OUTPUT_DIR/llmlingua2_m128/llmlingua_results.json"
    echo "  - $OUTPUT_DIR/llmlingua2_m32_qagnostic/llmlingua_results.json"
    echo "  - $OUTPUT_DIR/llmlingua1_m32/llmlingua_results.json"
    echo ""
    echo "To analyze results, run:"
    echo "  python latentwire/analyze_llmlingua_results.py --results_dir $OUTPUT_DIR"
    echo "=============================================================="

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
