#!/bin/bash
# run_sst2_agnews_baselines.sh
#
# BLOCKING EXPERIMENT: Get Llama text baselines for SST-2 and AG News
# These are required to compare Bridge vs sender ceiling
#
# Usage: git pull && PYTHONPATH=. bash run_sst2_agnews_baselines.sh
#
# Expected runtime: ~30 min on H100
# Expected output: Llama SST-2 ~93-95%, Llama AG News ~85-90%

set -e

OUTPUT_BASE="${OUTPUT_BASE:-runs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/sst2_agnews_baselines_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "SST-2 + AG NEWS TEXT BASELINES"
echo "=============================================="
echo "Log file: $LOG_FILE"
echo ""

{
    echo "[$(date)] Starting SST-2 text baselines..."

    # SST-2 baselines
    SST2_DIR="${OUTPUT_BASE}/sst2_baselines_${TIMESTAMP}"
    mkdir -p "$SST2_DIR"

    python telepathy/eval_text_relay_baseline.py \
        --sst2_text \
        --num_samples 200 \
        --output_dir "$SST2_DIR" \
        --gpu 0

    echo ""
    echo "[$(date)] Starting AG News text baselines..."

    # AG News baselines
    AGNEWS_DIR="${OUTPUT_BASE}/agnews_baselines_${TIMESTAMP}"
    mkdir -p "$AGNEWS_DIR"

    python telepathy/eval_text_relay_baseline.py \
        --agnews_text \
        --num_samples 200 \
        --output_dir "$AGNEWS_DIR" \
        --gpu 0

    echo ""
    echo "=============================================="
    echo "COMPLETE!"
    echo "=============================================="
    echo ""
    echo "Results saved to:"
    echo "  - $SST2_DIR/sst2_baselines.json"
    echo "  - $AGNEWS_DIR/agnews_baselines.json"
    echo ""

    # Print summary
    echo "SUMMARY:"
    echo "--------"
    if [ -f "$SST2_DIR/sst2_baselines.json" ]; then
        SST2_LLAMA=$(python -c "import json; print(json.load(open('$SST2_DIR/sst2_baselines.json'))['results']['llama']['accuracy'])" 2>/dev/null || echo "N/A")
        SST2_MISTRAL=$(python -c "import json; print(json.load(open('$SST2_DIR/sst2_baselines.json'))['results']['mistral']['accuracy'])" 2>/dev/null || echo "N/A")
        echo "SST-2 Llama: ${SST2_LLAMA}% (Bridge: 94.7%, Text-Relay: 71.0%)"
        echo "SST-2 Mistral: ${SST2_MISTRAL}%"
    fi

    if [ -f "$AGNEWS_DIR/agnews_baselines.json" ]; then
        AGNEWS_LLAMA=$(python -c "import json; print(json.load(open('$AGNEWS_DIR/agnews_baselines.json'))['results']['llama']['accuracy'])" 2>/dev/null || echo "N/A")
        AGNEWS_MISTRAL=$(python -c "import json; print(json.load(open('$AGNEWS_DIR/agnews_baselines.json'))['results']['mistral']['accuracy'])" 2>/dev/null || echo "N/A")
        echo "AG News Llama: ${AGNEWS_LLAMA}% (Bridge: 88.9%, Text-Relay: 64.5%)"
        echo "AG News Mistral: ${AGNEWS_MISTRAL}%"
    fi

    echo ""
    echo "[$(date)] Done!"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Full log saved to: $LOG_FILE"
