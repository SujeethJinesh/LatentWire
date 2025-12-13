#!/bin/bash
# run_trec.sh
#
# Train and evaluate bridge on TREC 6-class question type classification
# Part of paper generalization story: SST-2 (2), AG News (4), TREC (6), Banking77 (77)
# Expected runtime: ~30-45 minutes on 1 GPU

set -e

OUTPUT_DIR="${OUTPUT_DIR:-runs/trec_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/trec.log"

echo "============================================================"
echo "TREC 6-CLASS QUESTION TYPE CLASSIFICATION"
echo "============================================================"
echo "Classes: ABBR, DESC, ENTY, HUM, LOC, NUM"
echo "Soft tokens: 16"
echo "Steps: 2000"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================"

{
    python telepathy/train_telepathy_trec.py \
        --output_dir "$OUTPUT_DIR" \
        --soft_tokens 16 \
        --steps 2000 \
        --batch_size 8 \
        --lr 1e-4 \
        --eval_every 400 \
        --diversity_weight 0.1 \
        --gpu "$GPU"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results: $OUTPUT_DIR/trec_results.json"
echo "Checkpoint: $OUTPUT_DIR/bridge_trec.pt"
echo "Log: $LOG_FILE"
echo "============================================================"
