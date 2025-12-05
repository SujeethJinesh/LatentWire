#!/bin/bash
# run_overnight.sh
#
# OVERNIGHT EXPERIMENT SUITE
#
# Tests three hypotheses:
# 1. PRECISION: Can the bridge transmit exact data? (Passkey)
# 2. BANDWIDTH: Can 16 tokens handle 77 classes? (Banking77)
# 3. GENERALIZATION: Does it work on question types? (TREC)
#
# Expected runtime: ~3-4 hours total on 4xH100

set -e

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs}"
SOFT_TOKENS="${SOFT_TOKENS:-16}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/overnight_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_BASE"

echo "============================================================" | tee "$LOG_FILE"
echo "OVERNIGHT EXPERIMENT SUITE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Soft tokens: $SOFT_TOKENS" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Experiments:" | tee -a "$LOG_FILE"
echo "  1. Passkey   - Can bridge transmit exact 5-digit codes?" | tee -a "$LOG_FILE"
echo "  2. Banking77 - Can 16 tokens handle 77 classes?" | tee -a "$LOG_FILE"
echo "  3. TREC      - Does bridge generalize to question types?" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 1: PASSKEY (Precision Test)
# ============================================================
echo "[1/3] PASSKEY RETRIEVAL (Precision Test)" | tee -a "$LOG_FILE"
echo "Goal: Can the bridge transmit a 5-digit code exactly?" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

{
    python telepathy/train_telepathy_passkey.py \
        --output_dir "${OUTPUT_BASE}/passkey_${TIMESTAMP}" \
        --steps 1000 \
        --soft_tokens "$SOFT_TOKENS" \
        --batch_size 8 \
        --eval_every 200
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Passkey Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 2: BANKING77 (Bandwidth Test)
# ============================================================
echo "[2/3] BANKING77 (Bandwidth Stress Test)" | tee -a "$LOG_FILE"
echo "Goal: Can 16 tokens distinguish 77 banking intents?" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

{
    python telepathy/train_telepathy_banking77.py \
        --output_dir "${OUTPUT_BASE}/banking77_${TIMESTAMP}" \
        --steps 3000 \
        --soft_tokens "$SOFT_TOKENS" \
        --batch_size 8 \
        --eval_every 500
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Banking77 Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 3: TREC (Generalization Test)
# ============================================================
echo "[3/3] TREC (Generalization Test)" | tee -a "$LOG_FILE"
echo "Goal: Does bridge work on question type classification?" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

{
    python telepathy/train_telepathy_trec.py \
        --output_dir "${OUTPUT_BASE}/trec_${TIMESTAMP}" \
        --steps 2000 \
        --soft_tokens "$SOFT_TOKENS" \
        --batch_size 8 \
        --eval_every 400
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "TREC Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================" | tee -a "$LOG_FILE"
echo "OVERNIGHT SUITE COMPLETE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/passkey_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/banking77_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/trec_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "INTERPRETATION GUIDE:" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "PASSKEY (Precision):" | tee -a "$LOG_FILE"
echo "  < 10%  : Bridge is 'vibe only' - can't transmit data" | tee -a "$LOG_FILE"
echo "  10-80% : Bridge is lossy - try more tokens" | tee -a "$LOG_FILE"
echo "  > 80%  : High-fidelity channel!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "BANKING77 (77 classes):" | tee -a "$LOG_FILE"
echo "  < 30%  : Bandwidth insufficient for fine-grained" | tee -a "$LOG_FILE"
echo "  30-60% : Decent, but needs more tokens" | tee -a "$LOG_FILE"
echo "  > 60%  : Excellent - bridge has massive capacity" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "TREC (50 classes, question types):" | tee -a "$LOG_FILE"
echo "  < 20%  : Doesn't generalize to this domain" | tee -a "$LOG_FILE"
echo "  20-50% : Partial generalization" | tee -a "$LOG_FILE"
echo "  > 50%  : Excellent cross-domain transfer" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
