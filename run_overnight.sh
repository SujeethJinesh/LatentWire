#!/bin/bash
# run_overnight.sh
#
# COMPLETE EXPERIMENT SUITE
#
# Runs all experiments with interpretability analysis:
# 1. SST-2      - Binary sentiment (preserved data rerun)
# 2. AG News    - 4-class topic (preserved data rerun)
# 3. Passkey    - Precision test (can bridge transmit exact data?)
# 4. Banking77  - Bandwidth test (77 classes)
# 5. TREC       - Generalization test (question types)
#
# Expected runtime: ~5-6 hours total on 4xH100

set -e

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/overnight_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_BASE"

echo "============================================================" | tee "$LOG_FILE"
echo "COMPLETE EXPERIMENT SUITE (with Interpretability)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Experiments:" | tee -a "$LOG_FILE"
echo "  1. SST-2     - Binary sentiment classification" | tee -a "$LOG_FILE"
echo "  2. AG News   - 4-class topic classification" | tee -a "$LOG_FILE"
echo "  3. Passkey   - Can bridge transmit exact 5-digit codes?" | tee -a "$LOG_FILE"
echo "  4. Banking77 - Can 16 tokens handle 77 classes?" | tee -a "$LOG_FILE"
echo "  5. TREC      - Does bridge generalize to question types?" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 1: SST-2 (Sentiment - Paper Ready)
# ============================================================
echo "[1/5] SST-2 SENTIMENT (Binary Classification)" | tee -a "$LOG_FILE"
echo "Goal: Validate bridge on binary sentiment task" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

mkdir -p "${OUTPUT_BASE}/sst2_${TIMESTAMP}"
{
    python telepathy/train_telepathy_sst2.py \
        --source_layer 31 \
        --soft_tokens 8 \
        --steps 2000 \
        --batch_size 16 \
        --eval_every 200 \
        --save_path "${OUTPUT_BASE}/sst2_${TIMESTAMP}/bridge_sst2.pt"
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "SST-2 Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 2: AG NEWS (4-class - Paper Ready)
# ============================================================
echo "[2/5] AG NEWS (4-class Topic Classification)" | tee -a "$LOG_FILE"
echo "Goal: Validate bridge on 4-class topic task" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

mkdir -p "${OUTPUT_BASE}/agnews_${TIMESTAMP}"
{
    python telepathy/train_telepathy_agnews.py \
        --source_layer 31 \
        --soft_tokens 8 \
        --steps 3000 \
        --batch_size 16 \
        --eval_every 200 \
        --save_path "${OUTPUT_BASE}/agnews_${TIMESTAMP}/bridge_agnews.pt"
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "AG News Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 3: PASSKEY (Precision Test)
# ============================================================
echo "[3/5] PASSKEY RETRIEVAL (Precision Test)" | tee -a "$LOG_FILE"
echo "Goal: Can the bridge transmit a 5-digit code exactly?" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

{
    python telepathy/train_telepathy_passkey.py \
        --output_dir "${OUTPUT_BASE}/passkey_${TIMESTAMP}" \
        --steps 1000 \
        --soft_tokens 16 \
        --batch_size 8 \
        --eval_every 200
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Passkey Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 4: BANKING77 (Bandwidth Test)
# ============================================================
echo "[4/5] BANKING77 (Bandwidth Stress Test)" | tee -a "$LOG_FILE"
echo "Goal: Can 16 tokens distinguish 77 banking intents?" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

{
    python telepathy/train_telepathy_banking77.py \
        --output_dir "${OUTPUT_BASE}/banking77_${TIMESTAMP}" \
        --steps 3000 \
        --soft_tokens 16 \
        --batch_size 8 \
        --eval_every 500
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Banking77 Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 5: TREC (Generalization Test)
# ============================================================
echo "[5/5] TREC (Generalization Test)" | tee -a "$LOG_FILE"
echo "Goal: Does bridge work on question type classification?" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

{
    python telepathy/train_telepathy_trec.py \
        --output_dir "${OUTPUT_BASE}/trec_${TIMESTAMP}" \
        --steps 2000 \
        --soft_tokens 16 \
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
echo "EXPERIMENT SUITE COMPLETE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/sst2_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/agnews_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/passkey_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/banking77_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "  - ${OUTPUT_BASE}/trec_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "INTERPRETATION GUIDE:" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "SST-2 (2 classes):" | tee -a "$LOG_FILE"
echo "  Target: > 93% (match Mistral text baseline)" | tee -a "$LOG_FILE"
echo "  Previous best: 94.72%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "AG NEWS (4 classes):" | tee -a "$LOG_FILE"
echo "  Target: > 85% (exceed Mistral text baseline)" | tee -a "$LOG_FILE"
echo "  Previous best: 88.9%" | tee -a "$LOG_FILE"
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
echo "" | tee -a "$LOG_FILE"
echo "INTERPRETABILITY:" | tee -a "$LOG_FILE"
echo "  Each experiment outputs nearest vocabulary tokens" | tee -a "$LOG_FILE"
echo "  for each soft token - reveals what concepts transfer" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
