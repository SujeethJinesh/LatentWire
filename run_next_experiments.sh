#!/bin/bash
# run_next_experiments.sh
#
# NEXT EXPERIMENTS: Token Ablations + Text-Relay Baseline
#
# 1. Banking77 Token Ablation (16, 32, 64, 128 tokens)
# 2. Passkey Token Ablation (16, 32, 64, 128 tokens)
# 3. Text-Relay Baseline (SST-2, AG News)
#
# Expected runtime: ~3-4 hours on 4xH100

set -e

# Configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE}/next_experiments_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_BASE"

echo "============================================================" | tee "$LOG_FILE"
echo "NEXT EXPERIMENTS: Token Ablations + Text-Relay Baseline" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Experiments:" | tee -a "$LOG_FILE"
echo "  1. Banking77 Token Ablation (16, 32, 64, 128)" | tee -a "$LOG_FILE"
echo "  2. Passkey Token Ablation (16, 32, 64, 128)" | tee -a "$LOG_FILE"
echo "  3. Text-Relay Baseline (SST-2, AG News)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 1: BANKING77 TOKEN ABLATION
# ============================================================
echo "[1/3] BANKING77 TOKEN ABLATION" | tee -a "$LOG_FILE"
echo "Goal: Test if more tokens help with 77-class classification" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

for TOKENS in 16 32 64 128; do
    echo "" | tee -a "$LOG_FILE"
    echo ">>> Banking77 with ${TOKENS} tokens" | tee -a "$LOG_FILE"

    BANKING_DIR="${OUTPUT_BASE}/banking77_${TOKENS}tok_${TIMESTAMP}"
    mkdir -p "$BANKING_DIR"

    {
        python telepathy/train_telepathy_banking77.py \
            --output_dir "$BANKING_DIR" \
            --soft_tokens $TOKENS \
            --steps 3000 \
            --batch_size 8 \
            --eval_every 500
    } 2>&1 | tee -a "$LOG_FILE" | tee "${BANKING_DIR}/banking77_${TOKENS}tok.log"

    echo "Banking77 ${TOKENS} tokens complete: $(date)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Banking77 Ablation Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 2: PASSKEY TOKEN ABLATION
# ============================================================
echo "[2/3] PASSKEY TOKEN ABLATION" | tee -a "$LOG_FILE"
echo "Goal: Test if more tokens enable precision (5-digit codes)" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

for TOKENS in 16 32 64 128; do
    echo "" | tee -a "$LOG_FILE"
    echo ">>> Passkey with ${TOKENS} tokens" | tee -a "$LOG_FILE"

    PASSKEY_DIR="${OUTPUT_BASE}/passkey_${TOKENS}tok_${TIMESTAMP}"
    mkdir -p "$PASSKEY_DIR"

    {
        python telepathy/train_telepathy_passkey.py \
            --output_dir "$PASSKEY_DIR" \
            --soft_tokens $TOKENS \
            --steps 1000 \
            --batch_size 8 \
            --eval_every 200
    } 2>&1 | tee -a "$LOG_FILE" | tee "${PASSKEY_DIR}/passkey_${TOKENS}tok.log"

    echo "Passkey ${TOKENS} tokens complete: $(date)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Passkey Ablation Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 3: TEXT-RELAY BASELINE
# ============================================================
echo "[3/3] TEXT-RELAY BASELINE" | tee -a "$LOG_FILE"
echo "Goal: Compare bridge vs Llama-summarizes→text→Mistral" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

TEXT_RELAY_DIR="${OUTPUT_BASE}/text_relay_${TIMESTAMP}"
mkdir -p "$TEXT_RELAY_DIR"

{
    python telepathy/eval_text_relay_baseline.py \
        --output_dir "$TEXT_RELAY_DIR" \
        --num_samples 200
} 2>&1 | tee -a "$LOG_FILE" | tee "${TEXT_RELAY_DIR}/text_relay.log"

echo "" | tee -a "$LOG_FILE"
echo "Text-Relay Baseline Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  Banking77 ablation:" | tee -a "$LOG_FILE"
for TOKENS in 16 32 64 128; do
    echo "    - ${OUTPUT_BASE}/banking77_${TOKENS}tok_${TIMESTAMP}/" | tee -a "$LOG_FILE"
done
echo "  Passkey ablation:" | tee -a "$LOG_FILE"
for TOKENS in 16 32 64 128; do
    echo "    - ${OUTPUT_BASE}/passkey_${TOKENS}tok_${TIMESTAMP}/" | tee -a "$LOG_FILE"
done
echo "  Text-relay baseline:" | tee -a "$LOG_FILE"
echo "    - ${OUTPUT_BASE}/text_relay_${TIMESTAMP}/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Main log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Print quick summary of results
echo "" | tee -a "$LOG_FILE"
echo "QUICK RESULTS SUMMARY:" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Banking77 Token Ablation:" | tee -a "$LOG_FILE"
for TOKENS in 16 32 64 128; do
    JSON_FILE="${OUTPUT_BASE}/banking77_${TOKENS}tok_${TIMESTAMP}/banking77_results.json"
    if [ -f "$JSON_FILE" ]; then
        ACC=$(python -c "import json; print(json.load(open('$JSON_FILE'))['final_results']['accuracy'])" 2>/dev/null || echo "N/A")
        echo "  ${TOKENS} tokens: ${ACC}%" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Passkey Token Ablation:" | tee -a "$LOG_FILE"
for TOKENS in 16 32 64 128; do
    JSON_FILE="${OUTPUT_BASE}/passkey_${TOKENS}tok_${TIMESTAMP}/passkey_results.json"
    if [ -f "$JSON_FILE" ]; then
        EXACT=$(python -c "import json; print(json.load(open('$JSON_FILE'))['final_results']['exact_match'])" 2>/dev/null || echo "N/A")
        DIGIT=$(python -c "import json; print(json.load(open('$JSON_FILE'))['final_results']['digit_accuracy'])" 2>/dev/null || echo "N/A")
        echo "  ${TOKENS} tokens: ${EXACT}% exact, ${DIGIT}% digit" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Text-Relay Baseline:" | tee -a "$LOG_FILE"
JSON_FILE="${OUTPUT_BASE}/text_relay_${TIMESTAMP}/text_relay_results.json"
if [ -f "$JSON_FILE" ]; then
    SST2=$(python -c "import json; print(json.load(open('$JSON_FILE'))['results']['sst2']['accuracy'])" 2>/dev/null || echo "N/A")
    AGNEWS=$(python -c "import json; print(json.load(open('$JSON_FILE'))['results']['agnews']['accuracy'])" 2>/dev/null || echo "N/A")
    echo "  SST-2: ${SST2}% (Bridge: 94.7%, Mistral: 93.5%)" | tee -a "$LOG_FILE"
    echo "  AG News: ${AGNEWS}% (Bridge: 88.9%, Mistral: 70.5%)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
