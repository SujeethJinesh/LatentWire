#!/bin/bash
# run_next_experiments.sh
#
# NEXT EXPERIMENTS: Token Ablations + Text-Relay Baseline
# NOW WITH 4x PARALLELISM - one experiment per GPU!
#
# 1. Banking77 Token Ablation (16, 32, 64, 128 tokens) - 4 GPUs in parallel
# 2. Passkey Token Ablation (16, 32, 64, 128 tokens) - 4 GPUs in parallel
# 3. Text-Relay Baseline (SST-2, AG News)
#
# Expected runtime: ~1 hour on 4xH100 (4x speedup from parallel execution)

set -e

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

# Show latest progress from each parallel experiment
show_progress() {
    local experiment_type=$1
    shift
    local log_files=("$@")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PROGRESS UPDATE: $experiment_type @ $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            local name=$(basename "$log_file" .log)
            # Get the last training progress line or evaluation result
            local progress=$(grep -E "(Training:|Accuracy:|Exact Match:|Step)" "$log_file" 2>/dev/null | tail -1)
            if [ -n "$progress" ]; then
                printf "  %-20s %s\n" "$name:" "$progress"
            else
                # Show last non-empty line as fallback
                local last_line=$(tail -1 "$log_file" 2>/dev/null | head -c 60)
                printf "  %-20s %s\n" "$name:" "${last_line}..."
            fi
        fi
    done
    echo ""
}

# Monitor parallel jobs with periodic updates
monitor_jobs() {
    local experiment_type=$1
    local update_interval=${2:-30}  # seconds between updates
    shift 2
    local pids=("$@")

    echo ""
    echo "Monitoring ${#pids[@]} parallel jobs (updates every ${update_interval}s)..."
    echo "Press Ctrl+C to stop monitoring (jobs continue in background)"
    echo ""

    while true; do
        # Check if any jobs still running
        local running=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running++))
            fi
        done

        if [ $running -eq 0 ]; then
            echo "All $experiment_type jobs completed!"
            break
        fi

        sleep $update_interval

        # Show progress based on experiment type
        if [[ "$experiment_type" == "Banking77" ]]; then
            show_progress "$experiment_type" \
                "${OUTPUT_BASE}/banking77_16tok_${TIMESTAMP}/banking77_16tok.log" \
                "${OUTPUT_BASE}/banking77_32tok_${TIMESTAMP}/banking77_32tok.log" \
                "${OUTPUT_BASE}/banking77_64tok_${TIMESTAMP}/banking77_64tok.log" \
                "${OUTPUT_BASE}/banking77_128tok_${TIMESTAMP}/banking77_128tok.log"
        elif [[ "$experiment_type" == "Passkey" ]]; then
            show_progress "$experiment_type" \
                "${OUTPUT_BASE}/passkey_16tok_${TIMESTAMP}/passkey_16tok.log" \
                "${OUTPUT_BASE}/passkey_32tok_${TIMESTAMP}/passkey_32tok.log" \
                "${OUTPUT_BASE}/passkey_64tok_${TIMESTAMP}/passkey_64tok.log" \
                "${OUTPUT_BASE}/passkey_128tok_${TIMESTAMP}/passkey_128tok.log"
        fi
    done
}

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
echo "PARALLEL EXECUTION MODE: 4 GPUs, 1 experiment each" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Experiments:" | tee -a "$LOG_FILE"
echo "  1. Banking77 Token Ablation (16, 32, 64, 128) - PARALLEL" | tee -a "$LOG_FILE"
echo "  2. Passkey Token Ablation (16, 32, 64, 128) - PARALLEL" | tee -a "$LOG_FILE"
echo "  3. Text-Relay Baseline (SST-2, AG News)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 1: BANKING77 TOKEN ABLATION (4 GPUs in parallel)
# ============================================================
echo "[1/3] BANKING77 TOKEN ABLATION (PARALLEL)" | tee -a "$LOG_FILE"
echo "Goal: Test if more tokens help with 77-class classification" | tee -a "$LOG_FILE"
echo "Running 4 experiments in parallel across 4 GPUs..." | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

# Create output directories
TOKENS_LIST=(16 32 64 128)
for TOKENS in "${TOKENS_LIST[@]}"; do
    mkdir -p "${OUTPUT_BASE}/banking77_${TOKENS}tok_${TIMESTAMP}"
done

# Launch all 4 in parallel, each on its own GPU
echo "" | tee -a "$LOG_FILE"
echo ">>> Launching Banking77 experiments on GPUs 0-3..." | tee -a "$LOG_FILE"

python telepathy/train_telepathy_banking77.py \
    --output_dir "${OUTPUT_BASE}/banking77_16tok_${TIMESTAMP}" \
    --soft_tokens 16 --steps 3000 --batch_size 8 --eval_every 500 --gpu 0 \
    > "${OUTPUT_BASE}/banking77_16tok_${TIMESTAMP}/banking77_16tok.log" 2>&1 &
PID_B16=$!

python telepathy/train_telepathy_banking77.py \
    --output_dir "${OUTPUT_BASE}/banking77_32tok_${TIMESTAMP}" \
    --soft_tokens 32 --steps 3000 --batch_size 8 --eval_every 500 --gpu 1 \
    > "${OUTPUT_BASE}/banking77_32tok_${TIMESTAMP}/banking77_32tok.log" 2>&1 &
PID_B32=$!

python telepathy/train_telepathy_banking77.py \
    --output_dir "${OUTPUT_BASE}/banking77_64tok_${TIMESTAMP}" \
    --soft_tokens 64 --steps 3000 --batch_size 8 --eval_every 500 --gpu 2 \
    > "${OUTPUT_BASE}/banking77_64tok_${TIMESTAMP}/banking77_64tok.log" 2>&1 &
PID_B64=$!

python telepathy/train_telepathy_banking77.py \
    --output_dir "${OUTPUT_BASE}/banking77_128tok_${TIMESTAMP}" \
    --soft_tokens 128 --steps 3000 --batch_size 8 --eval_every 500 --gpu 3 \
    > "${OUTPUT_BASE}/banking77_128tok_${TIMESTAMP}/banking77_128tok.log" 2>&1 &
PID_B128=$!

echo "  GPU 0: Banking77 16 tokens (PID: $PID_B16)" | tee -a "$LOG_FILE"
echo "  GPU 1: Banking77 32 tokens (PID: $PID_B32)" | tee -a "$LOG_FILE"
echo "  GPU 2: Banking77 64 tokens (PID: $PID_B64)" | tee -a "$LOG_FILE"
echo "  GPU 3: Banking77 128 tokens (PID: $PID_B128)" | tee -a "$LOG_FILE"

# Monitor with progress updates every 30 seconds
monitor_jobs "Banking77" 30 $PID_B16 $PID_B32 $PID_B64 $PID_B128

# Ensure all complete
wait $PID_B16 $PID_B32 $PID_B64 $PID_B128 2>/dev/null || true

echo "" | tee -a "$LOG_FILE"
echo "Banking77 Ablation Complete: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 2: PASSKEY TOKEN ABLATION (4 GPUs in parallel)
# ============================================================
echo "[2/3] PASSKEY TOKEN ABLATION (PARALLEL)" | tee -a "$LOG_FILE"
echo "Goal: Test if more tokens enable precision (5-digit codes)" | tee -a "$LOG_FILE"
echo "Running 4 experiments in parallel across 4 GPUs..." | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

# Create output directories
for TOKENS in "${TOKENS_LIST[@]}"; do
    mkdir -p "${OUTPUT_BASE}/passkey_${TOKENS}tok_${TIMESTAMP}"
done

# Launch all 4 in parallel, each on its own GPU
echo "" | tee -a "$LOG_FILE"
echo ">>> Launching Passkey experiments on GPUs 0-3..." | tee -a "$LOG_FILE"

python telepathy/train_telepathy_passkey.py \
    --output_dir "${OUTPUT_BASE}/passkey_16tok_${TIMESTAMP}" \
    --soft_tokens 16 --steps 1000 --batch_size 8 --eval_every 200 --gpu 0 \
    > "${OUTPUT_BASE}/passkey_16tok_${TIMESTAMP}/passkey_16tok.log" 2>&1 &
PID_P16=$!

python telepathy/train_telepathy_passkey.py \
    --output_dir "${OUTPUT_BASE}/passkey_32tok_${TIMESTAMP}" \
    --soft_tokens 32 --steps 1000 --batch_size 8 --eval_every 200 --gpu 1 \
    > "${OUTPUT_BASE}/passkey_32tok_${TIMESTAMP}/passkey_32tok.log" 2>&1 &
PID_P32=$!

python telepathy/train_telepathy_passkey.py \
    --output_dir "${OUTPUT_BASE}/passkey_64tok_${TIMESTAMP}" \
    --soft_tokens 64 --steps 1000 --batch_size 8 --eval_every 200 --gpu 2 \
    > "${OUTPUT_BASE}/passkey_64tok_${TIMESTAMP}/passkey_64tok.log" 2>&1 &
PID_P64=$!

python telepathy/train_telepathy_passkey.py \
    --output_dir "${OUTPUT_BASE}/passkey_128tok_${TIMESTAMP}" \
    --soft_tokens 128 --steps 1000 --batch_size 8 --eval_every 200 --gpu 3 \
    > "${OUTPUT_BASE}/passkey_128tok_${TIMESTAMP}/passkey_128tok.log" 2>&1 &
PID_P128=$!

echo "  GPU 0: Passkey 16 tokens (PID: $PID_P16)" | tee -a "$LOG_FILE"
echo "  GPU 1: Passkey 32 tokens (PID: $PID_P32)" | tee -a "$LOG_FILE"
echo "  GPU 2: Passkey 64 tokens (PID: $PID_P64)" | tee -a "$LOG_FILE"
echo "  GPU 3: Passkey 128 tokens (PID: $PID_P128)" | tee -a "$LOG_FILE"

# Monitor with progress updates every 30 seconds
monitor_jobs "Passkey" 30 $PID_P16 $PID_P32 $PID_P64 $PID_P128

# Ensure all complete
wait $PID_P16 $PID_P32 $PID_P64 $PID_P128 2>/dev/null || true

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
        --num_samples 200 \
        --gpu 0
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
