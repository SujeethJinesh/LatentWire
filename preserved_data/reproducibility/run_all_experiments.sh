#!/bin/bash
# preserved_data/reproducibility/run_all_experiments.sh
#
# MASTER REPRODUCIBILITY SCRIPT
# Runs all experiments needed to reproduce paper results
#
# Usage:
#   git pull && rm -rf runs && PYTHONPATH=. bash preserved_data/reproducibility/run_all_experiments.sh
#
# Expected runtime: ~3 hours on 4xH100 (11 experiments)
# Expected output: runs/ directory with all results

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_BASE="${OUTPUT_BASE:-runs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${OUTPUT_BASE}/reproducibility_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_BASE"

# =============================================================================
# LOGGING HELPERS
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

section() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "$1" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
}

# =============================================================================
# START
# =============================================================================
section "REPRODUCIBILITY RUN - ALL EXPERIMENTS"
log "Started: $(date)"
log "Output: $OUTPUT_BASE"
log "Log: $MASTER_LOG"
log ""
log "Experiments to run:"
log "  1. SST-2 Bridge Training + Eval"
log "  2. AG News Bridge Training + Eval"
log "  3. TREC Bridge Training + Eval (6-class)"
log "  4. Banking77 Token Ablation (16, 32, 64, 128)"
log "  5. Banking77 Text Baselines (Mistral, Llama)"
log "  6. Banking77 Text-Relay"
log "  7. Passkey Token Ablation (16, 32, 64, 128)"
log "  8. Text-Relay Baselines (SST-2, AG News)"
log "  9. TREC Baselines (Text + Text-Relay)"
log "  10. SST-2/AG News Text Baselines (for super-additive comparison)"
log "  11. Latency Benchmark (Bridge vs Text-Relay vs Direct)"

# =============================================================================
# EXPERIMENT 1: SST-2 BRIDGE
# =============================================================================
section "[1/9] SST-2 BRIDGE TRAINING"
log "Expected: ~94.7% accuracy"

SST2_DIR="${OUTPUT_BASE}/sst2_${TIMESTAMP}"
mkdir -p "$SST2_DIR"

{
    python telepathy/train_telepathy_sst2.py \
        --source_layer 31 \
        --soft_tokens 8 \
        --steps 2000 \
        --batch_size 8 \
        --lr 1e-4 \
        --diversity_weight 0.1 \
        --output_dir "$SST2_DIR" \
        --gpu 0
} 2>&1 | tee "${SST2_DIR}/sst2_train.log"

log "SST-2 training complete"

# =============================================================================
# EXPERIMENT 2: AG NEWS BRIDGE
# =============================================================================
section "[2/9] AG NEWS BRIDGE TRAINING"
log "Expected: ~88.9% accuracy"

AGNEWS_DIR="${OUTPUT_BASE}/agnews_${TIMESTAMP}"
mkdir -p "$AGNEWS_DIR"

{
    python telepathy/train_telepathy_agnews.py \
        --source_layer 31 \
        --soft_tokens 8 \
        --steps 2000 \
        --batch_size 8 \
        --lr 1e-4 \
        --diversity_weight 0.1 \
        --output_dir "$AGNEWS_DIR" \
        --gpu 0
} 2>&1 | tee "${AGNEWS_DIR}/agnews_train.log"

log "AG News training complete"

# =============================================================================
# EXPERIMENT 3: TREC BRIDGE (6-class)
# =============================================================================
section "[3/9] TREC BRIDGE TRAINING (6-class)"
log "Expected: ~94.5% accuracy"

TREC_DIR="${OUTPUT_BASE}/trec_${TIMESTAMP}"
mkdir -p "$TREC_DIR"

{
    python telepathy/train_telepathy_trec.py \
        --soft_tokens 16 \
        --steps 2000 \
        --batch_size 8 \
        --lr 1e-4 \
        --diversity_weight 0.1 \
        --output_dir "$TREC_DIR" \
        --gpu 0
} 2>&1 | tee "${TREC_DIR}/trec_train.log"

log "TREC training complete"

# =============================================================================
# EXPERIMENT 4: BANKING77 TOKEN ABLATION (PARALLEL)
# =============================================================================
section "[4/9] BANKING77 TOKEN ABLATION (4 GPUs)"
log "Expected: 16tok=21.5%, 32tok=13.5%, 64tok=7.5%, 128tok=1%"

TOKENS_LIST=(16 32 64 128)
BANKING_PIDS=()

for i in "${!TOKENS_LIST[@]}"; do
    TOKENS=${TOKENS_LIST[$i]}
    GPU=$i
    DIR="${OUTPUT_BASE}/banking77_${TOKENS}tok_${TIMESTAMP}"
    mkdir -p "$DIR"

    log "  Launching Banking77 ${TOKENS}tok on GPU ${GPU}"

    python telepathy/train_telepathy_banking77.py \
        --output_dir "$DIR" \
        --soft_tokens $TOKENS \
        --steps 3000 \
        --batch_size 8 \
        --eval_every 500 \
        --diversity_weight 0.1 \
        --gpu $GPU \
        > "${DIR}/banking77_${TOKENS}tok.log" 2>&1 &
    BANKING_PIDS+=($!)
done

log "Waiting for Banking77 ablation to complete..."
for pid in "${BANKING_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

log "Banking77 ablation complete"

# =============================================================================
# EXPERIMENT 5: BANKING77 TEXT BASELINES
# =============================================================================
section "[5/9] BANKING77 TEXT BASELINES"
log "Expected: Mistral=19.5%, Llama=22.0%"

BANKING_BASELINE_DIR="${OUTPUT_BASE}/banking77_baselines_${TIMESTAMP}"
mkdir -p "$BANKING_BASELINE_DIR"

{
    python telepathy/eval_text_relay_baseline.py \
        --banking77 \
        --num_samples 200 \
        --output_dir "$BANKING_BASELINE_DIR" \
        --gpu 0
} 2>&1 | tee "${BANKING_BASELINE_DIR}/banking77_baselines.log"

log "Banking77 text baselines complete"

# =============================================================================
# EXPERIMENT 6: BANKING77 TEXT-RELAY
# =============================================================================
section "[6/9] BANKING77 TEXT-RELAY"
log "Expected: TBD (should be < Bridge 21.5%)"

BANKING_RELAY_DIR="${OUTPUT_BASE}/banking77_relay_${TIMESTAMP}"
mkdir -p "$BANKING_RELAY_DIR"

{
    python telepathy/eval_text_relay_baseline.py \
        --banking77_relay \
        --num_samples 200 \
        --output_dir "$BANKING_RELAY_DIR" \
        --gpu 0
} 2>&1 | tee "${BANKING_RELAY_DIR}/banking77_relay.log"

log "Banking77 text-relay complete"

# =============================================================================
# EXPERIMENT 7: PASSKEY TOKEN ABLATION (PARALLEL)
# =============================================================================
section "[7/9] PASSKEY TOKEN ABLATION (4 GPUs)"
log "Expected: 0% exact match, digit accuracy degrades with more tokens"

PASSKEY_PIDS=()

for i in "${!TOKENS_LIST[@]}"; do
    TOKENS=${TOKENS_LIST[$i]}
    GPU=$i
    DIR="${OUTPUT_BASE}/passkey_${TOKENS}tok_${TIMESTAMP}"
    mkdir -p "$DIR"

    log "  Launching Passkey ${TOKENS}tok on GPU ${GPU}"

    python telepathy/train_telepathy_passkey.py \
        --output_dir "$DIR" \
        --soft_tokens $TOKENS \
        --steps 1000 \
        --batch_size 8 \
        --eval_every 200 \
        --diversity_weight 0.1 \
        --gpu $GPU \
        > "${DIR}/passkey_${TOKENS}tok.log" 2>&1 &
    PASSKEY_PIDS+=($!)
done

log "Waiting for Passkey ablation to complete..."
for pid in "${PASSKEY_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

log "Passkey ablation complete"

# =============================================================================
# EXPERIMENT 8: TEXT-RELAY BASELINES (SST-2, AG NEWS)
# =============================================================================
section "[8/9] TEXT-RELAY BASELINES (SST-2, AG NEWS)"
log "Expected: SST-2=71%, AG News=64.5%"

TEXT_RELAY_DIR="${OUTPUT_BASE}/text_relay_${TIMESTAMP}"
mkdir -p "$TEXT_RELAY_DIR"

{
    python telepathy/eval_text_relay_baseline.py \
        --num_samples 200 \
        --output_dir "$TEXT_RELAY_DIR" \
        --gpu 0
} 2>&1 | tee "${TEXT_RELAY_DIR}/text_relay.log"

log "Text-relay baselines complete"

# =============================================================================
# EXPERIMENT 9: TREC BASELINES (Text + Text-Relay)
# =============================================================================
section "[9/9] TREC BASELINES (Text + Text-Relay)"
log "Expected: Mistral 43%, Llama 53.5%, Text-relay 58%, Bridge 94.5%"

TREC_BASELINE_DIR="${OUTPUT_BASE}/trec_baselines_${TIMESTAMP}"
mkdir -p "$TREC_BASELINE_DIR"

{
    # Part 1: Direct text baselines
    python telepathy/eval_text_relay_baseline.py \
        --trec \
        --num_samples 200 \
        --output_dir "$TREC_BASELINE_DIR" \
        --gpu 0

    # Part 2: Text-relay
    python telepathy/eval_text_relay_baseline.py \
        --trec_relay \
        --num_samples 200 \
        --output_dir "$TREC_BASELINE_DIR" \
        --gpu 0
} 2>&1 | tee "${TREC_BASELINE_DIR}/trec_baselines.log"

log "TREC baselines complete"

# =============================================================================
# EXPERIMENT 10: SST-2/AG NEWS TEXT BASELINES
# =============================================================================
section "[10/11] SST-2/AG NEWS TEXT BASELINES (Super-Additive)"
log "Expected: SST-2 Llama=92%, Mistral=88.5%; AG News both=79%"

SST2_AGNEWS_BASELINE_DIR="${OUTPUT_BASE}/sst2_agnews_baselines_${TIMESTAMP}"
mkdir -p "$SST2_AGNEWS_BASELINE_DIR"

{
    python telepathy/eval_text_relay_baseline.py \
        --sst2_text \
        --num_samples 200 \
        --output_dir "$SST2_AGNEWS_BASELINE_DIR" \
        --gpu 0

    python telepathy/eval_text_relay_baseline.py \
        --agnews_text \
        --num_samples 200 \
        --output_dir "$SST2_AGNEWS_BASELINE_DIR" \
        --gpu 0
} 2>&1 | tee "${SST2_AGNEWS_BASELINE_DIR}/sst2_agnews_baselines.log"

log "SST-2/AG News text baselines complete"

# =============================================================================
# EXPERIMENT 11: LATENCY BENCHMARK
# =============================================================================
section "[11/11] LATENCY BENCHMARK"
log "Expected: Bridge 5-10x faster than Text-Relay"

LATENCY_DIR="${OUTPUT_BASE}/latency_benchmark_${TIMESTAMP}"
mkdir -p "$LATENCY_DIR"

# Use SST-2 checkpoint for latency benchmark
SST2_CKPT="${SST2_DIR}/best_checkpoint.pt"
if [ -f "$SST2_CKPT" ]; then
    log "Using SST-2 checkpoint: $SST2_CKPT"
    {
        python telepathy/benchmark_latency.py \
            --checkpoint "$SST2_CKPT" \
            --num_trials 50 \
            --output_dir "$LATENCY_DIR" \
            --gpu 0
    } 2>&1 | tee "${LATENCY_DIR}/latency_benchmark.log"
else
    log "No SST-2 checkpoint found, running without Bridge timing"
    {
        python telepathy/benchmark_latency.py \
            --num_trials 50 \
            --output_dir "$LATENCY_DIR" \
            --gpu 0
    } 2>&1 | tee "${LATENCY_DIR}/latency_benchmark.log"
fi

log "Latency benchmark complete"

# =============================================================================
# SUMMARY
# =============================================================================
section "ALL EXPERIMENTS COMPLETE"
log "Finished: $(date)"
log ""
log "Results Summary:"
log ""

# SST-2 Results
if [ -f "${SST2_DIR}/sst2_results.json" ]; then
    SST2_ACC=$(python -c "import json; d=json.load(open('${SST2_DIR}/sst2_results.json')); print(d.get('final_accuracy', d.get('accuracy', 'N/A')))" 2>/dev/null || echo "N/A")
    log "SST-2 Bridge: ${SST2_ACC}% (expected: 94.7%)"
fi

# AG News Results
if [ -f "${AGNEWS_DIR}/agnews_results.json" ]; then
    AGNEWS_ACC=$(python -c "import json; d=json.load(open('${AGNEWS_DIR}/agnews_results.json')); print(d.get('final_accuracy', d.get('accuracy', 'N/A')))" 2>/dev/null || echo "N/A")
    log "AG News Bridge: ${AGNEWS_ACC}% (expected: 88.9%)"
fi

# TREC Results
if [ -f "${TREC_DIR}/trec_results.json" ]; then
    TREC_ACC=$(python -c "import json; d=json.load(open('${TREC_DIR}/trec_results.json')); print(d.get('final_results', {}).get('accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
    log "TREC Bridge (6-class): ${TREC_ACC}% (expected: 94.5%)"
fi

# TREC Baselines
if [ -f "${TREC_BASELINE_DIR}/trec_baselines.json" ]; then
    TREC_MISTRAL=$(python -c "import json; print(json.load(open('${TREC_BASELINE_DIR}/trec_baselines.json'))['results']['mistral']['accuracy'])" 2>/dev/null || echo "N/A")
    TREC_LLAMA=$(python -c "import json; print(json.load(open('${TREC_BASELINE_DIR}/trec_baselines.json'))['results']['llama']['accuracy'])" 2>/dev/null || echo "N/A")
    log ""
    log "TREC Text Baselines:"
    log "  Mistral: ${TREC_MISTRAL}%"
    log "  Llama: ${TREC_LLAMA}%"
fi
if [ -f "${TREC_BASELINE_DIR}/trec_relay_results.json" ]; then
    TREC_RELAY=$(python -c "import json; print(json.load(open('${TREC_BASELINE_DIR}/trec_relay_results.json'))['results']['trec']['accuracy'])" 2>/dev/null || echo "N/A")
    log "  Text-Relay: ${TREC_RELAY}%"
fi

# Banking77 Token Ablation
log ""
log "Banking77 Token Ablation:"
for TOKENS in 16 32 64 128; do
    JSON="${OUTPUT_BASE}/banking77_${TOKENS}tok_${TIMESTAMP}/banking77_results.json"
    if [ -f "$JSON" ]; then
        ACC=$(python -c "import json; print(json.load(open('$JSON'))['final_results']['accuracy'])" 2>/dev/null || echo "N/A")
        log "  ${TOKENS} tokens: ${ACC}%"
    fi
done

# Banking77 Baselines
if [ -f "${BANKING_BASELINE_DIR}/banking77_baselines.json" ]; then
    MISTRAL=$(python -c "import json; print(json.load(open('${BANKING_BASELINE_DIR}/banking77_baselines.json'))['results']['mistral']['accuracy'])" 2>/dev/null || echo "N/A")
    LLAMA=$(python -c "import json; print(json.load(open('${BANKING_BASELINE_DIR}/banking77_baselines.json'))['results']['llama']['accuracy'])" 2>/dev/null || echo "N/A")
    log ""
    log "Banking77 Text Baselines:"
    log "  Mistral: ${MISTRAL}% (expected: 19.5%)"
    log "  Llama: ${LLAMA}% (expected: 22.0%)"
fi

# Banking77 Text-Relay
if [ -f "${BANKING_RELAY_DIR}/banking77_relay_results.json" ]; then
    RELAY=$(python -c "import json; print(json.load(open('${BANKING_RELAY_DIR}/banking77_relay_results.json'))['results']['banking77']['accuracy'])" 2>/dev/null || echo "N/A")
    log ""
    log "Banking77 Text-Relay: ${RELAY}%"
fi

# Passkey Token Ablation
log ""
log "Passkey Token Ablation:"
for TOKENS in 16 32 64 128; do
    JSON="${OUTPUT_BASE}/passkey_${TOKENS}tok_${TIMESTAMP}/passkey_results.json"
    if [ -f "$JSON" ]; then
        EXACT=$(python -c "import json; print(json.load(open('$JSON'))['final_results']['exact_match'])" 2>/dev/null || echo "N/A")
        DIGIT=$(python -c "import json; print(json.load(open('$JSON'))['final_results']['digit_accuracy'])" 2>/dev/null || echo "N/A")
        log "  ${TOKENS} tokens: ${EXACT}% exact, ${DIGIT}% digit"
    fi
done

# Text-Relay Baselines
if [ -f "${TEXT_RELAY_DIR}/text_relay_results.json" ]; then
    SST2_RELAY=$(python -c "import json; print(json.load(open('${TEXT_RELAY_DIR}/text_relay_results.json'))['results']['sst2']['accuracy'])" 2>/dev/null || echo "N/A")
    AGNEWS_RELAY=$(python -c "import json; print(json.load(open('${TEXT_RELAY_DIR}/text_relay_results.json'))['results']['agnews']['accuracy'])" 2>/dev/null || echo "N/A")
    log ""
    log "Text-Relay Baselines:"
    log "  SST-2: ${SST2_RELAY}% (expected: 71.0%)"
    log "  AG News: ${AGNEWS_RELAY}% (expected: 64.5%)"
fi

log ""

# SST-2/AG News Text Baselines
if [ -f "${SST2_AGNEWS_BASELINE_DIR}/sst2_baselines.json" ]; then
    SST2_LLAMA=$(python -c "import json; print(json.load(open('${SST2_AGNEWS_BASELINE_DIR}/sst2_baselines.json'))['results']['llama']['accuracy'])" 2>/dev/null || echo "N/A")
    SST2_MISTRAL=$(python -c "import json; print(json.load(open('${SST2_AGNEWS_BASELINE_DIR}/sst2_baselines.json'))['results']['mistral']['accuracy'])" 2>/dev/null || echo "N/A")
    log "SST-2 Text Baselines:"
    log "  Llama: ${SST2_LLAMA}% (expected: 92.0%)"
    log "  Mistral: ${SST2_MISTRAL}% (expected: 88.5%)"
fi
if [ -f "${SST2_AGNEWS_BASELINE_DIR}/agnews_baselines.json" ]; then
    AGNEWS_LLAMA=$(python -c "import json; print(json.load(open('${SST2_AGNEWS_BASELINE_DIR}/agnews_baselines.json'))['results']['llama']['accuracy'])" 2>/dev/null || echo "N/A")
    AGNEWS_MISTRAL=$(python -c "import json; print(json.load(open('${SST2_AGNEWS_BASELINE_DIR}/agnews_baselines.json'))['results']['mistral']['accuracy'])" 2>/dev/null || echo "N/A")
    log ""
    log "AG News Text Baselines:"
    log "  Llama: ${AGNEWS_LLAMA}% (expected: 79.0%)"
    log "  Mistral: ${AGNEWS_MISTRAL}% (expected: 79.0%)"
fi

# Latency Benchmark
if [ -f "${LATENCY_DIR}/latency_benchmark.json" ]; then
    DIRECT=$(python -c "import json; print(f\"{json.load(open('${LATENCY_DIR}/latency_benchmark.json'))['direct_text']['total_ms']:.1f}\")" 2>/dev/null || echo "N/A")
    RELAY=$(python -c "import json; print(f\"{json.load(open('${LATENCY_DIR}/latency_benchmark.json'))['text_relay']['total_ms']:.1f}\")" 2>/dev/null || echo "N/A")
    log ""
    log "Latency Benchmark (50 trials):"
    log "  Direct Text: ${DIRECT}ms"
    log "  Text-Relay: ${RELAY}ms"
    if [ -n "$(python -c "import json; d=json.load(open('${LATENCY_DIR}/latency_benchmark.json')); print('yes' if 'bridge' in d else '')" 2>/dev/null)" ]; then
        BRIDGE=$(python -c "import json; print(f\"{json.load(open('${LATENCY_DIR}/latency_benchmark.json'))['bridge']['total_ms']:.1f}\")" 2>/dev/null || echo "N/A")
        SPEEDUP=$(python -c "import json; d=json.load(open('${LATENCY_DIR}/latency_benchmark.json')); print(f\"{d['text_relay']['total_ms']/d['bridge']['total_ms']:.1f}x\")" 2>/dev/null || echo "N/A")
        log "  Bridge: ${BRIDGE}ms (${SPEEDUP} faster than Text-Relay)"
    fi
fi

log ""
section "REPRODUCIBILITY RUN COMPLETE"
log "All results saved to: $OUTPUT_BASE"
log "Master log: $MASTER_LOG"
