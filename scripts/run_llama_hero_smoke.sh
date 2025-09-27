#!/usr/bin/env bash
set -euo pipefail

# run_llama_hero_smoke.sh
# Validation harness for hero checkpointing/resume and evaluation stages.
# - Runs Stage A/B for two short epochs with per-epoch checkpoints into runs/hero-smoke.
# - Runs Stage C evaluation once the first pass completes.
# - Re-runs Stage A/B in resume mode to confirm checkpoints reload cleanly.
# - Logs are written under runs/hero-smoke/ for inspection.

RUN_TAG="hero_smoke"
RUN_DIR="runs/${RUN_TAG}"
LOG_DIR="$RUN_DIR"
PRIMARY_LOG="${LOG_DIR}/pipeline_pass1_$(date +%Y%m%d_%H%M%S).log"
RESUME_LOG="${LOG_DIR}/pipeline_pass2_$(date +%Y%m%d_%H%M%S).log"
DIAG_LOG="${LOG_DIR}/diagnostics.jsonl"
mkdir -p "$LOG_DIR"
: > "$DIAG_LOG"

# Common tiny-settings for the smoke drill
TRAIN_SAMPLES_STAGEA=${TRAIN_SAMPLES_STAGEA:-128}
TRAIN_SAMPLES_STAGEB=${TRAIN_SAMPLES_STAGEB:-256}
EPOCHS_STAGEA=${EPOCHS_STAGEA:-2}
EPOCHS_STAGEB=${EPOCHS_STAGEB:-2}
BATCH_SIZE_STAGEA=${BATCH_SIZE_STAGEA:-8}
BATCH_SIZE_STAGEB=${BATCH_SIZE_STAGEB:-16}
SAMPLES=${SAMPLES:-32}

# First pass: fresh run with per-epoch checkpoints (save_every=steps_per_epoch).
RUN_TAG="$RUN_TAG" \
TRAIN_SAMPLES_STAGEA=$TRAIN_SAMPLES_STAGEA \
TRAIN_SAMPLES_STAGEB=$TRAIN_SAMPLES_STAGEB \
EPOCHS_STAGEA=$EPOCHS_STAGEA \
EPOCHS_STAGEB=$EPOCHS_STAGEB \
BATCH_SIZE_STAGEA=$BATCH_SIZE_STAGEA \
BATCH_SIZE_STAGEB=$BATCH_SIZE_STAGEB \
SAVE_EVERY_STAGEA=0 \
SAVE_EVERY_STAGEB=0 \
SAMPLES=$SAMPLES \
DIAGNOSTIC_LOG="$DIAG_LOG" \
  bash "$(dirname "$0")/run_llama_single.sh" --hero "$@" | tee "$PRIMARY_LOG"

# Second pass: rerun Stage A/B to ensure checkpoints resume rather than restart.
RUN_TAG="$RUN_TAG" \
TRAIN_SAMPLES_STAGEA=$TRAIN_SAMPLES_STAGEA \
TRAIN_SAMPLES_STAGEB=$TRAIN_SAMPLES_STAGEB \
EPOCHS_STAGEA=$EPOCHS_STAGEA \
EPOCHS_STAGEB=$EPOCHS_STAGEB \
BATCH_SIZE_STAGEA=$BATCH_SIZE_STAGEA \
BATCH_SIZE_STAGEB=$BATCH_SIZE_STAGEB \
SAVE_EVERY_STAGEA=0 \
SAVE_EVERY_STAGEB=0 \
SAMPLES=$SAMPLES \
DIAGNOSTIC_LOG="$DIAG_LOG" \
  bash "$(dirname "$0")/run_llama_single.sh" --hero "$@" | tee "$RESUME_LOG"

# After both passes, echo where the logs/checkpoints live.
echo
echo "Hero smoke run complete. Logs:"
echo "  Pass1: $PRIMARY_LOG"
echo "  Pass2: $RESUME_LOG"
echo "Diagnostics: $DIAG_LOG"
echo "Checkpoints: runs/hero-smoke/ckpt/stageA and stageB"
