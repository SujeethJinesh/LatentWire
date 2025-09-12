#!/usr/bin/env bash
set -euo pipefail

# Set output log file
LOG_FILE="eval_output_$(date +%Y%m%d_%H%M%S).log"

echo "Running all evaluations. Output will be saved to: $LOG_FILE"

{
  echo "========================================="
  echo "Starting evaluations at $(date)"
  echo "========================================="
  echo ""
  
  # First command
  echo "Running evaluation with sequential_eval and calibrate_prefix_rms..."
  echo "Command 1 started at $(date)"
  echo ""
  
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
  python -u latentwire/eval.py \
    --ckpt runs/squad_m16_scalereg_20250911_211615/ckpt \
    --dataset squad \
    --samples 200 \
    --max_new_tokens 8 \
    --latent_anchor_text "Answer: " \
    --out_dir runs/squad_m16_scalereg_20250911_211615/squad_eval_se_nc \
    --sequential_eval \
    --fresh_eval \
    --encoder_text_mode auto \
    --calibrate_prefix_rms \
    --prefix_gain 0.5 \
    --min_new_tokens 2 \
    --eos_ban_steps 6 \
    --first_token_top_p 0.9 \
    --first_token_temperature 0.7 \
    --debug 2>&1
  
  echo ""
  echo "========================================="
  echo "Starting prefix_gain sweep..."
  echo "========================================="
  echo ""
  
  # Second command - loop
  for PG in 0.5 1 2 4; do
    echo "Running evaluation with prefix_gain=${PG}..."
    echo "Started at $(date)"
    echo ""
    
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
    python -u latentwire/eval.py \
      --ckpt runs/squad_m16_scalereg_20250911_211615/ckpt \
      --dataset squad \
      --samples 200 \
      --max_new_tokens 8 \
      --latent_anchor_text "Answer: " \
      --out_dir runs/squad_m16_scalereg_20250911_211615/squad_eval_gain${PG} \
      --sequential_eval \
      --fresh_eval \
      --encoder_text_mode auto \
      --calibrate_prefix_rms \
      --prefix_gain $PG \
      --min_new_tokens 2 \
      --eos_ban_steps 6 \
      --first_token_top_p 0.9 \
      --first_token_temperature 0.7 \
      --debug 2>&1
    
    echo ""
    echo "Completed prefix_gain=${PG} at $(date)"
    echo "----------------------------------------"
    echo ""
  done
  
  echo "========================================="
  echo "All evaluations completed at $(date)"
  echo "========================================="
  
} | tee "$LOG_FILE"

echo ""
echo "All output has been saved to: $LOG_FILE"