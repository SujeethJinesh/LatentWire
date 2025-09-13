#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export PYTHONPATH=.

# Set run name and paths
RUN="8B_runs"
RUN_DIR="runs/$RUN"
LOG_FILE="${RUN_DIR}/full_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Create run directory if it doesn't exist
mkdir -p "$RUN_DIR"

echo "Starting LatentWire training and evaluation pipeline"
echo "Run ID: $RUN"
echo "Output will be saved to: $LOG_FILE"
echo ""

{
  echo "========================================="
  echo "Starting pipeline at $(date)"
  echo "========================================="
  echo ""
  
  # Training phase
  echo "========================================="
  echo "PHASE 1: TRAINING"
  echo "========================================="
  echo "Starting training at $(date)"
  echo "Checkpoint will be saved to: ${RUN_DIR}/ckpt"
  echo ""
  
  CUDA_VISIBLE_DEVICES=0,1 \
  python -u latentwire/train.py \
    --dataset squad --samples 87599 --epochs 8 --batch_size 128 \
    --encoder_type simple-st --encoder_use_chat_template \
    --latent_len 16 --d_z 256 --max_bytes 512 \
    --qwen_id Qwen/Qwen2.5-7B-Instruct \
    --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --warm_anchor_text "Answer: " \
    --lr 5e-5 \
    --scale_l2 0.05 --save_dir ${RUN_DIR}/ckpt --save_every 2000 \
    --save_training_stats --debug 2>&1
    # --grad_ckpt 
  TRAIN_EXIT_CODE=$?
  
  echo ""
  echo "Training completed at $(date) with exit code: $TRAIN_EXIT_CODE"
  
  if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Aborting pipeline"
    exit $TRAIN_EXIT_CODE
  fi
  
  echo ""
  echo "========================================="
  echo "PHASE 2: EVALUATION"
  echo "========================================="
  echo "Starting evaluation at $(date)"
  echo "Using checkpoint from: ${RUN_DIR}/ckpt"
  echo "Results will be saved to: ${RUN_DIR}/eval_squad"
  echo ""
  
  CUDA_VISIBLE_DEVICES=0,1 \
  python -u latentwire/eval.py \
    --ckpt ${RUN_DIR}/ckpt --dataset squad --samples 200 \
    --max_new_tokens 12 --latent_anchor_text "Answer: " \
    --sequential_eval --fresh_eval \
    --encoder_text_mode auto \
    --calibration embed_rms --prefix_gain 1.0 \
    --first_token_top_p 0.9 --first_token_temperature 0.7 \
    --min_new_tokens 2 --eos_ban_steps 6 \
    --out_dir ${RUN_DIR}/eval_squad --debug 2>&1

  EVAL_EXIT_CODE=$?
  
  echo ""
  echo "Evaluation completed at $(date) with exit code: $EVAL_EXIT_CODE"
  
  if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Evaluation failed with exit code $EVAL_EXIT_CODE"
  fi
  
  echo ""
  echo "========================================="
  echo "PIPELINE SUMMARY"
  echo "========================================="
  echo "Run ID: $RUN"
  echo "Started: $(head -n 20 "$LOG_FILE" | grep "Starting pipeline at" | cut -d' ' -f4-)"
  echo "Completed: $(date)"
  echo ""
  echo "Outputs:"
  echo "  Training checkpoint: ${RUN_DIR}/ckpt/"
  echo "  Training log: ${RUN_DIR}/train.log"
  echo "  Evaluation results: ${RUN_DIR}/eval_squad/"
  echo "  Evaluation log: ${RUN_DIR}/eval.log"
  echo "  Full pipeline log: $LOG_FILE"
  echo ""
  
  # Check if key output files exist
  if [ -f "${RUN_DIR}/ckpt/encoder.pt" ]; then
    echo "✓ Encoder checkpoint saved"
  else
    echo "✗ Encoder checkpoint missing"
  fi
  
  if [ -f "${RUN_DIR}/ckpt/adapter_llama.pt" ]; then
    echo "✓ Llama adapter checkpoint saved"
  else
    echo "✗ Llama adapter checkpoint missing"
  fi
  
  if [ -f "${RUN_DIR}/ckpt/adapter_qwen.pt" ]; then
    echo "✓ Qwen adapter checkpoint saved"
  else
    echo "✗ Qwen adapter checkpoint missing"
  fi
  
  if [ -f "${RUN_DIR}/eval_squad/metrics.json" ]; then
    echo "✓ Evaluation metrics saved"
    echo ""
    echo "Key metrics:"
    python -c "
import json
with open('${RUN_DIR}/eval_squad/metrics.json') as f:
    m = json.load(f)
    print(f\"  Compression: Llama {m['compression']['llama']:.1f}x, Qwen {m['compression']['qwen']:.1f}x\")
    print(f\"  Text F1: Llama {m['text']['llama']['f1']:.3f}, Qwen {m['text']['qwen']['f1']:.3f}\")
    print(f\"  Latent F1: Llama {m['latent']['llama']['f1']:.3f}, Qwen {m['latent']['qwen']['f1']:.3f}\")
" 2>/dev/null || echo "  (Could not parse metrics)"
  else
    echo "✗ Evaluation metrics missing"
  fi
  
  echo ""
  echo "========================================="
  echo "Pipeline completed at $(date)"
  echo "========================================="
  
} 2>&1 | tee "$LOG_FILE"

# Also save individual logs for easier access
grep -E "^\[|step \d+/|epoch \d+/|loss" "$LOG_FILE" > "${RUN_DIR}/train.log" 2>/dev/null || true
tail -n 500 "$LOG_FILE" | grep -A 1000 "PHASE 2: EVALUATION" > "${RUN_DIR}/eval.log" 2>/dev/null || true

echo ""
echo "All output has been saved to:"
echo "  Full log: $LOG_FILE"
echo "  Training log: ${RUN_DIR}/train.log"
echo "  Evaluation log: ${RUN_DIR}/eval.log"