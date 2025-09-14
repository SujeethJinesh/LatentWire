#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

RUN="8B_runs"
RUN_DIR="runs/$RUN"
LOG_FILE="${RUN_DIR}/full_pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RUN_DIR"

{
  echo "========================================="
  echo "Starting pipeline at $(date)"
  echo "========================================="
  echo ""

  echo "========================================="
  echo "PHASE 1: TRAINING (optional)"
  echo "========================================="
  echo "Checkpoint will be saved to: ${RUN_DIR}/ckpt"
  echo ""

  # Uncomment to retrain with the simple textual anchor
  # CUDA_VISIBLE_DEVICES=0,1 \
  # python -u latentwire/train.py \
  #   --dataset squad --samples 87599 --epochs 12 --batch_size 128 \
  #   --encoder_type simple-st --encoder_use_chat_template \
  #   --latent_len 16 --d_z 256 \
  #   --qwen_id Qwen/Qwen2.5-7B-Instruct \
  #   --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  #   --warm_anchor_text "Answer: " \
  #   --lr 5e-5 --scale_l2 0.05 \
  #   --adapter_rms_l2 0.0 --max_grad_norm 1.0 \
  #   --save_dir ${RUN_DIR}/ckpt --save_every 2000 \
  #   --save_training_stats --debug 2>&1

  echo ""
  echo "========================================="
  echo "PHASE 2: EVALUATION"
  echo "========================================="
  echo "Using checkpoint from: ${RUN_DIR}/ckpt"
  echo ""

  EV_DIR="${RUN_DIR}/eval_squad_answer_noBOS"
  mkdir -p "$EV_DIR"

  CUDA_VISIBLE_DEVICES=0,1 \
  python -u latentwire/eval.py \
    --ckpt ${RUN_DIR}/ckpt \
    --dataset squad --samples 200 \
    --max_new_tokens 12 \
    --latent_anchor_mode text --latent_anchor_text "Answer: " \
    --append_bos_after_prefix no \
    --fresh_eval \
    --encoder_text_mode auto \
    --calibration embed_rms --prefix_gain 1.0 \
    --first_token_top_p 1.0 --first_token_temperature 0.0 \
    --min_new_tokens 2 --eos_ban_steps 6 \
    --out_dir "${EV_DIR}" \
    --debug --debug_print_first 5 --debug_topk 10 --debug_topk_examples 2 2>&1

  echo ""
  echo "========================================="
  echo "PIPELINE SUMMARY"
  echo "========================================="
  echo "Run ID: $RUN"
  echo "Completed: $(date)"
  echo "Outputs:"
  echo "  Training checkpoint: ${RUN_DIR}/ckpt/"
  echo "  Evaluation results: ${EV_DIR}/"
  echo ""

  if [ -f "${EV_DIR}/metrics.json" ]; then
    echo "✓ Evaluation metrics saved at: ${EV_DIR}/metrics.json"
    echo ""
    echo "Key metrics:"
    python - <<'PY'
import json, os
p=os.environ.get("EV_DIR","") or "${EV_DIR}"
with open(os.path.join(p,"metrics.json")) as f:
    m=json.load(f)
def g(d,*ks):
    for k in ks: d=d.get(k,{})
    return d
print(f"  Compression: Llama {m['compression'].get('llama','-'):.1f}x | Qwen {m['compression'].get('qwen','-'):.1f}x")
print(f"  Text F1:     Llama {g(m,'text','llama').get('f1','-'):.3f} | Qwen {g(m,'text','qwen').get('f1','-'):.3f}")
print(f"  Latent F1:   Llama {g(m,'latent','llama').get('f1','-'):.3f} | Qwen {g(m,'latent','qwen').get('f1','-'):.3f}")
PY
  else
    echo "✗ Evaluation metrics missing"
  fi
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "All output has been saved to:"
echo "  Full log: $LOG_FILE"
echo "  Evaluation: ${EV_DIR}/metrics.json"
