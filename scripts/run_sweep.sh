#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

LLAMA=${1:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
QWEN=${2:-"Qwen/Qwen2-0.5B-Instruct"}
SAMPLES=${3:-5000}
VALS=${4:-1000}

echo "m,em_text_llama,f1_text_llama,em_lat_llama,f1_lat_llama,em_joint,f1_joint,comp_llama,comp_qwen,bytes,wall_text,wall_latent" > sweep.csv

for M in 8 12 16; do
  CK=ckpt_m${M}
  python latentwire/train.py --llama_id "$LLAMA" --qwen_id "$QWEN" --samples $SAMPLES --epochs 1 --batch_size 1 --latent_len $M --d_z 256 --max_bytes 512 --max_answer_tokens 32 --save_dir "./$CK"
  python latentwire/eval.py  --ckpt "./$CK" --samples $VALS --max_new_tokens 32 | tee eval_${M}.log
done
