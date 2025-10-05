#!/usr/bin/env bash
        set -euo pipefail
        MODEL=${MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
        python -u train.py --model_id "$MODEL" \
          --first_token_ce_weight 12.0 --K 8 --k_ce_weight 0.5 \
          --kd_first_k_weight 2.0 --kd_tau 2.0 --d_z 256
