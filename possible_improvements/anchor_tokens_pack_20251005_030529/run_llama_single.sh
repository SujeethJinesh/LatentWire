#!/usr/bin/env bash
        set -euo pipefail
        MODEL=${MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
        python -u train.py --model_id "$MODEL" \
          --first_token_ce_weight 8.0 --K 8 --k_ce_weight 0.5 \
          --kd_first_k_weight 1.0 --kd_tau 2.0 --anchor_window 64
