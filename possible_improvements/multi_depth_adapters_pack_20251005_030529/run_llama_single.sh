#!/usr/bin/env bash
        set -euo pipefail
        MODEL=${MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
        python -u train.py --model_id "$MODEL" \
          --adapter_layers 5,10,15 --d_z 256 \
          --first_token_ce_weight 10.0 --K 8 --k_ce_weight 1.0 \
          --kd_first_k_weight 1.0 --kd_tau 2.0
