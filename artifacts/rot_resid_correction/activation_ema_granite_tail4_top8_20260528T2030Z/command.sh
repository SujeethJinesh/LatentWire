#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/collect_om_driftrot_activation_ema.py --run-id activation_ema_granite_tail4_top8_20260528T2030Z --prompt-indices 4 --top-tensors 8 --dtype bfloat16
