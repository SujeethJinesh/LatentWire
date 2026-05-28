#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_phase9_mpred.py --model-key granite --run-id om_phase9_mpred_granite_reduced_20260527T201251Z --batch-size 2 --device auto --dtype bfloat16 --new-regimes mpred_top10_alpha_0_95 mpred_top5_alpha_0_95 mpred_random_alpha_top5 random_walk_top5
