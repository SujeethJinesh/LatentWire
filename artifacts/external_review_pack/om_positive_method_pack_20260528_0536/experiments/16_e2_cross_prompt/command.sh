#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_stage1_e2_cross_prompt_replication.py --run-id om_stage1_e2_mathonly_20260526T1202Z --models deepseek_r1_distill,falcon_h1,granite_small --math-only --math-count 30 --cap-hours 15 --device auto --dtype bfloat16
