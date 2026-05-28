#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_stage1_e3_paroquant_m11b_composition.py --run-id om_stage1_e3_granite_20260526T034442Z --device auto --dtype bfloat16 --cap-hours 15
