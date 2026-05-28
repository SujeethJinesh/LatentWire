#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase3/run_phase3_intervention.py --run-id om_phase3_20260509T212000Z --batch-size 4 --dtype bfloat16 --device auto --reuse-prequant-run-dir experimental/outlier_migrate/phase3/results/om_phase3_20260509T180200Z
