#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_driftrot_clip_subset.py --run-id om_driftrot_granite_clip_loose_confirmation_20260528T1735Z --candidate-id clip_loose --split-name confirmation --prompt-indices 4,7,9,10 --base-run-dir experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z --scale-clip-min 0.125 --scale-clip-max 8.0 --batch-size 1 --dtype bfloat16
