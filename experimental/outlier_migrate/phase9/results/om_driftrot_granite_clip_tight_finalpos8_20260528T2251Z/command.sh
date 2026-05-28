#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_driftrot_clip_subset.py --run-id om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z --candidate-id tight_clip_0p5_2p0_final_positive_trace --prompt-indices 8 --split-name confirmation --scale-clip-min 0.5 --scale-clip-max 2.0
