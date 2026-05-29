#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_phase9_msurface_sanity.py --run-id om_phase9_msurface_granite_internals_20260529T030958Z --prompt-indices 0,1 --positions 100,20000 --max-new-tokens 20000 --dtype bfloat16 --device auto
