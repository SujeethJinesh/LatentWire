#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
source .venv_gpu/bin/activate

echo "Prepared clip candidates only. Do not launch from this worker."
echo "Template:"
echo "python experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py --run-id <RUN_ID> --scale-clip-min <MIN> --scale-clip-max <MAX> --batch-size 1"
echo "python experimental/outlier_migrate/phase9/run_om_v1_paroquant_nemotron.py --run-id <RUN_ID> --scale-clip-min <MIN> --scale-clip-max <MAX> --batch-size 1 --dtype bfloat16"

