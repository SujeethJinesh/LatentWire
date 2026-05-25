#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---fast-verify}"
CONFIG="configs/granite_small.yaml"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "OutlierMigrate reproduction starting in ${MODE} mode"
echo "FAST_VERIFY expected runtime: under 5 minutes on CPU"

python src/scripts/verify_environment.py
python src/scripts/reproduce_set_leaving.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_static_union.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_m2.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_m10.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_m11.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_m11b.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_m18.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_m26.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_decdec.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_paroquant.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_kl_accumulation.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_fft.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_per_component.py --config "${CONFIG}" "${MODE}"
python src/scripts/reproduce_trace_heterogeneity.py --config "${CONFIG}" "${MODE}"

echo "OutlierMigrate reproduction completed"
