#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/materialize_om_driftrot_delta_columns.py --run-id delta_columns_granite_tail4_top8x32_20260528T2048Z --top-modules 8 --columns-per-module 32 --scale-clip-min 0.5 --scale-clip-max 2.0
