#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/collect_om_driftrot_residual_cache.py --run-id residual_cache_granite_tight_20260528T2025Z --scale-clip-min 0.5 --scale-clip-max 2.0 --row-chunk 128 --topn 128 --protected-columns-per-tensor 64
