#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-quantization/data}
OUT=${2:-quantization/analysis/regime_map}
python quantization/scripts/analyze_regime_map.py --runs-root "$ROOT" --output-dir "$OUT"
