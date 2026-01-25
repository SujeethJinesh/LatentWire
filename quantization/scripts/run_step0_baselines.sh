#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/projects/m000066/sujinesh/LatentWire}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$SCRIPT_DIR/run_step0_baselines.py" --project-root "$PROJECT_ROOT"
