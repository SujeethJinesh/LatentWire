#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
echo "C10 novelty audit artifacts:"
find artifacts/novelty_audit -maxdepth 1 -type f | sort
echo
echo "Novelty matrix:"
cat artifacts/novelty_audit/novelty_matrix.csv

