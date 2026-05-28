#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
echo "C11 repro_audit is a table/provenance artifact generated from existing packs."
echo "Validate CSVs with:"
echo "python - <<'PY'"
echo "import csv, pathlib"
echo "for p in pathlib.Path('artifacts/repro_audit').glob('*.csv'):"
echo "    with p.open() as f: list(csv.DictReader(f))"
echo "    print(p)"
echo "PY"

