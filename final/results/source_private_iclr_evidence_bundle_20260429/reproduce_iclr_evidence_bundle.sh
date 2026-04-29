#!/usr/bin/env bash
set -euo pipefail
./venv_arm64/bin/python scripts/build_source_private_rate_frontier.py --output-dir results/source_private_rate_frontier_20260429
./venv_arm64/bin/python scripts/build_source_private_kv_cache_baseline_table.py --output-dir results/source_private_kv_cache_baseline_table_20260429
./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py --examples 160 --candidates 4 --family-set all --seeds 29,31,37 --budget 2 --output-dir results/source_private_coded_label_risk_gate_20260429
./venv_arm64/bin/python scripts/build_source_private_pass_fail_ledger.py --output-dir results/source_private_pass_fail_ledger_20260429
find final -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > final/MANIFEST.sha256
shasum -a 256 -c final/MANIFEST.sha256
./venv_arm64/bin/python -m pytest tests/test_build_source_private_rate_frontier.py tests/test_build_source_private_kv_cache_baseline_table.py tests/test_run_source_private_coded_label_risk_gate.py tests/test_build_source_private_pass_fail_ledger.py -q
