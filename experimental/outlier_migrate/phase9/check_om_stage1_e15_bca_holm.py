#!/usr/bin/env python3
"""Check Stage 1 E15 BCa/Holm post-processing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experimental/outlier_migrate/phase9/results"
SCHEMA_VERSION = "om_stage1_e15_bca_holm_v1"
PASS = "PASS_E15_BCA_HOLM_COMPLETE"
FAIL_INFRA = "FAIL_INFRA_E15"


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def latest_output() -> Path:
    """Return the newest E15 output JSON under Phase 9 results."""
    candidates = list(RESULTS.glob("om_stage1_e15_bca_holm_*/e15_bca_holm.json"))
    if not candidates:
        raise FileNotFoundError(f"no E15 outputs found under {RESULTS}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def output_path(path: Path | None) -> Path:
    """Normalize a passed output file or run directory."""
    if path is None:
        return latest_output()
    resolved = path.resolve()
    return resolved / "e15_bca_holm.json" if resolved.is_dir() else resolved


def evaluate(path: Path) -> dict[str, Any]:
    """Validate an E15 JSON output."""
    reasons: list[str] = []
    if not path.is_file():
        reasons.append(f"missing output: {path}")
        payload: dict[str, Any] = {}
    else:
        payload = load_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("schema_version mismatch")
    if payload.get("missing_inputs"):
        reasons.append(f"missing inputs remain: {payload['missing_inputs']}")
    results = payload.get("results_by_packet", {})
    if not isinstance(results, dict) or not results:
        reasons.append("results_by_packet must be non-empty")
    test_count = 0
    for packet, regimes in results.items():
        if not isinstance(regimes, dict) or not regimes:
            reasons.append(f"{packet}: no regimes")
            continue
        for regime, stats in regimes.items():
            test_count += 1
            if stats.get("median_recovery") is None:
                reasons.append(f"{packet}:{regime}: missing median")
            ci = stats.get("bca_ci95", {})
            if ci.get("ci95_low") is None or ci.get("ci95_high") is None:
                reasons.append(f"{packet}:{regime}: missing BCa CI")
            if int(stats.get("included_trace_count", 0)) < 2:
                reasons.append(f"{packet}:{regime}: fewer than two included traces")
    tests = payload.get("holm_bonferroni", {}).get("tests", [])
    if len(tests) != test_count:
        reasons.append("Holm test count does not match result count")
    for row in tests:
        if "holm_adjusted_p" not in row or "holm_reject" not in row:
            reasons.append("Holm row missing adjusted p-value or reject flag")
            break
    return {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": FAIL_INFRA if reasons else PASS,
        "artifact_complete": not reasons,
        "output": str(path),
        "reasons": reasons,
        "test_count": test_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="E15 JSON output file or run directory.")
    args = parser.parse_args(argv)
    result = evaluate(output_path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
