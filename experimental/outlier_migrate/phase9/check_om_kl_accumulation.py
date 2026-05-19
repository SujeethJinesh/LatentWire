#!/usr/bin/env python3
"""Check Phase 9 KL accumulation packets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_kl_accumulation.md"

SCHEMA_VERSION = "om_phase9_kl_accumulation_v1"
TRACE_COUNT = 12
MAX_NEW_TOKENS = 20000
MODEL_ID = "ibm-granite/granite-4.0-h-small"
MODEL_SNAPSHOT = "b8c0982bab7fde4eb48110f5a069527c008fab39"

REGIMES = [
    "bf16_reference",
    "static_1pct",
    "decdec_reactive_top1_proxy",
    "m11_alpha_0_5",
]
QUANTIZED_REGIMES = [regime for regime in REGIMES if regime != "bf16_reference"]

PASS_DECISION = "PASS_KL_ACCUMULATION_REPORTED"
FAIL_INFRA = "FAIL_INFRA_KL_ACCUMULATION"

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "source_artifacts.json",
    "quantization_config.json",
    "kl_positions.json",
    "kl_rows.jsonl.gz",
    "kl_summary.json",
    "growth_model_fits.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
OPTIONAL_FILES = ["bf16_traces.jsonl.gz", "bf16_trace_manifest.json"]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"] + OPTIONAL_FILES


def dense_grid_positions() -> list[int]:
    values = set(range(1, 513))
    values.update(range(520, 2001, 10))
    values.update(range(2050, 10001, 50))
    values.update(range(10100, 20001, 100))
    return sorted(values)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_phase9_kl_")]
    if not candidates:
        raise FileNotFoundError(f"no KL accumulation result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in HASHED_FILES:
        path = run_dir / rel
        if not path.is_file():
            continue
        item = by_path.get(rel)
        if item is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


def validate_kl_rows(run_dir: Path, expected_positions: set[int], infra: list[str]) -> dict[str, Any]:
    counts = {(trace_index, regime): 0 for trace_index in range(TRACE_COUNT) for regime in REGIMES}
    sums = {regime: 0.0 for regime in REGIMES}
    total_by_regime = {regime: 0 for regime in REGIMES}
    seen_positions: dict[tuple[int, str], set[int]] = {
        (trace_index, regime): set() for trace_index in range(TRACE_COUNT) for regime in REGIMES
    }
    path = run_dir / "kl_rows.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            trace_index = int(row.get("prompt_index", -1))
            regime = str(row.get("regime"))
            position = int(row.get("decode_position", -1))
            kl = float(row.get("kl_bf16_q", float("nan")))
            if trace_index not in range(TRACE_COUNT):
                infra.append(f"kl_rows line {line_no}: invalid prompt_index {trace_index}")
                continue
            if regime not in REGIMES:
                infra.append(f"kl_rows line {line_no}: invalid regime {regime}")
                continue
            if position not in expected_positions:
                infra.append(f"kl_rows line {line_no}: unexpected position {position}")
                continue
            if not math.isfinite(kl) or kl < -1e-8:
                infra.append(f"kl_rows line {line_no}: invalid KL {kl}")
                continue
            counts[(trace_index, regime)] += 1
            seen_positions[(trace_index, regime)].add(position)
            sums[regime] += kl
            total_by_regime[regime] += 1
    for key, positions in seen_positions.items():
        if positions != expected_positions:
            missing = sorted(expected_positions - positions)
            extra = sorted(positions - expected_positions)
            infra.append(f"kl_rows incomplete for trace/regime {key}: missing {len(missing)}, extra {len(extra)}")
    return {
        "row_count": sum(counts.values()),
        "mean_kl_by_regime": {
            regime: (sums[regime] / total_by_regime[regime] if total_by_regime[regime] else None)
            for regime in REGIMES
        },
    }


def classify_best_fit(fits: dict[str, Any]) -> str:
    classes: list[str] = []
    for regime in QUANTIZED_REGIMES:
        best = fits.get("regime_fits", {}).get(regime, {}).get("best_fit_class")
        if best:
            classes.append(str(best))
    if not classes:
        return "inconclusive"
    if any(item == "superlinear" for item in classes):
        return "superlinear_or_compounding"
    if all(item in {"flat", "linear", "sublinear"} for item in classes):
        return "flat_linear_or_sublinear"
    return "mixed_or_inconclusive"


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    missing = [rel for rel in REQUIRED_FILES if not (run_dir / rel).is_file()]
    if missing:
        infra.append(f"missing required files: {missing}")

    positions_payload: dict[str, Any] = {}
    fits: dict[str, Any] = {}
    row_summary: dict[str, Any] = {}
    if not missing:
        positions_payload = load_json(run_dir / "kl_positions.json")
        positions = [int(value) for value in positions_payload.get("positions", [])]
        if positions_payload.get("dense_grid_fallback"):
            expected_positions = set(dense_grid_positions())
        else:
            expected_positions = set(range(1, MAX_NEW_TOKENS + 1))
        if set(positions) != expected_positions:
            infra.append("kl_positions does not match declared evaluation grid")
        fits = load_json(run_dir / "growth_model_fits.json")
        summary = load_json(run_dir / "kl_summary.json")
        for regime in REGIMES:
            if regime not in summary.get("regime_summary", {}):
                infra.append(f"kl_summary missing regime {regime}")
            if regime != "bf16_reference" and regime not in fits.get("regime_fits", {}):
                infra.append(f"growth_model_fits missing regime {regime}")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
        validate_artifact_hashes(run_dir, artifact_hashes, infra)
        if not infra:
            row_summary = validate_kl_rows(run_dir, expected_positions, infra)

    decision = FAIL_INFRA if infra else PASS_DECISION
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": not infra,
        "reasons": infra if infra else ["KL accumulation packet is complete"],
        "run_dir": str(run_dir),
        "trajectory_classification": None if infra else classify_best_fit(fits),
        "dense_grid_fallback": positions_payload.get("dense_grid_fallback") if positions_payload else None,
        "row_summary": row_summary,
    }
    write_json(run_dir / "checker_result.json", result)
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": decision,
        "artifact_complete": not infra,
        "reasons": result["reasons"],
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "artifact_check.json", artifact_check)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir or latest_run_dir()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
