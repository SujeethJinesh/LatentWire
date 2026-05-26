#!/usr/bin/env python3
"""Check Stage 1 E4 format-axis triangulation packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
SCHEMA_VERSION = "om_stage1_e4_format_axis_v1"

PASS = "PASS_E4_FORMAT_AXIS_COMPLETE"
PARTIAL = "PARTIAL_E4_FORMAT_AXIS"
INCOMPLETE = "INCOMPLETE_E4_FORMAT_AXIS"
FAIL_INFRA = "FAIL_INFRA_E4"

FORMATS = ("fp8", "w4a16", "nvfp4_w4a16", "nvfp4_w4a4")
BENCHMARKS = ("aime_2025", "math_500")
VALID_STATUSES = {"COMPLETED", "SKIPPED_INFRA", "PENDING_GPU_RUN"}
REQUIRED_FILES = (
    "environment.json",
    "command_metadata.json",
    "prompt_manifest.json",
    "kernel_probe.json",
    "e4_format_axis.json",
    "artifact_hashes.json",
)


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic UTF-8 JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_run_dir() -> Path:
    """Return the newest E4 result directory."""
    candidates = [path for path in RESULTS_DIR.glob("om_stage1_e4_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no E4 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_completed_metrics(format_id: str, row: dict[str, Any]) -> list[str]:
    """Validate metric rows for a completed format."""
    reasons: list[str] = []
    metrics = row.get("metrics_by_benchmark")
    if not isinstance(metrics, dict):
        return [f"{format_id}: completed row missing metrics_by_benchmark"]
    for benchmark in BENCHMARKS:
        item = metrics.get(benchmark)
        if not isinstance(item, dict):
            reasons.append(f"{format_id}:{benchmark}: missing metrics")
            continue
        for key in ("accuracy", "tokens_per_second", "peak_vram_gb"):
            value = item.get(key)
            if not isinstance(value, (int, float)):
                reasons.append(f"{format_id}:{benchmark}: {key} must be numeric")
            elif key == "accuracy" and not 0.0 <= float(value) <= 1.0:
                reasons.append(f"{format_id}:{benchmark}: accuracy outside [0, 1]")
            elif key != "accuracy" and float(value) < 0.0:
                reasons.append(f"{format_id}:{benchmark}: {key} is negative")
        if int(item.get("prompt_count", 0)) <= 0:
            reasons.append(f"{format_id}:{benchmark}: prompt_count must be positive")
    return reasons


def validate_packet(packet: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return infra reasons, completed formats, and skipped formats."""
    reasons: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []
    if packet.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
        reasons.append("e4_format_axis schema_version mismatch")
    if packet.get("model_id") != "ibm-granite/granite-4.0-h-small":
        reasons.append("E4 must target Granite-4.0-H-Small only")
    rows = packet.get("format_results")
    if not isinstance(rows, dict):
        return reasons + ["format_results must be a dict"], completed, skipped
    missing = sorted(set(FORMATS).difference(rows))
    extra = sorted(set(rows).difference(FORMATS))
    if missing:
        reasons.append(f"missing format rows: {missing}")
    if extra:
        reasons.append(f"unexpected format rows: {extra}")
    for format_id in FORMATS:
        row = rows.get(format_id)
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status not in VALID_STATUSES:
            reasons.append(f"{format_id}: invalid status {status!r}")
            continue
        if status == "COMPLETED":
            completed.append(format_id)
            reasons.extend(validate_completed_metrics(format_id, row))
        elif status == "SKIPPED_INFRA":
            skipped.append(format_id)
            if not row.get("skip_reason"):
                reasons.append(f"{format_id}: SKIPPED_INFRA requires skip_reason")
        elif not row.get("pending_reason"):
            reasons.append(f"{format_id}: PENDING_GPU_RUN requires pending_reason")
    return reasons, completed, skipped


def evaluate(run_dir: Path) -> dict[str, Any]:
    """Evaluate an E4 run directory."""
    infra = [f"missing required artifact: {rel}" for rel in REQUIRED_FILES if not (run_dir / rel).is_file()]
    packet: dict[str, Any] = {}
    completed: list[str] = []
    skipped: list[str] = []
    if not infra:
        packet = load_json(run_dir / "e4_format_axis.json")
        packet_reasons, completed, skipped = validate_packet(packet)
        infra.extend(packet_reasons)
    if infra:
        decision = FAIL_INFRA
        reasons = infra
    elif len(completed) == len(FORMATS):
        decision = PASS
        reasons = ["all preregistered formats completed"]
    elif completed:
        decision = PARTIAL
        reasons = [f"completed={completed}; skipped={skipped}"]
    else:
        decision = INCOMPLETE
        reasons = ["no format has completed benchmark metrics"]
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": decision in {PASS, PARTIAL},
        "run_dir": str(run_dir),
        "reasons": reasons,
        "completed_formats": completed,
        "skipped_formats": skipped,
        "headline": packet.get("headline") if packet else None,
    }
    write_json(run_dir / "checker_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.run_dir or latest_run_dir())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
