from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


CORE_FIELDS = (
    "accuracy",
    "target_accuracy",
    "best_control_accuracy",
    "matched_minus_target",
    "matched_minus_best_control",
)


OPTIONAL_EVIDENCE_FIELDS = (
    "valid_rate",
    "ci95_low_vs_target",
    "ci95_low_vs_comparator",
    "mean_payload_bytes",
)


LEDGER_COLUMNS = (
    "row_id",
    "contribution",
    "method",
    "surface",
    "status",
    "reviewer_bucket",
    "accuracy",
    "target_accuracy",
    "best_control_accuracy",
    "matched_minus_target",
    "matched_minus_best_control",
    "mean_payload_bytes",
    "valid_rate",
    "ci95_low_vs_target",
    "ci95_low_vs_comparator",
    "p50_latency_ms",
    "evidence_complete",
    "missing_evidence",
    "pass_reason",
    "pruning_reason",
    "comparator",
    "note",
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _missing_evidence(row: dict[str, Any]) -> list[str]:
    missing = [field for field in CORE_FIELDS if row.get(field) is None]
    if row.get("status") == "pass":
        missing.extend(field for field in OPTIONAL_EVIDENCE_FIELDS if row.get(field) is None)
    return missing


def _reviewer_bucket(row: dict[str, Any], missing: list[str]) -> str:
    if row["status"] != "pass":
        return "failed_or_pruned"
    if not missing and row.get("ci95_low_vs_target") is not None and row.get("ci95_low_vs_comparator") is not None:
        return "paper_ready_evidence"
    if row.get("matched_minus_best_control") is not None and row["matched_minus_best_control"] >= 0.15:
        return "positive_needs_more_evidence"
    return "weak_positive"


def _pass_reason(row: dict[str, Any], missing: list[str]) -> str:
    if row["status"] != "pass":
        return ""
    reasons = []
    delta_target = row.get("matched_minus_target")
    delta_control = row.get("matched_minus_best_control")
    if delta_target is not None:
        reasons.append(f"beats target by {delta_target:.3f}")
    if delta_control is not None:
        reasons.append(f"beats best destructive/control by {delta_control:.3f}")
    if row.get("ci95_low_vs_target") is not None:
        reasons.append(f"paired CI lower vs target {row['ci95_low_vs_target']:.3f}")
    if row.get("valid_rate") is not None:
        reasons.append(f"valid rate {row['valid_rate']:.3f}")
    if missing:
        reasons.append(f"missing reviewer evidence: {', '.join(missing)}")
    return "; ".join(reasons)


def _pruning_reason(row: dict[str, Any], missing: list[str]) -> str:
    if row["status"] == "pass":
        return ""
    reasons = []
    delta_target = row.get("matched_minus_target")
    delta_control = row.get("matched_minus_best_control")
    if delta_target is not None and delta_target <= 0:
        reasons.append(f"does not beat target ({delta_target:.3f})")
    if delta_control is not None and delta_control < 0.15:
        reasons.append(f"insufficient matched-control delta ({delta_control:.3f})")
    valid_rate = row.get("valid_rate")
    if valid_rate is not None and valid_rate < 0.95:
        reasons.append(f"valid rate below 0.95 ({valid_rate:.3f})")
    if missing:
        reasons.append(f"missing evidence: {', '.join(missing)}")
    if not reasons:
        reasons.append("source artifact marks row fail/near-miss")
    return "; ".join(reasons)


def _ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = _missing_evidence(row)
    bucket = _reviewer_bucket(row, missing)
    return {
        "row_id": row["row_id"],
        "contribution": row["contribution"],
        "method": row["method"],
        "surface": row["surface"],
        "status": row["status"],
        "reviewer_bucket": bucket,
        "accuracy": row.get("accuracy"),
        "target_accuracy": row.get("target_accuracy"),
        "best_control_accuracy": row.get("best_control_accuracy"),
        "matched_minus_target": row.get("matched_minus_target"),
        "matched_minus_best_control": row.get("matched_minus_best_control"),
        "mean_payload_bytes": row.get("mean_payload_bytes"),
        "valid_rate": row.get("valid_rate"),
        "ci95_low_vs_target": row.get("ci95_low_vs_target"),
        "ci95_low_vs_comparator": row.get("ci95_low_vs_comparator"),
        "p50_latency_ms": row.get("p50_latency_ms"),
        "evidence_complete": not missing,
        "missing_evidence": ",".join(missing),
        "pass_reason": _pass_reason(row, missing),
        "pruning_reason": _pruning_reason(row, missing),
        "comparator": row.get("comparator") or "",
        "note": row.get("note") or "",
    }


def build_pass_fail_ledger(*, frontier_json: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier = _read_json(frontier_json)
    rows = [_ledger_row(row) for row in frontier["rows"]]
    by_bucket = Counter(row["reviewer_bucket"] for row in rows)
    by_contribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_contribution[row["contribution"]][row["reviewer_bucket"]] += 1

    payload = {
        "gate": "source_private_pass_fail_ledger",
        "source_frontier": str(frontier_json),
        "total_rows": len(rows),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_contribution": {key: dict(sorted(value.items())) for key, value in sorted(by_contribution.items())},
        "paper_ready_rows": [row["row_id"] for row in rows if row["reviewer_bucket"] == "paper_ready_evidence"],
        "rows": rows,
        "interpretation": (
            "Reviewer-facing pass/fail ledger derived from the CPU systems frontier. "
            "Rows retain source-artifact status while adding explicit pass/pruning reasons and evidence gaps."
        ),
    }

    json_path = output_dir / "pass_fail_ledger.json"
    csv_path = output_dir / "pass_fail_ledger.csv"
    md_path = output_dir / "pass_fail_ledger.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in LEDGER_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "total_rows": len(rows),
        "by_bucket": payload["by_bucket"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Pass/Fail Ledger",
        "",
        f"- source frontier: `{payload['source_frontier']}`",
        f"- total rows: `{payload['total_rows']}`",
        "",
        "## Reviewer Buckets",
        "",
        "| Bucket | Rows |",
        "|---|---:|",
    ]
    for bucket, count in payload["by_bucket"].items():
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(
        [
            "",
            "## Contribution Summary",
            "",
            "| Contribution | Paper-ready | Positive needs more evidence | Weak positive | Failed/pruned |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for contribution, counts in payload["by_contribution"].items():
        lines.append(
            f"| {contribution} | {counts.get('paper_ready_evidence', 0)} | "
            f"{counts.get('positive_needs_more_evidence', 0)} | {counts.get('weak_positive', 0)} | "
            f"{counts.get('failed_or_pruned', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Paper-Ready Evidence Rows",
            "",
        ]
    )
    if payload["paper_ready_rows"]:
        lines.extend(f"- `{row_id}`" for row_id in payload["paper_ready_rows"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Highest-Risk Failed/Pruned Rows",
            "",
            "| Row | Contribution | Surface | Accuracy | Target | Best control | Reason |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    failed = [row for row in payload["rows"] if row["reviewer_bucket"] == "failed_or_pruned"]
    failed = sorted(
        failed,
        key=lambda row: (
            row["matched_minus_best_control"] if row["matched_minus_best_control"] is not None else -999.0,
            row["matched_minus_target"] if row["matched_minus_target"] is not None else -999.0,
        ),
        reverse=True,
    )[:20]
    for row in failed:
        lines.append(
            f"| `{row['row_id']}` | {row['contribution']} | {row['surface']} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['target_accuracy'])} | {_fmt(row['best_control_accuracy'])} | "
            f"{row['pruning_reason']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier-json",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_cpu_systems_frontier_20260429/cpu_systems_frontier.json"),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_pass_fail_ledger_20260429"))
    args = parser.parse_args()
    frontier = args.frontier_json if args.frontier_json.is_absolute() else ROOT / args.frontier_json
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_pass_fail_ledger(frontier_json=frontier, output_dir=output)
    print(json.dumps({"output_dir": str(output), "rows": payload["total_rows"], "by_bucket": payload["by_bucket"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
