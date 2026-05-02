from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

LEDGER_PATH = ROOT / "results/source_private_pass_fail_ledger_20260429/pass_fail_ledger.json"

MASKED_INNOVATION_SUMMARIES = [
    (
        "masked_innovation_anchor_relative_core_to_holdout",
        ROOT / "results/source_private_masked_innovation_receiver_20260429/core_to_holdout_seed29_30/summary.json",
    ),
    (
        "masked_innovation_shared_text_core_to_holdout",
        ROOT / "results/source_private_masked_innovation_receiver_20260429/core_to_holdout_shared_text_seed29_30/summary.json",
    ),
]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_oracle_from_note(note: str | None) -> float | None:
    if not note:
        return None
    match = re.search(r"full_diag_oracle=([0-9.]+)", note)
    return float(match.group(1)) if match else None


def _is_cross_family_ledger_row(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ["row_id", "contribution", "surface", "method", "note"]
    ).lower()
    return any(
        marker in haystack
        for marker in [
            "cross-family",
            "cross_family",
            "core_to_holdout",
            "holdout_to_core",
            "core -> holdout",
            "holdout -> core",
            "heldout-family",
            "holdout-eval",
        ]
    )


def _claim_status(row: dict[str, Any]) -> str:
    bucket = row.get("reviewer_bucket")
    status = row.get("status")
    evidence_complete = bool(row.get("evidence_complete"))
    if bucket == "failed_or_pruned" or status in {"fail", "near-miss"}:
        return "negative_boundary"
    if status == "pass" and not evidence_complete:
        return "asymmetric_or_incomplete_not_claimed"
    return "not_headline_claim"


def _ledger_rows() -> list[dict[str, Any]]:
    ledger = _read_json(LEDGER_PATH)
    rows: list[dict[str, Any]] = []
    for row in ledger["rows"]:
        if not _is_cross_family_ledger_row(row):
            continue
        oracle = _extract_oracle_from_note(row.get("note"))
        rows.append(
            {
                "row_id": row["row_id"],
                "source": str(LEDGER_PATH.relative_to(ROOT)),
                "contribution": row["contribution"],
                "method": row["method"],
                "surface": row["surface"],
                "budget_bytes": _float_or_none(row.get("mean_payload_bytes")),
                "accuracy": _float_or_none(row.get("accuracy")),
                "target_accuracy": _float_or_none(row.get("target_accuracy")),
                "best_control_accuracy": _float_or_none(row.get("best_control_accuracy")),
                "oracle_accuracy": oracle,
                "oracle_headroom_vs_target": None
                if oracle is None or row.get("target_accuracy") is None
                else oracle - float(row["target_accuracy"]),
                "matched_minus_target": _float_or_none(row.get("matched_minus_target")),
                "matched_minus_best_control": _float_or_none(row.get("matched_minus_best_control")),
                "status": row["status"],
                "reviewer_bucket": row["reviewer_bucket"],
                "evidence_complete": bool(row.get("evidence_complete")),
                "claim_status": _claim_status(row),
                "failure_or_boundary_reason": row.get("pruning_reason")
                or row.get("missing_evidence")
                or "asymmetric/incomplete positive row is not a headline claim",
            }
        )
    return rows


def _masked_innovation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_prefix, path in MASKED_INNOVATION_SUMMARIES:
        if not path.exists():
            continue
        summary = _read_json(path)
        for budget in summary["budget_summaries"]:
            accuracy = budget["matched_accuracy"]
            target = budget["best_no_source_accuracy"]
            best_control = budget["best_destructive_control_accuracy"]
            oracle = budget.get("full_diag_oracle_accuracy")
            delta_control = accuracy - best_control
            delta_target = accuracy - target
            pass_gate = bool(budget.get("pass_gate", False))
            rows.append(
                {
                    "row_id": f"{row_prefix}::budget{budget['budget_bytes']}",
                    "source": str(path.relative_to(ROOT)),
                    "contribution": "masked innovation receiver cross-family boundary",
                    "method": f"{budget['budget_bytes']}-byte masked innovation receiver",
                    "surface": f"{summary['train_family_set']} -> {summary['eval_family_set']}",
                    "budget_bytes": float(budget["budget_bytes"]),
                    "accuracy": accuracy,
                    "target_accuracy": target,
                    "best_control_accuracy": best_control,
                    "oracle_accuracy": oracle,
                    "oracle_headroom_vs_target": None if oracle is None else oracle - target,
                    "matched_minus_target": delta_target,
                    "matched_minus_best_control": delta_control,
                    "status": "pass" if pass_gate else "fail",
                    "reviewer_bucket": "failed_or_pruned" if not pass_gate else "positive_needs_more_evidence",
                    "evidence_complete": True,
                    "claim_status": "negative_boundary" if not pass_gate else "not_headline_claim",
                    "failure_or_boundary_reason": (
                        "full diagnostic oracle is high but learned source-private packet stays at target/control floor"
                        if not pass_gate and oracle and oracle >= 0.95
                        else "does not pass strict cross-family gate"
                    ),
                }
            )
    return rows


def _method_family(row: dict[str, Any]) -> str:
    contribution = row["contribution"].lower()
    if "wyner-ziv" in contribution:
        return "learned Wyner-Ziv / scalar syndrome"
    if "anchor-relative" in contribution:
        return "anchor-relative sparse packet"
    if "canonical rasp" in contribution:
        return "canonical RASP"
    if "consistent" in contribution:
        return "consistent posterior packet"
    if "masked innovation" in contribution:
        return "masked innovation receiver"
    if "learned target-preserving receiver" in contribution:
        return "learned target-preserving receiver"
    return row["contribution"]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({_method_family(row) for row in rows})
    by_family: dict[str, dict[str, Any]] = {}
    for family in families:
        family_rows = [row for row in rows if _method_family(row) == family]
        failed = [row for row in family_rows if row["claim_status"] == "negative_boundary"]
        incomplete = [row for row in family_rows if row["claim_status"] == "asymmetric_or_incomplete_not_claimed"]
        best_accuracy = max(row["accuracy"] for row in family_rows if row["accuracy"] is not None)
        best_delta_control = max(
            row["matched_minus_best_control"]
            for row in family_rows
            if row["matched_minus_best_control"] is not None
        )
        by_family[family] = {
            "rows": len(family_rows),
            "negative_boundary_rows": len(failed),
            "asymmetric_or_incomplete_rows": len(incomplete),
            "best_accuracy": best_accuracy,
            "best_delta_vs_best_control": best_delta_control,
            "claim_status": "not_claimed_cross_family",
        }
    oracle_rows = [row for row in rows if row["oracle_accuracy"] is not None and row["oracle_accuracy"] >= 0.95]
    return {
        "total_rows": len(rows),
        "families": len(families),
        "negative_boundary_rows": sum(row["claim_status"] == "negative_boundary" for row in rows),
        "asymmetric_or_incomplete_not_claimed_rows": sum(
            row["claim_status"] == "asymmetric_or_incomplete_not_claimed" for row in rows
        ),
        "oracle_headroom_rows": len(oracle_rows),
        "claim_ready_cross_family_methods": 0,
        "by_family": by_family,
        "interpretation": (
            "Cross-family latent/source-private learned communication is not a headline claim. "
            "Several rows retain oracle headroom, so the benchmark can represent source information, "
            "but current learned/static interfaces do not transfer it bidirectionally under controls."
        ),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Cross-Family Negative Boundary",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- total rows: `{h['total_rows']}`",
        f"- method families: `{h['families']}`",
        f"- claim-ready cross-family methods: `{h['claim_ready_cross_family_methods']}`",
        f"- oracle-headroom rows: `{h['oracle_headroom_rows']}`",
        "",
        "## Family Summary",
        "",
        "| Family | Rows | Negative boundary | Asymmetric/incomplete | Best accuracy | Best delta vs control | Claim status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for family, row in h["by_family"].items():
        lines.append(
            "| "
            f"{family} | {row['rows']} | {row['negative_boundary_rows']} | "
            f"{row['asymmetric_or_incomplete_rows']} | {_fmt(row['best_accuracy'])} | "
            f"{_fmt(row['best_delta_vs_best_control'])} | `{row['claim_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary Rows",
            "",
            "| Row | Surface | Method | Bytes | Acc | Target | Best control | Oracle | Claim status | Reason |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"`{row['row_id']}` | {row['surface']} | {row['method']} | "
            f"{_fmt(row['budget_bytes'], 1)} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['target_accuracy'])} | {_fmt(row['best_control_accuracy'])} | "
            f"{_fmt(row['oracle_accuracy'])} | `{row['claim_status']}` | "
            f"{row['failure_or_boundary_reason']} |"
        )
    lines.extend(["", f"Interpretation: {h['interpretation']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_boundary(*, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _ledger_rows() + _masked_innovation_rows()
    rows.sort(key=lambda row: (row["contribution"], row["surface"], row["budget_bytes"] or 0.0, row["row_id"]))
    headline = _summarize(rows)
    payload = {
        "gate": "source_private_cross_family_negative_boundary",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_ledger": str(LEDGER_PATH.relative_to(ROOT)),
        "input_summaries": [str(path.relative_to(ROOT)) for _, path in MASKED_INNOVATION_SUMMARIES if path.exists()],
        "pass_gate": (
            headline["claim_ready_cross_family_methods"] == 0
            and headline["negative_boundary_rows"] >= 10
            and headline["oracle_headroom_rows"] >= 2
        ),
        "headline": headline,
        "rows": rows,
        "pass_rule": (
            "This is a boundary artifact, not a positive-method gate: it passes if it "
            "records no claim-ready cross-family method, includes substantial negative "
            "rows, and preserves oracle-headroom diagnostics where available."
        ),
    }
    (output_dir / "cross_family_negative_boundary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "cross_family_negative_boundary.csv", rows)
    _write_markdown(output_dir / "cross_family_negative_boundary.md", payload)
    artifacts = [
        "cross_family_negative_boundary.json",
        "cross_family_negative_boundary.csv",
        "cross_family_negative_boundary.md",
        "manifest.json",
        "manifest.md",
    ]
    manifest = {
        "command": "./venv_arm64/bin/python scripts/build_source_private_cross_family_negative_boundary.py --output-dir "
        + str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir),
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "pass_gate": payload["pass_gate"],
        "python": sys.version,
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Cross-Family Negative Boundary Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- rows: `{len(rows)}`",
                f"- families: `{headline['families']}`",
                "",
                "## Artifacts",
                "",
                *[f"- `{artifact}`" for artifact in artifacts],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_boundary(output_dir=output_dir)
    if not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
