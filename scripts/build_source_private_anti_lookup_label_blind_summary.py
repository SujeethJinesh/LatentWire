from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import statistics
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_LABEL_BLIND_SUMMARIES = (
    "results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n8_label_blind/summary.json",
    "results/source_private_anti_lookup_label_blind_20260429/holdout_seed30_qwen3_n8_label_blind/summary.json",
    "results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n32_label_blind/summary.json",
    "results/source_private_anti_lookup_label_blind_20260429/holdout_seed30_qwen3_n32_label_blind/summary.json",
)
DEFAULT_POSITIVE_SUMMARIES = (
    "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
    "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
    "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
    "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
)

OPAQUE_CONDITIONS = (
    "matched_packet",
    "matched_byte_text_2",
    "random_same_byte_packet",
    "deranged_candidate_diag_table",
    "query_aware_diag_span",
    "structured_json_diag",
    "structured_free_text_diag",
    "full_hidden_log",
)

CSV_COLUMNS = (
    "surface",
    "candidate_view",
    "n",
    "target_accuracy",
    "matched_packet_accuracy",
    "max_opaque_accuracy",
    "max_opaque_minus_target",
    "max_opaque_ci95_high_vs_target",
    "max_opaque_strict_ci95_high_vs_target",
    "matched_packet_valid_rate",
    "exact_id_parity",
    "collapse_pass",
    "diagnostic_table_positive_packet_accuracy",
    "diagnostic_table_positive_target_accuracy",
    "diagnostic_table_positive_lift",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface(path: pathlib.Path, summary: dict[str, Any]) -> str:
    name = path.parent.name
    if name.startswith("core_"):
        return f"core n{summary['n']} {summary.get('candidate_view', 'unknown')}"
    if name.startswith("holdout_"):
        return f"holdout n{summary['n']} {summary.get('candidate_view', 'unknown')}"
    return f"{name} n{summary['n']}"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def _paired_deltas(rows_path: pathlib.Path, *, condition: str, field: str) -> list[float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(rows_path):
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = []
    for _, conditions in sorted(by_example.items()):
        if condition in conditions and "target_only" in conditions:
            deltas.append(float(bool(conditions[condition][field])) - float(bool(conditions["target_only"][field])))
    return deltas


def _collapse_cis(label_blind_path: pathlib.Path, *, samples: int = 2000, seed: int = 29) -> dict[str, Any]:
    rows_path = label_blind_path.parent / "endpoint_proxy_rows.jsonl"
    comparisons: dict[str, Any] = {}
    for index, condition in enumerate(OPAQUE_CONDITIONS):
        comparisons[condition] = {
            "delta_bootstrap95": _bootstrap_ci(
                _paired_deltas(rows_path, condition=condition, field="correct"),
                samples=samples,
                seed=seed + index * 1009,
            ),
            "strict_delta_bootstrap95": _bootstrap_ci(
                _paired_deltas(rows_path, condition=condition, field="strict_correct"),
                samples=samples,
                seed=seed + 17 + index * 1009,
            ),
        }
    return comparisons


def _row(label_blind_path: pathlib.Path, positive_path: pathlib.Path) -> dict[str, Any]:
    label_blind = _read_json(label_blind_path)
    positive = _read_json(positive_path)
    target = label_blind["metrics"]["target_only"]["accuracy"]
    max_opaque = max(label_blind["metrics"][condition]["accuracy"] for condition in OPAQUE_CONDITIONS)
    collapse_cis = _collapse_cis(label_blind_path)
    max_opaque_ci95_high = max(
        collapse_cis[condition]["delta_bootstrap95"]["ci95_high"] for condition in OPAQUE_CONDITIONS
    )
    max_opaque_strict_ci95_high = max(
        collapse_cis[condition]["strict_delta_bootstrap95"]["ci95_high"] for condition in OPAQUE_CONDITIONS
    )
    positive_packet = positive["metrics"]["matched_packet"]["accuracy"]
    positive_target = positive["metrics"]["target_only"]["accuracy"]
    row = {
        "surface": _surface(label_blind_path, label_blind),
        "candidate_view": label_blind.get("candidate_view"),
        "n": label_blind["n"],
        "target_accuracy": target,
        "matched_packet_accuracy": label_blind["metrics"]["matched_packet"]["accuracy"],
        "max_opaque_accuracy": max_opaque,
        "max_opaque_minus_target": max_opaque - target,
        "max_opaque_ci95_high_vs_target": max_opaque_ci95_high,
        "max_opaque_strict_ci95_high_vs_target": max_opaque_strict_ci95_high,
        "matched_packet_valid_rate": label_blind["metrics"]["matched_packet"]["valid_prediction_rate"],
        "exact_id_parity": bool(label_blind["exact_id_parity"]),
        "collapse_pass": (
            label_blind.get("candidate_view") == "label_blind"
            and bool(label_blind["exact_id_parity"])
            and label_blind["metrics"]["matched_packet"]["valid_prediction_rate"] >= 0.95
            and max_opaque <= target + 0.05
            and max_opaque_ci95_high <= 0.10
            and max_opaque_strict_ci95_high <= 0.10
            and positive_packet >= positive_target + 0.15
        ),
        "collapse_comparisons_vs_target": collapse_cis,
        "diagnostic_table_positive_packet_accuracy": positive_packet,
        "diagnostic_table_positive_target_accuracy": positive_target,
        "diagnostic_table_positive_lift": positive_packet - positive_target,
    }
    return row


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Anti-Lookup Label-Blind Summary",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- scale rung: `{payload['scale_rung']}`",
        "",
        "## Headline",
        "",
    ]
    for key, value in payload["headline"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Surface | Collapse pass | Target | Matched packet | Max opaque | Opaque-target | Max opaque CI high | Positive diagnostic-table lift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['surface']} | `{row['collapse_pass']}` | {row['target_accuracy']:.3f} | "
            f"{row['matched_packet_accuracy']:.3f} | {row['max_opaque_accuracy']:.3f} | "
            f"{row['max_opaque_minus_target']:.3f} | {row['max_opaque_ci95_high_vs_target']:.3f} | "
            f"{row['diagnostic_table_positive_lift']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_anti_lookup_summary(
    *,
    label_blind_summaries: list[pathlib.Path],
    positive_summaries: list[pathlib.Path],
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_paths = [_resolve(path) for path in label_blind_summaries]
    positive_paths = [_resolve(path) for path in positive_summaries]
    if len(label_paths) != len(positive_paths):
        raise ValueError("label_blind_summaries and positive_summaries must have equal length")
    rows = [_row(label_path, positive_path) for label_path, positive_path in zip(label_paths, positive_paths, strict=True)]
    headline = {
        "rows": len(rows),
        "collapse_pass_rows": sum(1 for row in rows if row["collapse_pass"]),
        "max_opaque_minus_target": max(row["max_opaque_minus_target"] for row in rows),
        "max_opaque_ci95_high_vs_target": max(row["max_opaque_ci95_high_vs_target"] for row in rows),
        "max_opaque_strict_ci95_high_vs_target": max(row["max_opaque_strict_ci95_high_vs_target"] for row in rows),
        "min_diagnostic_table_positive_lift": min(row["diagnostic_table_positive_lift"] for row in rows),
        "min_matched_packet_valid_rate": min(row["matched_packet_valid_rate"] for row in rows),
        "all_exact_id_parity": all(row["exact_id_parity"] for row in rows),
    }
    pass_gate = (
        headline["collapse_pass_rows"] == headline["rows"]
        and headline["max_opaque_minus_target"] <= 0.05
        and headline["max_opaque_ci95_high_vs_target"] <= 0.10
        and headline["max_opaque_strict_ci95_high_vs_target"] <= 0.10
        and headline["min_diagnostic_table_positive_lift"] >= 0.15
        and headline["min_matched_packet_valid_rate"] >= 0.95
        and headline["all_exact_id_parity"]
    )
    payload = {
        "gate": "source_private_anti_lookup_label_blind_summary",
        "pass_gate": pass_gate,
        "scale_rung": "strict-small anti-lookup stress" if max(row["n"] for row in rows) >= 32 else "micro smoke anti-lookup stress",
        "source_label_blind_summaries": [str(path.relative_to(ROOT)) for path in label_paths],
        "source_positive_summaries": [str(path.relative_to(ROOT)) for path in positive_paths],
        "headline": headline,
        "opaque_conditions": list(OPAQUE_CONDITIONS),
        "rows": rows,
        "interpretation": (
            "When candidate repair-key metadata and original labels are hidden, opaque diagnostic packets and text relays "
            "collapse to target accuracy while the diagnostic-table endpoint rows remain strongly positive. This weakens "
            "the leakage concern that hidden labels alone explain the positive endpoint row, but it also confirms the "
            "current method requires a public side-information table. It is not protocol-free semantic transfer. "
            "Bootstrap upper bounds here are one-sided collapse diagnostics, not positive-method CIs."
        ),
        "next_gate": (
            "Run n=160 core+holdout label-blind stress with paired uncertainty, then implement a learned/shared-dictionary "
            "receiver if the goal is a protocol-free or less table-shaped method."
        ),
    }

    json_path = output_dir / "anti_lookup_label_blind_summary.json"
    csv_path = output_dir / "anti_lookup_label_blind_summary.csv"
    md_path = output_dir / "anti_lookup_label_blind_summary.md"
    manifest_path = output_dir / "manifest.json"
    manifest_md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name, manifest_md_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": headline,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_md_path.write_text(
        "\n".join(
            [
                "# Anti-Lookup Label-Blind Summary Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- rows: `{headline['rows']}`",
                f"- max opaque minus target: `{headline['max_opaque_minus_target']:.3f}`",
                f"- max opaque CI95 high vs target: `{headline['max_opaque_ci95_high_vs_target']:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-blind-summary", action="append", type=pathlib.Path)
    parser.add_argument("--positive-summary", action="append", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_anti_lookup_label_blind_20260429"),
    )
    args = parser.parse_args()
    label_blind = args.label_blind_summary or [pathlib.Path(path) for path in DEFAULT_LABEL_BLIND_SUMMARIES]
    positive = args.positive_summary or [pathlib.Path(path) for path in DEFAULT_POSITIVE_SUMMARIES]
    output_dir = _resolve(args.output_dir)
    payload = build_anti_lookup_summary(
        label_blind_summaries=label_blind,
        positive_summaries=positive,
        output_dir=output_dir,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"], "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
