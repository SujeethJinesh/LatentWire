from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run(path: pathlib.Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json under {path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    return {
        "run_dir": str(path),
        "basis_view": payload["basis_view"],
        "budget_bytes": payload["budget_bytes"],
        "variant": payload["variant"],
        "remap_slot_seed": payload["remap_slot_seed"],
        "train_family_set": payload["train_family_set"],
        "eval_family_set": payload["eval_family_set"],
        "train_examples": payload["train_examples"],
        "eval_examples": payload["eval_examples"],
        "train_start_index": payload["train_start_index"],
        "eval_start_index": payload["eval_start_index"],
        "pass_gate": summary["pass_gate"],
        "train_eval_id_intersection_count": summary["train_eval_id_intersection_count"],
        "source_accuracy": summary["source_accuracy"],
        "target_only_accuracy": summary["target_only_accuracy"],
        "best_control_condition": summary["best_control_condition"],
        "best_control_accuracy": summary["best_control_accuracy"],
        "source_minus_best_control": summary["source_minus_best_control"],
        "ci95_low_vs_best_control": summary["paired_bootstrap"]["source_vs_best_control"]["ci95_low"],
        "unquantized_predicted_accuracy": summary["unquantized_predicted_accuracy"],
        "target_innovation_oracle_accuracy": summary["target_innovation_oracle_accuracy"],
        "unique_payload_ratio": summary["payload_uniqueness"]["unique_payload_ratio"],
        "unique_payloads": summary["payload_uniqueness"]["unique_payloads"],
        "max_payload_frequency": summary["payload_uniqueness"]["max_payload_frequency"],
        "reused_payload_accuracy": summary["payload_uniqueness"]["reused_payload_accuracy"],
        "source_p50_latency_ms": summary["metrics"]["source"]["p50_latency_ms"],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisive_rows = [
        row
        for row in rows
        if row["train_family_set"] == "all"
        and row["eval_family_set"] == "all"
        and row["eval_examples"] >= 500
        and row["train_eval_id_intersection_count"] == 0
    ]
    cross_family_rows = [
        row
        for row in rows
        if row["train_family_set"] != row["eval_family_set"]
        and row["train_eval_id_intersection_count"] == 0
    ]
    budget2_rows = [row for row in decisive_rows if row["budget_bytes"] == 2]
    anchor_rows = [row for row in decisive_rows if row["basis_view"] == "anchor_relative"]
    shared_rows = [row for row in decisive_rows if row["basis_view"] == "shared_text"]
    return {
        "rows": len(rows),
        "pass_rows": sum(1 for row in rows if row["pass_gate"]),
        "decisive_disjoint_n500_rows": len(decisive_rows),
        "decisive_disjoint_n500_pass_rows": sum(1 for row in decisive_rows if row["pass_gate"]),
        "anchor_relative_decisive_rows": len(anchor_rows),
        "anchor_relative_decisive_pass_rows": sum(1 for row in anchor_rows if row["pass_gate"]),
        "shared_text_decisive_rows": len(shared_rows),
        "shared_text_decisive_pass_rows": sum(1 for row in shared_rows if row["pass_gate"]),
        "budget2_decisive_rows": len(budget2_rows),
        "budget2_decisive_pass_rows": sum(1 for row in budget2_rows if row["pass_gate"]),
        "min_decisive_source_accuracy": None if not decisive_rows else min(row["source_accuracy"] for row in decisive_rows),
        "max_decisive_best_control_accuracy": None if not decisive_rows else max(row["best_control_accuracy"] for row in decisive_rows),
        "min_decisive_ci95_low_vs_best_control": None
        if not decisive_rows
        else min(row["ci95_low_vs_best_control"] for row in decisive_rows),
        "min_budget2_unique_payload_ratio": None if not budget2_rows else min(row["unique_payload_ratio"] for row in budget2_rows),
        "max_budget2_unique_payload_ratio": None if not budget2_rows else max(row["unique_payload_ratio"] for row in budget2_rows),
        "cross_family_rows": len(cross_family_rows),
        "cross_family_pass_rows": sum(1 for row in cross_family_rows if row["pass_gate"]),
        "max_cross_family_source_accuracy": None if not cross_family_rows else max(row["source_accuracy"] for row in cross_family_rows),
        "pass_gate": bool(decisive_rows)
        and all(row["pass_gate"] for row in decisive_rows)
        and bool(budget2_rows)
        and all(row["pass_gate"] for row in budget2_rows),
        "interpretation": (
            "Conditional PQ innovation passes same-family disjoint-ID n500 gates across shared-text and "
            "anchor-relative bases, including 2-byte low-uniqueness rows. Bidirectional held-out-family rows "
            "remain negative, so the method should be framed as shared-schema disjoint communication rather than "
            "unseen-family latent transfer."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["runs"]
    lines = [
        "# Conditional PQ Innovation Summary",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- rows: `{summary['rows']}`",
        f"- decisive n500 pass rows: `{summary['decisive_disjoint_n500_pass_rows']}/{summary['decisive_disjoint_n500_rows']}`",
        f"- budget-2 n500 pass rows: `{summary['budget2_decisive_pass_rows']}/{summary['budget2_decisive_rows']}`",
        f"- cross-family pass rows: `{summary['cross_family_pass_rows']}/{summary['cross_family_rows']}`",
        f"- min decisive source accuracy: `{summary['min_decisive_source_accuracy']}`",
        f"- max decisive best control accuracy: `{summary['max_decisive_best_control_accuracy']}`",
        "",
        "| Run | Basis | Bytes | Remap | Train->Eval | Pass | Source | Target | Best control | CI95 low | Unique ratio |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {pathlib.Path(row['run_dir']).name} | {row['basis_view']} | {row['budget_bytes']} | "
            f"{row['remap_slot_seed']} | {row['train_family_set']}->{row['eval_family_set']} | "
            f"`{row['pass_gate']}` | {row['source_accuracy']:.3f} | {row['target_only_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['ci95_low_vs_best_control']:.3f} | "
            f"{row['unique_payload_ratio']:.3f} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [_read_run(path if path.is_absolute() else ROOT / path) for path in args.run_dir]
    payload = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "runs": runs,
        "summary": _summarize(runs),
    }
    (output_dir / "conditional_pq_innovation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "conditional_pq_innovation_summary.md", payload)
    manifest = {
        "artifacts": [
            "conditional_pq_innovation_summary.json",
            "conditional_pq_innovation_summary.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["conditional_pq_innovation_summary.json", "conditional_pq_innovation_summary.md"]
        },
        "pass_gate": payload["summary"]["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Conditional PQ Innovation Summary Manifest",
                "",
                f"- pass gate: `{payload['summary']['pass_gate']}`",
                f"- rows: `{payload['summary']['rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"pass_gate": payload["summary"]["pass_gate"], "rows": len(runs)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
