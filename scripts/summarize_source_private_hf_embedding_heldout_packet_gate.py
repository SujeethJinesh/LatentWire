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


def _load_gate(path: pathlib.Path) -> dict[str, Any]:
    gate_path = path / "learned_synonym_dictionary_packet_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    return json.loads(gate_path.read_text(encoding="utf-8"))


def _load_direction_budget_rows(path: pathlib.Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for summary_path in sorted(path.glob("*/summary.json")):
        direction = summary_path.parent.name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for summary in payload.get("budget_summaries", []):
            rows[(direction, int(summary["budget_bytes"]))] = {
                "best_control_name": summary.get("best_control_name"),
                "top_atom_knockout_accuracy": summary.get("top_atom_knockout_accuracy"),
                "private_random_knockout_accuracy": summary.get("private_random_knockout_accuracy"),
                "top_atom_knockout_lift_reduction": summary.get("top_atom_knockout_lift_reduction"),
                "private_random_knockout_lift_reduction": summary.get("private_random_knockout_lift_reduction"),
            }
    return rows


def _row_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["learned_minus_best_control"]),
        float(row["learned_minus_target"]),
        float(row["paired_ci95_low_vs_target"]),
    )


def _flatten(run_dirs: list[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        payload = _load_gate(run_dir)
        budget_aux = _load_direction_budget_rows(run_dir)
        for row in payload["rows"]:
            aux = budget_aux.get((row["direction"], int(row["budget_bytes"])), {})
            rows.append(
                {
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "feature_model": payload.get("feature_model"),
                    "text_feature_mode": payload["text_feature_mode"],
                    "receiver_mode": payload["receiver_mode"],
                    "min_decision_score": payload["min_decision_score"],
                    "top_k": payload["top_k"],
                    "min_score": payload["min_score"],
                    "ridge": payload["ridge"],
                    **aux,
                    **row,
                }
            )
    return rows


def _semantic_anchor_summary(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    headline = payload.get("headline", {})
    return {
        "path": str(path.relative_to(ROOT)),
        "pass_gate": bool(payload.get("pass_gate", headline.get("all_seed_pass", False))),
        "rows": len(rows),
        "pass_rows": int(headline.get("pass_rows", sum(1 for row in rows if row["pass_gate"]))),
        "min_passing_accuracy": min(row["learned_synonym_dictionary_accuracy"] for row in rows if row["pass_gate"]),
        "min_passing_ci95_low_vs_target": float(
            headline.get(
                "min_passing_ci95_low_vs_target",
                min(row["paired_ci95_low_vs_target"] for row in rows if row["pass_gate"]),
            )
        ),
        "max_best_control_accuracy": max(row["best_control_accuracy"] for row in rows),
        "min_oracle_accuracy": float(
            headline.get(
                "min_passing_oracle_accuracy",
                min(row["oracle_learned_candidate_atoms_accuracy"] for row in rows),
            )
        ),
        "text_feature_mode": payload["text_feature_mode"],
    }


def summarize(rows: list[dict[str, Any]], semantic_anchor: dict[str, Any] | None) -> dict[str, Any]:
    pass_rows = [row for row in rows if row["pass_gate"]]
    near_miss_rows = [
        row
        for row in rows
        if row["controls_ok"]
        and row["learned_minus_target"] >= 0.15
        and row["learned_minus_best_control"] >= 0.10
        and row["paired_ci95_low_vs_target"] > 0.05
        and row["oracle_learned_candidate_atoms_accuracy"] >= 0.75
    ]
    best_rows = sorted(rows, key=_row_key, reverse=True)[:12]
    by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = by_run.setdefault(
            row["run_dir"],
            {
                "run_dir": row["run_dir"],
                "feature_model": row["feature_model"],
                "text_feature_mode": row["text_feature_mode"],
                "receiver_mode": row["receiver_mode"],
                "pass_rows": 0,
                "direction_pass": {"core_to_holdout": False, "holdout_to_core": False, "same_family_all": False},
                "max_accuracy": 0.0,
                "max_delta_vs_target": 0.0,
                "max_ci95_low_vs_target": -1.0,
                "min_best_control": 1.0,
                "min_oracle": 1.0,
                "max_private_random_knockout_lift_reduction": 0.0,
            },
        )
        if row["pass_gate"]:
            item["pass_rows"] += 1
            item["direction_pass"][row["direction"]] = True
        item["max_accuracy"] = max(item["max_accuracy"], row["learned_synonym_dictionary_accuracy"])
        item["max_delta_vs_target"] = max(item["max_delta_vs_target"], row["learned_minus_target"])
        item["max_ci95_low_vs_target"] = max(item["max_ci95_low_vs_target"], row["paired_ci95_low_vs_target"])
        item["min_best_control"] = min(item["min_best_control"], row["best_control_accuracy"])
        item["min_oracle"] = min(item["min_oracle"], row["oracle_learned_candidate_atoms_accuracy"])
        if row.get("private_random_knockout_lift_reduction") is not None:
            item["max_private_random_knockout_lift_reduction"] = max(
                item["max_private_random_knockout_lift_reduction"],
                row["private_random_knockout_lift_reduction"],
            )
    return {
        "rows": len(rows),
        "pass_rows": len(pass_rows),
        "near_miss_rows": len(near_miss_rows),
        "best_rows": best_rows,
        "runs": sorted(by_run.values(), key=lambda row: (row["pass_rows"], row["max_accuracy"]), reverse=True),
        "semantic_anchor_reference": semantic_anchor,
        "pass_gate": any(
            run["direction_pass"]["core_to_holdout"] and run["direction_pass"]["holdout_to_core"]
            for run in by_run.values()
        ),
        "interpretation": (
            "Frozen embedding receivers recover part of the semantic-anchor held-out packet signal with clean "
            "source-destroying controls, but none of the tested BGE/MiniLM variants clears the strict "
            "bidirectional gate. This weakens the hypothesis that a generic frozen text embedding alone can "
            "replace the explicit public semantic-anchor lexicon; the live next branch is a learned receiver "
            "or a better public ontology calibration layer, not another fixed-basis packet tweak."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    lines = [
        "# Frozen Embedding Held-Out Packet Summary",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- rows: `{summary['rows']}`",
        f"- pass rows: `{summary['pass_rows']}`",
        f"- near-miss rows: `{summary['near_miss_rows']}`",
    ]
    if summary["semantic_anchor_reference"] is not None:
        ref = summary["semantic_anchor_reference"]
        lines.extend(
            [
                f"- semantic-anchor reference pass rows: `{ref['pass_rows']}/{ref['rows']}`",
                f"- semantic-anchor min passing accuracy: `{ref['min_passing_accuracy']}`",
                f"- semantic-anchor min CI95 low: `{ref['min_passing_ci95_low_vs_target']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Model | Mode | Pass rows | Direction pass | Max acc | Max delta | Max CI95 low | Min oracle |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["runs"]:
        lines.append(
            f"| `{row['run_dir']}` | {row['feature_model']} | `{row['text_feature_mode']}` | "
            f"{row['pass_rows']} | `{row['direction_pass']}` | {row['max_accuracy']:.3f} | "
            f"{row['max_delta_vs_target']:.3f} | {row['max_ci95_low_vs_target']:.3f} | {row['min_oracle']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best Rows",
            "",
            "| Run | Direction | Budget | Pass | Acc | Target | Best ctrl | Delta | CI95 low | Oracle | Top knock | Random knock | Controls |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["best_rows"]:
        top_knock = row.get("top_atom_knockout_lift_reduction")
        random_knock = row.get("private_random_knockout_lift_reduction")
        lines.append(
            f"| `{row['run_dir']}` | {row['direction']} | {row['budget_bytes']} | `{row['pass_gate']}` | "
            f"{row['learned_synonym_dictionary_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['learned_minus_target']:.3f} | "
            f"{row['paired_ci95_low_vs_target']:.3f} | {row['oracle_learned_candidate_atoms_accuracy']:.3f} | "
            f"{fmt(top_knock)} | {fmt(random_knock)} | "
            f"`{row['controls_ok']}` |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, action="append", required=True)
    parser.add_argument(
        "--semantic-anchor-summary",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_semantic_anchor_heldout_medium_confirmation_20260430/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_hf_embedding_heldout_packet_gate_20260430/summary"),
    )
    args = parser.parse_args()
    run_dirs = [path if path.is_absolute() else ROOT / path for path in args.run_dir]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = (
        None
        if args.semantic_anchor_summary is None
        else args.semantic_anchor_summary
        if args.semantic_anchor_summary.is_absolute()
        else ROOT / args.semantic_anchor_summary
    )
    rows = _flatten(run_dirs)
    payload = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "run_dirs": [str(path.relative_to(ROOT)) for path in run_dirs],
        "summary": summarize(rows, _semantic_anchor_summary(semantic_path)),
        "rows": rows,
    }
    (output_dir / "hf_embedding_heldout_packet_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "hf_embedding_heldout_packet_summary.md", payload)
    manifest = {
        "artifacts": [
            "hf_embedding_heldout_packet_summary.json",
            "hf_embedding_heldout_packet_summary.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["hf_embedding_heldout_packet_summary.json", "hf_embedding_heldout_packet_summary.md"]
        },
        "pass_gate": payload["summary"]["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Frozen Embedding Held-Out Packet Manifest",
                "",
                f"- pass gate: `{payload['summary']['pass_gate']}`",
                f"- rows: `{payload['summary']['rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["summary"]["pass_gate"],
                "rows": payload["summary"]["rows"],
                "pass_rows": payload["summary"]["pass_rows"],
                "near_miss_rows": payload["summary"]["near_miss_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
