from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run(path: pathlib.Path) -> dict[str, Any]:
    run_dir = _resolve(path)
    payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    return {
        "run_dir": str(path),
        "pass_gate": bool(payload["pass_gate"]),
        "variant": payload["variant"],
        "remap_slot_seed": payload["remap_slot_seed"],
        "eval_examples": payload["eval_examples"],
        "train_eval_id_intersection_count": payload["train_eval_id_intersection_count"],
        "is_disjoint": payload["train_eval_id_intersection_count"] == 0,
        "learned_source_accuracy": summary["learned_source_accuracy"],
        "l2_source_accuracy": summary["l2_source_accuracy"],
        "target_only_accuracy": summary["target_only_accuracy"],
        "best_control_condition": summary["best_control_condition"],
        "best_control_accuracy": summary["best_control_accuracy"],
        "learned_minus_best_control": summary["learned_minus_best_control"],
        "learned_minus_target": summary["learned_minus_target"],
        "learned_minus_l2": summary["learned_minus_l2"],
        "deranged_public_table_accuracy": summary["learned_metrics"]["deranged_public_table"]["accuracy"],
        "ci95_low_vs_best_control": summary["paired_bootstrap"]["learned_source_vs_best_control"]["ci95_low"],
        "ci95_low_vs_target": summary["paired_bootstrap"]["learned_source_vs_target"]["ci95_low"],
        "matched_weight": payload["matched_weight"],
        "control_weight": payload["control_weight"],
        "target_weight": payload["target_weight"],
        "deranged_weight": payload["deranged_weight"],
        "random_rounds": payload["random_rounds"],
    }


def summarize_runs(*, run_dirs: list[pathlib.Path], output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_read_run(path) for path in run_dirs]
    overlap_rows = [row for row in rows if not row["is_disjoint"]]
    disjoint_rows = [row for row in rows if row["is_disjoint"]]
    low_control_overlap = [
        row
        for row in overlap_rows
        if row["matched_weight"] >= 10.0 and row["control_weight"] <= 0.5
    ]
    headline = {
        "rows": len(rows),
        "pass_rows": sum(1 for row in rows if row["pass_gate"]),
        "overlap_rows": len(overlap_rows),
        "overlap_pass_rows": sum(1 for row in overlap_rows if row["pass_gate"]),
        "disjoint_rows": len(disjoint_rows),
        "disjoint_pass_rows": sum(1 for row in disjoint_rows if row["pass_gate"]),
        "low_control_overlap_rows": len(low_control_overlap),
        "low_control_overlap_pass_rows": sum(1 for row in low_control_overlap if row["pass_gate"]),
        "min_low_control_overlap_learned_accuracy": min(
            (row["learned_source_accuracy"] for row in low_control_overlap),
            default=None,
        ),
        "max_low_control_overlap_best_control_accuracy": max(
            (row["best_control_accuracy"] for row in low_control_overlap),
            default=None,
        ),
        "min_low_control_overlap_ci95_low_vs_best_control": min(
            (row["ci95_low_vs_best_control"] for row in low_control_overlap),
            default=None,
        ),
        "max_disjoint_learned_accuracy": max((row["learned_source_accuracy"] for row in disjoint_rows), default=None),
        "max_disjoint_l2_accuracy": max((row["l2_source_accuracy"] for row in disjoint_rows), default=None),
        "pass_gate": bool(low_control_overlap)
        and all(row["pass_gate"] for row in low_control_overlap)
        and bool(disjoint_rows)
        and all(not row["pass_gate"] for row in disjoint_rows),
    }
    payload = {
        "gate": "source_private_pq_control_regularized_receiver_summary",
        "headline": headline,
        "rows": rows,
        "pass_gate": headline["pass_gate"],
        "interpretation": (
            "The low-control learned PQ receiver preserves deterministic PQ on the established exact-ID overlap "
            "surface across remaps, while disjoint train/eval IDs collapse the underlying PQ signal. This is a "
            "bounded positive diagnostic for learned reception and a stronger blocker against using PQ as an "
            "ICLR headline until the packet source is disjoint-safe."
        ),
    }
    json_path = output_dir / "pq_control_regularized_receiver_summary.json"
    md_path = output_dir / "pq_control_regularized_receiver_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [
            "pq_control_regularized_receiver_summary.json",
            "pq_control_regularized_receiver_summary.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "pq_control_regularized_receiver_summary.json": _sha256_file(json_path),
            "pq_control_regularized_receiver_summary.md": _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
        "headline": headline,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# PQ Control-Regularized Receiver Summary Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- rows: `{headline['rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# PQ Control-Regularized Receiver Summary",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{headline['rows']}`",
        f"- overlap pass rows: `{headline['overlap_pass_rows']}/{headline['overlap_rows']}`",
        f"- disjoint pass rows: `{headline['disjoint_pass_rows']}/{headline['disjoint_rows']}`",
        f"- min low-control overlap learned accuracy: `{_fmt(headline['min_low_control_overlap_learned_accuracy'])}`",
        f"- max low-control overlap best control accuracy: `{_fmt(headline['max_low_control_overlap_best_control_accuracy'])}`",
        f"- max disjoint L2 accuracy: `{_fmt(headline['max_disjoint_l2_accuracy'])}`",
        "",
        "| Run | Disjoint | Remap | Pass | Learned | L2 | Target | Best control | Deranged | CI low vs control |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_dir']}` | `{row['is_disjoint']}` | {row['remap_slot_seed']} | `{row['pass_gate']}` | "
            f"{row['learned_source_accuracy']:.3f} | {row['l2_source_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_control_condition']} "
            f"{row['best_control_accuracy']:.3f} | {row['deranged_public_table_accuracy']:.3f} | "
            f"{row['ci95_low_vs_best_control']:.3f} |"
        )
    lines.extend(["", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = summarize_runs(run_dirs=args.run_dir, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "pass_gate": payload["pass_gate"], "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
