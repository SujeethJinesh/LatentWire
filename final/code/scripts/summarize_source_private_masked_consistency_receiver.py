from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIRS = (
    pathlib.Path("results/source_private_masked_consistency_receiver_smoke_20260430/n64_seed29_30_budget6"),
    pathlib.Path("results/source_private_masked_consistency_receiver_smoke_20260430/n256_seed29_30_budget6"),
    pathlib.Path("results/source_private_masked_consistency_receiver_smoke_20260430/n256_seed31_32_budget6"),
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _run_row(run_dir: pathlib.Path) -> dict[str, Any]:
    run_dir = _resolve(run_dir)
    run_summary = _read_json(run_dir / "run_summary.json")
    summary = run_summary["summary"]
    return {
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "n": summary["n"],
        "pass_gate": summary["pass_gate"],
        "source_packet_pass": summary["source_packet_pass"],
        "learned_matched_accuracy": summary["learned_matched_accuracy"],
        "hamming_matched_accuracy": summary["hamming_matched_accuracy"],
        "target_only_accuracy": summary["target_only_accuracy"],
        "best_control_condition": summary["best_control_condition"],
        "best_control_accuracy": summary["best_control_accuracy"],
        "learned_minus_target": summary["learned_minus_target"],
        "learned_minus_best_control": summary["learned_minus_best_control"],
        "learned_minus_hamming": summary["learned_minus_hamming"],
        "ci95_low_vs_target": summary["paired_bootstrap"]["learned_matched_vs_target"]["ci95_low"],
        "ci95_low_vs_best_control": summary["paired_bootstrap"]["learned_matched_vs_best_control"]["ci95_low"],
        "exact_id_parity": summary["exact_id_parity"],
        "budget_bytes": run_summary["budget_bytes"],
        "feature_dim": run_summary["feature_dim"],
        "train_seed": run_summary["train_seed"],
        "eval_seed": run_summary["eval_seed"],
    }


def summarize_runs(*, run_dirs: list[pathlib.Path], output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_run_row(path) for path in run_dirs]
    n256_rows = [row for row in rows if int(row["n"]) == 256]
    headline = {
        "pass_gate": all(row["pass_gate"] for row in rows),
        "runs": len(rows),
        "n256_runs": len(n256_rows),
        "min_learned_matched_accuracy": min(row["learned_matched_accuracy"] for row in rows),
        "min_n256_learned_matched_accuracy": min(row["learned_matched_accuracy"] for row in n256_rows),
        "min_n256_lift_vs_target": min(row["learned_minus_target"] for row in n256_rows),
        "min_n256_lift_vs_best_control": min(row["learned_minus_best_control"] for row in n256_rows),
        "min_n256_ci95_low_vs_target": min(row["ci95_low_vs_target"] for row in n256_rows),
        "min_n256_ci95_low_vs_best_control": min(row["ci95_low_vs_best_control"] for row in n256_rows),
        "min_n256_learned_minus_hamming": min(row["learned_minus_hamming"] for row in n256_rows),
        "max_n256_learned_minus_hamming": max(row["learned_minus_hamming"] for row in n256_rows),
        "all_exact_id_parity": all(row["exact_id_parity"] for row in rows),
        "all_source_packet_pass": all(row["source_packet_pass"] for row in rows),
    }
    payload = {
        "gate": "source_private_masked_consistency_receiver_summary",
        "headline": headline,
        "rows": rows,
        "interpretation": (
            "A one-step learned masked-consistency receiver over 6-byte learned syndrome packets preserves most "
            "deterministic packet utility while suppressing destructive-control leakage. It is not yet a fully "
            "table-free semantic receiver because it uses public candidate/code features and is compared against "
            "deterministic Hamming decoding."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Source-Private Masked Consistency Receiver Summary",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- runs: `{headline['runs']}`",
        f"- n256 runs: `{headline['n256_runs']}`",
        f"- min n256 learned matched accuracy: `{headline['min_n256_learned_matched_accuracy']:.3f}`",
        f"- min n256 lift vs target: `{headline['min_n256_lift_vs_target']:.3f}`",
        f"- min n256 lift vs best control: `{headline['min_n256_lift_vs_best_control']:.3f}`",
        f"- min n256 CI95 low vs target: `{headline['min_n256_ci95_low_vs_target']:.3f}`",
        f"- min n256 CI95 low vs best control: `{headline['min_n256_ci95_low_vs_best_control']:.3f}`",
        f"- n256 learned-minus-Hamming range: `{headline['min_n256_learned_minus_hamming']:.3f}` to "
        f"`{headline['max_n256_learned_minus_hamming']:.3f}`",
        "",
        "| Run | n | pass | learned | Hamming | target | best control | lift vs control | CI low vs control | learned-Hamming |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_dir']}` | {row['n']} | `{row['pass_gate']}` | "
            f"{row['learned_matched_accuracy']:.3f} | {row['hamming_matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | `{row['best_control_condition']}` {row['best_control_accuracy']:.3f} | "
            f"{row['learned_minus_best_control']:.3f} | {row['ci95_low_vs_best_control']:.3f} | "
            f"{row['learned_minus_hamming']:.3f} |"
        )
    lines.extend(["", payload["interpretation"], ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "gate": payload["gate"],
                "headline": headline,
                "pass_gate": headline["pass_gate"],
                "artifacts": ["summary.json", "summary.md", "manifest.json", "manifest.md"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Masked Consistency Receiver Summary Manifest",
                "",
                f"- pass gate: `{headline['pass_gate']}`",
                f"- n256 runs: `{headline['n256_runs']}`",
                f"- min n256 learned matched accuracy: `{headline['min_n256_learned_matched_accuracy']:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_masked_consistency_receiver_smoke_20260430/summary"),
    )
    args = parser.parse_args()
    payload = summarize_runs(run_dirs=args.run_dir or list(DEFAULT_RUN_DIRS), output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
