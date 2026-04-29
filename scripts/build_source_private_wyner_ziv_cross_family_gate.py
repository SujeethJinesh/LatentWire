from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_tool_trace_compression_baselines import run_gate  # noqa: E402


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_from_budget(
    result: dict[str, Any],
    *,
    result_dir: pathlib.Path,
    direction: str,
    budget_row: dict[str, Any],
) -> dict[str, Any]:
    metrics = budget_row["metrics"]
    return {
        "direction": direction,
        "result_dir": str(result_dir),
        "budget_bytes": budget_row["budget_bytes"],
        "n": result["eval_examples"],
        "train_family_set": result["train_family_set"],
        "eval_family_set": result["eval_family_set"],
        "target_accuracy": budget_row["target_only_accuracy"],
        "best_no_source_accuracy": budget_row["best_no_source_accuracy"],
        "learned_syndrome_accuracy": budget_row["matched_accuracy"],
        "scalar_wyner_ziv_accuracy": budget_row["scalar_quantized_source_accuracy"],
        "qjl_residual_accuracy": budget_row["qjl_residual_source_accuracy"],
        "canonical_rasp_accuracy": budget_row["relative_canonical_score_source_accuracy"],
        "raw_source_sign_accuracy": metrics["raw_source_sign_sketch"]["accuracy"],
        "best_scalar_control_accuracy": max(
            metrics["scalar_label_shuffled_ridge"]["accuracy"],
            metrics["scalar_constrained_shuffled_source"]["accuracy"],
            metrics["scalar_answer_masked_source"]["accuracy"],
            metrics["random_same_byte"]["accuracy"],
        ),
        "best_canonical_rasp_control_accuracy": max(
            metrics["relative_canonical_label_shuffled_ridge"]["accuracy"],
            metrics["relative_canonical_constrained_shuffled_source"]["accuracy"],
            metrics["relative_canonical_order_mismatch_source"]["accuracy"],
            metrics["relative_canonical_answer_masked_source"]["accuracy"],
            metrics["relative_canonical_permuted_score_bytes"]["accuracy"],
            metrics["relative_canonical_random_same_byte"]["accuracy"],
        ),
        "query_aware_text_at_budget_accuracy": 0.25,
        "query_aware_oracle_bytes": 14.0,
        "packet_vs_query_aware_oracle_compression": 14.0 / budget_row["budget_bytes"],
        "scalar_controls_ok": budget_row["scalar_controls_ok"],
        "canonical_rasp_controls_ok": budget_row["relative_canonical_controls_ok"],
        "scalar_pass": budget_row["scalar_source_packet_pass"],
        "canonical_rasp_pass": budget_row["relative_canonical_source_packet_pass"],
        "exact_id_parity": result["exact_id_parity"],
    }


def _run_or_load(
    *,
    run_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_family_set: str,
    eval_family_set: str,
    budgets: list[int],
    feature_dim: int,
    train_seed: int,
    eval_seed: int,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and all((run_dir / f"predictions_budget{budget}.jsonl").exists() for budget in budgets):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_gate(
        output_dir=run_dir,
        train_examples=train_examples,
        eval_examples=eval_examples,
        train_family_set=train_family_set,
        eval_family_set=eval_family_set,
        candidates=4,
        feature_dim=feature_dim,
        budgets=budgets,
        train_seed=train_seed,
        eval_seed=eval_seed,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        packet_variants=["qjl_residual", "relative_scores_canonical"],
    )


def build_cross_family_gate(
    *,
    output_dir: pathlib.Path,
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("core_to_holdout", "core", "holdout", seed, seed + 1),
        ("holdout_to_core", "holdout", "core", seed + 1, seed),
    ]
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for direction, train_family, eval_family, train_seed, eval_seed in specs:
        run_dir = output_dir / direction
        result = _run_or_load(
            run_dir=run_dir,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_family_set=train_family,
            eval_family_set=eval_family,
            feature_dim=feature_dim,
            budgets=budgets,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )
        run_dirs.append(str(run_dir))
        for budget_row in result["budget_summaries"]:
            rows.append(_row_from_budget(result, result_dir=run_dir, direction=direction, budget_row=budget_row))
    directions = sorted({row["direction"] for row in rows})
    direction_pass = {
        direction: all(
            row["scalar_pass"]
            and row["scalar_wyner_ziv_accuracy"] >= row["target_accuracy"] + 0.15
            and row["scalar_wyner_ziv_accuracy"] - row["best_scalar_control_accuracy"] >= 0.15
            for row in rows
            if row["direction"] == direction
        )
        for direction in directions
    }
    payload = {
        "gate": "source_private_wyner_ziv_cross_family_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "directions": directions,
            "direction_pass": direction_pass,
            "pass_directions": sum(1 for ok in direction_pass.values() if ok),
            "budgets": budgets,
            "min_scalar_accuracy": min(row["scalar_wyner_ziv_accuracy"] for row in rows),
            "max_scalar_accuracy": max(row["scalar_wyner_ziv_accuracy"] for row in rows),
            "min_scalar_minus_control": min(row["scalar_wyner_ziv_accuracy"] - row["best_scalar_control_accuracy"] for row in rows),
            "max_payload_bytes": max(row["budget_bytes"] for row in rows),
        },
        "pass_gate": all(direction_pass.values()),
        "pass_rule": (
            "All 2/4/6-byte rows in both core->holdout and holdout->core must beat target by >=0.15 and best "
            "source-destroying scalar control by >=0.15. Otherwise cross-family remains failed/asymmetric."
        ),
    }
    (output_dir / "wyner_ziv_cross_family_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "wyner_ziv_cross_family_gate.md", payload)
    manifest = {
        "artifacts": [
            "wyner_ziv_cross_family_gate.json",
            "wyner_ziv_cross_family_gate.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "wyner_ziv_cross_family_gate.json": _sha256_file(output_dir / "wyner_ziv_cross_family_gate.json"),
            "wyner_ziv_cross_family_gate.md": _sha256_file(output_dir / "wyner_ziv_cross_family_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Wyner-Ziv Cross-Family Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Wyner-Ziv Cross-Family Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- direction pass: `{h['direction_pass']}`",
        f"- budgets: `{h['budgets']}`",
        f"- scalar accuracy range: `{h['min_scalar_accuracy']:.3f}-{h['max_scalar_accuracy']:.3f}`",
        f"- minimum scalar-control margin: `{h['min_scalar_minus_control']:.3f}`",
        "",
        "## Rows",
        "",
        "| Direction | Budget | N | Scalar WZ | Target | Best scalar control | Raw sign | QJL | Canonical RASP | Scalar pass | Canonical pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['direction']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['scalar_wyner_ziv_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_scalar_control_accuracy']:.3f} | {row['raw_source_sign_accuracy']:.3f} | "
            f"{_fmt(row['qjl_residual_accuracy'])} | {_fmt(row['canonical_rasp_accuracy'])} | "
            f"`{row['scalar_pass']}` | `{row['canonical_rasp_pass']}` |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_wyner_ziv_cross_family_gate_20260429"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_cross_family_gate(
        output_dir=output_dir,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
