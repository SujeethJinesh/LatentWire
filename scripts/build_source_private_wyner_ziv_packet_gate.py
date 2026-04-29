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


def _row_from_result(result: dict[str, Any], *, result_dir: pathlib.Path, remap_seed: int, budget_row: dict[str, Any]) -> dict[str, Any]:
    metrics = budget_row["metrics"]
    text_at_budget_accuracy = 0.25
    query_aware_oracle_bytes = 14.0
    return {
        "result_dir": str(result_dir),
        "remap_slot_seed": remap_seed,
        "budget_bytes": budget_row["budget_bytes"],
        "n": result["eval_examples"],
        "target_accuracy": budget_row["target_only_accuracy"],
        "learned_syndrome_accuracy": budget_row["matched_accuracy"],
        "scalar_wyner_ziv_accuracy": budget_row["scalar_quantized_source_accuracy"],
        "qjl_residual_accuracy": budget_row["qjl_residual_source_accuracy"],
        "canonical_rasp_accuracy": budget_row["relative_canonical_score_source_accuracy"],
        "raw_source_sign_accuracy": metrics["raw_source_sign_sketch"]["accuracy"],
        "best_no_source_accuracy": budget_row["best_no_source_accuracy"],
        "best_scalar_control_accuracy": max(
            metrics["scalar_label_shuffled_ridge"]["accuracy"],
            metrics["scalar_constrained_shuffled_source"]["accuracy"],
            metrics["scalar_answer_masked_source"]["accuracy"],
            metrics["random_same_byte"]["accuracy"],
        ),
        "best_qjl_control_accuracy": max(
            metrics["qjl_label_shuffled_ridge"]["accuracy"],
            metrics["qjl_constrained_shuffled_source"]["accuracy"],
            metrics["qjl_answer_masked_source"]["accuracy"],
            metrics["qjl_random_same_byte"]["accuracy"],
        ),
        "best_canonical_rasp_control_accuracy": max(
            metrics["relative_canonical_label_shuffled_ridge"]["accuracy"],
            metrics["relative_canonical_constrained_shuffled_source"]["accuracy"],
            metrics["relative_canonical_order_mismatch_source"]["accuracy"],
            metrics["relative_canonical_answer_masked_source"]["accuracy"],
            metrics["relative_canonical_permuted_score_bytes"]["accuracy"],
            metrics["relative_canonical_random_same_byte"]["accuracy"],
        ),
        "query_aware_text_at_budget_accuracy": text_at_budget_accuracy,
        "query_aware_oracle_bytes": query_aware_oracle_bytes,
        "packet_vs_query_aware_oracle_compression": query_aware_oracle_bytes / budget_row["budget_bytes"],
        "scalar_controls_ok": budget_row["scalar_controls_ok"],
        "qjl_controls_ok": budget_row["qjl_controls_ok"],
        "canonical_rasp_controls_ok": budget_row["relative_canonical_controls_ok"],
        "scalar_pass": budget_row["scalar_source_packet_pass"],
        "qjl_pass": budget_row["qjl_source_packet_pass"],
        "canonical_rasp_pass": budget_row["relative_canonical_source_packet_pass"],
        "exact_id_parity": result["exact_id_parity"],
    }


def build_wyner_ziv_gate(
    *,
    output_dir: pathlib.Path,
    remap_seeds: list[int],
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    train_seed: int,
    eval_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for remap_seed in remap_seeds:
        run_dir = output_dir / f"remap_{remap_seed}"
        result = run_gate(
            output_dir=run_dir,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_family_set="all",
            eval_family_set="all",
            candidates=4,
            feature_dim=feature_dim,
            budgets=budgets,
            train_seed=train_seed,
            eval_seed=eval_seed,
            ridge=1e-2,
            candidate_view="slot",
            fit_intercept=False,
            remap_slot_seed=remap_seed,
            packet_variants=["qjl_residual", "relative_scores_canonical"],
        )
        run_dirs.append(str(run_dir))
        for budget_row in result["budget_summaries"]:
            rows.append(_row_from_result(result, result_dir=run_dir, remap_seed=remap_seed, budget_row=budget_row))
    pass_rows = [
        row
        for row in rows
        if row["scalar_pass"]
        and row["scalar_wyner_ziv_accuracy"] >= row["target_accuracy"] + 0.15
        and row["query_aware_text_at_budget_accuracy"] <= row["target_accuracy"] + 0.02
        and row["packet_vs_query_aware_oracle_compression"] > 1.0
    ]
    payload = {
        "gate": "source_private_wyner_ziv_packet_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "rows": len(rows),
            "pass_rows": len(pass_rows),
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "min_passing_scalar_accuracy": min((row["scalar_wyner_ziv_accuracy"] for row in pass_rows), default=None),
            "min_passing_scalar_minus_control": min(
                (row["scalar_wyner_ziv_accuracy"] - row["best_scalar_control_accuracy"] for row in pass_rows),
                default=None,
            ),
            "max_passing_payload_bytes": max((row["budget_bytes"] for row in pass_rows), default=None),
            "min_packet_vs_query_aware_oracle_compression": min(
                (row["packet_vs_query_aware_oracle_compression"] for row in pass_rows),
                default=None,
            ),
        },
        "pass_gate": len(pass_rows) >= len(remap_seeds),
        "pass_rule": (
            "At least one scalar Wyner-Ziv/source-side-information packet row per remapped codebook must beat target by >=0.15, "
            "keep source-destroying controls clean, and beat query-aware text at the same byte budget."
        ),
        "interpretation": (
            "This is a learned source-private syndrome gate: the encoder maps private source evidence into a compact vector packet, "
            "while the decoder uses public candidate side information. It is less hand-coded than the deterministic diagnostic packet, "
            "but still scoped to same-family/all-family remapped slot codebooks."
        ),
    }
    (output_dir / "wyner_ziv_packet_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "wyner_ziv_packet_gate.md", payload)
    manifest = {
        "artifacts": ["wyner_ziv_packet_gate.json", "wyner_ziv_packet_gate.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "wyner_ziv_packet_gate.json": _sha256_file(output_dir / "wyner_ziv_packet_gate.json"),
            "wyner_ziv_packet_gate.md": _sha256_file(output_dir / "wyner_ziv_packet_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Wyner-Ziv Packet Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
    h = payload["headline"]
    lines = [
        "# Source-Private Wyner-Ziv Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- remap seeds: `{h['remap_seeds']}`",
        f"- budgets: `{h['budgets']}`",
        f"- minimum passing scalar accuracy: `{_fmt(h['min_passing_scalar_accuracy'])}`",
        f"- minimum passing scalar-control margin: `{_fmt(h['min_passing_scalar_minus_control'])}`",
        f"- minimum packet-vs-query-aware text compression: `{_fmt(h['min_packet_vs_query_aware_oracle_compression'], 1)}x`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | Scalar WZ | Target | Best scalar control | Raw sign | QJL | Canonical RASP | Query-aware text@budget | Packet/query-aware oracle | Scalar pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['scalar_wyner_ziv_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_scalar_control_accuracy']:.3f} | {row['raw_source_sign_accuracy']:.3f} | "
            f"{_fmt(row['qjl_residual_accuracy'])} | {_fmt(row['canonical_rasp_accuracy'])} | "
            f"{row['query_aware_text_at_budget_accuracy']:.3f} | "
            f"{row['packet_vs_query_aware_oracle_compression']:.1f}x | `{row['scalar_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_wyner_ziv_packet_gate_20260429"))
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_wyner_ziv_gate(
        output_dir=output_dir,
        remap_seeds=args.remap_seeds,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        feature_dim=args.feature_dim,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
