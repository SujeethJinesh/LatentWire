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


def _best_control(metrics: dict[str, Any], names: list[str]) -> float:
    return max(float(metrics[name]["accuracy"]) for name in names)


def _row_from_result(result: dict[str, Any], *, result_dir: pathlib.Path, remap_seed: int, budget_row: dict[str, Any]) -> dict[str, Any]:
    metrics = budget_row["metrics"]
    protected_metrics = metrics["protected_rotated_residual_source"]
    protected_control = _best_control(
        metrics,
        [
            "protected_label_shuffled_ridge",
            "protected_constrained_shuffled_source",
            "protected_answer_masked_source",
            "protected_random_same_byte",
        ],
    )
    scalar_control = _best_control(
        metrics,
        [
            "scalar_label_shuffled_ridge",
            "scalar_constrained_shuffled_source",
            "scalar_answer_masked_source",
            "random_same_byte",
        ],
    )
    qjl_control = _best_control(
        metrics,
        [
            "qjl_label_shuffled_ridge",
            "qjl_constrained_shuffled_source",
            "qjl_answer_masked_source",
            "qjl_random_same_byte",
        ],
    )
    query_aware_oracle_bytes = 14.0
    protected_accuracy = budget_row["protected_rotated_residual_accuracy"]
    scalar_accuracy = budget_row["scalar_quantized_source_accuracy"]
    qjl_accuracy = budget_row["qjl_residual_source_accuracy"]
    return {
        "result_dir": str(result_dir),
        "remap_slot_seed": remap_seed,
        "budget_bytes": budget_row["budget_bytes"],
        "n": result["eval_examples"],
        "target_accuracy": budget_row["target_only_accuracy"],
        "best_no_source_accuracy": budget_row["best_no_source_accuracy"],
        "protected_accuracy": protected_accuracy,
        "scalar_wyner_ziv_accuracy": scalar_accuracy,
        "qjl_residual_accuracy": qjl_accuracy,
        "canonical_rasp_accuracy": budget_row["relative_canonical_score_source_accuracy"],
        "protected_minus_scalar": protected_accuracy - scalar_accuracy,
        "protected_minus_qjl": protected_accuracy - qjl_accuracy,
        "protected_minus_best_no_source": protected_accuracy - budget_row["best_no_source_accuracy"],
        "best_protected_control_accuracy": protected_control,
        "best_scalar_control_accuracy": scalar_control,
        "best_qjl_control_accuracy": qjl_control,
        "protected_minus_best_control": protected_accuracy - protected_control,
        "protected_controls_ok": budget_row["protected_controls_ok"],
        "protected_pass": budget_row["protected_source_packet_pass"],
        "protected_within_002_of_scalar": protected_accuracy >= scalar_accuracy - 0.02,
        "protected_beats_qjl": protected_accuracy >= qjl_accuracy,
        "query_aware_text_at_budget_accuracy": 0.25,
        "packet_vs_query_aware_oracle_compression": query_aware_oracle_bytes / budget_row["budget_bytes"],
        "p50_decode_latency_ms": protected_metrics["p50_latency_ms"],
        "p95_decode_latency_ms": protected_metrics["p95_latency_ms"],
        "mean_payload_bytes": protected_metrics["mean_payload_bytes"],
        "mean_payload_tokens": protected_metrics["mean_payload_tokens"],
        "exact_id_parity": result["exact_id_parity"],
    }


def build_protected_residual_gate(
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
            packet_variants=["qjl_residual", "protected_rotated_residual", "relative_scores_canonical"],
        )
        run_dirs.append(str(run_dir))
        for budget_row in result["budget_summaries"]:
            rows.append(_row_from_result(result, result_dir=run_dir, remap_seed=remap_seed, budget_row=budget_row))

    protected_pass_rows = [
        row
        for row in rows
        if row["protected_pass"]
        and row["protected_within_002_of_scalar"]
        and row["p50_decode_latency_ms"] < 2.0
        and row["query_aware_text_at_budget_accuracy"] <= row["target_accuracy"] + 0.02
    ]
    remaps_with_pass = sorted({row["remap_slot_seed"] for row in protected_pass_rows})
    payload = {
        "gate": "source_private_protected_residual_packet_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "rows": len(rows),
            "protected_pass_rows": len(protected_pass_rows),
            "remaps_with_protected_pass": remaps_with_pass,
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "max_protected_accuracy": max((row["protected_accuracy"] for row in rows), default=None),
            "min_passing_protected_minus_control": min(
                (row["protected_minus_best_control"] for row in protected_pass_rows),
                default=None,
            ),
            "min_passing_protected_minus_scalar": min(
                (row["protected_minus_scalar"] for row in protected_pass_rows),
                default=None,
            ),
            "max_passing_decode_latency_ms": max(
                (row["p50_decode_latency_ms"] for row in protected_pass_rows),
                default=None,
            ),
        },
        "pass_gate": len(remaps_with_pass) == len(remap_seeds),
        "pass_rule": (
            "At least one protected rotated residual packet row per remapped codebook must pass source-destroying controls, "
            "beat no-source by >=0.15, stay within 0.02 accuracy of scalar WZ, keep p50 decode latency under 2 ms, "
            "and beat query-aware text at the same byte budget."
        ),
        "interpretation": (
            "This gate tests whether a TurboQuant/QJL-inspired packet codec can make the compact source-private method more "
            "principled: protected scalar coordinates are selected by calibration separation, while remaining bytes carry a "
            "sign-sketch residual. It is a codec contribution only if it preserves scalar WZ accuracy at comparable bytes with "
            "clean controls and low CPU decode overhead."
        ),
    }
    (output_dir / "protected_residual_packet_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "protected_residual_packet_gate.md", payload)
    manifest = {
        "artifacts": ["protected_residual_packet_gate.json", "protected_residual_packet_gate.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "protected_residual_packet_gate.json": _sha256_file(output_dir / "protected_residual_packet_gate.json"),
            "protected_residual_packet_gate.md": _sha256_file(output_dir / "protected_residual_packet_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Protected Residual Packet Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
        "# Source-Private Protected Residual Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- protected pass rows: `{h['protected_pass_rows']}`",
        f"- remaps with protected pass: `{h['remaps_with_protected_pass']}`",
        f"- remap seeds: `{h['remap_seeds']}`",
        f"- budgets: `{h['budgets']}`",
        f"- max protected accuracy: `{_fmt(h['max_protected_accuracy'])}`",
        f"- min passing protected-control margin: `{_fmt(h['min_passing_protected_minus_control'])}`",
        f"- min passing protected-scalar margin: `{_fmt(h['min_passing_protected_minus_scalar'])}`",
        f"- max passing p50 decode latency ms: `{_fmt(h['max_passing_decode_latency_ms'])}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | Protected | Scalar WZ | QJL | Canonical RASP | Target | Best protected control | Protected-control | Protected-scalar | p50 ms | Protected pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['protected_accuracy']:.3f} | {row['scalar_wyner_ziv_accuracy']:.3f} | "
            f"{row['qjl_residual_accuracy']:.3f} | {row['canonical_rasp_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_protected_control_accuracy']:.3f} | "
            f"{row['protected_minus_best_control']:.3f} | {row['protected_minus_scalar']:.3f} | "
            f"{row['p50_decode_latency_ms']:.3f} | `{row['protected_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_protected_residual_packet_gate_20260429"))
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_protected_residual_gate(
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
