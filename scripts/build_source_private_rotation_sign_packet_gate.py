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
    rotation_metrics = metrics["rotation_sign_source"]
    rotation_control = _best_control(
        metrics,
        [
            "rotation_sign_constrained_shuffled_source",
            "rotation_sign_answer_masked_source",
            "rotation_sign_permuted_bits",
            "rotation_sign_random_same_byte",
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
    query_aware_oracle_bytes = 14.0
    rotation_accuracy = budget_row["rotation_sign_source_accuracy"]
    scalar_accuracy = budget_row["scalar_quantized_source_accuracy"]
    return {
        "result_dir": str(result_dir),
        "remap_slot_seed": remap_seed,
        "budget_bytes": budget_row["budget_bytes"],
        "n": result["eval_examples"],
        "target_accuracy": budget_row["target_only_accuracy"],
        "best_no_source_accuracy": budget_row["best_no_source_accuracy"],
        "rotation_sign_accuracy": rotation_accuracy,
        "raw_source_sign_accuracy": metrics["raw_source_sign_sketch"]["accuracy"],
        "scalar_wyner_ziv_accuracy": scalar_accuracy,
        "protected_rotated_residual_accuracy": budget_row["protected_rotated_residual_accuracy"],
        "qjl_residual_accuracy": budget_row["qjl_residual_source_accuracy"],
        "best_rotation_sign_control_accuracy": rotation_control,
        "best_scalar_control_accuracy": scalar_control,
        "rotation_sign_minus_best_no_source": rotation_accuracy - budget_row["best_no_source_accuracy"],
        "rotation_sign_minus_best_control": rotation_accuracy - rotation_control,
        "rotation_sign_minus_scalar": rotation_accuracy - scalar_accuracy,
        "rotation_sign_minus_raw_source_sign": rotation_accuracy - metrics["raw_source_sign_sketch"]["accuracy"],
        "rotation_sign_controls_ok": budget_row["rotation_sign_controls_ok"],
        "rotation_sign_pass": budget_row["rotation_sign_source_packet_pass"],
        "query_aware_text_at_budget_accuracy": 0.25,
        "packet_vs_query_aware_oracle_compression": query_aware_oracle_bytes / budget_row["budget_bytes"],
        "p50_decode_latency_ms": rotation_metrics["p50_latency_ms"],
        "p95_decode_latency_ms": rotation_metrics["p95_latency_ms"],
        "mean_payload_bytes": rotation_metrics["mean_payload_bytes"],
        "mean_payload_tokens": rotation_metrics["mean_payload_tokens"],
        "exact_id_parity": result["exact_id_parity"],
    }


def build_rotation_sign_gate(
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
            packet_variants=["qjl_residual", "protected_rotated_residual", "rotation_sign"],
        )
        run_dirs.append(str(run_dir))
        for budget_row in result["budget_summaries"]:
            rows.append(_row_from_result(result, result_dir=run_dir, remap_seed=remap_seed, budget_row=budget_row))

    pass_rows = [
        row
        for row in rows
        if row["rotation_sign_pass"]
        and row["rotation_sign_minus_best_control"] >= 0.15
        and row["p50_decode_latency_ms"] < 2.0
        and row["query_aware_text_at_budget_accuracy"] <= row["target_accuracy"] + 0.02
        and row["packet_vs_query_aware_oracle_compression"] > 1.0
    ]
    remaps_with_pass = sorted({row["remap_slot_seed"] for row in pass_rows})
    payload = {
        "gate": "source_private_rotation_sign_packet_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "rows": len(rows),
            "pass_rows": len(pass_rows),
            "remaps_with_pass": remaps_with_pass,
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "max_rotation_sign_accuracy": max((row["rotation_sign_accuracy"] for row in rows), default=None),
            "min_passing_rotation_sign_minus_control": min(
                (row["rotation_sign_minus_best_control"] for row in pass_rows),
                default=None,
            ),
            "max_passing_decode_latency_ms": max((row["p50_decode_latency_ms"] for row in pass_rows), default=None),
        },
        "pass_gate": len(remaps_with_pass) == len(remap_seeds),
        "pass_rule": (
            "At least one rotation-sign source packet row per remapped codebook must beat no-source by >=0.15, "
            "beat its strongest source-destroying control by >=0.15, keep p50 CPU decode latency under 2 ms, "
            "and beat query-aware text at the same byte budget."
        ),
        "interpretation": (
            "This gate isolates a compression-native packet: the source sends only signs of random projections of its private "
            "evidence vector, and the target decodes by Hamming distance against public candidate side information. It is a "
            "publishable systems/codec contribution only if the same-bit constrained-shuffle, answer-masked, permuted-bit, "
            "and random controls collapse to target accuracy while the matched packet remains useful."
        ),
    }
    (output_dir / "rotation_sign_packet_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "rotation_sign_packet_gate.md", payload)
    manifest = {
        "artifacts": ["rotation_sign_packet_gate.json", "rotation_sign_packet_gate.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "rotation_sign_packet_gate.json": _sha256_file(output_dir / "rotation_sign_packet_gate.json"),
            "rotation_sign_packet_gate.md": _sha256_file(output_dir / "rotation_sign_packet_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Rotation-Sign Packet Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
        "# Source-Private Rotation-Sign Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- remaps with pass: `{h['remaps_with_pass']}`",
        f"- remap seeds: `{h['remap_seeds']}`",
        f"- budgets: `{h['budgets']}`",
        f"- max rotation-sign accuracy: `{_fmt(h['max_rotation_sign_accuracy'])}`",
        f"- min passing rotation-control margin: `{_fmt(h['min_passing_rotation_sign_minus_control'])}`",
        f"- max passing p50 decode latency ms: `{_fmt(h['max_passing_decode_latency_ms'])}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | Rotation-sign | Raw sign | Scalar WZ | Protected | QJL | Target | Best rotation control | Rotation-control | Rotation-scalar | p50 ms | Rotation pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['rotation_sign_accuracy']:.3f} | {row['raw_source_sign_accuracy']:.3f} | "
            f"{row['scalar_wyner_ziv_accuracy']:.3f} | {_fmt(row['protected_rotated_residual_accuracy'])} | "
            f"{_fmt(row['qjl_residual_accuracy'])} | {row['target_accuracy']:.3f} | "
            f"{row['best_rotation_sign_control_accuracy']:.3f} | "
            f"{row['rotation_sign_minus_best_control']:.3f} | {row['rotation_sign_minus_scalar']:.3f} | "
            f"{row['p50_decode_latency_ms']:.3f} | `{row['rotation_sign_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_rotation_sign_packet_gate_20260430"))
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_rotation_sign_gate(
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
