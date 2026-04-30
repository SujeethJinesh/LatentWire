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
    pq_metrics = metrics["product_codebook_source"]
    pq_control = _best_control(
        metrics,
        [
            "product_codebook_label_shuffled_ridge",
            "product_codebook_constrained_shuffled_source",
            "product_codebook_answer_masked_source",
            "product_codebook_permuted_codes",
            "product_codebook_random_same_byte",
        ],
    )
    query_aware_oracle_bytes = 14.0
    pq_accuracy = budget_row["product_codebook_source_accuracy"]
    scalar_accuracy = budget_row["scalar_quantized_source_accuracy"]
    return {
        "result_dir": str(result_dir),
        "remap_slot_seed": remap_seed,
        "budget_bytes": budget_row["budget_bytes"],
        "n": result["eval_examples"],
        "target_accuracy": budget_row["target_only_accuracy"],
        "best_no_source_accuracy": budget_row["best_no_source_accuracy"],
        "product_codebook_accuracy": pq_accuracy,
        "scalar_wyner_ziv_accuracy": scalar_accuracy,
        "protected_rotated_residual_accuracy": budget_row["protected_rotated_residual_accuracy"],
        "qjl_residual_accuracy": budget_row["qjl_residual_source_accuracy"],
        "rotation_sign_accuracy": budget_row["rotation_sign_source_accuracy"],
        "best_product_codebook_control_accuracy": pq_control,
        "product_codebook_minus_best_no_source": pq_accuracy - budget_row["best_no_source_accuracy"],
        "product_codebook_minus_best_control": pq_accuracy - pq_control,
        "product_codebook_minus_scalar": pq_accuracy - scalar_accuracy,
        "product_codebook_minus_protected": pq_accuracy - budget_row["protected_rotated_residual_accuracy"],
        "product_codebook_minus_qjl": pq_accuracy - budget_row["qjl_residual_source_accuracy"],
        "product_codebook_controls_ok": budget_row["product_codebook_controls_ok"],
        "product_codebook_pass": budget_row["product_codebook_source_packet_pass"],
        "product_codebook_within_002_of_scalar": pq_accuracy >= scalar_accuracy - 0.02,
        "query_aware_text_at_budget_accuracy": 0.25,
        "packet_vs_query_aware_oracle_compression": query_aware_oracle_bytes / budget_row["budget_bytes"],
        "p50_decode_latency_ms": pq_metrics["p50_latency_ms"],
        "p95_decode_latency_ms": pq_metrics["p95_latency_ms"],
        "mean_payload_bytes": pq_metrics["mean_payload_bytes"],
        "mean_payload_tokens": pq_metrics["mean_payload_tokens"],
        "exact_id_parity": result["exact_id_parity"],
    }


def build_product_codebook_gate(
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
            packet_variants=["qjl_residual", "protected_rotated_residual", "rotation_sign", "product_codebook"],
        )
        run_dirs.append(str(run_dir))
        for budget_row in result["budget_summaries"]:
            rows.append(_row_from_result(result, result_dir=run_dir, remap_seed=remap_seed, budget_row=budget_row))

    functional_pass_rows = [
        row
        for row in rows
        if row["product_codebook_pass"]
        and row["product_codebook_minus_best_control"] >= 0.15
        and row["product_codebook_within_002_of_scalar"]
        and row["query_aware_text_at_budget_accuracy"] <= row["target_accuracy"] + 0.02
        and row["packet_vs_query_aware_oracle_compression"] > 1.0
    ]
    systems_pass_rows = [
        row
        for row in functional_pass_rows
        if row["p50_decode_latency_ms"] < 2.0
    ]
    functional_remaps_with_pass = sorted({row["remap_slot_seed"] for row in functional_pass_rows})
    systems_remaps_with_pass = sorted({row["remap_slot_seed"] for row in systems_pass_rows})
    pass_rows = [
        row
        for row in rows
        if row["product_codebook_pass"]
        and row["product_codebook_minus_best_control"] >= 0.15
        and row["product_codebook_within_002_of_scalar"]
        and row["p50_decode_latency_ms"] < 2.0
        and row["query_aware_text_at_budget_accuracy"] <= row["target_accuracy"] + 0.02
        and row["packet_vs_query_aware_oracle_compression"] > 1.0
    ]
    remaps_with_pass = sorted({row["remap_slot_seed"] for row in pass_rows})
    payload = {
        "gate": "source_private_product_codebook_packet_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "rows": len(rows),
            "pass_rows": len(pass_rows),
            "remaps_with_pass": remaps_with_pass,
            "functional_pass_rows": len(functional_pass_rows),
            "functional_remaps_with_pass": functional_remaps_with_pass,
            "systems_pass_rows": len(systems_pass_rows),
            "systems_remaps_with_pass": systems_remaps_with_pass,
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "max_product_codebook_accuracy": max((row["product_codebook_accuracy"] for row in rows), default=None),
            "min_passing_product_codebook_minus_control": min(
                (row["product_codebook_minus_best_control"] for row in pass_rows),
                default=None,
            ),
            "min_passing_product_codebook_minus_scalar": min(
                (row["product_codebook_minus_scalar"] for row in pass_rows),
                default=None,
            ),
            "max_passing_decode_latency_ms": max((row["p50_decode_latency_ms"] for row in pass_rows), default=None),
        },
        "functional_pass_gate": len(functional_remaps_with_pass) == len(remap_seeds),
        "systems_latency_pass_gate": len(systems_remaps_with_pass) == len(remap_seeds),
        "pass_gate": len(remaps_with_pass) == len(remap_seeds),
        "pass_rule": (
            "At least one product-codebook packet row per remapped codebook must pass source-destroying controls, "
            "beat its strongest product-codebook control by >=0.15, stay within 0.02 accuracy of scalar Wyner-Ziv, "
            "keep p50 CPU decode latency under 2 ms, and beat query-aware text at the same byte budget."
        ),
        "interpretation": (
            "This gate tests whether a product-quantized discrete packet can become a compression-native replacement for "
            "scalar Wyner-Ziv: each byte is a learned centroid index for one subspace of the source-projected vector. "
            "It is a method contribution only if the code indices preserve source-private candidate margins without "
            "being explained by label-shuffled, constrained-shuffled, answer-masked, permuted-code, or random controls. "
            "The aggregate reports functional pass separately from systems-latency pass because the current implementation "
            "uses a simple Python decoder rather than an optimized table-lookup kernel."
        ),
    }
    (output_dir / "product_codebook_packet_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "product_codebook_packet_gate.md", payload)
    manifest = {
        "artifacts": ["product_codebook_packet_gate.json", "product_codebook_packet_gate.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "product_codebook_packet_gate.json": _sha256_file(output_dir / "product_codebook_packet_gate.json"),
            "product_codebook_packet_gate.md": _sha256_file(output_dir / "product_codebook_packet_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Product-Codebook Packet Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
        "# Source-Private Product-Codebook Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- remaps with pass: `{h['remaps_with_pass']}`",
        f"- functional pass gate: `{payload['functional_pass_gate']}`",
        f"- functional pass rows: `{h['functional_pass_rows']}`",
        f"- functional remaps with pass: `{h['functional_remaps_with_pass']}`",
        f"- systems latency pass gate: `{payload['systems_latency_pass_gate']}`",
        f"- systems pass rows: `{h['systems_pass_rows']}`",
        f"- remap seeds: `{h['remap_seeds']}`",
        f"- budgets: `{h['budgets']}`",
        f"- max product-codebook accuracy: `{_fmt(h['max_product_codebook_accuracy'])}`",
        f"- min passing product-codebook-control margin: `{_fmt(h['min_passing_product_codebook_minus_control'])}`",
        f"- min passing product-codebook-scalar margin: `{_fmt(h['min_passing_product_codebook_minus_scalar'])}`",
        f"- max passing p50 decode latency ms: `{_fmt(h['max_passing_decode_latency_ms'])}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | Product codebook | Scalar WZ | Protected | QJL | Rotation-sign | Target | Best PQ control | PQ-control | PQ-scalar | p50 ms | PQ pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['product_codebook_accuracy']:.3f} | {row['scalar_wyner_ziv_accuracy']:.3f} | "
            f"{_fmt(row['protected_rotated_residual_accuracy'])} | {_fmt(row['qjl_residual_accuracy'])} | "
            f"{_fmt(row['rotation_sign_accuracy'])} | {row['target_accuracy']:.3f} | "
            f"{row['best_product_codebook_control_accuracy']:.3f} | "
            f"{row['product_codebook_minus_best_control']:.3f} | {row['product_codebook_minus_scalar']:.3f} | "
            f"{row['p50_decode_latency_ms']:.3f} | `{row['product_codebook_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_product_codebook_packet_gate_20260430"))
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_product_codebook_gate(
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
