from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRODUCT_CONDITION = "product_codebook_source"
CONTROL_CONDITIONS = [
    "product_codebook_label_shuffled_ridge",
    "product_codebook_constrained_shuffled_source",
    "product_codebook_answer_masked_source",
    "product_codebook_permuted_codes",
    "product_codebook_random_same_byte",
]
REFERENCE_CONDITIONS = [
    "target_only",
    "scalar_quantized_source",
    "qjl_residual_source",
    "protected_rotated_residual_source",
    "rotation_sign_source",
]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return float(ordered[index])


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    return {
        "mean": sum(values) / n,
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rows_by_example(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["example_id"], {})[row["condition"]] = row
    return grouped


def _paired_values(
    grouped: dict[str, dict[str, dict[str, Any]]],
    *,
    method: str,
    baseline: str,
) -> list[float]:
    values: list[float] = []
    for example_id in sorted(grouped):
        conditions = grouped[example_id]
        if method not in conditions or baseline not in conditions:
            continue
        values.append(float(bool(conditions[method]["correct"])) - float(bool(conditions[baseline]["correct"])))
    return values


def _accuracy(rows: list[dict[str, Any]], condition: str) -> float:
    condition_rows = [row for row in rows if row["condition"] == condition]
    if not condition_rows:
        return 0.0
    return sum(1 for row in condition_rows if row["correct"]) / len(condition_rows)


def _paired_counts(values: list[float]) -> dict[str, int]:
    return {
        "wins": sum(1 for value in values if value > 0),
        "losses": sum(1 for value in values if value < 0),
        "ties": sum(1 for value in values if value == 0),
    }


def _summarize_file(
    path: pathlib.Path,
    *,
    remap_seed: int,
    budget: int,
    prior_row: dict[str, Any] | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    grouped = _rows_by_example(rows)
    exact_id_parity = all(
        len({row["example_id"] for row in rows if row["condition"] == condition}) == len(grouped)
        for condition in [PRODUCT_CONDITION, *CONTROL_CONDITIONS, *REFERENCE_CONDITIONS]
        if any(row["condition"] == condition for row in rows)
    )
    control_accuracies = {condition: _accuracy(rows, condition) for condition in CONTROL_CONDITIONS}
    best_control = max(control_accuracies, key=control_accuracies.get)
    comparisons: dict[str, Any] = {}
    for baseline in [*REFERENCE_CONDITIONS, *CONTROL_CONDITIONS, best_control]:
        if baseline in comparisons:
            continue
        values = _paired_values(grouped, method=PRODUCT_CONDITION, baseline=baseline)
        comparisons[baseline] = {
            "accuracy": _accuracy(rows, baseline),
            "paired_delta": _bootstrap_ci(values, samples=bootstrap_samples, seed=seed + len(comparisons) * 1009),
            "paired_counts": _paired_counts(values),
        }
    product_accuracy = _accuracy(rows, PRODUCT_CONDITION)
    best_control_ci = comparisons[best_control]["paired_delta"]
    target_ci = comparisons["target_only"]["paired_delta"]
    row_pass = (
        bool(prior_row.get("product_codebook_pass", True) if prior_row else True)
        and exact_id_parity
        and best_control_ci["ci95_low"] > 0.10
        and target_ci["ci95_low"] > 0.15
        and product_accuracy >= comparisons["scalar_quantized_source"]["accuracy"] - 0.02
    )
    return {
        "remap_slot_seed": remap_seed,
        "budget_bytes": budget,
        "path": str(path),
        "n": len(grouped),
        "exact_id_parity": exact_id_parity,
        "prior_gate_functional_pass": prior_row.get("product_codebook_pass") if prior_row else None,
        "product_codebook_accuracy": product_accuracy,
        "best_product_codebook_control": best_control,
        "best_product_codebook_control_accuracy": control_accuracies[best_control],
        "target_accuracy": comparisons["target_only"]["accuracy"],
        "scalar_wyner_ziv_accuracy": comparisons["scalar_quantized_source"]["accuracy"],
        "product_minus_target": product_accuracy - comparisons["target_only"]["accuracy"],
        "product_minus_best_control": product_accuracy - control_accuracies[best_control],
        "product_minus_scalar": product_accuracy - comparisons["scalar_quantized_source"]["accuracy"],
        "paired_vs_target_ci95_low": target_ci["ci95_low"],
        "paired_vs_best_control_ci95_low": best_control_ci["ci95_low"],
        "paired_vs_scalar_ci95_low": comparisons["scalar_quantized_source"]["paired_delta"]["ci95_low"],
        "paired_comparisons": comparisons,
        "uncertainty_pass": row_pass,
    }


def summarize_uncertainty(
    *,
    product_gate_dir: pathlib.Path,
    product_gate_json: pathlib.Path | None,
    output_dir: pathlib.Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_rows: dict[tuple[int, int], dict[str, Any]] = {}
    if product_gate_json and product_gate_json.exists():
        gate_payload = json.loads(product_gate_json.read_text(encoding="utf-8"))
        gate_rows = {
            (int(row["remap_slot_seed"]), int(row["budget_bytes"])): row
            for row in gate_payload.get("rows", [])
        }
    rows: list[dict[str, Any]] = []
    for path in sorted(product_gate_dir.glob("remap_*/predictions_budget*.jsonl")):
        remap_seed = int(path.parent.name.removeprefix("remap_"))
        budget = int(path.stem.removeprefix("predictions_budget"))
        prior_row = gate_rows.get((remap_seed, budget))
        rows.append(
            _summarize_file(
                path,
                remap_seed=remap_seed,
                budget=budget,
                prior_row=prior_row,
                bootstrap_samples=bootstrap_samples,
                seed=seed + remap_seed * 31 + budget * 1009,
            )
        )
    pass_remaps = sorted({row["remap_slot_seed"] for row in rows if row["uncertainty_pass"]})
    payload = {
        "gate": "source_private_product_codebook_uncertainty",
        "rows": rows,
        "headline": {
            "rows": len(rows),
            "pass_rows": sum(1 for row in rows if row["uncertainty_pass"]),
            "remaps_with_pass": pass_remaps,
            "min_passing_ci95_low_vs_target": min(
                (row["paired_vs_target_ci95_low"] for row in rows if row["uncertainty_pass"]),
                default=None,
            ),
            "min_passing_ci95_low_vs_best_control": min(
                (row["paired_vs_best_control_ci95_low"] for row in rows if row["uncertainty_pass"]),
                default=None,
            ),
            "min_paired_ci95_low_vs_scalar": min((row["paired_vs_scalar_ci95_low"] for row in rows), default=None),
            "max_product_codebook_accuracy": max((row["product_codebook_accuracy"] for row in rows), default=None),
        },
        "pass_gate": len(pass_remaps) >= 3,
        "pass_rule": (
            "At least one row per remapped codebook must have exact ID parity, paired CI95 low >0.15 versus target-only, "
            "paired CI95 low >0.10 versus the best product-codebook destructive control, and stay within 0.02 accuracy "
            "of scalar Wyner-Ziv."
        ),
        "interpretation": (
            "This summary tests whether product-codebook gains are stable at the example-paired level rather than only "
            "aggregate accuracies. It is intentionally stricter against source-destroying controls than against scalar WZ: "
            "PQ must prove source-causal lift, while scalar WZ remains a strong adjacent codec comparator."
        ),
        "inputs": {
            "product_gate_dir": str(product_gate_dir),
            "product_gate_json": None if product_gate_json is None else str(product_gate_json),
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
    }
    json_path = output_dir / "product_codebook_uncertainty.json"
    md_path = output_dir / "product_codebook_uncertainty.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": ["product_codebook_uncertainty.json", "product_codebook_uncertainty.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "product_codebook_uncertainty.json": _sha256_file(json_path),
            "product_codebook_uncertainty.md": _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Product-Codebook Uncertainty Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
        "# Source-Private Product-Codebook Paired Uncertainty",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- remaps with pass: `{h['remaps_with_pass']}`",
        f"- min passing CI95 low vs target: `{_fmt(h['min_passing_ci95_low_vs_target'])}`",
        f"- min passing CI95 low vs best control: `{_fmt(h['min_passing_ci95_low_vs_best_control'])}`",
        f"- min CI95 low vs scalar: `{_fmt(h['min_paired_ci95_low_vs_scalar'])}`",
        f"- max product-codebook accuracy: `{_fmt(h['max_product_codebook_accuracy'])}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | PQ | Target | Best control | Scalar | PQ-target | PQ-control CI low | PQ-target CI low | PQ-scalar CI low | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['product_codebook_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_product_codebook_control_accuracy']:.3f} | {row['scalar_wyner_ziv_accuracy']:.3f} | "
            f"{row['product_minus_target']:.3f} | {row['paired_vs_best_control_ci95_low']:.3f} | "
            f"{row['paired_vs_target_ci95_low']:.3f} | {row['paired_vs_scalar_ci95_low']:.3f} | "
            f"`{row['uncertainty_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], "", "## Pass Rule", "", payload["pass_rule"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-gate-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_product_codebook_packet_gate_20260430"))
    parser.add_argument(
        "--product-gate-json",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_product_codebook_packet_gate_20260430/product_codebook_packet_gate.json"),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_product_codebook_uncertainty_20260430"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=991)
    args = parser.parse_args()

    payload = summarize_uncertainty(
        product_gate_dir=args.product_gate_dir,
        product_gate_json=args.product_gate_json,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
