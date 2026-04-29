from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(result_dir: pathlib.Path, budget: int) -> dict[str, dict[str, bool]]:
    rows_by_example: dict[str, dict[str, bool]] = {}
    with (result_dir / f"predictions_budget{budget}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_example.setdefault(row["example_id"], {})[row["condition"]] = bool(row["correct"])
    return rows_by_example


def _paired_deltas(rows: dict[str, dict[str, bool]], *, method: str, baseline: str) -> list[float]:
    return [
        float(conditions[method]) - float(conditions[baseline])
        for _, conditions in sorted(rows.items())
        if method in conditions and baseline in conditions
    ]


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "ci95_low": means[int(0.025 * (samples - 1))],
        "ci95_high": means[int(0.975 * (samples - 1))],
    }


def _control_conditions(method_condition: str) -> list[str]:
    if method_condition == "relative_score_source":
        return [
            "relative_label_shuffled_ridge",
            "relative_constrained_shuffled_source",
            "relative_answer_masked_source",
            "relative_random_same_byte",
            "relative_order_mismatch_source",
            "relative_permuted_score_bytes",
        ]
    if method_condition == "relative_canonical_score_source":
        return [
            "relative_canonical_label_shuffled_ridge",
            "relative_canonical_constrained_shuffled_source",
            "relative_canonical_answer_masked_source",
            "relative_canonical_random_same_byte",
            "relative_canonical_order_mismatch_source",
            "relative_canonical_permuted_score_bytes",
        ]
    raise ValueError(f"unsupported method condition {method_condition!r}")


def _summarize_dir(
    result_dir: pathlib.Path,
    *,
    budget: int,
    bootstrap_samples: int,
    seed: int,
    method_condition: str,
) -> dict[str, Any]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    budget_row = next(row for row in summary["budget_summaries"] if row["budget_bytes"] == budget)
    metrics = budget_row["metrics"]
    rows = _load_predictions(result_dir, budget)
    controls = _control_conditions(method_condition)
    baselines = [
        "target_only",
        "scalar_quantized_source",
        "raw_source_sign_sketch",
        *controls,
    ]
    paired = {
        baseline: _bootstrap_ci(
            _paired_deltas(rows, method=method_condition, baseline=baseline),
            samples=bootstrap_samples,
            seed=seed + idx * 1009,
        )
        for idx, baseline in enumerate(baselines)
    }
    strict_controls = ["target_only", *controls]
    return {
        "result_dir": str(result_dir),
        "pass_gate": summary["pass_gate"],
        "remap_slot_seed": summary.get("remap_slot_seed"),
        "budget_bytes": budget,
        "method_condition": method_condition,
        "relative_payload_bytes": metrics[method_condition]["mean_payload_bytes"],
        "scalar_payload_bytes": metrics["scalar_quantized_source"]["mean_payload_bytes"],
        "relative_accuracy": metrics[method_condition]["accuracy"],
        "scalar_accuracy": metrics["scalar_quantized_source"]["accuracy"],
        "target_accuracy": metrics["target_only"]["accuracy"],
        "raw_sign_accuracy": metrics["raw_source_sign_sketch"]["accuracy"],
        "relative_p50_latency_ms": metrics[method_condition]["p50_latency_ms"],
        "scalar_p50_latency_ms": metrics["scalar_quantized_source"]["p50_latency_ms"],
        "relative_minus_scalar": metrics[method_condition]["accuracy"] - metrics["scalar_quantized_source"]["accuracy"],
        "relative_minus_best_strict_control": metrics[method_condition]["accuracy"]
        - max(metrics[name]["accuracy"] for name in strict_controls),
        "paired_bootstrap": paired,
    }


def run_summary(
    *,
    result_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    budget: int,
    bootstrap_samples: int,
    seed: int,
    method_condition: str = "relative_score_source",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _summarize_dir(
            path if path.is_absolute() else ROOT / path,
            budget=budget,
            bootstrap_samples=bootstrap_samples,
            seed=seed + i * 2003,
            method_condition=method_condition,
        )
        for i, path in enumerate(result_dirs)
    ]
    remap_rows = [row for row in rows if row["remap_slot_seed"] is not None]
    comparison_rows = remap_rows or rows
    payload = {
        "gate": "source_private_relative_score_bootstrap",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "budget_bytes": budget,
        "bootstrap_samples": bootstrap_samples,
        "method_condition": method_condition,
        "result_dirs": [str(path) for path in result_dirs],
        "rows": rows,
        "mean_relative_accuracy": statistics.fmean(row["relative_accuracy"] for row in rows),
        "mean_relative_minus_scalar": statistics.fmean(row["relative_minus_scalar"] for row in rows),
        "mean_remap_relative_minus_scalar": statistics.fmean(row["relative_minus_scalar"] for row in comparison_rows),
        "min_relative_vs_target_ci95_low": min(row["paired_bootstrap"]["target_only"]["ci95_low"] for row in rows),
        "min_relative_vs_scalar_ci95_low": min(row["paired_bootstrap"]["scalar_quantized_source"]["ci95_low"] for row in comparison_rows),
        "pass_gate": (
            min(row["paired_bootstrap"]["target_only"]["ci95_low"] for row in rows) > 0.15
            and statistics.fmean(row["relative_minus_scalar"] for row in comparison_rows) >= 0.0
        ),
        "pass_rule": "Relative score packet should keep positive paired lower bound versus target-only and positive mean remap delta versus scalar at equal actual bytes.",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "summary.json": _sha256_file(output_dir / "summary.json"),
            "summary.md": _sha256_file(output_dir / "summary.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Relative Score Bootstrap Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Relative Score Bootstrap",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- method condition: `{payload['method_condition']}`",
        f"- budget bytes: `{payload['budget_bytes']}`",
        f"- bootstrap samples: `{payload['bootstrap_samples']}`",
        f"- mean relative accuracy: `{payload['mean_relative_accuracy']:.3f}`",
        f"- mean relative minus scalar: `{payload['mean_relative_minus_scalar']:.3f}`",
        f"- mean remap relative minus scalar: `{payload['mean_remap_relative_minus_scalar']:.3f}`",
        f"- min relative vs target CI95 low: `{payload['min_relative_vs_target_ci95_low']:.3f}`",
        f"- min remap relative vs scalar CI95 low: `{payload['min_relative_vs_scalar_ci95_low']:.3f}`",
        "",
        "| Result | Remap | Relative | Scalar | Target | Raw sign | Relative - scalar CI95 | Relative - target CI95 | Bytes rel/scalar | p50 rel/scalar ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        scalar_ci = row["paired_bootstrap"]["scalar_quantized_source"]
        target_ci = row["paired_bootstrap"]["target_only"]
        lines.append(
            f"| `{pathlib.Path(row['result_dir']).name}` | `{row['remap_slot_seed']}` | "
            f"{row['relative_accuracy']:.3f} | {row['scalar_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['raw_sign_accuracy']:.3f} | "
            f"[{scalar_ci['ci95_low']:.3f}, {scalar_ci['ci95_high']:.3f}] | "
            f"[{target_ci['ci95_low']:.3f}, {target_ci['ci95_high']:.3f}] | "
            f"{row['relative_payload_bytes']:.1f}/{row['scalar_payload_bytes']:.1f} | "
            f"{row['relative_p50_latency_ms']:.2f}/{row['scalar_p50_latency_ms']:.2f} |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--method-condition", choices=["relative_score_source", "relative_canonical_score_source"], default="relative_score_source")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_summary(
        result_dirs=args.result_dirs,
        output_dir=output_dir,
        budget=args.budget,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        method_condition=args.method_condition,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
