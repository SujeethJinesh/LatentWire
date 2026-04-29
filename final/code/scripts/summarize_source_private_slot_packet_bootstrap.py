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
    path = result_dir / f"predictions_budget{budget}.jsonl"
    rows_by_example: dict[str, dict[str, bool]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_id = row["example_id"]
            rows_by_example.setdefault(example_id, {})[row["condition"]] = bool(row["correct"])
    return rows_by_example


def _paired_deltas(rows: dict[str, dict[str, bool]], *, baseline_condition: str) -> list[float]:
    return [
        float(conditions["scalar_quantized_source"]) - float(conditions[baseline_condition])
        for _, conditions in sorted(rows.items())
        if "scalar_quantized_source" in conditions and baseline_condition in conditions
    ]


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": lo,
        "ci95_high": hi,
    }


def _summarize_dir(result_dir: pathlib.Path, *, budget: int, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    budget_row = next(row for row in summary["budget_summaries"] if row["budget_bytes"] == budget)
    metrics = budget_row["metrics"]
    rows = _load_predictions(result_dir, budget)
    baselines = [
        "target_only",
        "random_same_byte",
        "scalar_answer_masked_source",
        "scalar_constrained_shuffled_source",
        "scalar_label_shuffled_ridge",
        "raw_source_sign_sketch",
    ]
    paired = {
        baseline: _bootstrap_ci(
            _paired_deltas(rows, baseline_condition=baseline),
            samples=bootstrap_samples,
            seed=seed + idx * 1009,
        )
        for idx, baseline in enumerate(baselines)
    }
    strict_control_names = [
        "target_only",
        "random_same_byte",
        "scalar_answer_masked_source",
        "scalar_constrained_shuffled_source",
        "scalar_label_shuffled_ridge",
    ]
    best_strict_control = max(metrics[name]["accuracy"] for name in strict_control_names)
    return {
        "result_dir": str(result_dir),
        "pass_gate": summary["pass_gate"],
        "train_family_set": summary["train_family_set"],
        "eval_family_set": summary["eval_family_set"],
        "candidate_view": summary["candidate_view"],
        "fit_intercept": summary["fit_intercept"],
        "remap_slot_seed": summary.get("remap_slot_seed"),
        "n": metrics["scalar_quantized_source"]["n"],
        "scalar_accuracy": metrics["scalar_quantized_source"]["accuracy"],
        "target_accuracy": metrics["target_only"]["accuracy"],
        "raw_sign_accuracy": metrics["raw_source_sign_sketch"]["accuracy"],
        "best_strict_control_accuracy": best_strict_control,
        "scalar_minus_best_strict_control": metrics["scalar_quantized_source"]["accuracy"] - best_strict_control,
        "paired_bootstrap": paired,
    }


def run_summary(
    *,
    result_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    budget: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _summarize_dir(path if path.is_absolute() else ROOT / path, budget=budget, bootstrap_samples=bootstrap_samples, seed=seed + i * 2003)
        for i, path in enumerate(result_dirs)
    ]
    weighted_scalar = statistics.fmean(row["scalar_accuracy"] for row in rows)
    weighted_delta = statistics.fmean(row["scalar_minus_best_strict_control"] for row in rows)
    min_target_ci = min(row["paired_bootstrap"]["target_only"]["ci95_low"] for row in rows)
    min_raw_ci = min(row["paired_bootstrap"]["raw_source_sign_sketch"]["ci95_low"] for row in rows)
    payload = {
        "gate": "source_private_slot_packet_bootstrap",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "budget_bytes": budget,
        "bootstrap_samples": bootstrap_samples,
        "result_dirs": [str(path) for path in result_dirs],
        "rows": rows,
        "mean_scalar_accuracy": weighted_scalar,
        "mean_scalar_minus_best_strict_control": weighted_delta,
        "min_target_delta_ci95_low": min_target_ci,
        "min_raw_sign_delta_ci95_low": min_raw_ci,
        "pass_gate": min_target_ci > 0.15 and weighted_delta > 0.15,
        "pass_rule": "Every row should have paired bootstrap lower bound >0.15 versus target-only, and mean scalar-minus-best-strict-control should exceed 0.15. Raw sign sketch is reported as a compression baseline but not part of strict no-source controls.",
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
        "\n".join(
            [
                "# Source-Private Slot Packet Bootstrap Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- rows: `{len(rows)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Slot Packet Bootstrap",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- budget bytes: `{payload['budget_bytes']}`",
        f"- bootstrap samples: `{payload['bootstrap_samples']}`",
        f"- mean scalar accuracy: `{payload['mean_scalar_accuracy']:.3f}`",
        f"- mean scalar minus best strict control: `{payload['mean_scalar_minus_best_strict_control']:.3f}`",
        f"- min target delta CI95 low: `{payload['min_target_delta_ci95_low']:.3f}`",
        f"- min raw-sign delta CI95 low: `{payload['min_raw_sign_delta_ci95_low']:.3f}`",
        "",
        "| Result | Pass | Remap | Scalar | Target | Best strict control | Raw sign | Delta target CI95 | Delta raw CI95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        target_ci = row["paired_bootstrap"]["target_only"]
        raw_ci = row["paired_bootstrap"]["raw_source_sign_sketch"]
        lines.append(
            f"| `{pathlib.Path(row['result_dir']).name}` | `{row['pass_gate']}` | "
            f"`{row['remap_slot_seed']}` | {row['scalar_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_strict_control_accuracy']:.3f} | "
            f"{row['raw_sign_accuracy']:.3f} | "
            f"[{target_ci['ci95_low']:.3f}, {target_ci['ci95_high']:.3f}] | "
            f"[{raw_ci['ci95_low']:.3f}, {raw_ci['ci95_high']:.3f}] |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_summary(
        result_dirs=args.result_dirs,
        output_dir=output_dir,
        budget=args.budget,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
