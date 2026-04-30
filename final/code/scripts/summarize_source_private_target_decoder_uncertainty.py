from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

METHOD = "matched_packet"
CONTROL_CONDITIONS = [
    "shuffled_packet",
    "random_same_byte",
    "structured_json_2byte",
    "structured_free_text_2byte",
]
BASELINE_CONDITIONS = ["target_only", *CONTROL_CONDITIONS]


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return float(ordered[index])


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


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
        if method in conditions and baseline in conditions:
            values.append(float(bool(conditions[method]["correct"])) - float(bool(conditions[baseline]["correct"])))
    return values


def _paired_counts(values: list[float]) -> dict[str, int]:
    return {
        "wins": sum(1 for value in values if value > 0),
        "losses": sum(1 for value in values if value < 0),
        "ties": sum(1 for value in values if value == 0),
    }


def _comparison(
    grouped: dict[str, dict[str, dict[str, Any]]],
    *,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = _paired_values(grouped, method=METHOD, baseline=baseline)
    return {
        "baseline": baseline,
        "n": len(values),
        "paired_delta_bootstrap95": _bootstrap_ci(values, samples=samples, seed=seed),
        "paired_counts": _paired_counts(values),
    }


def summarize_run(
    *,
    result_dir: pathlib.Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result_dir = result_dir if result_dir.is_absolute() else ROOT / result_dir
    summary_path = result_dir / "summary.json"
    rows_path = result_dir / "target_predictions.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(rows_path)
    grouped = _rows_by_example(rows)
    metrics = summary["metrics"]
    active_controls = [name for name in CONTROL_CONDITIONS if name in metrics]
    best_control = max(active_controls, key=lambda name: metrics[name]["accuracy"])
    comparisons: dict[str, Any] = {}
    for index, baseline in enumerate(dict.fromkeys(["target_only", *active_controls, best_control])):
        comparisons[baseline] = _comparison(
            grouped,
            baseline=baseline,
            samples=bootstrap_samples,
            seed=seed + index * 1009,
        )
    target_ci = comparisons["target_only"]["paired_delta_bootstrap95"]
    control_ci = comparisons[best_control]["paired_delta_bootstrap95"]
    return {
        "result_dir": str(result_dir),
        "surface": result_dir.name,
        "n": summary["n"],
        "pass_gate": summary["pass_gate"],
        "exact_id_parity": summary["exact_id_parity"],
        "matched_accuracy": summary["matched_accuracy"],
        "target_only_accuracy": summary["target_only_accuracy"],
        "best_control": best_control,
        "best_control_accuracy": metrics[best_control]["accuracy"],
        "matched_minus_target": summary["matched_minus_target"],
        "matched_minus_best_control": summary["matched_minus_best_control"],
        "matched_valid_prediction_rate": metrics[METHOD]["valid_prediction_rate"],
        "matched_p50_latency_ms": metrics[METHOD]["p50_latency_ms"],
        "matched_mean_generated_tokens": metrics[METHOD]["mean_generated_tokens"],
        "matched_mean_payload_bytes": metrics[METHOD]["mean_payload_bytes"],
        "paired_vs_target_ci95_low": target_ci["ci95_low"],
        "paired_vs_best_control_ci95_low": control_ci["ci95_low"],
        "comparisons": comparisons,
        "summary_sha256": _sha256_file(summary_path),
        "rows_sha256": _sha256_file(rows_path),
    }


def run_summary(
    *,
    result_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_run(result_dir=path, bootstrap_samples=bootstrap_samples, seed=seed + index * 2003)
        for index, path in enumerate(result_dirs)
    ]
    target_lows = [row["paired_vs_target_ci95_low"] for row in rows]
    control_lows = [row["paired_vs_best_control_ci95_low"] for row in rows]
    payload = {
        "gate": "source_private_target_decoder_uncertainty",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "bootstrap_samples": bootstrap_samples,
        "result_dirs": [str(path) for path in result_dirs],
        "rows": rows,
        "headline": {
            "rows": len(rows),
            "pass_rows": sum(1 for row in rows if row["pass_gate"]),
            "min_matched_minus_target": min((row["matched_minus_target"] for row in rows), default=None),
            "min_matched_minus_best_control": min((row["matched_minus_best_control"] for row in rows), default=None),
            "min_paired_ci95_low_vs_target": min(target_lows) if target_lows else None,
            "min_paired_ci95_low_vs_best_control": min(control_lows) if control_lows else None,
            "min_valid_prediction_rate": min((row["matched_valid_prediction_rate"] for row in rows), default=None),
            "max_p50_latency_ms": max((row["matched_p50_latency_ms"] for row in rows), default=None),
        },
        "pass_gate": (
            bool(rows)
            and all(row["pass_gate"] for row in rows)
            and all(row["exact_id_parity"] for row in rows)
            and all(row["matched_valid_prediction_rate"] >= 0.95 for row in rows)
            and min(target_lows) > 0.10
            and min(control_lows) > 0.10
        ),
        "pass_rule": (
            "Every target-decoder run must pass its point gate, preserve exact-ID parity, have matched valid prediction "
            "rate >=0.95, and have paired CI95 lower bounds above +0.10 versus both target-only and the strongest "
            "source-destroying or matched-byte text control."
        ),
        "interpretation": (
            "This gate asks whether a frozen target LLM can consume the compact source packet beyond target priors and "
            "same-byte/source-destroyed controls. It is a reviewer-defense receiver gate, not a systems-speed claim."
        ),
    }
    json_path = output_dir / "target_decoder_uncertainty.json"
    md_path = output_dir / "target_decoder_uncertainty.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": ["target_decoder_uncertainty.json", "target_decoder_uncertainty.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "target_decoder_uncertainty.json": _sha256_file(json_path),
            "target_decoder_uncertainty.md": _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Target-Decoder Uncertainty Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
        "# Source-Private Target-Decoder Paired Uncertainty",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- min matched-target: `{_fmt(h['min_matched_minus_target'])}`",
        f"- min matched-best-control: `{_fmt(h['min_matched_minus_best_control'])}`",
        f"- min CI95 low vs target: `{_fmt(h['min_paired_ci95_low_vs_target'])}`",
        f"- min CI95 low vs best control: `{_fmt(h['min_paired_ci95_low_vs_best_control'])}`",
        f"- min valid prediction rate: `{_fmt(h['min_valid_prediction_rate'])}`",
        f"- max p50 latency ms: `{_fmt(h['max_p50_latency_ms'])}`",
        "",
        "## Rows",
        "",
        "| Surface | N | Matched | Target | Best control | Matched-target | Matched-control | CI low target | CI low control | Valid | p50 ms | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['surface']} | {row['n']} | {row['matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | "
            f"{row['matched_minus_target']:.3f} | {row['matched_minus_best_control']:.3f} | "
            f"{row['paired_vs_target_ci95_low']:.3f} | {row['paired_vs_best_control_ci95_low']:.3f} | "
            f"{row['matched_valid_prediction_rate']:.3f} | {row['matched_p50_latency_ms']:.1f} | "
            f"`{row['pass_gate']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], "", "## Pass Rule", "", payload["pass_rule"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()
    payload = run_summary(
        result_dirs=args.result_dirs,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
