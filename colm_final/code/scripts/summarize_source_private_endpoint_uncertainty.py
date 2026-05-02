from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import statistics
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


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
    return ordered[index]


def _rows_by_example(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault(row["example_id"], {})[row["condition"]] = row
    return paired


def _paired_values(
    paired: dict[str, dict[str, dict[str, Any]]],
    *,
    method: str,
    baseline: str,
    field: str = "correct",
) -> list[float]:
    values: list[float] = []
    for _, conditions in sorted(paired.items()):
        if method in conditions and baseline in conditions:
            values.append(float(bool(conditions[method][field])) - float(bool(conditions[baseline][field])))
    return values


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


def _exact_sign_p_value(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(0, min(wins, losses) + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def _paired_counts(values: list[float]) -> dict[str, Any]:
    wins = sum(1 for value in values if value > 0)
    losses = sum(1 for value in values if value < 0)
    ties = sum(1 for value in values if value == 0)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "exact_sign_p_two_sided": _exact_sign_p_value(wins, losses),
    }


def _comparison(
    paired: dict[str, dict[str, dict[str, Any]]],
    *,
    method: str,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = _paired_values(paired, method=method, baseline=baseline, field="correct")
    strict_values = _paired_values(paired, method=method, baseline=baseline, field="strict_correct")
    return {
        "baseline": baseline,
        "n": len(values),
        "delta_bootstrap95": _bootstrap_ci(values, samples=samples, seed=seed),
        "strict_delta_bootstrap95": _bootstrap_ci(strict_values, samples=samples, seed=seed + 17),
        "paired_counts": _paired_counts(values),
        "strict_paired_counts": _paired_counts(strict_values),
    }


def summarize_run(
    *,
    result_dir: pathlib.Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result_dir = result_dir if result_dir.is_absolute() else ROOT / result_dir
    summary_path = result_dir / "summary.json"
    rows_path = result_dir / "endpoint_proxy_rows.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(rows_path)
    paired = _rows_by_example(rows)
    metrics = summary["metrics"]
    destructive = ["matched_byte_text_2", "random_same_byte_packet", "deranged_candidate_diag_table"]
    best_control = max(destructive, key=lambda name: metrics[name]["accuracy"])
    baselines = [
        "target_only",
        "matched_byte_text_2",
        "random_same_byte_packet",
        "deranged_candidate_diag_table",
        best_control,
        "query_aware_diag_span",
        "structured_json_diag",
        "structured_free_text_diag",
        "full_hidden_log",
    ]
    comparisons: dict[str, Any] = {}
    for index, baseline in enumerate(dict.fromkeys(baselines)):
        comparisons[baseline] = _comparison(
            paired,
            method="matched_packet",
            baseline=baseline,
            samples=bootstrap_samples,
            seed=seed + index * 1009,
        )
    packet = metrics["matched_packet"]
    target = metrics["target_only"]
    full_log = metrics["full_hidden_log"]
    query = metrics["query_aware_diag_span"]
    return {
        "result_dir": str(result_dir),
        "surface": result_dir.name,
        "pass_gate": summary["pass_gate"],
        "n": summary["n"],
        "prompt_style": summary["prompt_style"],
        "packet_accuracy": packet["accuracy"],
        "packet_strict_accuracy": packet["strict_accuracy"],
        "target_accuracy": target["accuracy"],
        "best_source_destroying_control": best_control,
        "best_source_destroying_control_accuracy": metrics[best_control]["accuracy"],
        "packet_valid_rate": packet["valid_prediction_rate"],
        "packet_payload_bytes": packet["mean_payload_bytes"],
        "query_payload_bytes": query["mean_payload_bytes"],
        "full_log_payload_bytes": full_log["mean_payload_bytes"],
        "packet_vs_query_payload_compression": summary["packet_vs_query_payload_compression"],
        "packet_vs_full_log_payload_compression": summary["packet_vs_full_log_payload_compression"],
        "full_log_ttft_delta_vs_packet_ms": summary["full_log_ttft_delta_vs_packet_ms"],
        "full_log_e2e_delta_vs_packet_ms": summary["full_log_e2e_delta_vs_packet_ms"],
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
    target_lows = [row["comparisons"]["target_only"]["delta_bootstrap95"]["ci95_low"] for row in rows]
    control_lows = [
        row["comparisons"][row["best_source_destroying_control"]]["delta_bootstrap95"]["ci95_low"]
        for row in rows
    ]
    strict_target_lows = [row["comparisons"]["target_only"]["strict_delta_bootstrap95"]["ci95_low"] for row in rows]
    payload = {
        "gate": "source_private_endpoint_uncertainty",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "bootstrap_samples": bootstrap_samples,
        "result_dirs": [str(path) for path in result_dirs],
        "rows": rows,
        "min_packet_vs_target_ci95_low": min(target_lows) if target_lows else 0.0,
        "min_packet_vs_best_control_ci95_low": min(control_lows) if control_lows else 0.0,
        "min_strict_packet_vs_target_ci95_low": min(strict_target_lows) if strict_target_lows else 0.0,
        "pass_gate": (
            bool(rows)
            and all(row["pass_gate"] for row in rows)
            and all(row["packet_valid_rate"] >= 0.95 for row in rows)
            and min(target_lows) > 0.0
            and min(control_lows) > 0.0
            and min(strict_target_lows) > 0.0
            and all(row["full_log_ttft_delta_vs_packet_ms"] > 0.0 for row in rows)
        ),
        "pass_rule": (
            "All endpoint rows must pass their strict gate, packet valid rate must be >=0.95, paired "
            "bootstrap lower bounds versus target and best source-destroying control must be positive, "
            "strict-label packet-vs-target lower bound must be positive, and full-log p50 TTFT must be "
            "slower than the packet. Query-aware/structured relays are reported as rate-quality "
            "comparators, not required accuracy losses."
        ),
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
        "\n".join(["# Source-Private Endpoint Uncertainty Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def _format_ci(ci: dict[str, float]) -> str:
    return f"[{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]"


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Endpoint Uncertainty",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- bootstrap samples: `{payload['bootstrap_samples']}`",
        f"- min packet vs target CI95 low: `{payload['min_packet_vs_target_ci95_low']:.3f}`",
        f"- min packet vs best-control CI95 low: `{payload['min_packet_vs_best_control_ci95_low']:.3f}`",
        f"- min strict packet vs target CI95 low: `{payload['min_strict_packet_vs_target_ci95_low']:.3f}`",
        "",
        "| Surface | N | Packet | Strict packet | Target | Best control | Valid | Packet-target CI | Packet-control CI | Strict packet-target CI | Full-log TTFT delta ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        target_ci = row["comparisons"]["target_only"]["delta_bootstrap95"]
        strict_target_ci = row["comparisons"]["target_only"]["strict_delta_bootstrap95"]
        control_ci = row["comparisons"][row["best_source_destroying_control"]]["delta_bootstrap95"]
        lines.append(
            f"| `{row['surface']}` | {row['n']} | {row['packet_accuracy']:.3f} | "
            f"{row['packet_strict_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_source_destroying_control_accuracy']:.3f} | {row['packet_valid_rate']:.3f} | "
            f"{_format_ci(target_ci)} | {_format_ci(control_ci)} | {_format_ci(strict_target_ci)} | "
            f"{row['full_log_ttft_delta_vs_packet_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Rate/Quality Comparators",
            "",
            "| Surface | Query-aware accuracy | Packet-query CI | Query bytes / packet bytes | Structured JSON accuracy | Free-text accuracy | Full-log accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        query_ci = row["comparisons"]["query_aware_diag_span"]["delta_bootstrap95"]
        lines.append(
            f"| `{row['surface']}` | "
            f"{row['packet_accuracy'] - query_ci['mean']:.3f} | "
            f"{_format_ci(query_ci)} | {row['query_payload_bytes']:.1f}/{row['packet_payload_bytes']:.1f} | "
            f"{row['packet_accuracy'] - row['comparisons']['structured_json_diag']['delta_bootstrap95']['mean']:.3f} | "
            f"{row['packet_accuracy'] - row['comparisons']['structured_free_text_diag']['delta_bootstrap95']['mean']:.3f} | "
            f"{row['packet_accuracy'] - row['comparisons']['full_hidden_log']['delta_bootstrap95']['mean']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260429)
    args = parser.parse_args()
    payload = run_summary(
        result_dirs=args.result_dirs,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
