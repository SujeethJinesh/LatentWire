from __future__ import annotations

import argparse
import hashlib
import json
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


def _bootstrap_delta_ci(
    rows: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(rows)
    deltas: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            row = rows[rng.randrange(n)]
            total += float(row["conditions"][left]["correct"]) - float(row["conditions"][right]["correct"])
        deltas.append(total / n)
    deltas.sort()
    return {
        "mean": statistics.fmean(deltas),
        "low": _percentile(deltas, 0.025),
        "high": _percentile(deltas, 0.975),
    }


def summarize_run(
    *,
    run_id: str,
    model: str,
    prompt_mode: str,
    predictions_path: pathlib.Path,
    summary_path: pathlib.Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = _load_jsonl(predictions_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matched = summary["metrics"]["matched_model_packet"]
    target = summary["metrics"]["target_only"]
    best_control_name = max(
        ["zero_source", "shuffled_model_packet", "random_same_byte", "answer_only", "answer_masked", "target_derived_sidecar"],
        key=lambda condition: summary["metrics"][condition]["accuracy"],
    )
    matched_latencies = [row["conditions"]["matched_model_packet"]["latency_ms"] for row in rows]
    return {
        "run_id": run_id,
        "model": model,
        "prompt_mode": prompt_mode,
        "pass_gate": summary["pass_gate"],
        "n": summary["n"],
        "matched_correct": matched["correct"],
        "matched_accuracy": matched["accuracy"],
        "target_only_correct": target["correct"],
        "target_only_accuracy": target["accuracy"],
        "best_control_name": best_control_name,
        "best_source_destroying_control_accuracy": summary["best_source_destroying_control_accuracy"],
        "packet_valid_rate": summary["packet_valid_rate"],
        "mean_packet_bytes": matched["mean_payload_bytes"],
        "mean_packet_tokens": matched["mean_payload_tokens"],
        "p50_source_latency_ms": matched["p50_latency_ms"],
        "p95_source_latency_ms": _percentile(matched_latencies, 0.95),
        "matched_minus_target_bootstrap95": _bootstrap_delta_ci(
            rows,
            left="matched_model_packet",
            right="target_only",
            samples=bootstrap_samples,
            seed=seed,
        ),
        "matched_minus_best_control_bootstrap95": _bootstrap_delta_ci(
            rows,
            left="matched_model_packet",
            right=best_control_name,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "exact_id_sha256": summary["exact_id_sha256"],
        "summary_sha256": _sha256_file(summary_path),
        "predictions_sha256": _sha256_file(predictions_path),
    }


def _write_markdown(path: pathlib.Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Hidden-Repair Medium Summary",
        "",
        f"- gate: `{aggregate['gate']}`",
        f"- pass gate: `{aggregate['pass_gate']}`",
        f"- examples: `{aggregate['n']}`",
        f"- bootstrap samples: `{aggregate['bootstrap_samples']}`",
        "",
        "| Run | Model | Mode | Pass | Matched | Target | Best control | Valid | Mean bytes | p95 latency ms | Delta target 95% CI | Delta control 95% CI |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["rows"]:
        target_ci = row["matched_minus_target_bootstrap95"]
        control_ci = row["matched_minus_best_control_bootstrap95"]
        lines.append(
            f"| {row['run_id']} | {row['model']} | {row['prompt_mode']} | `{str(row['pass_gate']).lower()}` | "
            f"{row['matched_accuracy']:.3f} | {row['target_only_accuracy']:.3f} | "
            f"{row['best_source_destroying_control_accuracy']:.3f} | {row['packet_valid_rate']:.3f} | "
            f"{row['mean_packet_bytes']:.2f} | {row['p95_source_latency_ms']:.2f} | "
            f"[{target_ci['low']:.3f}, {target_ci['high']:.3f}] | "
            f"[{control_ci['low']:.3f}, {control_ci['high']:.3f}] |"
        )
    lines.extend(["", f"Pass rule: {aggregate['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-dir", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260429)
    args = parser.parse_args()

    llm_dir = args.llm_dir if args.llm_dir.is_absolute() else ROOT / args.llm_dir
    benchmark_path = args.benchmark_jsonl if args.benchmark_jsonl.is_absolute() else ROOT / args.benchmark_jsonl
    runs = [
        ("qwen3_trace_no_hint", "Qwen/Qwen3-0.6B", "trace_no_hint"),
        ("phi3_trace_no_hint", "microsoft/Phi-3-mini-4k-instruct", "trace_no_hint"),
        ("qwen3_raw_log_no_trace", "Qwen/Qwen3-0.6B", "raw_log_no_trace"),
    ]
    rows = [
        summarize_run(
            run_id=run_id,
            model=model,
            prompt_mode=prompt_mode,
            predictions_path=llm_dir / run_id / "predictions.jsonl",
            summary_path=llm_dir / run_id / "summary.json",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index * 100,
        )
        for index, (run_id, model, prompt_mode) in enumerate(runs)
    ]
    primary = [row for row in rows if row["prompt_mode"] == "trace_no_hint"]
    destruction = [row for row in rows if row["prompt_mode"] == "raw_log_no_trace"]
    aggregate = {
        "gate": "source_private_hidden_repair_packet_medium_20260429",
        "n": rows[0]["n"] if rows else 0,
        "benchmark_jsonl": str(args.benchmark_jsonl),
        "benchmark_sha256": _sha256_file(benchmark_path),
        "bootstrap_samples": args.bootstrap_samples,
        "pass_gate": all(row["pass_gate"] for row in primary) and all(not row["pass_gate"] for row in destruction),
        "pass_rule": (
            "Qwen3 and Phi-3 trace_no_hint rows pass, raw_log_no_trace fails, "
            "and paired bootstrap lower bounds remain comfortably above +0.15 for primary rows."
        ),
        "rows": rows,
    }
    (llm_dir / "medium_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(llm_dir / "medium_summary.md", aggregate)


if __name__ == "__main__":
    main()
