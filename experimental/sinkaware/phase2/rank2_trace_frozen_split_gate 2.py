"""Trace-level frozen split repeat for all-head rank-2 SinkAware.

Earlier stability gates randomized token-level train/test splits. This gate
holds out whole traces, fits all-head rank-2 on train traces, and evaluates on
held-out traces. It is still Mac-local drift evidence, not quality or speed.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import torch

try:
    from .real_qk_sink_softmax_output_probe import (
        OUT_DIR,
        ROOT,
        _collect_fit_samples,
        _evaluate_output_errors,
        _fit_layer_predictors,
        _load_texts,
        _mean_ci95,
        _paired_head_improvements,
        _summarize,
    )
    from .real_query_sink_probe import DEFAULT_TRACES
except ImportError:  # pragma: no cover - supports direct script execution.
    from real_qk_sink_softmax_output_probe import (
        OUT_DIR,
        ROOT,
        _collect_fit_samples,
        _evaluate_output_errors,
        _fit_layer_predictors,
        _load_texts,
        _mean_ci95,
        _paired_head_improvements,
        _summarize,
    )
    from real_query_sink_probe import DEFAULT_TRACES


def _split_trace_indices(n_traces: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if n_traces < 3:
        raise ValueError("need at least three traces for a trace-level split")
    train_count = int(train_fraction * n_traces)
    train_count = min(max(2, train_count), n_traces - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_traces, generator=generator).tolist()
    return permutation[:train_count], permutation[train_count:]


def _aggregate_seed_rows(seed_rows: list[dict[str, object]]) -> dict[str, dict[str, float] | dict[str, object]]:
    output_improvements = [
        float(row["output_rel_l2_improvement_vs_position"])
        for row in seed_rows
    ]
    sink_mass_improvements = [
        float(row["sink_mass_mae_improvement_vs_position"])
        for row in seed_rows
    ]
    attention_improvements = [
        float(row["attention_l1_improvement_vs_position"])
        for row in seed_rows
    ]
    head_win_rates = [
        float(row["paired_head_vs_position"]["rank2"]["output_rel_l2_win_rate"])
        for row in seed_rows
    ]
    return {
        "output_rel_l2_improvement_vs_position": _mean_ci95(output_improvements),
        "sink_mass_mae_improvement_vs_position": _mean_ci95(sink_mass_improvements),
        "attention_l1_improvement_vs_position": _mean_ci95(attention_improvements),
        "output_rel_l2_head_win_rate": _mean_ci95(head_win_rates),
        "all_trace_splits_rank2_beats_position": {
            "value": all(value > 0.0 for value in output_improvements),
            "n": len(output_improvements),
        },
        "min_output_rel_l2_improvement": {
            "value": min(output_improvements) if output_improvements else float("nan"),
        },
    }


def _status(seed_rows: list[dict[str, object]], aggregate: dict[str, object]) -> str:
    output = aggregate["output_rel_l2_improvement_vs_position"]
    all_positive = bool(aggregate["all_trace_splits_rank2_beats_position"]["value"])
    min_improvement = float(aggregate["min_output_rel_l2_improvement"]["value"])
    if all_positive and min_improvement > 0.015 and output["mean"] > 0.02:
        return "ALIVE but bounded; all-head rank-2 beats position-only across trace-level frozen splits."
    if output["mean"] > 0.0:
        return "WEAKLY ALIVE; trace-level mean improvement is positive but below the promotion margin."
    return "WEAKENED; all-head rank-2 does not survive trace-level frozen splits."


def _run(
    model_name: str,
    max_traces: int,
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    texts = _load_texts(DEFAULT_TRACES, max_traces)
    seed_rows: list[dict[str, object]] = []
    modes = ("position", "rank2")
    for seed in seeds:
        train_indices, test_indices = _split_trace_indices(len(texts), train_fraction, seed)
        train_texts = [texts[idx] for idx in train_indices]
        test_texts = [texts[idx] for idx in test_indices]
        fit_samples, train_meta = _collect_fit_samples(model_name, train_texts, max_length, sink_tokens)
        predictors = {}
        splits = {}
        train_samples_per_layer = {}
        for layer, tensors in fit_samples.items():
            splits[layer] = 0
            train_samples_per_layer[layer] = int(tensors["q"].shape[0])
            predictors[layer] = _fit_layer_predictors(
                tensors["q"],
                tensors["pos"],
                tensors["sink_logits"],
                ranks=(2,),
            )
        rows, head_rows = _evaluate_output_errors(
            model_name,
            test_texts,
            max_length,
            sink_tokens,
            predictors,
            splits,
            modes,
        )
        summary = _summarize(rows)
        paired = _paired_head_improvements(head_rows)
        position = summary["position"]
        rank2 = summary["rank2"]
        seed_rows.append(
            {
                "seed": seed,
                "train_trace_indices": train_indices,
                "test_trace_indices": test_indices,
                "train_traces": len(train_texts),
                "test_traces": len(test_texts),
                "train_samples": train_meta["n_samples"],
                "train_samples_per_layer_mean": mean(train_samples_per_layer.values()),
                "summary": summary,
                "paired_head_vs_position": paired,
                "output_rel_l2_improvement_vs_position": position["output_rel_l2"] - rank2["output_rel_l2"],
                "sink_mass_mae_improvement_vs_position": position["sink_mass_mae"] - rank2["sink_mass_mae"],
                "attention_l1_improvement_vs_position": position["attention_l1"] - rank2["attention_l1"],
            }
        )
    aggregate = _aggregate_seed_rows(seed_rows)
    return {
        "model_name": model_name,
        "max_traces": len(texts),
        "max_length": max_length,
        "sink_tokens": sink_tokens,
        "train_fraction": train_fraction,
        "seeds": list(seeds),
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "status": _status(seed_rows, aggregate),
    }


def _write_markdown(result: dict[str, object]) -> None:
    aggregate = result["aggregate"]
    output = aggregate["output_rel_l2_improvement_vs_position"]
    sink_mass = aggregate["sink_mass_mae_improvement_vs_position"]
    attention = aggregate["attention_l1_improvement_vs_position"]
    head_win = aggregate["output_rel_l2_head_win_rate"]
    lines = [
        "# SinkAware All-Rank2 Trace-Level Frozen Split Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['max_traces']}",
        f"- max length: {result['max_length']}",
        f"- sink tokens: {result['sink_tokens']}",
        f"- train fraction: {result['train_fraction']}",
        f"- seeds: {', '.join(str(seed) for seed in result['seeds'])}",
        "",
        "This gate freezes whole traces into train and held-out sets for each seed. It fits all-head rank-2 on train traces and evaluates output drift on held-out traces. It keeps all non-sink QK scores exact and makes no GPU speed claim.",
        "",
        "## Aggregate Across Trace Splits",
        "",
        "| Metric | Mean improvement | 95% CI |",
        "|---|---:|---:|",
        f"| output rel-L2 vs position | {output['mean']:.4f} | +/- {output['ci95']:.4f} |",
        f"| sink-mass MAE vs position | {sink_mass['mean']:.4f} | +/- {sink_mass['ci95']:.4f} |",
        f"| attention L1 vs position | {attention['mean']:.4f} | +/- {attention['ci95']:.4f} |",
        f"| layer-head output win rate | {head_win['mean']:.3f} | +/- {head_win['ci95']:.3f} |",
        f"| minimum output rel-L2 improvement | {aggregate['min_output_rel_l2_improvement']['value']:.4f} | |",
        "",
        "Positive improvement means rank-2 has lower error than position-only.",
        "",
        "## Per Trace Split",
        "",
        "| Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["seed_rows"]:
        position = row["summary"]["position"]
        rank2 = row["summary"]["rank2"]
        paired = row["paired_head_vs_position"]["rank2"]
        lines.append(
            "| {seed} | {train_traces} | {test_traces} | {position:.4f} | {rank2:.4f} | {improvement:+.4f} | {win_rate:.3f} |".format(
                seed=row["seed"],
                train_traces=row["train_traces"],
                test_traces=row["test_traces"],
                position=position["output_rel_l2"],
                rank2=rank2["output_rel_l2"],
                improvement=row["output_rel_l2_improvement_vs_position"],
                win_rate=paired["output_rel_l2_win_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is stronger than token-level randomized splits because no text trace appears in both train and held-out sets. It still only measures Mac-local attention-output drift, not downstream quality or GPU latency.",
        ]
    )
    (OUT_DIR / "rank2_trace_frozen_split_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--max-traces", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    cache_dir = ROOT / ".debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run(
        args.model_name,
        args.max_traces,
        args.max_length,
        args.sink_tokens,
        args.train_fraction,
        tuple(args.seeds),
    )
    (OUT_DIR / "rank2_trace_frozen_split_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
