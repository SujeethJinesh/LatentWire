"""Split/seed stability gate for all-head rank-2 SinkAware.

The aggregate softmax/output probe keeps all-rank2 weakly alive, while the
simple validation head selector failed. This gate asks a narrower question
before GPU work: does all-head rank-2 still beat position-only when the
token-level train/test split is randomized across seeds?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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


def _split_indices(n_samples: int, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_samples < 3:
        raise ValueError("need at least three samples for a train/test split")
    train_count = int(train_fraction * n_samples)
    train_count = min(max(2, train_count), n_samples - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator)
    return permutation[:train_count], permutation[train_count:]


def _aggregate_seed_rows(seed_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
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
        "all_seeds_rank2_beats_position": {
            "value": all(value > 0.0 for value in output_improvements),
            "n": len(output_improvements),
        },
    }


def _status(seed_rows: list[dict[str, object]], aggregate: dict[str, dict[str, float]]) -> str:
    output = aggregate["output_rel_l2_improvement_vs_position"]
    all_positive = bool(aggregate["all_seeds_rank2_beats_position"]["value"])
    if all_positive and output["mean"] > 0.015:
        return "ALIVE but still weak; all-head rank-2 beats position-only across randomized split seeds."
    if output["mean"] > 0.0:
        return "WEAKLY ALIVE; mean split-seed improvement is positive but not stable across every seed."
    return "WEAKENED; all-head rank-2 does not beat position-only under split-seed repeats."


def _run(
    model_name: str,
    max_traces: int,
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    texts = _load_texts(DEFAULT_TRACES, max_traces)
    fit_samples, meta = _collect_fit_samples(model_name, texts, max_length, sink_tokens)
    seed_rows: list[dict[str, object]] = []
    modes = ("position", "rank2")
    for seed in seeds:
        predictors = {}
        train_splits = {}
        test_indices: dict[int, set[int]] = {}
        train_counts = {}
        test_counts = {}
        for layer, tensors in fit_samples.items():
            n_samples = int(tensors["q"].shape[0])
            train_idx, test_idx = _split_indices(n_samples, train_fraction, seed + 1009 * int(layer))
            train_splits[layer] = 0
            test_indices[layer] = set(int(idx) for idx in test_idx.tolist())
            train_counts[layer] = int(train_idx.numel())
            test_counts[layer] = int(test_idx.numel())
            predictors[layer] = _fit_layer_predictors(
                tensors["q"][train_idx],
                tensors["pos"][train_idx],
                tensors["sink_logits"][train_idx],
                ranks=(2,),
            )

        rows, head_rows = _evaluate_output_errors(
            model_name,
            texts,
            max_length,
            sink_tokens,
            predictors,
            train_splits,
            modes,
            eval_indices=test_indices,
        )
        summary = _summarize(rows)
        paired = _paired_head_improvements(head_rows)
        position = summary["position"]
        rank2 = summary["rank2"]
        seed_rows.append(
            {
                "seed": seed,
                "train_samples_per_layer_mean": mean(train_counts.values()),
                "test_samples_per_layer_mean": mean(test_counts.values()),
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
        "n_traces": len(texts),
        "n_samples": meta["n_samples"],
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
        "# SinkAware All-Rank2 Split/Seed Stability Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['n_traces']}",
        f"- max length: {result['max_length']}",
        f"- sink tokens: {result['sink_tokens']}",
        f"- train fraction: {result['train_fraction']}",
        f"- seeds: {', '.join(str(seed) for seed in result['seeds'])}",
        "",
        "This gate randomizes the token-level train/test split across seeds and evaluates all-head rank-2 against the position-only predictor. It keeps all non-sink QK scores exact and makes no GPU speed claim.",
        "",
        "## Aggregate Across Seeds",
        "",
        "| Metric | Mean improvement | 95% CI |",
        "|---|---:|---:|",
        f"| output rel-L2 vs position | {output['mean']:.4f} | +/- {output['ci95']:.4f} |",
        f"| sink-mass MAE vs position | {sink_mass['mean']:.4f} | +/- {sink_mass['ci95']:.4f} |",
        f"| attention L1 vs position | {attention['mean']:.4f} | +/- {attention['ci95']:.4f} |",
        f"| layer-head output win rate | {head_win['mean']:.3f} | +/- {head_win['ci95']:.3f} |",
        "",
        "Positive improvement means rank-2 has lower error than position-only.",
        "",
        "## Per Seed",
        "",
        "| Seed | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["seed_rows"]:
        position = row["summary"]["position"]
        rank2 = row["summary"]["rank2"]
        paired = row["paired_head_vs_position"]["rank2"]
        lines.append(
            "| {seed} | {position:.4f} | {rank2:.4f} | {improvement:+.4f} | {win_rate:.3f} |".format(
                seed=row["seed"],
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
            "The simple validation head selector remains ruled out. This gate only tests whether the all-head rank-2 row is repeatable enough to justify interpreter/GPU work.",
        ]
    )
    (OUT_DIR / "rank2_split_stability_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--max-traces", type=int, default=48)
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
    (OUT_DIR / "rank2_split_stability_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
