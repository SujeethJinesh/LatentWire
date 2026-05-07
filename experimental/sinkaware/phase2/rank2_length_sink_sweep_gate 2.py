"""Sequence-length and sink-count sweep for all-head rank-2 SinkAware.

This is the next Mac-local gate after the split/seed repeat. It keeps the
method fixed as all-head rank-2 and asks whether the positive aggregate result
survives small changes in sequence length and fixed sink-token count.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean

try:
    from .rank2_split_stability_gate import _run as _run_split_gate
    from .real_qk_sink_softmax_output_probe import OUT_DIR, ROOT, _mean_ci95
except ImportError:  # pragma: no cover - supports direct script execution.
    from rank2_split_stability_gate import _run as _run_split_gate
    from real_qk_sink_softmax_output_probe import OUT_DIR, ROOT, _mean_ci95


def _config_summary(config_result: dict[str, object]) -> dict[str, object]:
    seed_rows = config_result["seed_rows"]
    position_outputs = [
        float(row["summary"]["position"]["output_rel_l2"])
        for row in seed_rows
    ]
    rank2_outputs = [
        float(row["summary"]["rank2"]["output_rel_l2"])
        for row in seed_rows
    ]
    aggregate = config_result["aggregate"]
    return {
        "max_length": config_result["max_length"],
        "sink_tokens": config_result["sink_tokens"],
        "position_output_rel_l2_mean": mean(position_outputs),
        "rank2_output_rel_l2_mean": mean(rank2_outputs),
        "output_rel_l2_improvement": aggregate["output_rel_l2_improvement_vs_position"],
        "sink_mass_mae_improvement": aggregate["sink_mass_mae_improvement_vs_position"],
        "attention_l1_improvement": aggregate["attention_l1_improvement_vs_position"],
        "output_rel_l2_head_win_rate": aggregate["output_rel_l2_head_win_rate"],
        "all_seeds_rank2_beats_position": aggregate["all_seeds_rank2_beats_position"]["value"],
    }


def _aggregate_configs(config_summaries: list[dict[str, object]]) -> dict[str, object]:
    output_improvements = [
        float(row["output_rel_l2_improvement"]["mean"])
        for row in config_summaries
    ]
    head_win_rates = [
        float(row["output_rel_l2_head_win_rate"]["mean"])
        for row in config_summaries
    ]
    return {
        "output_rel_l2_improvement_across_configs": _mean_ci95(output_improvements),
        "output_rel_l2_head_win_rate_across_configs": _mean_ci95(head_win_rates),
        "configs_all_seeds_positive": {
            "value": all(bool(row["all_seeds_rank2_beats_position"]) for row in config_summaries),
            "n": len(config_summaries),
        },
        "min_output_rel_l2_improvement": min(output_improvements),
    }


def _status(config_summaries: list[dict[str, object]], aggregate: dict[str, object]) -> str:
    all_positive = bool(aggregate["configs_all_seeds_positive"]["value"])
    min_improvement = float(aggregate["min_output_rel_l2_improvement"])
    if all_positive and min_improvement > 0.015:
        return "ALIVE but bounded; all-head rank-2 beats position-only across the length/sink sweep."
    if min_improvement > 0.0:
        return "WEAKLY ALIVE; all configs are positive on average, but at least one is below the margin."
    return "WEAKENED; all-head rank-2 does not survive the length/sink sweep."


def _run(
    model_name: str,
    max_traces: int,
    max_lengths: tuple[int, ...],
    sink_tokens: tuple[int, ...],
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    config_results = []
    config_summaries = []
    for max_length in max_lengths:
        for sink_count in sink_tokens:
            config_result = _run_split_gate(
                model_name,
                max_traces,
                max_length,
                sink_count,
                train_fraction,
                seeds,
            )
            config_results.append(config_result)
            config_summaries.append(_config_summary(config_result))
    aggregate = _aggregate_configs(config_summaries)
    return {
        "model_name": model_name,
        "max_traces": max_traces,
        "max_lengths": list(max_lengths),
        "sink_tokens": list(sink_tokens),
        "train_fraction": train_fraction,
        "seeds": list(seeds),
        "config_summaries": config_summaries,
        "config_results": config_results,
        "aggregate": aggregate,
        "status": _status(config_summaries, aggregate),
    }


def _write_markdown(result: dict[str, object]) -> None:
    aggregate = result["aggregate"]
    output = aggregate["output_rel_l2_improvement_across_configs"]
    head_win = aggregate["output_rel_l2_head_win_rate_across_configs"]
    lines = [
        "# SinkAware All-Rank2 Length/Sink Sweep",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- max traces: {result['max_traces']}",
        f"- max lengths: {', '.join(str(value) for value in result['max_lengths'])}",
        f"- sink tokens: {', '.join(str(value) for value in result['sink_tokens'])}",
        f"- train fraction: {result['train_fraction']}",
        f"- seeds: {', '.join(str(seed) for seed in result['seeds'])}",
        "",
        "This sweep keeps the method fixed as all-head rank-2 and reuses the randomized token split gate for each configuration. It keeps all non-sink QK scores exact and makes no GPU speed claim.",
        "",
        "## Aggregate Across Configurations",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---:|---:|",
        f"| output rel-L2 improvement vs position | {output['mean']:.4f} | +/- {output['ci95']:.4f} |",
        f"| layer-head output win rate | {head_win['mean']:.3f} | +/- {head_win['ci95']:.3f} |",
        f"| minimum config output improvement | {aggregate['min_output_rel_l2_improvement']:.4f} | |",
        "",
        "Positive improvement means rank-2 has lower error than position-only.",
        "",
        "## Per Configuration",
        "",
        "| Max length | Sink tokens | Position output rel-L2 | Rank2 output rel-L2 | Improvement | 95% CI | Head win rate | All seeds positive? |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["config_summaries"]:
        output = row["output_rel_l2_improvement"]
        head_win = row["output_rel_l2_head_win_rate"]
        lines.append(
            "| {max_length} | {sink_tokens} | {position:.4f} | {rank2:.4f} | {improvement:.4f} | +/- {ci95:.4f} | {head_win:.3f} | {all_positive} |".format(
                max_length=row["max_length"],
                sink_tokens=row["sink_tokens"],
                position=row["position_output_rel_l2_mean"],
                rank2=row["rank2_output_rel_l2_mean"],
                improvement=output["mean"],
                ci95=output["ci95"],
                head_win=head_win["mean"],
                all_positive="yes" if row["all_seeds_rank2_beats_position"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The exact static branch remains killed, and simple validation head selection remains ruled out. This sweep only decides whether all-head rank-2 quality is stable enough to justify interpreter/GPU work.",
        ]
    )
    (OUT_DIR / "rank2_length_sink_sweep_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--max-traces", type=int, default=48)
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--sink-tokens", type=int, nargs="+", default=[2, 4])
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
        tuple(args.max_lengths),
        tuple(args.sink_tokens),
        args.train_fraction,
        tuple(args.seeds),
    )
    (OUT_DIR / "rank2_length_sink_sweep_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
