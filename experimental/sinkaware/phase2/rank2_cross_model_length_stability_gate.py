"""Cross-model length stability gate for all-head rank-2 SinkAware.

This Mac-local gate broadens the held-out/cross-family falsification row by
running the same whole-trace split protocol at multiple sequence lengths. It
keeps predictors per model and per length, so this is not cross-model transfer,
downstream quality, or a speed claim.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any

try:
    from .rank2_cross_model_falsification_gate import _run as _run_cross_model_gate
    from .real_qk_sink_softmax_output_probe import OUT_DIR, ROOT, _mean_ci95
except ImportError:  # pragma: no cover - supports direct script execution.
    from rank2_cross_model_falsification_gate import _run as _run_cross_model_gate
    from real_qk_sink_softmax_output_probe import OUT_DIR, ROOT, _mean_ci95


def _length_summary(length_result: dict[str, Any]) -> dict[str, Any]:
    model_summaries = []
    for model_row in length_result["model_results"]:
        aggregate = model_row["aggregate"]
        position_outputs = [
            float(seed_row["summary"]["position"]["output_rel_l2"])
            for seed_row in model_row["seed_rows"]
        ]
        rank2_outputs = [
            float(seed_row["summary"]["rank2"]["output_rel_l2"])
            for seed_row in model_row["seed_rows"]
        ]
        model_summaries.append(
            {
                "model_name": model_row["model_name"],
                "model_family": model_row["model_family"],
                "position_output_rel_l2_mean": mean(position_outputs),
                "rank2_output_rel_l2_mean": mean(rank2_outputs),
                "output_rel_l2_improvement": aggregate["output_rel_l2_improvement_vs_position"],
                "sink_mass_mae_improvement": aggregate["sink_mass_mae_improvement_vs_position"],
                "attention_l1_improvement": aggregate["attention_l1_improvement_vs_position"],
                "output_rel_l2_head_win_rate": aggregate["output_rel_l2_head_win_rate"],
                "all_seeds_rank2_beats_position": aggregate["all_seeds_rank2_beats_position"]["value"],
                "min_seed_output_rel_l2_improvement": aggregate["min_output_rel_l2_improvement"]["value"],
            }
        )
    output_means = [
        float(row["output_rel_l2_improvement"]["mean"])
        for row in model_summaries
    ]
    return {
        "max_length": length_result["max_length"],
        "model_summaries": model_summaries,
        "output_rel_l2_improvement_across_models": _mean_ci95(output_means),
        "all_models_positive": all(value > 0.0 for value in output_means),
        "min_model_output_rel_l2_improvement": min(output_means) if output_means else float("nan"),
    }


def _aggregate_lengths(length_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    model_length_rows = [
        model_row
        for length_row in length_summaries
        for model_row in length_row["model_summaries"]
    ]
    output_improvements = [
        float(row["output_rel_l2_improvement"]["mean"])
        for row in model_length_rows
    ]
    head_win_rates = [
        float(row["output_rel_l2_head_win_rate"]["mean"])
        for row in model_length_rows
    ]
    return {
        "output_rel_l2_improvement_across_model_lengths": _mean_ci95(output_improvements),
        "output_rel_l2_head_win_rate_across_model_lengths": _mean_ci95(head_win_rates),
        "all_model_lengths_positive": {
            "value": all(value > 0.0 for value in output_improvements),
            "n": len(output_improvements),
        },
        "all_seeds_positive": {
            "value": all(bool(row["all_seeds_rank2_beats_position"]) for row in model_length_rows),
            "n": len(model_length_rows),
        },
        "min_model_length_output_rel_l2_improvement": {
            "value": min(output_improvements) if output_improvements else float("nan"),
        },
    }


def _status(aggregate: dict[str, Any]) -> str:
    all_rows_positive = bool(aggregate["all_model_lengths_positive"]["value"])
    all_seeds_positive = bool(aggregate["all_seeds_positive"]["value"])
    min_improvement = float(aggregate["min_model_length_output_rel_l2_improvement"]["value"])
    output = aggregate["output_rel_l2_improvement_across_model_lengths"]
    if all_rows_positive and all_seeds_positive and min_improvement > 0.015 and output["mean"] > 0.02:
        return "ALIVE but bounded; rank-2 survives cross-family length stability."
    if all_rows_positive:
        return "WEAKLY ALIVE; every model/length row is positive, but at least one row is below the promotion margin."
    return "WEAKENED; rank-2 fails at least one model/length row."


def _run(
    model_names: tuple[str, ...],
    max_traces: int,
    max_lengths: tuple[int, ...],
    sink_tokens: int,
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    length_results = [
        _run_cross_model_gate(model_names, max_traces, max_length, sink_tokens, train_fraction, seeds)
        for max_length in max_lengths
    ]
    length_summaries = [_length_summary(row) for row in length_results]
    aggregate = _aggregate_lengths(length_summaries)
    return {
        "model_names": list(model_names),
        "max_traces": max_traces,
        "max_lengths": list(max_lengths),
        "sink_tokens": sink_tokens,
        "train_fraction": train_fraction,
        "seeds": list(seeds),
        "length_summaries": length_summaries,
        "length_results": length_results,
        "aggregate": aggregate,
        "status": _status(aggregate),
    }


def _write_markdown(result: dict[str, Any]) -> None:
    aggregate = result["aggregate"]
    output = aggregate["output_rel_l2_improvement_across_model_lengths"]
    head_win = aggregate["output_rel_l2_head_win_rate_across_model_lengths"]
    lines = [
        "# SinkAware Rank-2 Cross-Model Length Stability Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- models: {', '.join(f'`{name}`' for name in result['model_names'])}",
        f"- traces: {result['max_traces']}",
        f"- max lengths: {', '.join(str(value) for value in result['max_lengths'])}",
        f"- sink tokens: {result['sink_tokens']}",
        f"- train fraction: {result['train_fraction']}",
        f"- seeds: {', '.join(str(seed) for seed in result['seeds'])}",
        "",
        "This gate broadens the held-out/cross-family falsification row across sequence lengths. It fits all-head rank-2 predictors separately per model and length on whole-trace train splits, evaluates held-out traces against a position-only predictor, keeps non-sink QK scores exact, and makes no GPU speed or downstream-quality claim.",
        "",
        "## Aggregate Across Model/Length Rows",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---:|---:|",
        f"| output rel-L2 improvement vs position | {output['mean']:.4f} | +/- {output['ci95']:.4f} |",
        f"| layer-head output win rate | {head_win['mean']:.3f} | +/- {head_win['ci95']:.3f} |",
        f"| minimum model/length output improvement | {aggregate['min_model_length_output_rel_l2_improvement']['value']:.4f} | |",
        f"| model/length rows positive | {aggregate['all_model_lengths_positive']['n']} / {aggregate['all_model_lengths_positive']['n'] if aggregate['all_model_lengths_positive']['value'] else 'mixed'} | |",
        "",
        "Positive improvement means rank-2 has lower error than position-only.",
        "",
        "## Per Model And Length",
        "",
        "| Max length | Model | Family | Position output rel-L2 | Rank2 output rel-L2 | Improvement | 95% CI | Head win rate | All seeds positive? |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for length_row in result["length_summaries"]:
        for model_row in length_row["model_summaries"]:
            output = model_row["output_rel_l2_improvement"]
            head_win = model_row["output_rel_l2_head_win_rate"]
            lines.append(
                "| {max_length} | {model} | {family} | {position:.4f} | {rank2:.4f} | {improvement:+.4f} | +/- {ci95:.4f} | {head_win:.3f} | {all_positive} |".format(
                    max_length=length_row["max_length"],
                    model=model_row["model_name"],
                    family=model_row["model_family"],
                    position=model_row["position_output_rel_l2_mean"],
                    rank2=model_row["rank2_output_rel_l2_mean"],
                    improvement=output["mean"],
                    ci95=output["ci95"],
                    head_win=head_win["mean"],
                    all_positive="yes" if model_row["all_seeds_rank2_beats_position"] else "no",
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This gate strengthens the Mac-local decision surface by requiring both GPT2-family and OPT-family rows to stay positive across length. It remains bounded attention-output drift evidence only; it does not establish predictor transfer, GPU speed, or downstream quality preservation.",
        ]
    )
    (OUT_DIR / "rank2_cross_model_length_stability_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", nargs="+", default=["distilgpt2", "facebook/opt-125m"])
    parser.add_argument("--max-traces", type=int, default=48)
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[64, 96])
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
        tuple(args.model_names),
        args.max_traces,
        tuple(args.max_lengths),
        args.sink_tokens,
        args.train_fraction,
        tuple(args.seeds),
    )
    (OUT_DIR / "rank2_cross_model_length_stability_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
