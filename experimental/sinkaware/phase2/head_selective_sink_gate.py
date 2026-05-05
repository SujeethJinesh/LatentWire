"""Head-selective SinkAware gate on Mac-local GPT2-style Q/K/V traces.

This gate tests whether the mixed per-head result can be made useful without a
GPU by selecting rank-2 sink-logit approximation only for heads where a
validation split beats the position-only predictor.  It then evaluates that
selection rule on a held-out test split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

try:
    from .real_qk_sink_softmax_output_probe import (
        METRIC_KEYS,
        OUT_DIR,
        PREDICTOR_MODES,
        ROOT,
        _collect_fit_samples,
        _evaluate_output_errors,
        _fit_layer_predictors,
        _load_texts,
        _summarize,
    )
    from .real_query_sink_probe import DEFAULT_TRACES
except ImportError:  # pragma: no cover - supports direct script execution.
    from real_qk_sink_softmax_output_probe import (
        METRIC_KEYS,
        OUT_DIR,
        PREDICTOR_MODES,
        ROOT,
        _collect_fit_samples,
        _evaluate_output_errors,
        _fit_layer_predictors,
        _load_texts,
        _summarize,
    )
    from real_query_sink_probe import DEFAULT_TRACES


def _select_heads(
    validation_heads: dict[int, dict[int, dict[str, dict[str, float]]]],
    *,
    metric: str = "output_rel_l2",
    baseline: str = "position",
    candidate: str = "rank2",
    min_improvement: float = 0.0,
) -> dict[int, set[int]]:
    selected: dict[int, set[int]] = {}
    for layer, by_head in validation_heads.items():
        selected[layer] = set()
        for head, by_mode in by_head.items():
            improvement = by_mode[baseline][metric] - by_mode[candidate][metric]
            if improvement > min_improvement:
                selected[layer].add(head)
    return selected


def _mixed_summary(
    test_heads: dict[int, dict[int, dict[str, dict[str, float]]]],
    selected: dict[int, set[int]],
    *,
    baseline: str = "position",
    candidate: str = "rank2",
) -> dict[str, float]:
    values = {metric: [] for metric in METRIC_KEYS}
    for layer, by_head in test_heads.items():
        for head, by_mode in by_head.items():
            mode = candidate if head in selected.get(layer, set()) else baseline
            for metric in METRIC_KEYS:
                values[metric].append(by_mode[mode][metric])
    return {metric: mean(metric_values) for metric, metric_values in values.items()}


def _count_selected(selected: dict[int, set[int]]) -> int:
    return sum(len(heads) for heads in selected.values())


def _run(
    model_name: str,
    max_traces: int,
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, object]:
    texts = _load_texts(DEFAULT_TRACES, max_traces)
    fit_samples, meta = _collect_fit_samples(model_name, texts, max_length, sink_tokens)
    ranks = (1, 2, 4, 8)
    predictors = {}
    train_splits = {}
    eval_ranges_validation = {}
    eval_ranges_test = {}
    for layer, tensors in fit_samples.items():
        n = int(tensors["q"].shape[0])
        train_end = max(8, int(train_fraction * n))
        validation_end = max(train_end + 1, int((train_fraction + validation_fraction) * n))
        validation_end = min(validation_end, n - 1)
        train_splits[layer] = train_end
        eval_ranges_validation[layer] = (train_end, validation_end)
        eval_ranges_test[layer] = (validation_end, None)
        predictors[layer] = _fit_layer_predictors(
            tensors["q"][:train_end],
            tensors["pos"][:train_end],
            tensors["sink_logits"][:train_end],
            ranks,
        )

    validation_rows, validation_heads = _evaluate_output_errors(
        model_name,
        texts,
        max_length,
        sink_tokens,
        predictors,
        train_splits,
        PREDICTOR_MODES,
        eval_ranges=eval_ranges_validation,
    )
    test_rows, test_heads = _evaluate_output_errors(
        model_name,
        texts,
        max_length,
        sink_tokens,
        predictors,
        train_splits,
        PREDICTOR_MODES,
        eval_ranges=eval_ranges_test,
    )
    selected = _select_heads(validation_heads)
    validation_summary = _summarize(validation_rows)
    test_summary = _summarize(test_rows)
    mixed = _mixed_summary(test_heads, selected)
    position = test_summary["position"]
    rank2 = test_summary["rank2"]
    selected_count = _count_selected(selected)
    total_heads = sum(len(by_head) for by_head in test_heads.values())
    improvement = position["output_rel_l2"] - mixed["output_rel_l2"]
    if improvement >= 0.015 and mixed["output_rel_l2"] <= rank2["output_rel_l2"]:
        status = "ALIVE; validation-selected rank-2 heads improve held-out output drift over both position-only and all-rank2."
    elif improvement > 0:
        status = "WEAKLY ALIVE; validation-selected rank-2 heads improve held-out output drift over position-only but do not dominate all-rank2."
    else:
        status = "WEAKENED; validation-selected rank-2 heads do not improve held-out output drift."
    return {
        "model_name": model_name,
        "n_traces": len(texts),
        "n_samples": meta["n_samples"],
        "sink_tokens": sink_tokens,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "selected_rank2_heads": selected_count,
        "total_layer_heads": total_heads,
        "selected_fraction": selected_count / total_heads,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "test_head_selective_rank2": mixed,
        "heldout_output_rel_l2_improvement_vs_position": improvement,
        "heldout_output_rel_l2_margin_vs_all_rank2": rank2["output_rel_l2"] - mixed["output_rel_l2"],
        "selected_heads": {str(layer): sorted(heads) for layer, heads in selected.items()},
        "status": status,
    }


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# SinkAware Head-Selective Rank-2 Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['n_traces']}",
        f"- sink tokens: {result['sink_tokens']}",
        f"- selected rank-2 heads: {result['selected_rank2_heads']} / {result['total_layer_heads']} ({result['selected_fraction']:.3f})",
        "",
        "The rule fits predictors on the train split, selects rank-2 heads on a validation split when rank-2 beats position-only on output rel-L2, then evaluates the mixed policy on a held-out split.",
        "",
        "| Test policy | Sink-logit RMSE | Sink-mass MAE | Attention L1 | Output rel-L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in [
        ("position", result["test_summary"]["position"]),
        ("rank2_all_heads", result["test_summary"]["rank2"]),
        ("rank2_validation_selected", result["test_head_selective_rank2"]),
    ]:
        lines.append(
            "| {name} | {sink_logit_rmse:.4f} | {sink_mass_mae:.4f} | {attention_l1:.4f} | {output_rel_l2:.4f} |".format(
                name=name,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- held-out output rel-L2 improvement vs position-only: `{result['heldout_output_rel_l2_improvement_vs_position']:+.4f}`",
            f"- held-out output rel-L2 margin vs all-rank2: `{result['heldout_output_rel_l2_margin_vs_all_rank2']:+.4f}` (positive means selected is better)",
            "",
            "A GPU paper should not claim speed from this result. It only decides whether a head-selective approximation is worth native implementation.",
        ]
    )
    (OUT_DIR / "head_selective_sink_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--max-traces", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
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
        args.validation_fraction,
    )
    (OUT_DIR / "head_selective_sink_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "selected_rank2_heads": result["selected_rank2_heads"]}, indent=2))


if __name__ == "__main__":
    main()
