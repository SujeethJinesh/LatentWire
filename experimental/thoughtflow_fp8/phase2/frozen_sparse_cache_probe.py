"""Frozen CPU sparse-KV quality probe for ThoughtFlow-FP8.

This gate evaluates only pre-selected policies on a larger saved-trace slice.
It does not tune, sweep, or select policies on this slice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .kv_drop_quality_probe import (
        SparseSweepConfig,
        _make_sparse_sweep_policy,
        _paired_deltas,
        _prepare_rows,
        _prune_cache,
        _summary,
    )
    from .perplexity_impact_proxy import POLICIES, thoughtflow_saliency_recent
    from .run_real_trace_retention import OUT_DIR, ROOT
    from .simulate_phase_retention import Token, rkv_like, thin_kv_like
except ImportError:  # pragma: no cover - supports direct script execution.
    from kv_drop_quality_probe import (
        SparseSweepConfig,
        _make_sparse_sweep_policy,
        _paired_deltas,
        _prepare_rows,
        _prune_cache,
        _summary,
    )
    from perplexity_impact_proxy import POLICIES, thoughtflow_saliency_recent
    from run_real_trace_retention import OUT_DIR, ROOT
    from simulate_phase_retention import Token, rkv_like, thin_kv_like


FROZEN_SPARSE_CONFIG = SparseSweepConfig(
    recent_fraction=0.55,
    phase_bonus=0.05,
    math_bonus=0.12,
    protect_anchors=2,
)
FROZEN_SPARSE_POLICY_NAME = FROZEN_SPARSE_CONFIG.name


def _frozen_policies():
    return {
        "rkv_like": rkv_like,
        "thin_kv_like": thin_kv_like,
        "longflow_like": POLICIES["longflow_like"],
        "thoughtflow_saliency_recent": thoughtflow_saliency_recent,
        FROZEN_SPARSE_POLICY_NAME: _make_sparse_sweep_policy(FROZEN_SPARSE_CONFIG),
    }


def _score_continuation_with_cache(
    model: AutoModelForCausalLM,
    prefix_outputs,
    prefix_len: int,
    continuation_ids: list[int],
    kept: set[int],
) -> tuple[float, int]:
    continuation = torch.tensor([continuation_ids], dtype=torch.long)
    pruned_cache = _prune_cache(prefix_outputs.past_key_values, kept)
    position_ids = torch.arange(prefix_len, prefix_len + len(continuation_ids)).reshape(1, -1)
    with torch.no_grad():
        outputs = model(
            input_ids=continuation,
            past_key_values=pruned_cache,
            position_ids=position_ids,
            labels=continuation,
            use_cache=False,
        )
    scored_tokens = max(0, len(continuation_ids) - 1)
    return float(outputs.loss.item()), scored_tokens


def _run(
    model_name: str,
    keep_fraction: float,
    max_traces: int,
    max_length: int,
    continuation_tokens: int,
) -> dict[str, object]:
    cache_dir = ROOT / "experimental/thoughtflow_fp8/.debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    prepared = _prepare_rows(tokenizer, max_traces, max_length, continuation_tokens)
    policies = _frozen_policies()
    rows = []
    with torch.no_grad():
        for row in prepared:
            prefix_ids = row["prefix_ids"]
            continuation_ids = row["continuation_ids"]
            trace = row["trace"]
            assert isinstance(prefix_ids, list)
            assert isinstance(continuation_ids, list)
            assert isinstance(trace, list)

            prefix = torch.tensor([prefix_ids], dtype=torch.long)
            prefix_outputs = model(input_ids=prefix, use_cache=True)
            full_kept = set(range(len(prefix_ids)))
            full_loss, scored_tokens = _score_continuation_with_cache(
                model,
                prefix_outputs,
                len(prefix_ids),
                continuation_ids,
                full_kept,
            )
            rows.append(
                {
                    "trace_id": row["trace_id"],
                    "policy": "full_cache",
                    "keep_rate": 1.0,
                    "retained_prefix_tokens": len(prefix_ids),
                    "continuation_tokens": scored_tokens,
                    "nll": full_loss,
                    "delta_nll_vs_full": 0.0,
                }
            )
            budget = max(1, math.ceil(len(prefix_ids) * keep_fraction))
            for name, policy in policies.items():
                kept = policy(trace, budget)
                loss, scored_tokens = _score_continuation_with_cache(
                    model,
                    prefix_outputs,
                    len(prefix_ids),
                    continuation_ids,
                    kept,
                )
                rows.append(
                    {
                        "trace_id": row["trace_id"],
                        "policy": name,
                        "keep_rate": len(kept) / len(prefix_ids),
                        "retained_prefix_tokens": len(kept),
                        "continuation_tokens": scored_tokens,
                        "nll": loss,
                        "delta_nll_vs_full": loss - full_loss,
                    }
                )
    summary = _summary(rows)
    paired_vs_rkv = _paired_deltas(rows, baseline_policy="rkv_like")
    paired_vs_thin = _paired_deltas(rows, baseline_policy="thin_kv_like")
    return {
        "model_name": model_name,
        "keep_fraction": keep_fraction,
        "max_traces": max_traces,
        "max_length": max_length,
        "continuation_tokens": continuation_tokens,
        "n_scored_traces": int(summary.get("full_cache", {}).get("n_traces", 0)),
        "frozen_policy_names": ["thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME],
        "rows": rows,
        "summary": summary,
        "paired_delta_nll_vs_rkv_like": paired_vs_rkv,
        "paired_delta_nll_vs_thin_kv_like": paired_vs_thin,
        "status": _status(summary, paired_vs_rkv, paired_vs_thin),
    }


def _best_frozen_policy(summary: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    candidates = {
        policy: summary[policy]
        for policy in ("thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME)
        if policy in summary
    }
    return min(candidates.items(), key=lambda item: item[1]["nll"])


def _status(
    summary: dict[str, dict[str, float]],
    paired_vs_rkv: dict[str, dict[str, float]],
    paired_vs_thin: dict[str, dict[str, float]],
) -> str:
    best_name, best_metrics = _best_frozen_policy(summary)
    strongest_other_name, strongest_other = min(
        ((policy, metrics) for policy, metrics in summary.items() if policy not in {"full_cache", "thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME}),
        key=lambda item: item[1]["nll"],
    )
    margin = strongest_other["nll"] - best_metrics["nll"]
    rkv_ci_high = paired_vs_rkv.get(best_name, {}).get("ci95_high", float("inf"))
    thin_ci_high = paired_vs_thin.get(best_name, {}).get("ci95_high", float("inf"))
    if margin >= 0.03 and rkv_ci_high < 0.0 and thin_ci_high < 0.0:
        return f"ALIVE on frozen sparse-cache probe; {best_name} beats {strongest_other_name} by {margin:.3f} NLL with paired CIs."
    if margin >= 0.03:
        return f"MIXED on frozen sparse-cache probe; {best_name} beats {strongest_other_name} by {margin:.3f} NLL but paired uncertainty remains."
    return f"MIXED on frozen sparse-cache probe; {best_name} remains inside 0.03 NLL vs {strongest_other_name}."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 Frozen Sparse-Cache Probe",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- scored traces: {result['n_scored_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        f"- continuation tokens: {result['continuation_tokens']}",
        f"- frozen ThoughtFlow policies: `{', '.join(result['frozen_policy_names'])}`",
        "",
        "This larger slice freezes the two current ThoughtFlow candidates and performs no policy selection or retuning.",
        "The model processes the full prefix once per trace, prunes the returned KV cache, and scores the continuation from the sparse cache.",
        "",
        "| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, metrics in sorted(result["summary"].items(), key=lambda item: item[1]["nll"]):
        lines.append(
            "| {policy} | {n_traces:.0f} | {keep_rate:.3f} | {nll:.3f} | {delta_nll_vs_full:.3f} |".format(
                policy=policy,
                **metrics,
            )
        )
    for section_title, key, baseline, metric in (
        ("Paired Delta vs R-KV-like", "paired_delta_nll_vs_rkv_like", "rkv_like", "mean_delta_nll_minus_rkv_like"),
        ("Paired Delta vs ThinKV-like", "paired_delta_nll_vs_thin_kv_like", "thin_kv_like", "mean_delta_nll_minus_thin_kv_like"),
    ):
        lines.extend(
            [
                "",
                f"## {section_title}",
                "",
                f"Negative means lower continuation NLL than {baseline} on the same trace.",
                "",
                "| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for policy, metrics in sorted(result[key].items(), key=lambda item: item[1][metric]):
            lines.append(
                "| {policy} | {n_pairs:.0f} | {mean:+.3f} | {low:+.3f} | {high:+.3f} |".format(
                    policy=policy,
                    n_pairs=metrics["n_pairs"],
                    mean=metrics[metric],
                    low=metrics["ci95_low"],
                    high=metrics["ci95_high"],
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Promote only if a frozen ThoughtFlow policy beats both R-KV-like and ThinKV-like by at least 0.03 NLL with paired CIs below zero.",
        ]
    )
    (OUT_DIR / "frozen_sparse_cache_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=74)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run(args.model_name, args.keep_fraction, args.max_traces, args.max_length, args.continuation_tokens)
    (OUT_DIR / "frozen_sparse_cache_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
