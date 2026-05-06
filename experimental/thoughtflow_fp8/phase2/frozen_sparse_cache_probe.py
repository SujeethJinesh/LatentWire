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
RDU_POLICY_NAME = "rdu_topk"
RDU_SECOND_BUCKET_WEIGHT = 0.5
RDU_BUCKETS = (
    ("b0_8_15", 8, 15),
    ("b1_16_31", 16, 31),
    ("b2_32_63", 32, 63),
    ("b3_64_inf", 64, None),
)
RDU_LABELS = ("anchor", "phase", "math_state")


def _frozen_policies():
    return {
        "rkv_like": rkv_like,
        "thin_kv_like": thin_kv_like,
        "longflow_like": POLICIES["longflow_like"],
        "thoughtflow_saliency_recent": thoughtflow_saliency_recent,
        FROZEN_SPARSE_POLICY_NAME: _make_sparse_sweep_policy(FROZEN_SPARSE_CONFIG),
    }


def _rdu_scores_from_attentions(attentions) -> list[dict[str, object]]:
    if not attentions:
        raise ValueError("rdu_topk requires prefill attentions")
    if any(attention is None for attention in attentions):
        raise ValueError("rdu_topk requires concrete prefill attention tensors; use an eager attention backend")
    layer_attentions = [attention.detach().float().squeeze(0) for attention in attentions]
    stacked = torch.stack(layer_attentions, dim=0)
    if stacked.ndim != 4:
        raise ValueError(f"expected stacked attentions with shape [layers, heads, q, i], got {tuple(stacked.shape)}")
    layer_count, head_count, prefix_len, key_len = stacked.shape
    if prefix_len != key_len:
        raise ValueError(f"expected square prefill attention, got query={prefix_len} key={key_len}")

    scores = []
    for idx in range(prefix_len):
        masses: dict[str, float] = {}
        for bucket_name, min_lag, max_lag in RDU_BUCKETS:
            q_start = idx + min_lag
            q_end = prefix_len - 1 if max_lag is None else min(prefix_len - 1, idx + max_lag)
            query_count = max(0, q_end - q_start + 1)
            if query_count == 0:
                mass = 0.0
            else:
                numerator = stacked[:, :, q_start : q_end + 1, idx].sum().item()
                normalizer = max(1.0, math.sqrt(query_count * layer_count * head_count))
                mass = float(numerator / normalizer)
            masses[bucket_name] = mass
        ranked_masses = sorted(masses.items(), key=lambda item: item[1], reverse=True)
        top_mass = ranked_masses[0][1]
        second_mass = ranked_masses[1][1] if len(ranked_masses) > 1 else 0.0
        primary_bucket = ranked_masses[0][0] if top_mass > 0.0 else "none"
        scores.append(
            {
                "index": idx,
                "rdu": top_mass + RDU_SECOND_BUCKET_WEIGHT * second_mass,
                "bucket_masses": masses,
                "primary_bucket": primary_bucket,
            }
        )
    return scores


def _rdu_topk_from_attentions(attentions, budget: int) -> tuple[set[int], list[dict[str, object]]]:
    scores = _rdu_scores_from_attentions(attentions)
    budget = min(max(0, budget), len(scores))
    ranked = sorted(scores, key=lambda item: (-float(item["rdu"]), int(item["index"])))
    return {int(item["index"]) for item in ranked[:budget]}, scores


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _rdu_retention_telemetry(trace: list[Token], kept: set[int], scores: list[dict[str, object]]) -> dict[str, object]:
    labels = {}
    for label in RDU_LABELS:
        total = sum(1 for token in trace if token.label == label)
        retained = sum(1 for idx, token in enumerate(trace) if token.label == label and idx in kept)
        labels[label] = {
            "total": float(total),
            "retained": float(retained),
            "retention_rate": _rate(retained, total),
        }

    bucket_names = [bucket[0] for bucket in RDU_BUCKETS] + ["none"]
    recurrence_buckets = {}
    for bucket_name in bucket_names:
        indices = [idx for idx, score in enumerate(scores) if score["primary_bucket"] == bucket_name]
        retained = sum(1 for idx in indices if idx in kept)
        recurrence_buckets[bucket_name] = {
            "total": float(len(indices)),
            "retained": float(retained),
            "retention_rate": _rate(retained, len(indices)),
        }

    rdu_values = [float(score["rdu"]) for score in scores]
    return {
        "labels": labels,
        "recurrence_buckets": recurrence_buckets,
        "rdu_summary": {
            "mean_rdu": float(sum(rdu_values) / len(rdu_values)) if rdu_values else 0.0,
            "max_rdu": max(rdu_values) if rdu_values else 0.0,
            "nonzero_tokens": float(sum(1 for value in rdu_values if value > 0.0)),
        },
    }


def _aggregate_rdu_telemetry(per_trace: list[dict[str, object]]) -> dict[str, object]:
    aggregate = {
        "labels": {label: {"total": 0.0, "retained": 0.0, "retention_rate": 0.0} for label in RDU_LABELS},
        "recurrence_buckets": {
            bucket_name: {"total": 0.0, "retained": 0.0, "retention_rate": 0.0}
            for bucket_name in [bucket[0] for bucket in RDU_BUCKETS] + ["none"]
        },
        "rdu_summary": {"mean_rdu": 0.0, "max_rdu": 0.0, "nonzero_tokens": 0.0},
    }
    if not per_trace:
        return aggregate

    for item in per_trace:
        telemetry = item["telemetry"]
        for section_name in ("labels", "recurrence_buckets"):
            for key, metrics in telemetry[section_name].items():
                aggregate[section_name][key]["total"] += float(metrics["total"])
                aggregate[section_name][key]["retained"] += float(metrics["retained"])
        aggregate["rdu_summary"]["mean_rdu"] += float(telemetry["rdu_summary"]["mean_rdu"])
        aggregate["rdu_summary"]["max_rdu"] = max(
            aggregate["rdu_summary"]["max_rdu"],
            float(telemetry["rdu_summary"]["max_rdu"]),
        )
        aggregate["rdu_summary"]["nonzero_tokens"] += float(telemetry["rdu_summary"]["nonzero_tokens"])

    for section_name in ("labels", "recurrence_buckets"):
        for metrics in aggregate[section_name].values():
            metrics["retention_rate"] = _rate(int(metrics["retained"]), int(metrics["total"]))
    aggregate["rdu_summary"]["mean_rdu"] /= len(per_trace)
    return aggregate


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


def _load_model_for_prefill_attentions(model_name: str) -> AutoModelForCausalLM:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:  # pragma: no cover - older transformers versions.
        return AutoModelForCausalLM.from_pretrained(model_name)


def _run(
    model_name: str,
    keep_fraction: float,
    max_traces: int,
    max_length: int,
    continuation_tokens: int,
    trace_paths: list[Path] | None = None,
) -> dict[str, object]:
    cache_dir = ROOT / "experimental/thoughtflow_fp8/.debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_model_for_prefill_attentions(model_name)
    model.eval()
    prepared = _prepare_rows(tokenizer, max_traces, max_length, continuation_tokens, trace_paths)
    policies = _frozen_policies()
    rows = []
    rdu_per_trace = []
    with torch.no_grad():
        for row in prepared:
            prefix_ids = row["prefix_ids"]
            continuation_ids = row["continuation_ids"]
            trace = row["trace"]
            assert isinstance(prefix_ids, list)
            assert isinstance(continuation_ids, list)
            assert isinstance(trace, list)

            prefix = torch.tensor([prefix_ids], dtype=torch.long)
            prefix_outputs = model(input_ids=prefix, use_cache=True, output_attentions=True)
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
            rdu_kept, rdu_scores = _rdu_topk_from_attentions(prefix_outputs.attentions, budget)
            rdu_loss, scored_tokens = _score_continuation_with_cache(
                model,
                prefix_outputs,
                len(prefix_ids),
                continuation_ids,
                rdu_kept,
            )
            rows.append(
                {
                    "trace_id": row["trace_id"],
                    "policy": RDU_POLICY_NAME,
                    "keep_rate": len(rdu_kept) / len(prefix_ids),
                    "retained_prefix_tokens": len(rdu_kept),
                    "continuation_tokens": scored_tokens,
                    "nll": rdu_loss,
                    "delta_nll_vs_full": rdu_loss - full_loss,
                }
            )
            rdu_per_trace.append(
                {
                    "trace_id": row["trace_id"],
                    "telemetry": _rdu_retention_telemetry(trace, rdu_kept, rdu_scores),
                }
            )
    summary = _summary(rows)
    paired_vs_rkv = _paired_deltas(rows, baseline_policy="rkv_like")
    paired_vs_thin = _paired_deltas(rows, baseline_policy="thin_kv_like")
    return {
        "model_name": model_name,
        "input_paths": [
            str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)
            for path in (trace_paths or [])
        ],
        "keep_fraction": keep_fraction,
        "max_traces": max_traces,
        "max_length": max_length,
        "continuation_tokens": continuation_tokens,
        "n_scored_traces": int(summary.get("full_cache", {}).get("n_traces", 0)),
        "frozen_policy_names": ["thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME, RDU_POLICY_NAME],
        "rows": rows,
        "summary": summary,
        "paired_delta_nll_vs_rkv_like": paired_vs_rkv,
        "paired_delta_nll_vs_thin_kv_like": paired_vs_thin,
        "rdu_topk_telemetry": {
            "aggregate": _aggregate_rdu_telemetry(rdu_per_trace),
            "per_trace": rdu_per_trace,
        },
        "status": _status(summary, paired_vs_rkv, paired_vs_thin),
    }


def _best_frozen_policy(summary: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    candidates = {
        policy: summary[policy]
        for policy in ("thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME, RDU_POLICY_NAME)
        if policy in summary
    }
    return min(candidates.items(), key=lambda item: item[1]["nll"])


def _status(
    summary: dict[str, dict[str, float]],
    paired_vs_rkv: dict[str, dict[str, float]],
    paired_vs_thin: dict[str, dict[str, float]],
) -> str:
    if RDU_POLICY_NAME in summary:
        rdu_metrics = summary[RDU_POLICY_NAME]
        margin_vs_rkv = summary["rkv_like"]["nll"] - rdu_metrics["nll"]
        margin_vs_thin = summary["thin_kv_like"]["nll"] - rdu_metrics["nll"]
        rkv_ci_high = paired_vs_rkv.get(RDU_POLICY_NAME, {}).get("ci95_high", float("inf"))
        thin_ci_high = paired_vs_thin.get(RDU_POLICY_NAME, {}).get("ci95_high", float("inf"))
        if margin_vs_rkv >= 0.03 and margin_vs_thin >= 0.03 and rkv_ci_high < 0.0 and thin_ci_high < 0.0:
            return (
                "ALIVE on frozen sparse-cache probe; rdu_topk clears the preregistered "
                f"promotion rule with margins {margin_vs_rkv:.3f} vs R-KV-like and {margin_vs_thin:.3f} vs ThinKV-like."
            )
        return (
            "KILLED on frozen sparse-cache probe; rdu_topk fails the preregistered "
            f"promotion rule with margins {margin_vs_rkv:.3f} vs R-KV-like and {margin_vs_thin:.3f} vs ThinKV-like."
        )

    best_name, best_metrics = _best_frozen_policy(summary)
    strongest_other_name, strongest_other = min(
        (
            (policy, metrics)
            for policy, metrics in summary.items()
            if policy not in {"full_cache", "thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME}
        ),
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
        "This larger slice freezes the stopped ThoughtFlow candidates plus the one pre-registered `rdu_topk` successor and performs no policy selection or retuning.",
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
    if "rdu_topk_telemetry" in result:
        telemetry = result["rdu_topk_telemetry"]["aggregate"]
        lines.extend(
            [
                "",
                "## RDU Top-K Telemetry",
                "",
                "Telemetry is reported after selection and is not used by the policy.",
                "",
                "### Label Retention",
                "",
                "| Label | Total | Retained | Retention rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, metrics in telemetry["labels"].items():
            lines.append(
                "| {label} | {total:.0f} | {retained:.0f} | {retention_rate:.3f} |".format(
                    label=label,
                    **metrics,
                )
            )
        lines.extend(
            [
                "",
                "### Recurrence-Distance Buckets",
                "",
                "| Primary bucket | Tokens | Retained | Retention rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for bucket, metrics in telemetry["recurrence_buckets"].items():
            lines.append(
                "| {bucket} | {total:.0f} | {retained:.0f} | {retention_rate:.3f} |".format(
                    bucket=bucket,
                    **metrics,
                )
            )
        lines.extend(
            [
                "",
                "### Score Summary",
                "",
                "| Mean per-trace RDU | Max RDU | Nonzero-token count |",
                "|---:|---:|---:|",
                "| {mean_rdu:.3f} | {max_rdu:.3f} | {nonzero_tokens:.0f} |".format(
                    **telemetry["rdu_summary"],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Promote `rdu_topk` only if it beats both R-KV-like and ThinKV-like by at least 0.03 NLL with paired CIs below zero.",
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
