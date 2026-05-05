"""CPU sparse-KV continuation quality probe for ThoughtFlow-FP8.

Unlike the retained-text proxy, this script runs the full prefix once, prunes
the returned past_key_values according to a retention policy, and scores the
held-out continuation from the pruned cache. It is still a Mac-local quality
gate, not a GPU kernel or latency benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .perplexity_impact_proxy import POLICIES, _token_piece_trace, thoughtflow_saliency_recent
    from .policy_sweep import SweepConfig, _make_policy
    from .run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT, _load_traces
    from .simulate_phase_retention import Token, rkv_like, thin_kv_like
except ImportError:  # pragma: no cover - supports direct script execution.
    from perplexity_impact_proxy import POLICIES, _token_piece_trace, thoughtflow_saliency_recent
    from policy_sweep import SweepConfig, _make_policy
    from run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT, _load_traces
    from simulate_phase_retention import Token, rkv_like, thin_kv_like


def _texts(max_traces: int) -> list[str]:
    trace_items = _load_traces(DEFAULT_TRACES)
    return [" ".join(token.text for token in item["trace"]) for item in trace_items[:max_traces]]


def _prepare_rows(tokenizer: AutoTokenizer, max_traces: int, max_length: int, continuation_tokens: int) -> list[dict[str, object]]:
    rows = []
    for trace_id, text in enumerate(_texts(max_traces)):
        encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)
        token_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if len(token_ids) < continuation_tokens + 8:
            continue
        split = len(token_ids) - continuation_tokens
        rows.append(
            {
                "trace_id": trace_id,
                "prefix_ids": token_ids[:split],
                "continuation_ids": token_ids[split:],
                "trace": _token_piece_trace(tokenizer, token_ids[:split]),
            }
        )
    return rows


def _legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    return past_key_values


def _prune_cache(past_key_values, kept: set[int]):
    indices = torch.tensor(sorted(kept), dtype=torch.long)
    pruned = []
    for key, value in _legacy_cache(past_key_values):
        pruned.append((key.index_select(2, indices), value.index_select(2, indices)))
    if hasattr(past_key_values, "layers"):
        return DynamicCache(ddp_cache_data=tuple(pruned))
    return tuple(pruned)


def _score_continuation_from_cache(
    model: AutoModelForCausalLM,
    prefix_ids: list[int],
    continuation_ids: list[int],
    kept: set[int],
) -> tuple[float, int]:
    prefix = torch.tensor([prefix_ids], dtype=torch.long)
    continuation = torch.tensor([continuation_ids], dtype=torch.long)
    with torch.no_grad():
        prefix_outputs = model(input_ids=prefix, use_cache=True)
        pruned_cache = _prune_cache(prefix_outputs.past_key_values, kept)
        position_ids = torch.arange(len(prefix_ids), len(prefix_ids) + len(continuation_ids)).reshape(1, -1)
        outputs = model(
            input_ids=continuation,
            past_key_values=pruned_cache,
            position_ids=position_ids,
            labels=continuation,
            use_cache=False,
        )
    # With a past cache, HF causal-LM loss scores continuation tokens after the
    # first continuation token. This is consistent across all compared policies.
    scored_tokens = max(0, len(continuation_ids) - 1)
    return float(outputs.loss.item()), scored_tokens


def _policy_set() -> dict[str, Callable[[list[Token], int], set[int]]]:
    policies = {
        "rkv_like": rkv_like,
        "thin_kv_like": thin_kv_like,
        "thoughtflow_saliency_recent": thoughtflow_saliency_recent,
        "thoughtflow_sweep_best": _make_policy(
            SweepConfig(
                recent_fraction=0.55,
                phase_bonus=0.00,
                math_bonus=0.18,
                protect_anchors=4,
            )
        ),
    }
    for name in ("longflow_like", "thoughtflow", "thoughtflow_recent"):
        policies[name] = POLICIES[name]
    return policies


def _run(model_name: str, keep_fraction: float, max_traces: int, max_length: int, continuation_tokens: int) -> dict[str, object]:
    cache_dir = ROOT / "experimental/thoughtflow_fp8/.debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    prepared = _prepare_rows(tokenizer, max_traces, max_length, continuation_tokens)
    policies = _policy_set()
    rows = []
    for row in prepared:
        prefix_ids = row["prefix_ids"]
        continuation_ids = row["continuation_ids"]
        trace = row["trace"]
        assert isinstance(prefix_ids, list)
        assert isinstance(continuation_ids, list)
        assert isinstance(trace, list)
        full_kept = set(range(len(prefix_ids)))
        full_loss, scored_tokens = _score_continuation_from_cache(model, prefix_ids, continuation_ids, full_kept)
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
            loss, scored_tokens = _score_continuation_from_cache(model, prefix_ids, continuation_ids, kept)
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
    paired_deltas = _paired_deltas(rows, baseline_policy="rkv_like")
    return {
        "model_name": model_name,
        "keep_fraction": keep_fraction,
        "max_traces": max_traces,
        "max_length": max_length,
        "continuation_tokens": continuation_tokens,
        "n_scored_traces": int(summary.get("full_cache", {}).get("n_traces", 0)),
        "rows": rows,
        "summary": summary,
        "paired_delta_nll_vs_rkv_like": paired_deltas,
        "status": _status(summary),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "n_traces": float(len(policy_rows)),
            "keep_rate": mean(float(row["keep_rate"]) for row in policy_rows),
            "nll": mean(float(row["nll"]) for row in policy_rows),
            "delta_nll_vs_full": mean(float(row["delta_nll_vs_full"]) for row in policy_rows),
        }
    return summary


def _paired_deltas(
    rows: list[dict[str, object]],
    *,
    baseline_policy: str,
    bootstrap_samples: int = 1000,
    seed: int = 17,
) -> dict[str, dict[str, float]]:
    by_policy = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), {})[int(row["trace_id"])] = float(row["nll"])
    if baseline_policy not in by_policy:
        return {}
    rng = random.Random(seed)
    result = {}
    for policy, by_trace in by_policy.items():
        if policy == baseline_policy:
            continue
        trace_ids = sorted(set(by_trace) & set(by_policy[baseline_policy]))
        if not trace_ids:
            continue
        deltas = [by_trace[trace_id] - by_policy[baseline_policy][trace_id] for trace_id in trace_ids]
        boot = []
        for _ in range(bootstrap_samples):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boot.append(mean(sample))
        boot.sort()
        result[policy] = {
            "n_pairs": float(len(deltas)),
            "mean_delta_nll_minus_rkv_like": mean(deltas),
            "ci95_low": boot[int(0.025 * (bootstrap_samples - 1))],
            "ci95_high": boot[int(0.975 * (bootstrap_samples - 1))],
        }
    return result


def _status(summary: dict[str, dict[str, float]]) -> str:
    compressed = {policy: metrics for policy, metrics in summary.items() if policy != "full_cache"}
    thought = {policy: metrics for policy, metrics in compressed.items() if policy.startswith("thoughtflow")}
    best_thought_name, best_thought = min(thought.items(), key=lambda item: item[1]["nll"])
    best_other_name, best_other = min(
        ((policy, metrics) for policy, metrics in compressed.items() if not policy.startswith("thoughtflow")),
        key=lambda item: item[1]["nll"],
    )
    margin = best_other["nll"] - best_thought["nll"]
    if margin >= 0.03:
        return f"ALIVE on CPU sparse-KV probe; {best_thought_name} beats {best_other_name} by {margin:.3f} NLL."
    if margin >= -0.03:
        return f"MIXED on CPU sparse-KV probe; {best_thought_name} ties {best_other_name} within 0.03 NLL."
    return f"WEAKENED on CPU sparse-KV probe; {best_other_name} beats {best_thought_name} by {-margin:.3f} NLL."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 CPU Sparse-KV Drop Quality Probe",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- scored traces: {result['n_scored_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        f"- continuation tokens: {result['continuation_tokens']}",
        "",
        "This probe runs the full prefix once, prunes the returned KV cache according to each policy, and scores the continuation from the pruned cache.",
        "It is CPU-only quality evidence, not a Triton/CUDA performance result.",
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
    lines.extend(
        [
            "",
            "## Paired Delta vs R-KV-like",
            "",
            "Negative means lower continuation NLL than R-KV-like on the same trace.",
            "",
            "| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy, metrics in sorted(
        result["paired_delta_nll_vs_rkv_like"].items(),
        key=lambda item: item[1]["mean_delta_nll_minus_rkv_like"],
    ):
        lines.append(
            "| {policy} | {n_pairs:.0f} | {mean_delta_nll_minus_rkv_like:+.3f} | {ci95_low:+.3f} | {ci95_high:+.3f} |".format(
                policy=policy,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is the closest Mac gate to actual cache dropping. Advance only if a train-fixed ThoughtFlow-family policy beats R-KV-like and ThinKV-like on matched-budget continuation NLL with paired uncertainty.",
        ]
    )
    (OUT_DIR / "kv_drop_quality_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run(args.model_name, args.keep_fraction, args.max_traces, args.max_length, args.continuation_tokens)
    (OUT_DIR / "kv_drop_quality_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
