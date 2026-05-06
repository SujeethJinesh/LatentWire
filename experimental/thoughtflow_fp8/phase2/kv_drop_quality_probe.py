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
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class SparseSweepConfig:
    recent_fraction: float
    phase_bonus: float
    math_bonus: float
    protect_anchors: int

    @property
    def name(self) -> str:
        return (
            f"tf_sparse_r{self.recent_fraction:.2f}_p{self.phase_bonus:.2f}_"
            f"m{self.math_bonus:.2f}_a{self.protect_anchors}"
        )


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


def _sparse_sweep_configs() -> list[SparseSweepConfig]:
    configs = []
    for recent_fraction in (0.50, 0.55, 0.60):
        for phase_bonus in (0.05, 0.10):
            for math_bonus in (0.12, 0.18):
                for protect_anchors in (2, 4):
                    configs.append(
                        SparseSweepConfig(
                            recent_fraction=recent_fraction,
                            phase_bonus=phase_bonus,
                            math_bonus=math_bonus,
                            protect_anchors=protect_anchors,
                        )
                    )
    return configs


def _make_sparse_sweep_policy(config: SparseSweepConfig) -> Callable[[list[Token], int], set[int]]:
    def policy(trace: list[Token], budget: int) -> set[int]:
        anchor_candidates = [idx for idx, token in enumerate(trace) if token.label == "anchor"]
        kept = set(anchor_candidates[: min(config.protect_anchors, budget)])
        recent_budget = max(1, int(round(budget * config.recent_fraction)))
        recent = set(range(max(0, len(trace) - recent_budget), len(trace)))
        recent_slots = max(0, budget - len(kept))
        if recent_slots:
            kept |= set(sorted(recent)[-recent_slots:])
        remaining = max(0, budget - len(kept))

        def score(idx: int) -> tuple[float, int]:
            token = trace[idx]
            bonus = 0.0
            if token.label == "phase":
                bonus += config.phase_bonus
            if token.label == "math_state":
                bonus += config.math_bonus
            return token.importance + bonus, -idx

        if remaining:
            filler = sorted(
                [idx for idx in range(len(trace)) if idx not in kept],
                key=lambda idx: (-score(idx)[0], score(idx)[1]),
            )
            kept |= set(filler[:remaining])
        return kept

    return policy


def _score_policy_rows(
    model: AutoModelForCausalLM,
    prepared_rows: list[dict[str, object]],
    policy_name: str,
    policy: Callable[[list[Token], int], set[int]],
    keep_fraction: float,
) -> list[dict[str, object]]:
    rows = []
    for row in prepared_rows:
        prefix_ids = row["prefix_ids"]
        continuation_ids = row["continuation_ids"]
        trace = row["trace"]
        assert isinstance(prefix_ids, list)
        assert isinstance(continuation_ids, list)
        assert isinstance(trace, list)
        budget = max(1, math.ceil(len(prefix_ids) * keep_fraction))
        kept = policy(trace, budget)
        loss, scored_tokens = _score_continuation_from_cache(model, prefix_ids, continuation_ids, kept)
        rows.append(
            {
                "trace_id": row["trace_id"],
                "policy": policy_name,
                "keep_rate": len(kept) / len(prefix_ids),
                "retained_prefix_tokens": len(kept),
                "continuation_tokens": scored_tokens,
                "nll": loss,
            }
        )
    return rows


def _train_fixed_sparse_sweep(
    model: AutoModelForCausalLM,
    prepared: list[dict[str, object]],
    keep_fraction: float,
) -> dict[str, object]:
    train_rows = [row for idx, row in enumerate(prepared) if idx % 2 == 0]
    heldout_rows = [row for idx, row in enumerate(prepared) if idx % 2 == 1]
    configs = _sparse_sweep_configs()
    train_scored = []
    for config in configs:
        train_scored.extend(
            _score_policy_rows(
                model,
                train_rows,
                config.name,
                _make_sparse_sweep_policy(config),
                keep_fraction,
            )
        )
    train_summary = _summary(train_scored)
    best_name = min((config.name for config in configs), key=lambda name: train_summary[name]["nll"])
    best_config = next(config for config in configs if config.name == best_name)

    heldout_scored = []
    baselines = {
        "rkv_like": rkv_like,
        "thin_kv_like": thin_kv_like,
        "thoughtflow_saliency_recent": thoughtflow_saliency_recent,
        best_name: _make_sparse_sweep_policy(best_config),
    }
    for name, policy in baselines.items():
        heldout_scored.extend(_score_policy_rows(model, heldout_rows, name, policy, keep_fraction))

    heldout_summary = _summary(heldout_scored)
    paired_vs_rkv = _paired_deltas(heldout_scored, baseline_policy="rkv_like")
    paired_vs_thin = _paired_deltas(heldout_scored, baseline_policy="thin_kv_like")
    best_nll = heldout_summary[best_name]["nll"]
    strongest_other_name, strongest_other = min(
        ((name, metrics) for name, metrics in heldout_summary.items() if not name.startswith("thoughtflow") and not name.startswith("tf_sparse")),
        key=lambda item: item[1]["nll"],
    )
    margin = strongest_other["nll"] - best_nll
    rkv_ci_high = paired_vs_rkv.get(best_name, {}).get("ci95_high", float("inf"))
    thin_ci_high = paired_vs_thin.get(best_name, {}).get("ci95_high", float("inf"))
    if margin >= 0.03 and rkv_ci_high < 0.0 and thin_ci_high < 0.0:
        status = "ALIVE on train-fixed sparse sweep; held-out policy clears mean margin and paired CIs."
    elif margin >= 0.03:
        status = "MIXED on train-fixed sparse sweep; held-out policy clears mean margin but not paired uncertainty."
    else:
        status = "MIXED on train-fixed sparse sweep; held-out policy remains inside the 0.03 NLL margin."
    return {
        "n_train_traces": len(train_rows),
        "n_heldout_traces": len(heldout_rows),
        "configs": [asdict(config) | {"name": config.name} for config in configs],
        "best_config": asdict(best_config) | {"name": best_name},
        "train_summary": train_summary,
        "heldout_summary": heldout_summary,
        "heldout_rows": heldout_scored,
        "paired_delta_nll_vs_rkv_like": paired_vs_rkv,
        "paired_delta_nll_vs_thin_kv_like": paired_vs_thin,
        "heldout_margin_vs_strongest_non_thoughtflow": margin,
        "strongest_non_thoughtflow_policy": strongest_other_name,
        "status": status,
    }


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
    paired_deltas_vs_thin = _paired_deltas(rows, baseline_policy="thin_kv_like")
    sparse_sweep = _train_fixed_sparse_sweep(model, prepared, keep_fraction)
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
        "paired_delta_nll_vs_thin_kv_like": paired_deltas_vs_thin,
        "train_fixed_sparse_sweep": sparse_sweep,
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
            "delta_nll_vs_full": mean(float(row.get("delta_nll_vs_full", 0.0)) for row in policy_rows),
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
            f"mean_delta_nll_minus_{baseline_policy}": mean(deltas),
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
            "## Paired Delta vs ThinKV-like",
            "",
            "Negative means lower continuation NLL than ThinKV-like on the same trace.",
            "",
            "| Policy | Pairs | Mean delta NLL | 95% CI low | 95% CI high |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy, metrics in sorted(
        result["paired_delta_nll_vs_thin_kv_like"].items(),
        key=lambda item: item[1]["mean_delta_nll_minus_thin_kv_like"],
    ):
        lines.append(
            "| {policy} | {n_pairs:.0f} | {mean_delta_nll_minus_thin_kv_like:+.3f} | {ci95_low:+.3f} | {ci95_high:+.3f} |".format(
                policy=policy,
                **metrics,
            )
        )
    sweep = result["train_fixed_sparse_sweep"]
    lines.extend(
        [
            "",
            "## Train-Fixed Sparse Sweep",
            "",
            f"Status: **{sweep['status']}**",
            "",
            f"- train traces: {sweep['n_train_traces']}",
            f"- held-out traces: {sweep['n_heldout_traces']}",
            f"- best train-selected policy: `{sweep['best_config']['name']}`",
            f"- strongest non-ThoughtFlow held-out baseline: `{sweep['strongest_non_thoughtflow_policy']}`",
            f"- held-out NLL margin vs strongest non-ThoughtFlow baseline: {sweep['heldout_margin_vs_strongest_non_thoughtflow']:+.3f}",
            "",
            "| Policy | Held-out traces | Keep rate | NLL |",
            "|---|---:|---:|---:|",
        ]
    )
    for policy, metrics in sorted(sweep["heldout_summary"].items(), key=lambda item: item[1]["nll"]):
        lines.append(
            "| {policy} | {n_traces:.0f} | {keep_rate:.3f} | {nll:.3f} |".format(
                policy=policy,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "| Policy | Baseline | Mean delta NLL | 95% CI low | 95% CI high |",
            "|---|---|---:|---:|---:|",
        ]
    )
    best_name = sweep["best_config"]["name"]
    reported_sweep_policies = []
    for policy in ("thoughtflow_saliency_recent", best_name):
        if policy in sweep["heldout_summary"] and policy not in reported_sweep_policies:
            reported_sweep_policies.append(policy)
    for policy in reported_sweep_policies:
        for baseline_key, baseline_name, metric_name in (
            ("paired_delta_nll_vs_rkv_like", "rkv_like", "mean_delta_nll_minus_rkv_like"),
            ("paired_delta_nll_vs_thin_kv_like", "thin_kv_like", "mean_delta_nll_minus_thin_kv_like"),
        ):
            metrics = sweep[baseline_key][policy]
            lines.append(
                "| {policy} | {baseline} | {mean:+.3f} | {low:+.3f} | {high:+.3f} |".format(
                    policy=policy,
                    baseline=baseline_name,
                    mean=metrics[metric_name],
                    low=metrics["ci95_low"],
                    high=metrics["ci95_high"],
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
