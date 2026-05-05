"""Held-out policy sweep for ThoughtFlow-FP8 retained-context NLL.

The previous hand-tuned successor nearly tied the R-KV-like retained-prefix
proxy. This script performs a small pre-registered Mac-local sweep over
recency/phase/math bonuses, selects on a train split, and reports on a held-out
split. It is still a text-prefix proxy, not sparse-KV decoding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .perplexity_impact_proxy import (
        _continuation_nll,
        _select_context_ids,
        _token_piece_trace,
        thoughtflow_saliency_recent,
    )
    from .run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT, _load_traces
    from .simulate_phase_retention import Token, rkv_like, thin_kv_like
except ImportError:  # pragma: no cover - supports direct script execution.
    from perplexity_impact_proxy import (
        _continuation_nll,
        _select_context_ids,
        _token_piece_trace,
        thoughtflow_saliency_recent,
    )
    from run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT, _load_traces
    from simulate_phase_retention import Token, rkv_like, thin_kv_like


@dataclass(frozen=True)
class SweepConfig:
    recent_fraction: float
    phase_bonus: float
    math_bonus: float
    protect_anchors: int

    @property
    def name(self) -> str:
        return (
            f"tf_sweep_r{self.recent_fraction:.2f}_p{self.phase_bonus:.2f}_"
            f"m{self.math_bonus:.2f}_a{self.protect_anchors}"
        )


def _make_policy(config: SweepConfig) -> Callable[[list[Token], int], set[int]]:
    def policy(trace: list[Token], budget: int) -> set[int]:
        anchor_candidates = [idx for idx, token in enumerate(trace) if token.label == "anchor"]
        anchors = set(anchor_candidates[: min(config.protect_anchors, budget)])
        recent_budget = max(1, int(round(budget * config.recent_fraction)))
        recent = set(range(max(0, len(trace) - recent_budget), len(trace)))
        kept = set(sorted(anchors, key=lambda idx: idx)[:budget])
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


def _configs() -> list[SweepConfig]:
    configs = []
    for recent_fraction in (0.45, 0.55, 0.65):
        for phase_bonus in (0.00, 0.10):
            for math_bonus in (0.18, 0.30):
                configs.append(
                    SweepConfig(
                        recent_fraction=recent_fraction,
                        phase_bonus=phase_bonus,
                        math_bonus=math_bonus,
                        protect_anchors=4,
                    )
                )
    return configs


def _load_texts(max_traces: int) -> list[str]:
    trace_items = _load_traces(DEFAULT_TRACES)
    return [" ".join(token.text for token in item["trace"]) for item in trace_items[:max_traces]]


def _prepare_trace_rows(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    continuation_tokens: int,
) -> list[dict[str, object]]:
    rows = []
    for trace_id, text in enumerate(texts):
        encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)
        token_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if len(token_ids) < continuation_tokens + 8:
            continue
        split = len(token_ids) - continuation_tokens
        prefix_ids = token_ids[:split]
        continuation_ids = token_ids[split:]
        if len(prefix_ids) < 8 or len(continuation_ids) < 2:
            continue
        rows.append(
            {
                "trace_id": trace_id,
                "token_ids": token_ids,
                "prefix_ids": prefix_ids,
                "continuation_ids": continuation_ids,
                "trace": _token_piece_trace(tokenizer, prefix_ids),
            }
        )
    return rows


def _score_policy(
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
        retained_prefix = _select_context_ids(prefix_ids, kept)
        scored_ids = retained_prefix + continuation_ids
        loss, scored_tokens = _continuation_nll(model, scored_ids, len(retained_prefix))
        rows.append(
            {
                "trace_id": row["trace_id"],
                "policy": policy_name,
                "retained_prefix_tokens": len(retained_prefix),
                "keep_rate": len(retained_prefix) / len(prefix_ids),
                "continuation_tokens": scored_tokens,
                "nll": loss,
                "ppl": math.exp(min(loss, 20.0)),
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "n_traces": float(len(policy_rows)),
            "keep_rate": mean(float(row["keep_rate"]) for row in policy_rows),
            "nll": mean(float(row["nll"]) for row in policy_rows),
            "ppl": mean(float(row["ppl"]) for row in policy_rows),
        }
    return summary


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

    prepared = _prepare_trace_rows(tokenizer, _load_texts(max_traces), max_length, continuation_tokens)
    train_rows = [row for idx, row in enumerate(prepared) if idx % 2 == 0]
    heldout_rows = [row for idx, row in enumerate(prepared) if idx % 2 == 1]
    configs = _configs()

    train_scored = []
    heldout_scored = []
    baselines = {
        "rkv_like": rkv_like,
        "thin_kv_like": thin_kv_like,
        "thoughtflow_saliency_recent": thoughtflow_saliency_recent,
    }
    for name, policy in baselines.items():
        train_scored.extend(_score_policy(model, train_rows, name, policy, keep_fraction))
        heldout_scored.extend(_score_policy(model, heldout_rows, name, policy, keep_fraction))

    for config in configs:
        policy = _make_policy(config)
        train_scored.extend(_score_policy(model, train_rows, config.name, policy, keep_fraction))

    train_summary = _summarize(train_scored)
    sweep_summaries = {config.name: train_summary[config.name] for config in configs}
    best_name = min(sweep_summaries, key=lambda name: sweep_summaries[name]["nll"])
    best_config = next(config for config in configs if config.name == best_name)
    heldout_scored.extend(_score_policy(model, heldout_rows, best_name, _make_policy(best_config), keep_fraction))
    heldout_summary = _summarize(heldout_scored)
    best_nll = heldout_summary[best_name]["nll"]
    rkv_nll = heldout_summary["rkv_like"]["nll"]
    margin = rkv_nll - best_nll
    if margin >= 0.03:
        status = "ALIVE on held-out policy sweep; best ThoughtFlow policy beats R-KV-like by >=0.03 NLL."
    elif margin >= -0.03:
        status = "MIXED on held-out policy sweep; best ThoughtFlow policy ties R-KV-like within 0.03 NLL."
    else:
        status = "WEAKENED on held-out policy sweep; R-KV-like remains better on held-out traces."
    return {
        "model_name": model_name,
        "keep_fraction": keep_fraction,
        "max_traces": max_traces,
        "max_length": max_length,
        "continuation_tokens": continuation_tokens,
        "n_train_traces": len(train_rows),
        "n_heldout_traces": len(heldout_rows),
        "configs": [asdict(config) | {"name": config.name} for config in configs],
        "best_config": asdict(best_config) | {"name": best_name},
        "train_summary": train_summary,
        "heldout_summary": heldout_summary,
        "heldout_margin_vs_rkv_nll": margin,
        "status": status,
    }


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 Held-Out Policy Sweep",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- train traces: {result['n_train_traces']}",
        f"- held-out traces: {result['n_heldout_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        f"- best train-selected policy: `{result['best_config']['name']}`",
        f"- held-out NLL margin vs R-KV-like: {result['heldout_margin_vs_rkv_nll']:+.3f} (positive means ThoughtFlow is better)",
        "",
        "## Held-Out Summary",
        "",
        "| Policy | Traces | Keep rate | NLL | PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, metrics in sorted(result["heldout_summary"].items(), key=lambda item: item[1]["nll"]):
        lines.append(
            "| {policy} | {n_traces:.0f} | {keep_rate:.3f} | {nll:.3f} | {ppl:.1f} |".format(
                policy=policy,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is still a text-prefix proxy, not sparse-KV decoding. A positive workshop method requires the train-selected ThoughtFlow-family policy to beat R-KV-like on held-out continuation NLL, then validate under real hidden/KV telemetry.",
        ]
    )
    (OUT_DIR / "policy_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    args = parser.parse_args()
    result = _run(
        args.model_name,
        args.keep_fraction,
        args.max_traces,
        args.max_length,
        args.continuation_tokens,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "policy_sweep.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "best_config": result["best_config"]}, indent=2))


if __name__ == "__main__":
    main()
