"""Mac-local retained-context perplexity proxy for ThoughtFlow-FP8.

This is not real sparse-KV decoding. It compresses a trace prefix into retained
tokens, appends a held-out continuation, and scores only the continuation NLL.
The goal is to detect whether a retention policy preserves useful context
better than matched-budget proxies before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .run_real_trace_retention import (
        DEFAULT_TRACES,
        MATH_RE,
        OUT_DIR,
        PHASE_WORDS,
        ROOT,
        _clean,
        _load_traces,
    )
    from .simulate_phase_retention import Token, longflow_like, rkv_like, thin_kv_like, thoughtflow
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_real_trace_retention import (
        DEFAULT_TRACES,
        MATH_RE,
        OUT_DIR,
        PHASE_WORDS,
        ROOT,
        _clean,
        _load_traces,
    )
    from simulate_phase_retention import Token, longflow_like, rkv_like, thin_kv_like, thoughtflow


def thoughtflow_recent(trace: list[Token], budget: int) -> set[int]:
    """Successor policy: protect phase/anchors, but reserve budget for recency.

    Earlier ThoughtFlow kept phase markers but lost continuation NLL to an
    R-KV-like sink+recent proxy. This bounded successor tests whether adding a
    recency reserve revives quality without discarding the interpretable
    anchor/phase constraint entirely.
    """

    anchors = {idx for idx, token in enumerate(trace) if token.label == "anchor"}
    phase = {idx for idx, token in enumerate(trace) if token.label == "phase"}
    recent_budget = max(1, budget // 3)
    recent = set(range(max(0, len(trace) - recent_budget), len(trace)))
    kept = anchors | recent
    remaining = max(0, budget - len(kept))
    if remaining:
        phase_sorted = sorted(phase - kept, key=lambda idx: (-trace[idx].importance, idx))
        kept |= set(phase_sorted[:remaining])
    remaining = max(0, budget - len(kept))
    if remaining:
        filler = sorted(
            [idx for idx in range(len(trace)) if idx not in kept],
            key=lambda idx: (-trace[idx].importance, idx),
        )
        kept |= set(filler[:remaining])
    if len(kept) > budget:
        protected = anchors | (recent & kept)
        overflow = sorted(
            [idx for idx in kept if idx not in protected],
            key=lambda idx: (trace[idx].importance, -idx),
        )
        for idx in overflow[: len(kept) - budget]:
            kept.remove(idx)
    return kept


def thoughtflow_saliency_recent(trace: list[Token], budget: int) -> set[int]:
    """Successor policy: protect anchors, reserve recency, then rank salient states.

    This is a stricter revival attempt than `thoughtflow_recent`: phase markers
    are no longer always protected. Instead, they receive a bonus and must
    compete with math-state and high-importance reasoning tokens after anchors
    and a recent reserve have been kept. If this still loses to the R-KV-like
    sink+recent proxy, the current phase-aware story is weak.
    """

    anchors = {idx for idx, token in enumerate(trace) if token.label == "anchor"}
    recent_budget = max(1, budget // 2)
    recent = set(range(max(0, len(trace) - recent_budget), len(trace)))
    kept = anchors | recent
    remaining = max(0, budget - len(kept))
    if remaining:
        def score(idx: int) -> tuple[float, int]:
            token = trace[idx]
            label_bonus = 0.0
            if token.label == "math_state":
                label_bonus = 0.18
            elif token.label == "phase":
                label_bonus = 0.10
            return (token.importance + label_bonus, -idx)

        filler = sorted(
            [idx for idx in range(len(trace)) if idx not in kept],
            key=lambda idx: (-score(idx)[0], score(idx)[1]),
        )
        kept |= set(filler[:remaining])
    if len(kept) > budget:
        protected = anchors | (recent & kept)
        overflow = sorted(
            [idx for idx in kept if idx not in protected],
            key=lambda idx: (trace[idx].importance, -idx),
        )
        for idx in overflow[: len(kept) - budget]:
            kept.remove(idx)
    return kept


POLICIES = {
    "longflow_like": longflow_like,
    "thin_kv_like": thin_kv_like,
    "rkv_like": rkv_like,
    "thoughtflow": thoughtflow,
    "thoughtflow_recent": thoughtflow_recent,
    "thoughtflow_saliency_recent": thoughtflow_saliency_recent,
}


def _token_piece_trace(tokenizer: AutoTokenizer, token_ids: list[int]) -> list[Token]:
    trace: list[Token] = []
    for idx, token_id in enumerate(token_ids):
        piece = tokenizer.decode([int(token_id)])
        clean = _clean(piece)
        if idx < 4:
            label = "anchor"
            importance = 1.0 - 0.05 * idx
        elif clean in PHASE_WORDS or clean.startswith("step") or clean[:1].isdigit():
            label = "phase"
            importance = 0.82
        elif MATH_RE.search(piece):
            label = "math_state"
            importance = 0.72
        else:
            label = "reason"
            importance = 0.48
        trace.append(Token(piece, label, importance))
    return trace


def _select_context_ids(prefix_ids: list[int], kept: set[int]) -> list[int]:
    return [token_id for idx, token_id in enumerate(prefix_ids) if idx in kept]


def _continuation_nll(
    model: AutoModelForCausalLM,
    input_ids: list[int],
    continuation_start: int,
) -> tuple[float, int]:
    if continuation_start >= len(input_ids):
        raise ValueError("continuation_start must leave at least one continuation token")
    tensor = torch.tensor([input_ids], dtype=torch.long)
    labels = tensor.clone()
    labels[:, :continuation_start] = -100
    with torch.no_grad():
        output = model(input_ids=tensor, labels=labels)
    continuation_tokens = int((labels != -100).sum().item())
    return float(output.loss.item()), continuation_tokens


def _build_rows(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    keep_fraction: float,
    max_length: int,
    continuation_tokens: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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

        full_loss, scored_tokens = _continuation_nll(model, token_ids, split)
        rows.append(
            {
                "trace_id": trace_id,
                "policy": "full_context",
                "prefix_tokens": len(prefix_ids),
                "retained_prefix_tokens": len(prefix_ids),
                "keep_rate": 1.0,
                "continuation_tokens": scored_tokens,
                "nll": full_loss,
                "delta_nll_vs_full": 0.0,
                "ppl": math.exp(min(full_loss, 20.0)),
            }
        )

        trace = _token_piece_trace(tokenizer, prefix_ids)
        budget = max(1, math.ceil(len(prefix_ids) * keep_fraction))
        for policy_name, policy in POLICIES.items():
            kept = policy(trace, budget)
            retained_prefix = _select_context_ids(prefix_ids, kept)
            scored_ids = retained_prefix + continuation_ids
            loss, scored_tokens = _continuation_nll(model, scored_ids, len(retained_prefix))
            rows.append(
                {
                    "trace_id": trace_id,
                    "policy": policy_name,
                    "prefix_tokens": len(prefix_ids),
                    "retained_prefix_tokens": len(retained_prefix),
                    "keep_rate": len(retained_prefix) / len(prefix_ids),
                    "continuation_tokens": scored_tokens,
                    "nll": loss,
                    "delta_nll_vs_full": loss - full_loss,
                    "ppl": math.exp(min(loss, 20.0)),
                }
            )
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "n_traces": len(policy_rows),
            "keep_rate": mean(float(row["keep_rate"]) for row in policy_rows),
            "nll": mean(float(row["nll"]) for row in policy_rows),
            "delta_nll_vs_full": mean(float(row["delta_nll_vs_full"]) for row in policy_rows),
            "ppl": mean(float(row["ppl"]) for row in policy_rows),
        }
    return summary


def _status(summary: dict[str, object]) -> str:
    compressed = {policy: metrics for policy, metrics in summary.items() if policy != "full_context"}
    thought_policies = {policy: metrics for policy, metrics in compressed.items() if policy.startswith("thoughtflow")}
    best_thought_policy, best_thought = min(
        thought_policies.items(),
        key=lambda item: float(item[1]["nll"]),
    )
    best_other_nll = min(
        float(metrics["nll"]) for policy, metrics in compressed.items() if not policy.startswith("thoughtflow")
    )
    margin = best_other_nll - float(best_thought["nll"])
    if margin >= 0.03:
        return f"ALIVE on retained-context NLL proxy via {best_thought_policy}; next gate is real sparse-KV or cache-dropping validation."
    if margin >= -0.03:
        return f"MIXED on retained-context NLL proxy via {best_thought_policy}; successor ties matched-budget proxies, not enough for GPU claims."
    return f"WEAKENED on retained-context NLL proxy via {best_thought_policy}; matched-budget proxies score continuation better."


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

    trace_items = _load_traces(DEFAULT_TRACES)
    texts = [" ".join(token.text for token in item["trace"]) for item in trace_items[:max_traces]]
    rows = _build_rows(model, tokenizer, texts, keep_fraction, max_length, continuation_tokens)
    summary = _summary(rows)
    return {
        "model_name": model_name,
        "keep_fraction": keep_fraction,
        "max_traces": max_traces,
        "max_length": max_length,
        "continuation_tokens": continuation_tokens,
        "n_scored_traces": summary.get("full_context", {}).get("n_traces", 0),
        "rows": rows,
        "summary": summary,
        "status": _status(summary),
    }


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 Retained-Context Perplexity Proxy",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- scored traces: {result['n_scored_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        f"- max length: {result['max_length']}",
        f"- continuation tokens: {result['continuation_tokens']}",
        "",
        "This is a Mac-local quality proxy, not sparse-KV decoding. It compresses",
        "the trace prefix, appends the same held-out continuation, and scores only",
        "continuation NLL. Full context is a reference row, not a matched-budget baseline.",
        "",
        "| Policy | Traces | Keep rate | NLL | Delta NLL vs full | PPL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, metrics in result["summary"].items():
        lines.append(
            "| {policy} | {n_traces:d} | {keep_rate:.3f} | {nll:.3f} | {delta_nll_vs_full:.3f} | {ppl:.1f} |".format(
                policy=policy, **metrics
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Advance only if a ThoughtFlow-family policy beats all matched-budget compressed proxies on continuation NLL.",
            "A tie or loss keeps the current branch mixed/weakened and defers GPU/KV work until a sharper policy exists.",
        ]
    )
    (OUT_DIR / "perplexity_impact_proxy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    (OUT_DIR / "perplexity_impact_proxy.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
