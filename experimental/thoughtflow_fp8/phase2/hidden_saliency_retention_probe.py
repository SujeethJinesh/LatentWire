"""Hidden/KV saliency telemetry for ThoughtFlow-FP8 on Mac."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT
    from .perplexity_impact_proxy import _token_piece_trace, thoughtflow_saliency_recent
    from .policy_sweep import SweepConfig, _make_policy
    from .simulate_phase_retention import _recall, longflow_like, rkv_like, thin_kv_like, thoughtflow
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT
    from perplexity_impact_proxy import _token_piece_trace, thoughtflow_saliency_recent
    from policy_sweep import SweepConfig, _make_policy
    from simulate_phase_retention import _recall, longflow_like, rkv_like, thin_kv_like, thoughtflow


REAL_SALIENCY_POLICIES = {
    "attention_received_topk",
    "hidden_norm_topk",
    "key_norm_topk",
    "value_norm_topk",
    "kv_norm_topk",
}


def _load_texts(paths: list[Path], max_traces: int) -> list[str]:
    texts: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = str(row.get("prediction") or row.get("text") or "")
                if len(text.split()) >= 8:
                    texts.append(text)
                if len(texts) >= max_traces:
                    return texts
    return texts


def _saliency_topk(saliency: torch.Tensor, budget: int) -> set[int]:
    return set(torch.topk(saliency, k=min(budget, saliency.numel())).indices.tolist())


def _attention_received_saliency(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    saliency = None
    for attention in attentions:
        # [batch, heads, query, key] -> key saliency.
        layer_saliency = attention[0].mean(dim=0).sum(dim=0).detach().float()
        saliency = layer_saliency if saliency is None else saliency + layer_saliency
    assert saliency is not None
    return saliency / len(attentions)


def _hidden_norm_saliency(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return hidden_states[-1][0].detach().float().norm(dim=-1)


def _token_norm(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.detach().float()
    if values.ndim == 4:
        values = values[0]
    if values.ndim != 3:
        raise ValueError(f"expected KV tensor with shape [batch, heads, seq, dim] or [heads, seq, dim], got {tuple(tensor.shape)}")
    return values.norm(dim=-1).mean(dim=0)


def _kv_norm_saliency(past_key_values: object) -> dict[str, torch.Tensor]:
    key_total = None
    value_total = None
    layer_count = 0
    layers = getattr(past_key_values, "layers", None)
    iterable = layers if layers is not None else past_key_values
    for layer in iterable:
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            key_tensor = layer.keys
            value_tensor = layer.values
        else:
            key_tensor, value_tensor = layer
        key_norm = _token_norm(key_tensor)
        value_norm = _token_norm(value_tensor)
        key_total = key_norm if key_total is None else key_total + key_norm
        value_total = value_norm if value_total is None else value_total + value_norm
        layer_count += 1
    if layer_count == 0 or key_total is None or value_total is None:
        raise ValueError("past_key_values did not contain KV layers")
    key_mean = key_total / layer_count
    value_mean = value_total / layer_count
    return {
        "key_norm_topk": key_mean,
        "value_norm_topk": value_mean,
        "kv_norm_topk": (key_mean + value_mean) / 2,
    }


def _paired_delta(
    rows: list[dict[str, object]],
    policy_a: str,
    policy_b: str,
    metric: str,
) -> dict[str, float]:
    a_by_trace = {int(row["trace_id"]): float(row[metric]) for row in rows if row["policy"] == policy_a}
    b_by_trace = {int(row["trace_id"]): float(row[metric]) for row in rows if row["policy"] == policy_b}
    trace_ids = sorted(set(a_by_trace) & set(b_by_trace))
    deltas = [a_by_trace[trace_id] - b_by_trace[trace_id] for trace_id in trace_ids]
    if not deltas:
        return {"n": 0.0, "mean": 0.0, "stderr": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    delta_mean = mean(deltas)
    if len(deltas) == 1:
        stderr = 0.0
    else:
        variance = sum((delta - delta_mean) ** 2 for delta in deltas) / (len(deltas) - 1)
        stderr = math.sqrt(variance / len(deltas))
    return {
        "n": float(len(deltas)),
        "mean": delta_mean,
        "stderr": stderr,
        "ci95_low": delta_mean - 1.96 * stderr,
        "ci95_high": delta_mean + 1.96 * stderr,
    }


def _best_policy(summary: dict[str, object], candidates: set[str]) -> str:
    present = [policy for policy in candidates if policy in summary]
    if not present:
        raise ValueError(f"none of the candidate policies are present: {sorted(candidates)}")
    return max(
        present,
        key=lambda policy: (
            float(summary[policy]["phase_recall"]) + float(summary[policy]["math_state_recall"]),
            float(summary[policy]["phase_recall"]),
            float(summary[policy]["math_state_recall"]),
        ),
    )


def _diagnostics(summary: dict[str, object], rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    thought_policies = {policy for policy in summary if policy.startswith("thoughtflow")}
    best_thought = _best_policy(summary, thought_policies)
    best_saliency = _best_policy(summary, REAL_SALIENCY_POLICIES)
    non_thought = {policy for policy in summary if not policy.startswith("thoughtflow")}
    best_non_thought = _best_policy(summary, non_thought)
    thought_metrics = summary[best_thought]
    saliency_metrics = summary[best_saliency]
    other_metrics = summary[best_non_thought]
    paired = {}
    if rows is not None:
        paired = {
            metric: _paired_delta(rows, best_thought, best_saliency, metric)
            for metric in ("anchor_recall", "phase_recall", "math_state_recall")
        }
    return {
        "best_thoughtflow_policy": best_thought,
        "best_real_saliency_policy": best_saliency,
        "best_non_thoughtflow_policy": best_non_thought,
        "phase_margin_vs_real_saliency": float(thought_metrics["phase_recall"]) - float(saliency_metrics["phase_recall"]),
        "math_state_margin_vs_real_saliency": float(thought_metrics["math_state_recall"]) - float(saliency_metrics["math_state_recall"]),
        "phase_margin_vs_best_non_thoughtflow": float(thought_metrics["phase_recall"]) - float(other_metrics["phase_recall"]),
        "math_state_margin_vs_best_non_thoughtflow": float(thought_metrics["math_state_recall"]) - float(other_metrics["math_state_recall"]),
        "paired_delta_vs_real_saliency": paired,
    }


def _policies() -> dict[str, Callable]:
    return {
        "longflow_like": longflow_like,
        "thin_kv_like": thin_kv_like,
        "rkv_like": rkv_like,
        "thoughtflow": thoughtflow,
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


def _run(model_name: str, keep_fraction: float, max_traces: int, max_length: int) -> dict[str, object]:
    cache_dir = ROOT / "experimental/thoughtflow_fp8/.debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    texts = _load_texts(DEFAULT_TRACES, max_traces)
    rows = []
    policies = _policies()
    with torch.no_grad():
        for trace_id, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encoded["input_ids"][0]
            if input_ids.numel() < 8:
                continue
            trace = _token_piece_trace(tokenizer, [int(token_id) for token_id in input_ids.tolist()])
            outputs = model(**encoded, output_attentions=True, output_hidden_states=True, use_cache=True)
            saliencies = {
                "attention_received_topk": _attention_received_saliency(outputs.attentions),
                "hidden_norm_topk": _hidden_norm_saliency(outputs.hidden_states),
            }
            saliencies.update(_kv_norm_saliency(outputs.past_key_values))
            n = min([len(trace), input_ids.numel()] + [saliency.numel() for saliency in saliencies.values()])
            trace = trace[:n]
            budget = max(1, math.ceil(n * keep_fraction))
            for name, policy in policies.items():
                kept = policy(trace, budget)
                rows.append(_row(trace_id, name, trace, kept, budget))
            for name, saliency in saliencies.items():
                kept = _saliency_topk(saliency[:n], budget)
                rows.append(_row(trace_id, name, trace, kept, budget))
    summary = _summary(rows)
    diagnostics = _diagnostics(summary, rows)
    result = {
        "model_name": model_name,
        "keep_fraction": keep_fraction,
        "n_traces": len({int(row["trace_id"]) for row in rows}),
        "rows": rows,
        "summary": summary,
        "diagnostics": diagnostics,
        "status": _status(summary),
    }
    return result


def _row(trace_id: int, policy: str, trace, kept: set[int], budget: int) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "policy": policy,
        "tokens": len(trace),
        "budget": budget,
        "keep_rate": len(kept) / len(trace),
        "anchor_recall": _recall(trace, kept, "anchor"),
        "phase_recall": _recall(trace, kept, "phase"),
        "math_state_recall": _recall(trace, kept, "math_state"),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    summary = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            metric: mean(float(row[metric]) for row in policy_rows)
            for metric in ("keep_rate", "anchor_recall", "phase_recall", "math_state_recall")
        }
    return summary


def _status(summary: dict[str, object]) -> str:
    diagnostics = _diagnostics(summary)
    phase_margin = float(diagnostics["phase_margin_vs_best_non_thoughtflow"])
    math_margin = float(diagnostics["math_state_margin_vs_best_non_thoughtflow"])
    real_phase_margin = float(diagnostics["phase_margin_vs_real_saliency"])
    real_math_margin = float(diagnostics["math_state_margin_vs_real_saliency"])
    best_thought = diagnostics["best_thoughtflow_policy"]
    best_saliency = diagnostics["best_real_saliency_policy"]
    if phase_margin >= 0.05 and math_margin >= 0.03:
        return f"ALIVE; {best_thought} beats all local hidden/KV retention proxies on phase and math-state recall."
    if real_phase_margin >= 0.05 and real_math_margin >= -0.03:
        return f"MIXED; {best_thought} beats {best_saliency} on phase recall but does not clear the full proxy set."
    return f"WEAKENED; {best_thought} does not beat the strongest hidden/KV saliency proxy on protected-token telemetry."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 Hidden-Saliency Retention Probe",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['n_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        "",
        "This uses real distilgpt2 attention, final-hidden norms, key norms, value norms, and combined KV norms as CPU hidden/KV saliency proxies.",
        "It is not a cache-compression accuracy result and not a GPU benchmark.",
        f"- best ThoughtFlow-family policy: `{result['diagnostics']['best_thoughtflow_policy']}`",
        f"- strongest real-saliency proxy: `{result['diagnostics']['best_real_saliency_policy']}`",
        f"- phase margin vs real saliency: {result['diagnostics']['phase_margin_vs_real_saliency']:+.3f}",
        f"- math-state margin vs real saliency: {result['diagnostics']['math_state_margin_vs_real_saliency']:+.3f}",
        "",
        "| Policy | Keep rate | Anchor recall | Phase recall | Math-state recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, metrics in result["summary"].items():
        lines.append(
            "| {policy} | {keep_rate:.3f} | {anchor_recall:.3f} | {phase_recall:.3f} | {math_state_recall:.3f} |".format(
                policy=policy, **metrics
            )
        )
    paired = result["diagnostics"]["paired_delta_vs_real_saliency"]
    lines.extend(
        [
            "",
            "## Paired Margins",
            "",
            "Mean paired recall delta for the best ThoughtFlow-family policy minus the strongest real-saliency proxy.",
            "",
            "| Metric | Traces | Mean delta | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, stats in paired.items():
        lines.append(
            "| {metric} | {n:.0f} | {mean:+.3f} | [{ci95_low:+.3f}, {ci95_high:+.3f}] |".format(
                metric=metric,
                **stats,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Advance ThoughtFlow only if a train-fixed ThoughtFlow-family policy beats real hidden/KV saliency proxies on both phase/control and math-state recall.",
            "A phase-only win is not enough; it can be a marker-preservation artifact rather than useful cache retention.",
        ]
    )
    (OUT_DIR / "hidden_saliency_retention_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()
    result = _run(args.model_name, args.keep_fraction, args.max_traces, args.max_length)
    (OUT_DIR / "hidden_saliency_retention_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
