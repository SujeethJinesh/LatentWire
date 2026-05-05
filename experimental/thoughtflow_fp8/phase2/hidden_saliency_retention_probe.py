"""Hidden/attention saliency proxy for ThoughtFlow-FP8 on Mac."""

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
    from .run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT, _label_trace
    from .simulate_phase_retention import _recall, longflow_like, rkv_like, thin_kv_like, thoughtflow
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_real_trace_retention import DEFAULT_TRACES, OUT_DIR, ROOT, _label_trace
    from simulate_phase_retention import _recall, longflow_like, rkv_like, thin_kv_like, thoughtflow


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


def _trace_from_tokens(tokenizer: AutoTokenizer, input_ids: torch.Tensor):
    pieces = [tokenizer.decode([int(token_id)]) for token_id in input_ids.tolist()]
    return _label_trace(" ".join(pieces))


def _attention_received_saliency(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    saliency = None
    for attention in attentions:
        # [batch, heads, query, key] -> key saliency.
        layer_saliency = attention[0].mean(dim=0).sum(dim=0).detach().float()
        saliency = layer_saliency if saliency is None else saliency + layer_saliency
    assert saliency is not None
    return saliency / len(attentions)


def _run(model_name: str, keep_fraction: float, max_traces: int, max_length: int) -> dict[str, object]:
    cache_dir = ROOT / ".debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    texts = _load_texts(DEFAULT_TRACES, max_traces)
    rows = []
    policies = {
        "longflow_like": longflow_like,
        "thin_kv_like": thin_kv_like,
        "rkv_like": rkv_like,
        "thoughtflow": thoughtflow,
    }
    with torch.no_grad():
        for trace_id, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encoded["input_ids"][0]
            if input_ids.numel() < 8:
                continue
            trace = _trace_from_tokens(tokenizer, input_ids)
            # Tokenizer re-decoding may drop pure whitespace; align conservatively.
            n = min(len(trace), input_ids.numel())
            trace = trace[:n]
            outputs = model(**encoded, output_attentions=True)
            saliency = _attention_received_saliency(outputs.attentions)[:n]
            budget = max(1, math.ceil(n * keep_fraction))
            for name, policy in policies.items():
                kept = policy(trace, budget)
                rows.append(_row(trace_id, name, trace, kept, budget))
            kept = _saliency_topk(saliency, budget)
            rows.append(_row(trace_id, "attention_received_topk", trace, kept, budget))
    summary = _summary(rows)
    result = {
        "model_name": model_name,
        "keep_fraction": keep_fraction,
        "n_traces": len(texts),
        "rows": rows,
        "summary": summary,
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
    thought = summary["thoughtflow"]
    saliency = summary["attention_received_topk"]
    best_other = max(float(metrics["phase_recall"]) for policy, metrics in summary.items() if policy != "thoughtflow")
    if float(thought["phase_recall"]) >= best_other + 0.05:
        return "ALIVE; protected phase markers beat all local retention proxies."
    if float(thought["phase_recall"]) >= float(saliency["phase_recall"]) + 0.05:
        return "MIXED; beats attention-saliency proxy but still ties the strongest importance proxy."
    return "WEAKENED against attention-saliency proxy; current phase markers are not enough."


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
        "This uses attention-received mass as a CPU hidden/KV saliency proxy.",
        "It is not a cache-compression accuracy result and not a GPU benchmark.",
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
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Advance ThoughtFlow only if the protected-token policy beats a hidden/KV saliency proxy.",
            "If it loses or ties, the current phase-marker heuristic should not go to GPU work.",
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
