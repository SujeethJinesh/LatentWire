"""Real Q/K sink-logit probe for SinkAware approximate revival.

This is stronger than predicting attention sink mass: it computes GPT-style
query/key sink logits from a small causal LM and tests whether query-side
low-rank features can predict those logits without recomputing full `QK_sink`.
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
    from .real_query_sink_probe import DEFAULT_TRACES, OUT_DIR, ROOT, _linear_r2, _load_texts, _r2, _rank_features
except ImportError:  # pragma: no cover - supports direct script execution.
    from real_query_sink_probe import DEFAULT_TRACES, OUT_DIR, ROOT, _linear_r2, _load_texts, _r2, _rank_features


def _split_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq_len, width = tensor.shape
    head_dim = width // n_heads
    return tensor.view(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)


def _extract_samples(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
) -> list[dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("real_qk_sink_logit_probe currently supports GPT2-style models only")
    model.eval()
    samples: list[dict[str, object]] = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            seq_len = encoded["input_ids"].shape[1]
            if seq_len <= sink_tokens + 2:
                continue
            outputs = model(**encoded, output_hidden_states=True)
            query_positions = torch.arange(sink_tokens, seq_len)
            norm_pos = (query_positions.float() / max(1, seq_len - 1)).unsqueeze(1)
            for layer_idx, block in enumerate(model.transformer.h):
                hidden = outputs.hidden_states[layer_idx].detach().float()
                qkv = block.attn.c_attn(hidden)
                q, k, _ = qkv.split(model.config.n_embd, dim=2)
                q_heads = _split_heads(q, model.config.n_head)
                k_heads = _split_heads(k, model.config.n_head)
                head_dim = q_heads.shape[-1]
                sink_keys = k_heads[0, :, :sink_tokens, :]
                query_vecs = q_heads[0, :, query_positions, :]
                logits = torch.einsum("hqd,hsd->hqs", query_vecs, sink_keys) / math.sqrt(head_dim)
                target = logits.mean(dim=(0, 2)).float()
                hidden_query = hidden[0, query_positions].float()
                for h, y, p in zip(hidden_query, target, norm_pos):
                    samples.append({"layer": layer_idx, "hidden": h, "position": p, "target": y})
    return samples


def _evaluate(samples: list[dict[str, object]], ranks: tuple[int, ...]) -> dict[str, object]:
    rows = []
    for layer in sorted({int(sample["layer"]) for sample in samples}):
        layer_samples = [sample for sample in samples if int(sample["layer"]) == layer]
        n = len(layer_samples)
        split = int(0.67 * n)
        hidden = torch.stack([sample["hidden"] for sample in layer_samples])
        position = torch.stack([sample["position"] for sample in layer_samples])
        target = torch.stack([sample["target"] for sample in layer_samples]).float()
        train_h, test_h = hidden[:split], hidden[split:]
        train_p, test_p = position[:split], position[split:]
        train_y, test_y = target[:split], target[split:]
        row: dict[str, object] = {
            "layer": layer,
            "n_samples": n,
            "static_r2": _r2(test_y, train_y.mean().expand_as(test_y)),
            "position_r2": _linear_r2(train_p, train_y, test_p, test_y),
        }
        for rank in ranks:
            train_rank, test_rank = _rank_features(train_h, test_h, rank)
            row[f"rank{rank}_hidden_r2"] = _linear_r2(train_rank, train_y, test_rank, test_y)
            row[f"rank{rank}_hidden_plus_pos_r2"] = _linear_r2(
                torch.cat([train_rank, train_p], dim=1),
                train_y,
                torch.cat([test_rank, test_p], dim=1),
                test_y,
            )
        rows.append(row)
    metrics = [key for key in rows[0] if key.endswith("_r2")]
    summary = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    return {"rows": rows, "summary": summary, "status": _status(summary)}


def _status(summary: dict[str, object]) -> str:
    best_hidden_pos = max(value for key, value in summary.items() if key.endswith("hidden_plus_pos_r2"))
    position = float(summary["position_r2"])
    if best_hidden_pos >= position + 0.05 and best_hidden_pos > 0.25:
        return "ALIVE as approximate QK-sink predictor; next gate is full per-head error and kernel cost model."
    return "WEAKENED; query-side low-rank QK-sink prediction is not strong enough."


def _write_markdown(result: dict[str, object]) -> None:
    summary = result["summary"]
    lines = [
        "# SinkAware Real Q/K Sink-Logit Probe",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['n_traces']}",
        f"- token samples: {result['n_samples']}",
        f"- sink tokens: {result['sink_tokens']}",
        "",
        "This computes GPT-style Q/K sink logits on CPU and tests a query-side low-rank approximation.",
        "It is not an exact-kernel result and not a GPU timing result.",
        "",
        "| Static R2 | Position R2 | Rank-1 hidden R2 | Rank-2 hidden R2 | Rank-4 hidden R2 | Rank-8 hidden R2 | Best hidden+pos R2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {static_r2:.3f} | {position_r2:.3f} | {rank1_hidden_r2:.3f} | {rank2_hidden_r2:.3f} | {rank4_hidden_r2:.3f} | {rank8_hidden_r2:.3f} | {best_hidden_pos:.3f} |".format(
            best_hidden_pos=max(value for key, value in summary.items() if key.endswith("hidden_plus_pos_r2")),
            **summary,
        ),
        "",
        "## Per Layer",
        "",
        "| Layer | Samples | Static R2 | Position R2 | Rank-8 hidden R2 | Rank-8 hidden+pos R2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {layer} | {n_samples} | {static_r2:.3f} | {position_r2:.3f} | {rank8_hidden_r2:.3f} | {rank8_hidden_plus_pos_r2:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is the strongest Mac-local SinkAware revival evidence so far if hidden+position beats position-only on real Q/K logits.",
            "The exact static-prior branch remains killed. The live branch is approximate low-rank QK-sink prediction or a fused exact path with a cost model.",
        ]
    )
    (OUT_DIR / "real_qk_sink_logit_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--max-traces", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--sink-tokens", type=int, default=4)
    args = parser.parse_args()

    cache_dir = ROOT / ".debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    texts = _load_texts(DEFAULT_TRACES, args.max_traces)
    samples = _extract_samples(args.model_name, texts, args.max_length, args.sink_tokens)
    result = _evaluate(samples, ranks=(1, 2, 4, 8))
    result.update(
        {
            "model_name": args.model_name,
            "n_traces": len(texts),
            "n_samples": len(samples),
            "sink_tokens": args.sink_tokens,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_qk_sink_logit_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
