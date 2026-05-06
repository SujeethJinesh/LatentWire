"""Per-head SinkAware softmax/output error probe on real GPT2-style Q/K/V.

This is the Mac-local gate after QK-sink-logit predictability.  It keeps the
non-sink attention scores exact and replaces only the first `sink_tokens` logits
with a per-head low-rank query predictor, then measures held-out softmax drift
and attention-output error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .real_qk_sink_logit_probe import _split_heads
    from .real_query_sink_probe import DEFAULT_TRACES, OUT_DIR, ROOT, _load_texts, _rank_features
except ImportError:  # pragma: no cover - supports direct script execution.
    from real_qk_sink_logit_probe import _split_heads
    from real_query_sink_probe import DEFAULT_TRACES, OUT_DIR, ROOT, _load_texts, _rank_features


def _ridge_weights(x: torch.Tensor, y: torch.Tensor, ridge_scale: float = 1e-3) -> torch.Tensor:
    x_aug = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    ridge = ridge_scale * torch.eye(x_aug.shape[1], dtype=x_aug.dtype)
    return torch.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y)


def _fit_layer_predictors(
    train_q: torch.Tensor,
    train_pos: torch.Tensor,
    train_sink_logits: torch.Tensor,
    ranks: tuple[int, ...],
) -> dict[str, object]:
    """Fit static, position-only, and per-head low-rank+position predictors.

    Shapes:
    - train_q: [tokens, heads, head_dim]
    - train_pos: [tokens, 1]
    - train_sink_logits: [tokens, heads, sink_tokens]
    """

    n_heads = train_q.shape[1]
    models: dict[str, object] = {
        "static": train_sink_logits.mean(dim=0),
        "position": [],
        "rank": {rank: [] for rank in ranks},
    }
    for head in range(n_heads):
        head_q = train_q[:, head, :]
        head_y = train_sink_logits[:, head, :]
        position_weights = _ridge_weights(train_pos, head_y)
        models["position"].append({"weights": position_weights})
        for rank in ranks:
            rank_train, _ = _rank_features(head_q, head_q, rank)
            mean_q = head_q.mean(dim=0, keepdim=True)
            centered = head_q - mean_q
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[: min(rank, vh.shape[0])].T
            features = torch.cat([rank_train, train_pos], dim=1)
            weights = _ridge_weights(features, head_y)
            models["rank"][rank].append({"mean": mean_q.squeeze(0), "basis": basis, "weights": weights})
    return models


def _predict_sink_logits(models: dict[str, object], q_heads: torch.Tensor, pos: torch.Tensor, mode: str) -> torch.Tensor:
    n_heads = q_heads.shape[0]
    if mode == "static":
        return models["static"].clone()

    preds = []
    if mode == "position":
        x = pos.reshape(1, 1)
        for head in range(n_heads):
            weights = models["position"][head]["weights"]
            preds.append((torch.cat([x, torch.ones(1, 1)], dim=1) @ weights).squeeze(0))
        return torch.stack(preds)

    if not mode.startswith("rank"):
        raise ValueError(f"unknown predictor mode: {mode}")
    rank = int(mode.removeprefix("rank"))
    for head in range(n_heads):
        head_model = models["rank"][rank][head]
        centered = q_heads[head : head + 1] - head_model["mean"].reshape(1, -1)
        rank_features = centered @ head_model["basis"]
        features = torch.cat([rank_features, pos.reshape(1, 1)], dim=1)
        weights = head_model["weights"]
        preds.append((torch.cat([features, torch.ones(1, 1)], dim=1) @ weights).squeeze(0))
    return torch.stack(preds)


METRIC_KEYS = ("sink_logit_rmse", "sink_mass_mae", "attention_l1", "output_rel_l2")
PREDICTOR_MODES = ("static", "position", "rank1", "rank2", "rank4", "rank8")


def _attention_error_metrics(
    exact_logits: torch.Tensor,
    approx_sink_logits: torch.Tensor,
    values: torch.Tensor,
    sink_tokens: int,
) -> dict[str, float]:
    approx_logits = exact_logits.clone()
    approx_logits[:, :sink_tokens] = approx_sink_logits
    exact_probs = torch.softmax(exact_logits, dim=-1)
    approx_probs = torch.softmax(approx_logits, dim=-1)
    exact_out = torch.einsum("hl,hld->hd", exact_probs, values)
    approx_out = torch.einsum("hl,hld->hd", approx_probs, values)
    return {
        "sink_logit_rmse": float(torch.sqrt(torch.mean((exact_logits[:, :sink_tokens] - approx_sink_logits) ** 2))),
        "sink_mass_mae": float(
            torch.mean(torch.abs(exact_probs[:, :sink_tokens].sum(dim=-1) - approx_probs[:, :sink_tokens].sum(dim=-1)))
        ),
        "attention_l1": float(torch.mean(torch.sum(torch.abs(exact_probs - approx_probs), dim=-1))),
        "output_rel_l2": float(torch.linalg.norm(exact_out - approx_out) / torch.linalg.norm(exact_out).clamp_min(1e-8)),
    }


def _attention_error_metrics_by_head(
    exact_logits: torch.Tensor,
    approx_sink_logits: torch.Tensor,
    values: torch.Tensor,
    sink_tokens: int,
) -> list[dict[str, float]]:
    """Return the same drift metrics without averaging across heads."""

    approx_logits = exact_logits.clone()
    approx_logits[:, :sink_tokens] = approx_sink_logits
    exact_probs = torch.softmax(exact_logits, dim=-1)
    approx_probs = torch.softmax(approx_logits, dim=-1)
    exact_out = torch.einsum("hl,hld->hd", exact_probs, values)
    approx_out = torch.einsum("hl,hld->hd", approx_probs, values)
    rows = []
    for head in range(exact_logits.shape[0]):
        rows.append(
            {
                "sink_logit_rmse": float(
                    torch.sqrt(torch.mean((exact_logits[head, :sink_tokens] - approx_sink_logits[head]) ** 2))
                ),
                "sink_mass_mae": float(
                    torch.abs(
                        exact_probs[head, :sink_tokens].sum(dim=-1)
                        - approx_probs[head, :sink_tokens].sum(dim=-1)
                    )
                ),
                "attention_l1": float(torch.sum(torch.abs(exact_probs[head] - approx_probs[head]), dim=-1)),
                "output_rel_l2": float(
                    torch.linalg.norm(exact_out[head] - approx_out[head])
                    / torch.linalg.norm(exact_out[head]).clamp_min(1e-8)
                ),
            }
        )
    return rows


def _collect_fit_samples(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("real_qk_sink_softmax_output_probe currently supports GPT2-style models only")
    model.eval()
    layer_q: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_pos: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_sink_logits: dict[int, list[torch.Tensor]] = defaultdict(list)
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
                q_heads = _split_heads(q, model.config.n_head)[0]
                k_heads = _split_heads(k, model.config.n_head)[0]
                head_dim = q_heads.shape[-1]
                sink_keys = k_heads[:, :sink_tokens, :]
                query_vecs = q_heads[:, query_positions, :].permute(1, 0, 2)
                sink_logits = torch.einsum("qhd,hsd->qhs", query_vecs, sink_keys) / math.sqrt(head_dim)
                layer_q[layer_idx].append(query_vecs)
                layer_pos[layer_idx].append(norm_pos)
                layer_sink_logits[layer_idx].append(sink_logits)

    samples = {}
    for layer in sorted(layer_q):
        samples[layer] = {
            "q": torch.cat(layer_q[layer], dim=0),
            "pos": torch.cat(layer_pos[layer], dim=0),
            "sink_logits": torch.cat(layer_sink_logits[layer], dim=0),
        }
    meta = {"n_layers": len(samples), "n_samples": sum(int(row["q"].shape[0]) for row in samples.values())}
    return samples, meta


def _evaluate_output_errors(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
    predictors: dict[int, dict[str, object]],
    splits: dict[int, int],
    modes: tuple[str, ...],
    eval_ranges: dict[int, tuple[int, int | None]] | None = None,
    eval_indices: dict[int, set[int]] | None = None,
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, dict[int, dict[str, dict[str, float]]]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    layer_offsets = defaultdict(int)
    sums: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    head_sums: dict[int, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    head_counts: dict[int, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            seq_len = encoded["input_ids"].shape[1]
            if seq_len <= sink_tokens + 2:
                continue
            outputs = model(**encoded, output_hidden_states=True)
            query_positions = torch.arange(sink_tokens, seq_len)
            for layer_idx, block in enumerate(model.transformer.h):
                hidden = outputs.hidden_states[layer_idx].detach().float()
                qkv = block.attn.c_attn(hidden)
                q, k, v = qkv.split(model.config.n_embd, dim=2)
                q_heads = _split_heads(q, model.config.n_head)[0]
                k_heads = _split_heads(k, model.config.n_head)[0]
                v_heads = _split_heads(v, model.config.n_head)[0]
                head_dim = q_heads.shape[-1]
                denom = max(1, seq_len - 1)
                for query_pos in query_positions.tolist():
                    sample_idx = layer_offsets[layer_idx]
                    layer_offsets[layer_idx] += 1
                    if eval_indices is not None and sample_idx not in eval_indices[layer_idx]:
                        continue
                    if eval_ranges:
                        start, end = eval_ranges.get(layer_idx, (splits[layer_idx], None))
                    elif eval_indices is not None:
                        start, end = 0, None
                    else:
                        start, end = splits[layer_idx], None
                    if sample_idx < start or (end is not None and sample_idx >= end):
                        continue
                    q_vec = q_heads[:, query_pos, :]
                    keys = k_heads[:, : query_pos + 1, :]
                    values = v_heads[:, : query_pos + 1, :]
                    exact_logits = torch.einsum("hd,hld->hl", q_vec, keys) / math.sqrt(head_dim)
                    norm_pos = torch.tensor([query_pos / denom], dtype=torch.float32)
                    for mode in modes:
                        approx_sink_logits = _predict_sink_logits(predictors[layer_idx], q_vec, norm_pos, mode)
                        metrics = _attention_error_metrics(exact_logits, approx_sink_logits, values, sink_tokens)
                        for key, value in metrics.items():
                            sums[layer_idx][mode][key] += value
                        counts[layer_idx][mode] += 1
                        for head, head_metrics in enumerate(
                            _attention_error_metrics_by_head(exact_logits, approx_sink_logits, values, sink_tokens)
                        ):
                            for key, value in head_metrics.items():
                                head_sums[layer_idx][head][mode][key] += value
                            head_counts[layer_idx][head][mode] += 1

    rows = {}
    for layer, by_mode in sums.items():
        rows[layer] = {}
        for mode, metric_sums in by_mode.items():
            rows[layer][mode] = {key: value / counts[layer][mode] for key, value in metric_sums.items()}
    head_rows = {}
    for layer, by_head in head_sums.items():
        head_rows[layer] = {}
        for head, by_mode in by_head.items():
            head_rows[layer][head] = {}
            for mode, metric_sums in by_mode.items():
                head_rows[layer][head][mode] = {
                    key: value / head_counts[layer][head][mode] for key, value in metric_sums.items()
                }
    return rows, head_rows


def _summarize(rows: dict[int, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    modes = sorted({mode for by_mode in rows.values() for mode in by_mode})
    summary = {}
    for mode in modes:
        metric_keys = rows[next(iter(rows))][mode].keys()
        summary[mode] = {metric: mean(rows[layer][mode][metric] for layer in rows) for metric in metric_keys}
    return summary


def _summarize_heads(head_rows: dict[int, dict[int, dict[str, dict[str, float]]]]) -> dict[str, dict[str, float]]:
    modes = sorted({mode for by_head in head_rows.values() for by_mode in by_head.values() for mode in by_mode})
    summary = {}
    for mode in modes:
        summary[mode] = {
            metric: mean(
                head_rows[layer][head][mode][metric]
                for layer in head_rows
                for head in head_rows[layer]
            )
            for metric in METRIC_KEYS
        }
    return summary


def _mean_ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci95": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "ci95": 0.0}
    return {"mean": mean(values), "ci95": 1.96 * stdev(values) / math.sqrt(len(values))}


def _paired_head_improvements(
    head_rows: dict[int, dict[int, dict[str, dict[str, float]]]],
    *,
    baseline: str = "position",
) -> dict[str, dict[str, float]]:
    """Summarize layer-head paired improvements over a baseline predictor.

    Positive values are better because all metrics are errors.
    """

    modes = sorted({mode for by_head in head_rows.values() for by_mode in by_head.values() for mode in by_mode})
    result = {}
    for mode in modes:
        if mode == baseline:
            continue
        mode_result: dict[str, float] = {}
        n_layer_heads = 0
        for metric in METRIC_KEYS:
            improvements = []
            for layer in head_rows:
                for head in head_rows[layer]:
                    if baseline not in head_rows[layer][head] or mode not in head_rows[layer][head]:
                        continue
                    improvements.append(
                        head_rows[layer][head][baseline][metric] - head_rows[layer][head][mode][metric]
                    )
            stats = _mean_ci95(improvements)
            wins = sum(value > 0.0 for value in improvements)
            n_layer_heads = max(n_layer_heads, len(improvements))
            mode_result[f"{metric}_improvement_mean"] = stats["mean"]
            mode_result[f"{metric}_improvement_ci95"] = stats["ci95"]
            mode_result[f"{metric}_win_rate"] = wins / len(improvements) if improvements else float("nan")
        mode_result["n_layer_heads"] = n_layer_heads
        result[mode] = mode_result
    return result


def _status(
    summary: dict[str, dict[str, float]],
    paired_head_vs_position: dict[str, dict[str, float]] | None = None,
) -> str:
    rank2 = summary.get("rank2")
    position = summary.get("position")
    if rank2 and position and rank2["output_rel_l2"] < position["output_rel_l2"] and rank2["output_rel_l2"] <= 0.15:
        if paired_head_vs_position:
            paired_rank2 = paired_head_vs_position.get("rank2", {})
            improvement = paired_rank2.get("output_rel_l2_improvement_mean", 0.0)
            ci95 = paired_rank2.get("output_rel_l2_improvement_ci95", 0.0)
            win_rate = paired_rank2.get("output_rel_l2_win_rate", 0.0)
            if improvement <= ci95 or win_rate < 0.5:
                return "WEAKLY ALIVE for GPU gate; aggregate rank-2 improves, but paired per-head gains are concentrated."
        return "ALIVE for GPU gate; rank-2 improves output error over position-only with bounded drift."
    if rank2 and rank2["output_rel_l2"] <= 0.25:
        return "WEAKLY ALIVE; rank-2 output drift is bounded but not clearly better than position-only."
    return "WEAKENED; low-rank sink-logit prediction causes too much softmax/output drift."


def _write_markdown(result: dict[str, object]) -> None:
    summary = result["summary"]
    lines = [
        "# SinkAware Per-Head Softmax/Output Error Probe",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['n_traces']}",
        f"- held-out token samples per layer: {result['heldout_samples_per_layer']}",
        f"- sink tokens: {result['sink_tokens']}",
        "",
        "This probe keeps all non-sink QK scores exact and replaces only fixed sink-token logits.",
        "It measures softmax drift and attention-output error on held-out tokens. It is not a GPU timing result.",
        "",
        "## Mean Across Layers",
        "",
        "| Predictor | Sink-logit RMSE | Sink-mass MAE | Attention L1 | Output rel-L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in PREDICTOR_MODES:
        metrics = summary[mode]
        lines.append(
            "| {mode} | {sink_logit_rmse:.4f} | {sink_mass_mae:.4f} | {attention_l1:.4f} | {output_rel_l2:.4f} |".format(
                mode=mode,
                **metrics,
            )
        )
    paired = result["paired_head_vs_position"]
    rank2 = paired["rank2"]
    lines.extend(
        [
            "",
            "## Layer-Head Paired Improvement vs Position-Only",
            "",
            f"Layer-head cells: {int(rank2['n_layer_heads'])}. Positive values are lower error than the position-only predictor.",
            "",
            "| Predictor | Output rel-L2 improvement | 95% CI | Output win rate | Sink-mass MAE improvement | Attention L1 improvement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in ["static", "rank1", "rank2", "rank4", "rank8"]:
        metrics = paired[mode]
        lines.append(
            "| {mode} | {output_rel_l2_improvement_mean:.4f} | +/- {output_rel_l2_improvement_ci95:.4f} | {output_rel_l2_win_rate:.3f} | {sink_mass_mae_improvement_mean:.4f} | {attention_l1_improvement_mean:.4f} |".format(
                mode=mode,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Exact static sink reuse remains killed because sink logits are query-dependent.",
            "Rank-2 is the only current low-rank compromise that stays below exact four-sink QK cost; its aggregate improvement and weak paired per-head gains are the Mac-local evidence for a correctness gate.",
            "The relevant question is whether a cheap per-head low-rank approximation preserves softmax and output quality well enough to justify a native kernel gate.",
        ]
    )
    (OUT_DIR / "real_qk_sink_softmax_output_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    fit_samples, meta = _collect_fit_samples(args.model_name, texts, args.max_length, args.sink_tokens)
    ranks = (1, 2, 4, 8)
    predictors = {}
    splits = {}
    for layer, tensors in fit_samples.items():
        split = int(0.67 * tensors["q"].shape[0])
        splits[layer] = split
        predictors[layer] = _fit_layer_predictors(
            tensors["q"][:split],
            tensors["pos"][:split],
            tensors["sink_logits"][:split],
            ranks,
        )
    modes = PREDICTOR_MODES
    rows, head_rows = _evaluate_output_errors(
        args.model_name, texts, args.max_length, args.sink_tokens, predictors, splits, modes
    )
    summary = _summarize(rows)
    head_summary = _summarize_heads(head_rows)
    paired_head_vs_position = _paired_head_improvements(head_rows)
    result = {
        "model_name": args.model_name,
        "n_traces": len(texts),
        "n_samples": meta["n_samples"],
        "heldout_samples_per_layer": min(int(fit_samples[layer]["q"].shape[0]) - splits[layer] for layer in fit_samples),
        "sink_tokens": args.sink_tokens,
        "rows": rows,
        "head_rows": head_rows,
        "summary": summary,
        "head_summary": head_summary,
        "paired_head_vs_position": paired_head_vs_position,
        "status": _status(summary, paired_head_vs_position),
    }
    (OUT_DIR / "real_qk_sink_softmax_output_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
