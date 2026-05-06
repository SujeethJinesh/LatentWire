"""Downstream loss/control gate for approximate SinkAware attention.

This Phase 3 gate moves beyond isolated attention-output drift. It patches the
actual GPT2/OPT attention modules during a full causal-LM forward pass and
compares:

- exact baseline attention,
- exact sink-logit replacement through the SinkAware path as a no-op control,
- position-only sink-logit replacement,
- rank-2 sink-logit replacement.

Predictors are still fit separately per model and split. This is not
cross-model predictor transfer, downstream benchmark success, or GPU evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import types
from contextlib import contextmanager
from statistics import mean
from typing import Any, Iterator

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from ..phase2.rank2_cross_model_falsification_gate import (
        _collect_fit_samples,
        _model_family,
        _split_heads,
    )
    from ..phase2.rank2_trace_frozen_split_gate import _split_trace_indices
    from ..phase2.real_qk_sink_softmax_output_probe import (
        OUT_DIR as PHASE2_OUT_DIR,
        ROOT,
        _fit_layer_predictors,
        _mean_ci95,
        _predict_sink_logits,
    )
    from ..phase2.real_query_sink_probe import DEFAULT_TRACES, _load_texts
except ImportError:  # pragma: no cover - supports direct script execution.
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.append(str(ROOT / "experimental/sinkaware/phase2"))
    from rank2_cross_model_falsification_gate import (  # type: ignore
        _collect_fit_samples,
        _model_family,
        _split_heads,
    )
    from rank2_trace_frozen_split_gate import _split_trace_indices  # type: ignore
    from real_qk_sink_softmax_output_probe import (  # type: ignore
        OUT_DIR as PHASE2_OUT_DIR,
        _fit_layer_predictors,
        _mean_ci95,
        _predict_sink_logits,
    )
    from real_query_sink_probe import DEFAULT_TRACES, _load_texts  # type: ignore


OUT_DIR = PHASE2_OUT_DIR.parent / "phase3"
PATCH_MODES = ("exact", "position", "rank2")


def _sink_predictions_for_query(
    *,
    predictors: dict[int, dict[str, object]],
    layer_idx: int,
    predictor_q: torch.Tensor,
    keys: torch.Tensor,
    query_pos: int,
    sink_tokens: int,
    mode: str,
    scaling: float,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return replacement sink logits with any additive mask preserved.

    Shapes:
    - predictor_q: [batch, heads, seq, head_dim], unscaled for predictor modes.
    - keys: [batch, heads, seq, head_dim].
    """

    if mode == "exact":
        predicted = torch.matmul(
            predictor_q[:, :, query_pos : query_pos + 1, :].float(),
            keys[:, :, :sink_tokens, :].float().transpose(-1, -2),
        ).squeeze(2) * scaling
    elif mode in {"position", "rank2"}:
        denom = max(1, predictor_q.shape[2] - 1)
        pos = torch.tensor([query_pos / denom], dtype=torch.float32, device=predictor_q.device)
        predicted_rows = []
        for batch_idx in range(predictor_q.shape[0]):
            predicted_rows.append(
                _predict_sink_logits(
                    predictors[layer_idx],
                    predictor_q[batch_idx, :, query_pos, :].detach().float(),
                    pos,
                    mode,
                )
            )
        predicted = torch.stack(predicted_rows, dim=0)
    else:
        raise ValueError(f"unknown SinkAware patch mode: {mode}")

    if attention_mask is not None:
        predicted = predicted + attention_mask[:, :, query_pos, :sink_tokens].to(predicted.dtype)
    return predicted


def _replace_fixed_sink_logits(
    attn_logits: torch.Tensor,
    *,
    predictors: dict[int, dict[str, object]],
    layer_idx: int,
    predictor_q: torch.Tensor,
    keys: torch.Tensor,
    sink_tokens: int,
    mode: str,
    scaling: float,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Replace only fixed-sink columns for valid post-sink query positions."""

    if sink_tokens <= 0:
        raise ValueError("sink_tokens must be positive")
    if predictor_q.shape[2] <= sink_tokens or attn_logits.shape[-1] <= sink_tokens:
        return attn_logits
    patched = attn_logits.clone()
    last_query = min(predictor_q.shape[2], attn_logits.shape[-2])
    for query_pos in range(sink_tokens, last_query):
        patched[:, :, query_pos, :sink_tokens] = _sink_predictions_for_query(
            predictors=predictors,
            layer_idx=layer_idx,
            predictor_q=predictor_q,
            keys=keys,
            query_pos=query_pos,
            sink_tokens=sink_tokens,
            mode=mode,
            scaling=scaling,
            attention_mask=attention_mask,
        ).to(patched.dtype)
    return patched


def _gpt2_sinkaware_forward(
    attn: torch.nn.Module,
    *,
    layer_idx: int,
    predictors: dict[int, dict[str, object]],
    mode: str,
    sink_tokens: int,
):
    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        past_key_values: object | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None or encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise RuntimeError("downstream_quality_control_gate only supports full self-attention forwards")

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        key_shape = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(key_shape).transpose(1, 2)
        value_states = value_states.view(key_shape).transpose(1, 2)
        query_shape = (*query_states.shape[:-1], -1, self.head_dim)
        query_heads = query_states.view(query_shape).transpose(1, 2)

        attn_logits = torch.matmul(query_heads, key_states.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask
        attn_logits = _replace_fixed_sink_logits(
            attn_logits,
            predictors=predictors,
            layer_idx=layer_idx,
            predictor_q=query_heads,
            keys=key_states,
            sink_tokens=sink_tokens,
            mode=mode,
            scaling=float(self.scaling),
            attention_mask=attention_mask,
        )
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights.type(value_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout.p if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights if output_attentions else None

    return types.MethodType(forward, attn)


def _opt_sinkaware_forward(
    attn: torch.nn.Module,
    *,
    layer_idx: int,
    predictors: dict[int, dict[str, object]],
    mode: str,
    sink_tokens: int,
):
    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        past_key_values: object | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None:
            raise RuntimeError("downstream_quality_control_gate only supports full self-attention forwards")

        batch_size, target_len, _ = hidden_states.size()
        query_raw = self.q_proj(hidden_states)
        query_scaled = query_raw * self.scaling
        query_states = query_scaled.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        predictor_q = query_raw.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(query_states, key_states.transpose(-1, -2))
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask
        attn_logits = _replace_fixed_sink_logits(
            attn_logits,
            predictors=predictors,
            layer_idx=layer_idx,
            predictor_q=predictor_q,
            keys=key_states,
            sink_tokens=sink_tokens,
            mode=mode,
            scaling=float(self.scaling),
            attention_mask=attention_mask,
        )
        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=0.0 if not self.training else self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, target_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None

    return types.MethodType(forward, attn)


@contextmanager
def _sinkaware_attention_patch(
    model: torch.nn.Module,
    predictors: dict[int, dict[str, object]],
    *,
    mode: str,
    sink_tokens: int,
) -> Iterator[None]:
    family = _model_family(model)
    originals: list[tuple[torch.nn.Module, object]] = []
    if family == "gpt2":
        layers = [block.attn for block in model.transformer.h]
        patcher = _gpt2_sinkaware_forward
    elif family == "opt":
        layers = [layer.self_attn for layer in model.model.decoder.layers]
        patcher = _opt_sinkaware_forward
    else:  # pragma: no cover - guarded by _model_family.
        raise AssertionError(f"unsupported family: {family}")

    try:
        for layer_idx, attn in enumerate(layers):
            if layer_idx not in predictors:
                continue
            originals.append((attn, attn.forward))
            attn.forward = patcher(
                attn,
                layer_idx=layer_idx,
                predictors=predictors,
                mode=mode,
                sink_tokens=sink_tokens,
            )
        yield
    finally:
        for module, original in originals:
            module.forward = original


def _fit_predictors(
    model_name: str,
    train_texts: list[str],
    max_length: int,
    sink_tokens: int,
) -> dict[int, dict[str, object]]:
    fit_samples, _ = _collect_fit_samples(model_name, train_texts, max_length, sink_tokens)
    return {
        layer: _fit_layer_predictors(
            tensors["q"],
            tensors["pos"],
            tensors["sink_logits"],
            ranks=(2,),
        )
        for layer, tensors in fit_samples.items()
    }


def _token_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    exact_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    shifted_logits = logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    n_tokens = int(labels.numel())
    result = {
        "nll_sum": float(losses.sum()),
        "n_tokens": float(n_tokens),
        "loss": float(losses.mean()),
    }
    if exact_logits is not None:
        exact_shifted = exact_logits[:, :-1, :].float()
        exact_log_probs = F.log_softmax(exact_shifted, dim=-1)
        mode_log_probs = F.log_softmax(shifted_logits, dim=-1)
        exact_probs = exact_log_probs.exp()
        kl_by_token = (exact_probs * (exact_log_probs - mode_log_probs)).sum(dim=-1)
        exact_top1 = exact_shifted.argmax(dim=-1)
        mode_top1 = shifted_logits.argmax(dim=-1)
        result.update(
            {
                "kl_sum": float(kl_by_token.sum()),
                "top1_disagree_count": float((exact_top1 != mode_top1).sum()),
            }
        )
    return result


def _empty_mode_totals() -> dict[str, float]:
    return {
        "nll_sum": 0.0,
        "n_tokens": 0.0,
        "kl_sum": 0.0,
        "top1_disagree_count": 0.0,
    }


def _accumulate(target: dict[str, float], metrics: dict[str, float]) -> None:
    for key in target:
        target[key] += float(metrics.get(key, 0.0))


def _finalize_mode(totals: dict[str, float], baseline_loss: float | None = None) -> dict[str, float]:
    n_tokens = max(1.0, totals["n_tokens"])
    loss = totals["nll_sum"] / n_tokens
    row = {
        "loss": loss,
        "n_tokens": totals["n_tokens"],
        "loss_delta_vs_exact": 0.0 if baseline_loss is None else loss - baseline_loss,
    }
    if baseline_loss is not None:
        row.update(
            {
                "abs_loss_delta_vs_exact": abs(loss - baseline_loss),
                "mean_kl_to_exact": totals["kl_sum"] / n_tokens,
                "top1_disagreement_rate": totals["top1_disagree_count"] / n_tokens,
            }
        )
    return row


def _evaluate_seed(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    seed: int,
) -> dict[str, Any]:
    train_indices, test_indices = _split_trace_indices(len(texts), train_fraction, seed)
    train_texts = [texts[idx] for idx in train_indices]
    test_texts = [texts[idx] for idx in test_indices]
    predictors = _fit_predictors(model_name, train_texts, max_length, sink_tokens)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").float()
    model.eval()
    family = _model_family(model)
    baseline_totals = _empty_mode_totals()
    mode_totals = {mode: _empty_mode_totals() for mode in PATCH_MODES}

    with torch.no_grad():
        for text in test_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] <= sink_tokens + 2:
                continue
            exact_logits = model(**encoded, use_cache=False).logits.detach()
            _accumulate(baseline_totals, _token_metrics(exact_logits, input_ids))
            for mode in PATCH_MODES:
                with _sinkaware_attention_patch(model, predictors, mode=mode, sink_tokens=sink_tokens):
                    mode_logits = model(**encoded, use_cache=False).logits.detach()
                _accumulate(mode_totals[mode], _token_metrics(mode_logits, input_ids, exact_logits=exact_logits))

    baseline = _finalize_mode(baseline_totals)
    modes = {
        mode: _finalize_mode(totals, baseline_loss=baseline["loss"])
        for mode, totals in mode_totals.items()
    }
    return {
        "seed": seed,
        "model_name": model_name,
        "model_family": family,
        "train_traces": len(train_texts),
        "heldout_traces": len(test_texts),
        "train_trace_indices": train_indices,
        "heldout_trace_indices": test_indices,
        "exact_baseline": baseline,
        "modes": modes,
        "rank2_abs_loss_delta_improvement_vs_position": (
            modes["position"]["abs_loss_delta_vs_exact"] - modes["rank2"]["abs_loss_delta_vs_exact"]
        ),
        "rank2_kl_improvement_vs_position": modes["position"]["mean_kl_to_exact"] - modes["rank2"]["mean_kl_to_exact"],
        "rank2_top1_improvement_vs_position": (
            modes["position"]["top1_disagreement_rate"] - modes["rank2"]["top1_disagreement_rate"]
        ),
    }


def _aggregate_seed_rows(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_abs_loss = [float(row["modes"]["exact"]["abs_loss_delta_vs_exact"]) for row in seed_rows]
    exact_kl = [float(row["modes"]["exact"]["mean_kl_to_exact"]) for row in seed_rows]
    rank2_loss_improvements = [
        float(row["rank2_abs_loss_delta_improvement_vs_position"])
        for row in seed_rows
    ]
    rank2_kl_improvements = [float(row["rank2_kl_improvement_vs_position"]) for row in seed_rows]
    rank2_top1_improvements = [float(row["rank2_top1_improvement_vs_position"]) for row in seed_rows]
    rank2_abs_loss = [float(row["modes"]["rank2"]["abs_loss_delta_vs_exact"]) for row in seed_rows]
    position_abs_loss = [float(row["modes"]["position"]["abs_loss_delta_vs_exact"]) for row in seed_rows]
    return {
        "exact_abs_loss_delta_vs_baseline": _mean_ci95(exact_abs_loss),
        "exact_mean_kl_to_baseline": _mean_ci95(exact_kl),
        "position_abs_loss_delta_vs_baseline": _mean_ci95(position_abs_loss),
        "rank2_abs_loss_delta_vs_baseline": _mean_ci95(rank2_abs_loss),
        "rank2_abs_loss_delta_improvement_vs_position": _mean_ci95(rank2_loss_improvements),
        "rank2_kl_improvement_vs_position": _mean_ci95(rank2_kl_improvements),
        "rank2_top1_disagreement_improvement_vs_position": _mean_ci95(rank2_top1_improvements),
        "all_seeds_exact_noop_ok": {
            "value": all(value <= 1e-5 for value in exact_abs_loss) and all(value <= 1e-7 for value in exact_kl),
            "n": len(seed_rows),
        },
        "all_seeds_rank2_closer_by_loss": {
            "value": all(value > 0.0 for value in rank2_loss_improvements),
            "n": len(seed_rows),
        },
        "all_seeds_rank2_closer_by_kl": {
            "value": all(value > 0.0 for value in rank2_kl_improvements),
            "n": len(seed_rows),
        },
        "min_rank2_abs_loss_delta_improvement_vs_position": {
            "value": min(rank2_loss_improvements) if rank2_loss_improvements else float("nan"),
        },
    }


def _model_status(aggregate: dict[str, Any]) -> str:
    exact_ok = bool(aggregate["all_seeds_exact_noop_ok"]["value"])
    loss_ok = bool(aggregate["all_seeds_rank2_closer_by_loss"]["value"])
    kl_ok = bool(aggregate["all_seeds_rank2_closer_by_kl"]["value"])
    mean_loss_improvement = float(aggregate["rank2_abs_loss_delta_improvement_vs_position"]["mean"])
    if exact_ok and loss_ok and kl_ok:
        return "ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL."
    if exact_ok and (loss_ok or mean_loss_improvement > 0.0):
        return "WEAKLY ALIVE on this model; rank-2 improves loss drift but not every downstream control."
    if not exact_ok:
        return "INVALID CONTROL on this model; exact replacement is not a no-op."
    return "WEAKENED on this model; downstream controls do not favor rank-2 over position-only."


def _run_model(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    seed_rows = [
        _evaluate_seed(model_name, texts, max_length, sink_tokens, train_fraction, seed)
        for seed in seeds
    ]
    aggregate = _aggregate_seed_rows(seed_rows)
    family = seed_rows[0]["model_family"] if seed_rows else "unknown"
    return {
        "model_name": model_name,
        "model_family": family,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "status": _model_status(aggregate),
    }


def _aggregate_models(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    model_loss_improvements = [
        float(row["aggregate"]["rank2_abs_loss_delta_improvement_vs_position"]["mean"])
        for row in model_results
    ]
    model_kl_improvements = [
        float(row["aggregate"]["rank2_kl_improvement_vs_position"]["mean"])
        for row in model_results
    ]
    return {
        "rank2_abs_loss_delta_improvement_across_models": _mean_ci95(model_loss_improvements),
        "rank2_kl_improvement_across_models": _mean_ci95(model_kl_improvements),
        "all_models_exact_noop_ok": {
            "value": all(bool(row["aggregate"]["all_seeds_exact_noop_ok"]["value"]) for row in model_results),
            "n": len(model_results),
        },
        "all_models_rank2_closer_by_loss": {
            "value": all(value > 0.0 for value in model_loss_improvements),
            "n": len(model_results),
        },
        "all_models_rank2_closer_by_kl": {
            "value": all(value > 0.0 for value in model_kl_improvements),
            "n": len(model_results),
        },
        "min_model_rank2_abs_loss_delta_improvement": {
            "value": min(model_loss_improvements) if model_loss_improvements else float("nan"),
        },
    }


def _status(aggregate: dict[str, Any]) -> str:
    exact_ok = bool(aggregate["all_models_exact_noop_ok"]["value"])
    loss_ok = bool(aggregate["all_models_rank2_closer_by_loss"]["value"])
    kl_ok = bool(aggregate["all_models_rank2_closer_by_kl"]["value"])
    if exact_ok and loss_ok and kl_ok:
        return "ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke."
    if not exact_ok:
        return "INVALID CONTROL; exact SinkAware replacement did not reproduce exact attention."
    if loss_ok:
        return "WEAKLY ALIVE; rank-2 improves downstream loss drift but not every behavior control."
    return "WEAKENED; downstream GPT2/OPT controls do not favor rank-2 over position-only."


def _run(
    model_names: tuple[str, ...],
    max_traces: int,
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    texts = _load_texts(DEFAULT_TRACES, max_traces)
    model_results = [
        _run_model(model_name, texts, max_length, sink_tokens, train_fraction, seeds)
        for model_name in model_names
    ]
    aggregate = _aggregate_models(model_results)
    return {
        "model_names": list(model_names),
        "max_traces": len(texts),
        "max_length": max_length,
        "sink_tokens": sink_tokens,
        "train_fraction": train_fraction,
        "seeds": list(seeds),
        "model_results": model_results,
        "aggregate": aggregate,
        "status": _status(aggregate),
    }


def _write_markdown(result: dict[str, Any], output_path: os.PathLike[str] | str | None = None) -> None:
    aggregate = result["aggregate"]
    loss = aggregate["rank2_abs_loss_delta_improvement_across_models"]
    kl = aggregate["rank2_kl_improvement_across_models"]
    lines = [
        "# SinkAware Downstream Quality/Control Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- models: {', '.join(f'`{name}`' for name in result['model_names'])}",
        f"- traces: {result['max_traces']}",
        f"- max length: {result['max_length']}",
        f"- sink tokens: {result['sink_tokens']}",
        f"- train fraction: {result['train_fraction']}",
        f"- seeds: {', '.join(str(seed) for seed in result['seeds'])}",
        "",
        "This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.",
        "",
        "## Aggregate Across Models",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---:|---:|",
        f"| rank-2 absolute loss-delta improvement vs position-only | {loss['mean']:.6f} | +/- {loss['ci95']:.6f} |",
        f"| rank-2 KL-to-exact improvement vs position-only | {kl['mean']:.6f} | +/- {kl['ci95']:.6f} |",
        f"| minimum model loss-delta improvement | {aggregate['min_model_rank2_abs_loss_delta_improvement']['value']:.6f} | |",
        "",
        "Positive improvement means rank-2 is closer to exact baseline behavior than position-only.",
        "",
        "## Per Model",
        "",
        "| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["model_results"]:
        agg = row["aggregate"]
        lines.append(
            "| {model} | {family} | {status} | {exact:.8f} | {position:.8f} | {rank2:.8f} | {loss_imp:+.8f} | {kl_imp:+.8f} | {exact_ok} |".format(
                model=row["model_name"],
                family=row["model_family"],
                status=row["status"],
                exact=agg["exact_abs_loss_delta_vs_baseline"]["mean"],
                position=agg["position_abs_loss_delta_vs_baseline"]["mean"],
                rank2=agg["rank2_abs_loss_delta_vs_baseline"]["mean"],
                loss_imp=agg["rank2_abs_loss_delta_improvement_vs_position"]["mean"],
                kl_imp=agg["rank2_kl_improvement_vs_position"]["mean"],
                exact_ok="yes" if agg["all_seeds_exact_noop_ok"]["value"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Per Seed",
            "",
            "| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_row in result["model_results"]:
        for seed_row in model_row["seed_rows"]:
            exact_loss = seed_row["exact_baseline"]["loss"]
            modes = seed_row["modes"]
            lines.append(
                "| {model} | {seed} | {heldout} | {exact_loss:.6f} | {exact_delta:+.8f} | {position_delta:+.8f} | {rank2_delta:+.8f} | {position_kl:.8f} | {rank2_kl:.8f} | {position_top1:.4f} | {rank2_top1:.4f} |".format(
                    model=model_row["model_name"],
                    seed=seed_row["seed"],
                    heldout=seed_row["heldout_traces"],
                    exact_loss=exact_loss,
                    exact_delta=modes["exact"]["loss_delta_vs_exact"],
                    position_delta=modes["position"]["loss_delta_vs_exact"],
                    rank2_delta=modes["rank2"]["loss_delta_vs_exact"],
                    position_kl=modes["position"]["mean_kl_to_exact"],
                    rank2_kl=modes["rank2"]["mean_kl_to_exact"],
                    position_top1=modes["position"]["top1_disagreement_rate"],
                    rank2_top1=modes["rank2"]["top1_disagreement_rate"],
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.",
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "downstream_quality_control_gate.md" if output_path is None else Path(output_path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", nargs="+", default=["distilgpt2", "facebook/opt-125m"])
    parser.add_argument("--max-traces", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--artifact-stem", default="downstream_quality_control_gate")
    args = parser.parse_args()

    cache_dir = ROOT / ".debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run(
        tuple(args.model_names),
        args.max_traces,
        args.max_length,
        args.sink_tokens,
        args.train_fraction,
        tuple(args.seeds),
    )
    json_path = OUT_DIR / f"{args.artifact_stem}.json"
    md_path = OUT_DIR / f"{args.artifact_stem}.md"
    json_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(result, md_path)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
