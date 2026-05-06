"""Small cross-model falsification gate for all-head rank-2 SinkAware.

This gate is the Mac-local fallback when Triton interpreter execution is not
available. It asks whether the all-head rank-2 approximation survives a
held-out model family under the same whole-trace split protocol. Predictors are
fit per model; this is not cross-model predictor transfer and not a speed
claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .rank2_trace_frozen_split_gate import _split_trace_indices
    from .real_qk_sink_softmax_output_probe import (
        OUT_DIR,
        ROOT,
        _attention_error_metrics,
        _attention_error_metrics_by_head,
        _fit_layer_predictors,
        _mean_ci95,
        _paired_head_improvements,
        _predict_sink_logits,
        _summarize,
    )
    from .real_query_sink_probe import DEFAULT_TRACES, _load_texts
except ImportError:  # pragma: no cover - supports direct script execution.
    from rank2_trace_frozen_split_gate import _split_trace_indices
    from real_qk_sink_softmax_output_probe import (
        OUT_DIR,
        ROOT,
        _attention_error_metrics,
        _attention_error_metrics_by_head,
        _fit_layer_predictors,
        _mean_ci95,
        _paired_head_improvements,
        _predict_sink_logits,
        _summarize,
    )
    from real_query_sink_probe import DEFAULT_TRACES, _load_texts


def _split_heads(projected: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Return [heads, seq, head_dim] from a [1, seq, hidden] projection."""

    if projected.ndim != 3 or projected.shape[0] != 1:
        raise ValueError(f"expected [1, seq, hidden] projection, got {tuple(projected.shape)}")
    batch, seq_len, hidden = projected.shape
    if hidden % n_heads != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by {n_heads} heads")
    return projected.view(batch, seq_len, n_heads, hidden // n_heads).permute(0, 2, 1, 3)[0]


def _model_family(model: torch.nn.Module) -> str:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        first_layer = model.model.decoder.layers[0]
        if all(hasattr(first_layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            return "opt"
    raise RuntimeError("rank2_cross_model_falsification_gate supports GPT2-style and OPT-style models only")


def _n_layers(model: torch.nn.Module) -> int:
    family = _model_family(model)
    if family == "gpt2":
        return len(model.transformer.h)
    if family == "opt":
        return len(model.model.decoder.layers)
    raise AssertionError(f"unhandled model family: {family}")


def _project_qkv(
    model: torch.nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project model hidden states into raw Q/K/V heads for one layer."""

    family = _model_family(model)
    hidden = hidden_states[layer_idx].detach().float()
    if family == "gpt2":
        block = model.transformer.h[layer_idx]
        attn_input = block.ln_1(hidden)
        qkv = block.attn.c_attn(attn_input)
        q, k, v = qkv.split(model.config.n_embd, dim=2)
        n_heads = int(model.config.n_head)
        return _split_heads(q, n_heads), _split_heads(k, n_heads), _split_heads(v, n_heads)

    if family == "opt":
        layer = model.model.decoder.layers[layer_idx]
        attn_input = hidden
        if getattr(model.config, "do_layer_norm_before", False) and hasattr(layer, "self_attn_layer_norm"):
            attn_input = layer.self_attn_layer_norm(attn_input)
        n_heads = int(model.config.num_attention_heads)
        q = layer.self_attn.q_proj(attn_input)
        k = layer.self_attn.k_proj(attn_input)
        v = layer.self_attn.v_proj(attn_input)
        return _split_heads(q, n_heads), _split_heads(k, n_heads), _split_heads(v, n_heads)

    raise AssertionError(f"unhandled model family: {family}")


def _collect_fit_samples(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).float()
    family = _model_family(model)
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
            for layer_idx in range(_n_layers(model)):
                q_heads, k_heads, _ = _project_qkv(model, outputs.hidden_states, layer_idx)
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
    return samples, {
        "family": family,
        "n_layers": len(samples),
        "n_samples": sum(int(row["q"].shape[0]) for row in samples.values()),
    }


def _evaluate_output_errors(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
    predictors: dict[int, dict[str, object]],
    modes: tuple[str, ...],
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, dict[int, dict[str, dict[str, float]]]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).float()
    model.eval()

    sums: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    head_sums: dict[int, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    head_counts: dict[int, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            seq_len = encoded["input_ids"].shape[1]
            if seq_len <= sink_tokens + 2:
                continue
            outputs = model(**encoded, output_hidden_states=True)
            query_positions = torch.arange(sink_tokens, seq_len)
            denom = max(1, seq_len - 1)
            for layer_idx in range(_n_layers(model)):
                if layer_idx not in predictors:
                    continue
                q_heads, k_heads, v_heads = _project_qkv(model, outputs.hidden_states, layer_idx)
                head_dim = q_heads.shape[-1]
                for query_pos in query_positions.tolist():
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


def _run_model(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
    train_fraction: float,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    seed_rows = []
    family = None
    modes = ("position", "rank2")
    for seed in seeds:
        train_indices, test_indices = _split_trace_indices(len(texts), train_fraction, seed)
        train_texts = [texts[idx] for idx in train_indices]
        test_texts = [texts[idx] for idx in test_indices]
        fit_samples, meta = _collect_fit_samples(model_name, train_texts, max_length, sink_tokens)
        family = meta["family"]
        predictors = {
            layer: _fit_layer_predictors(
                tensors["q"],
                tensors["pos"],
                tensors["sink_logits"],
                ranks=(2,),
            )
            for layer, tensors in fit_samples.items()
        }
        rows, head_rows = _evaluate_output_errors(model_name, test_texts, max_length, sink_tokens, predictors, modes)
        summary = _summarize(rows)
        paired = _paired_head_improvements(head_rows)
        position = summary["position"]
        rank2 = summary["rank2"]
        seed_rows.append(
            {
                "seed": seed,
                "train_traces": len(train_texts),
                "test_traces": len(test_texts),
                "train_trace_indices": train_indices,
                "test_trace_indices": test_indices,
                "train_samples": meta["n_samples"],
                "summary": summary,
                "paired_head_vs_position": paired,
                "output_rel_l2_improvement_vs_position": position["output_rel_l2"] - rank2["output_rel_l2"],
                "sink_mass_mae_improvement_vs_position": position["sink_mass_mae"] - rank2["sink_mass_mae"],
                "attention_l1_improvement_vs_position": position["attention_l1"] - rank2["attention_l1"],
            }
        )
    aggregate = _aggregate_seed_rows(seed_rows)
    return {
        "model_name": model_name,
        "model_family": family,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "status": _model_status(aggregate),
    }


def _aggregate_seed_rows(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_improvements = [
        float(row["output_rel_l2_improvement_vs_position"])
        for row in seed_rows
    ]
    sink_mass_improvements = [
        float(row["sink_mass_mae_improvement_vs_position"])
        for row in seed_rows
    ]
    attention_improvements = [
        float(row["attention_l1_improvement_vs_position"])
        for row in seed_rows
    ]
    head_win_rates = [
        float(row["paired_head_vs_position"]["rank2"]["output_rel_l2_win_rate"])
        for row in seed_rows
    ]
    return {
        "output_rel_l2_improvement_vs_position": _mean_ci95(output_improvements),
        "sink_mass_mae_improvement_vs_position": _mean_ci95(sink_mass_improvements),
        "attention_l1_improvement_vs_position": _mean_ci95(attention_improvements),
        "output_rel_l2_head_win_rate": _mean_ci95(head_win_rates),
        "all_seeds_rank2_beats_position": {
            "value": all(value > 0.0 for value in output_improvements),
            "n": len(output_improvements),
        },
        "min_output_rel_l2_improvement": {
            "value": min(output_improvements) if output_improvements else float("nan"),
        },
    }


def _model_status(aggregate: dict[str, Any]) -> str:
    output = aggregate["output_rel_l2_improvement_vs_position"]
    all_positive = bool(aggregate["all_seeds_rank2_beats_position"]["value"])
    min_improvement = float(aggregate["min_output_rel_l2_improvement"]["value"])
    if all_positive and min_improvement > 0.015 and output["mean"] > 0.02:
        return "ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits."
    if output["mean"] > 0.0:
        return "WEAKLY ALIVE on this model; mean improvement is positive but below the promotion margin."
    return "WEAKENED on this model; rank-2 does not beat position-only."


def _aggregate_models(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    output_means = [
        float(row["aggregate"]["output_rel_l2_improvement_vs_position"]["mean"])
        for row in model_results
    ]
    return {
        "output_rel_l2_improvement_across_models": _mean_ci95(output_means),
        "all_models_positive": {
            "value": all(value > 0.0 for value in output_means),
            "n": len(output_means),
        },
        "min_model_output_rel_l2_improvement": {
            "value": min(output_means) if output_means else float("nan"),
        },
    }


def _status(aggregate: dict[str, Any]) -> str:
    all_positive = bool(aggregate["all_models_positive"]["value"])
    min_improvement = float(aggregate["min_model_output_rel_l2_improvement"]["value"])
    output = aggregate["output_rel_l2_improvement_across_models"]
    if all_positive and min_improvement > 0.015 and output["mean"] > 0.02:
        return "ALIVE but bounded; rank-2 survives the small held-out/cross-family falsification smoke gate."
    if all_positive:
        return "WEAKLY ALIVE; all models are positive, but at least one row is below the promotion margin."
    return "WEAKENED; rank-2 fails at least one held-out/cross-family model row."


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


def _write_markdown(result: dict[str, Any]) -> None:
    aggregate = result["aggregate"]
    output = aggregate["output_rel_l2_improvement_across_models"]
    lines = [
        "# SinkAware Rank-2 Held-Out/Cross-Family Falsification Gate",
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
        "This is a smallest Mac-feasible falsification gate after the Triton interpreter path was blocked locally. It fits all-head rank-2 predictors separately per model on whole-trace train splits and evaluates held-out traces against a position-only predictor. It does not transfer predictors across models and makes no GPU speed claim.",
        "",
        "## Aggregate Across Models",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---:|---:|",
        f"| output rel-L2 improvement vs position | {output['mean']:.4f} | +/- {output['ci95']:.4f} |",
        f"| minimum model output improvement | {aggregate['min_model_output_rel_l2_improvement']['value']:.4f} | |",
        "",
        "Positive improvement means rank-2 has lower error than position-only.",
        "",
        "## Per Model",
        "",
        "| Model | Family | Status | Output improvement | 95% CI | Head win rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in result["model_results"]:
        model_output = row["aggregate"]["output_rel_l2_improvement_vs_position"]
        head_win = row["aggregate"]["output_rel_l2_head_win_rate"]
        lines.append(
            "| {model} | {family} | {status} | {improvement:.4f} | +/- {ci95:.4f} | {head_win:.3f} |".format(
                model=row["model_name"],
                family=row["model_family"],
                status=row["status"],
                improvement=model_output["mean"],
                ci95=model_output["ci95"],
                head_win=head_win["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Per Seed",
            "",
            "| Model | Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_row in result["model_results"]:
        for seed_row in model_row["seed_rows"]:
            position = seed_row["summary"]["position"]
            rank2 = seed_row["summary"]["rank2"]
            paired = seed_row["paired_head_vs_position"]["rank2"]
            lines.append(
                "| {model} | {seed} | {train_traces} | {test_traces} | {position:.4f} | {rank2:.4f} | {improvement:+.4f} | {win_rate:.3f} |".format(
                    model=model_row["model_name"],
                    seed=seed_row["seed"],
                    train_traces=seed_row["train_traces"],
                    test_traces=seed_row["test_traces"],
                    position=position["output_rel_l2"],
                    rank2=rank2["output_rel_l2"],
                    improvement=seed_row["output_rel_l2_improvement_vs_position"],
                    win_rate=paired["output_rel_l2_win_rate"],
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This smoke gate is weaker than the 48-trace distilgpt2 frozen split because it uses a smaller slice, but it is stronger as a falsification attempt because it includes a held-out OPT-family model. Passing this gate keeps the branch alive only as bounded Mac-local evidence; promotion still requires larger cross-family repeats or Triton interpreter correctness.",
        ]
    )
    (OUT_DIR / "rank2_cross_model_falsification_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-names", nargs="+", default=["distilgpt2", "facebook/opt-125m"])
    parser.add_argument("--max-traces", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
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
    (OUT_DIR / "rank2_cross_model_falsification_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
