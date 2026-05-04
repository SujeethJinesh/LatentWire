from __future__ import annotations

"""Strict ARC-Challenge behavior-residual Sparse Resonance Packet gate.

This gate tests a receiver/objective change after source-only and target-aligned
PCA packets failed.  Instead of reconstructing hidden coordinates, it fits a
train-only source-hidden-to-target-behavior residual map and transmits a sparse
candidate-local residual packet.  The receiver decodes the packet as a small
target score correction, then the usual source-private destructive controls ask
whether the packet carries causal source information.
"""

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as soft_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_behavior_residual_packet_gate_20260504_"
    "tinyllama_to_qwen3_disagreement"
)
DEFAULT_VALIDATION = soft_gate.DEFAULT_VALIDATION
DEFAULT_TEST = soft_gate.DEFAULT_TEST
DEFAULT_SOURCE_FAMILY_GATE_DIR = soft_gate.DEFAULT_SOURCE_FAMILY_GATE_DIR
DEFAULT_TINY_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_TINY_VALIDATION_SCORE_CACHE
DEFAULT_TINY_TEST_SCORE_CACHE = soft_gate.DEFAULT_TINY_TEST_SCORE_CACHE
DEFAULT_QWEN_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_QWEN_VALIDATION_SCORE_CACHE
DEFAULT_QWEN_TEST_SCORE_CACHE = soft_gate.DEFAULT_QWEN_TEST_SCORE_CACHE
DEFAULT_TINY_MODEL = pathlib.Path(
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
    "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)
DEFAULT_QWEN3_MODEL = pathlib.Path(
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)

MATCHED_CONDITION = "matched_behavior_residual_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll",
    "candidate_derangement",
    "packet_only_source_index",
    "source_rank_control",
    "source_score_control",
    "same_byte_visible_text",
    "qwen_substituted_packet",
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
STRICT_REQUIRED_CONTROLS = CONTROL_CONDITIONS


@dataclass(frozen=True)
class RidgeScalarMap:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    weights: np.ndarray
    ridge: float
    fit_mse: float
    fit_r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        standardized = (values - self.x_mean) / self.x_std
        return (standardized @ self.weights + self.y_mean).astype(np.float64, copy=False)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _load_score_rows(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    return {str(row["content_id"]): row for row in soft_gate._read_jsonl(path)}


def _source_scores_for_row(row: arc_gate.ArcRow, cache: dict[str, dict[str, Any]]) -> list[float]:
    cached = cache.get(row.content_id)
    if cached is None:
        raise ValueError(f"missing score cache row content_id={row.content_id}")
    scores = [float(value) for value in cached.get("source_scores", ())]
    if len(scores) != len(row.choices):
        raise ValueError(f"source score length mismatch content_id={row.content_id}")
    return scores


def _source_prediction_for_row(row: arc_gate.ArcRow, cache: dict[str, dict[str, Any]]) -> int:
    cached = cache.get(row.content_id)
    if cached is None:
        raise ValueError(f"missing prediction cache row content_id={row.content_id}")
    selected = int(cached["source_selected_index"])
    if selected < 0 or selected >= len(row.choices):
        raise ValueError(f"invalid source_selected_index content_id={row.content_id}")
    return selected


def _prediction(scores: Sequence[float]) -> int:
    return int(max(range(len(scores)), key=lambda index: (float(scores[index]), -index)))


def _margin(scores: Sequence[float], answer_index: int) -> float:
    gold = float(scores[int(answer_index)])
    distractors = [float(score) for index, score in enumerate(scores) if index != int(answer_index)]
    return gold - max(distractors) if distractors else gold


def _softmax(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray([float(score) for score in scores], dtype=np.float64)
    shifted = values - float(values.max()) if values.size else values
    exps = np.exp(shifted)
    total = float(exps.sum())
    if total <= 1e-12:
        return np.full_like(values, 1.0 / max(values.size, 1))
    return exps / total


def _entropy(scores: Sequence[float]) -> float:
    probs = _softmax(scores)
    return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))) if probs.size else 0.0


def _source_index_scores(choice_count: int, selected_index: int) -> list[float]:
    return preflight._source_index_scores(choice_count, selected_index)


def _source_rank_scores(raw_scores: Sequence[float]) -> list[float]:
    return preflight._source_rank_scores(raw_scores)


def _centered_source_score_control(raw_scores: Sequence[float]) -> list[float]:
    return preflight._centered_source_score_control(raw_scores)


def _row_offsets(rows: Sequence[arc_gate.ArcRow]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for row in rows:
        end = start + len(row.choices)
        offsets.append((start, end))
        start = end
    return offsets


def _candidate_targets(rows: Sequence[arc_gate.ArcRow], target_scores: Sequence[Sequence[float]]) -> np.ndarray:
    targets: list[float] = []
    for row, scores in zip(rows, target_scores, strict=True):
        probs = _softmax(scores)
        for candidate_index in range(len(row.choices)):
            gold = 1.0 if candidate_index == int(row.answer_index) else 0.0
            targets.append(float(gold - probs[candidate_index]))
    return np.asarray(targets, dtype=np.float64)


def _target_score_features(rows: Sequence[arc_gate.ArcRow], target_scores: Sequence[Sequence[float]]) -> np.ndarray:
    features: list[list[float]] = []
    for row, scores in zip(rows, target_scores, strict=True):
        values = np.asarray(scores, dtype=np.float64)
        probs = _softmax(scores)
        order = sorted(range(len(values)), key=lambda index: (-values[index], index))
        ranks = {index: rank for rank, index in enumerate(order)}
        centered = values - float(values.mean())
        scale = float(centered.std())
        if not math.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        for candidate_index in range(len(row.choices)):
            features.append(
                [
                    float(values[candidate_index]),
                    float(centered[candidate_index] / scale),
                    float(probs[candidate_index]),
                    float(len(row.choices) - ranks[candidate_index]) / float(max(len(row.choices), 1)),
                    float(_margin(scores, candidate_index)),
                ]
            )
    return np.asarray(features, dtype=np.float64)


def _fit_ridge_scalar_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_indices: np.ndarray,
    ridge: float,
) -> RidgeScalarMap:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("features must be rank-2 and targets rank-1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature/target row mismatch")
    if fit.size == 0:
        raise ValueError("fit_indices must not be empty")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    x_mean = x[fit].mean(axis=0, keepdims=True)
    x_std = x[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    x_std_all = (x - x_mean) / x_std
    x_fit = x_std_all[fit]
    y_mean = float(y[fit].mean())
    y_fit = y[fit] - y_mean
    kernel = x_fit @ x_fit.T
    if ridge > 0.0:
        kernel = kernel + float(ridge) * np.eye(kernel.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(kernel, y_fit)
    weights = x_fit.T @ alpha
    pred_fit = x_fit @ weights + y_mean
    mse = float(np.mean(np.square(y[fit] - pred_fit)))
    baseline = float(np.mean(np.square(y[fit] - y_mean)))
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - mse / baseline
    return RidgeScalarMap(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        weights=weights,
        ridge=float(ridge),
        fit_mse=mse,
        fit_r2=float(r2),
    )


def _rows_from_candidate_values(rows: Sequence[arc_gate.ArcRow], values: np.ndarray) -> list[np.ndarray]:
    row_values: list[np.ndarray] = []
    for start, end in _row_offsets(rows):
        row = np.asarray(values[start:end], dtype=np.float64)
        row_values.append(row - float(row.mean()) if row.size else row)
    return row_values


def _quantize_sparse_row_packets(
    row_coeffs: Sequence[np.ndarray],
    *,
    fit_row_count: int,
    top_k: int,
    quant_bits: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if top_k < 1:
        raise ValueError("packet_top_k must be at least 1")
    if quant_bits < 1:
        raise ValueError("packet_bits must be at least 1")
    fit_sparse_values: list[float] = []
    max_choices = 0
    sparse_masks: list[np.ndarray] = []
    for row_index, coeffs in enumerate(row_coeffs):
        values = np.asarray(coeffs, dtype=np.float64)
        max_choices = max(max_choices, int(values.size))
        retained = min(int(top_k), int(values.size))
        mask = np.zeros(values.shape, dtype=bool)
        if retained > 0:
            top = np.argsort(-np.abs(values))[:retained]
            mask[top] = True
        sparse_masks.append(mask)
        if row_index < int(fit_row_count):
            fit_sparse_values.extend(float(abs(value)) for value in values[mask])
    scale = float(max(fit_sparse_values)) if fit_sparse_values else 0.0
    signed_levels = int((2 ** (int(quant_bits) - 1)) - 1)
    if scale <= 1e-12 or signed_levels < 1:
        step = 0.0
    else:
        step = scale / float(signed_levels)
    decoded: list[np.ndarray] = []
    nonzero_counts: list[int] = []
    energy_ratios: list[float] = []
    for values, mask in zip(row_coeffs, sparse_masks, strict=True):
        sparse = np.zeros_like(np.asarray(values, dtype=np.float64))
        sparse[mask] = np.asarray(values, dtype=np.float64)[mask]
        if step <= 0.0:
            quantized = np.zeros_like(sparse, dtype=np.int64)
            dequantized = np.zeros_like(sparse)
        else:
            quantized = np.clip(np.rint(sparse / step), -signed_levels, signed_levels).astype(np.int64)
            dequantized = quantized.astype(np.float64) * step
        decoded.append(dequantized)
        nonzero_counts.append(int(np.count_nonzero(quantized)))
        denom = float(np.square(values).sum())
        energy_ratios.append(float(np.square(dequantized).sum() / denom) if denom > 1e-12 else 0.0)
    atom_id_bits = int(math.ceil(math.log2(max(max_choices, 2))))
    packet_bits_per_row = int(min(top_k, max_choices) * (atom_id_bits + int(quant_bits)))
    return decoded, {
        "kind": "candidate_local_behavior_residual_packet",
        "top_k": int(top_k),
        "quant_bits": int(quant_bits),
        "atom_id_bits": int(atom_id_bits),
        "max_choices": int(max_choices),
        "packet_bits_per_row": int(packet_bits_per_row),
        "packet_bytes_per_row": float(packet_bits_per_row / 8.0),
        "framed_packet_bytes_per_row": int(math.ceil(packet_bits_per_row / 8.0)),
        "cache_line_bytes_per_row_64b": int(math.ceil(max(1, math.ceil(packet_bits_per_row / 8.0)) / 64.0) * 64),
        "dma_bytes_per_row_128b": int(math.ceil(max(1, math.ceil(packet_bits_per_row / 8.0)) / 128.0) * 128),
        "quant_scale": float(scale),
        "quant_step": float(step),
        "fit_mean_nonzero_coefficients": float(statistics.fmean(nonzero_counts[:fit_row_count]))
        if fit_row_count
        else 0.0,
        "fit_sparse_energy_ratio": float(statistics.fmean(energy_ratios[:fit_row_count]))
        if fit_row_count
        else 0.0,
    }


def _fused_scores(target_scores: Sequence[float], residual: Sequence[float], *, residual_weight: float) -> list[float]:
    target = np.asarray(target_scores, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if target.shape != correction.shape:
        raise ValueError("target/residual score shape mismatch")
    return [float(value) for value in target + float(residual_weight) * correction]


def _choose_residual_weight(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
    residuals: Sequence[np.ndarray],
) -> dict[str, Any]:
    candidates = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0]
    best: dict[str, Any] | None = None
    for weight in candidates:
        correct = 0
        margins: list[float] = []
        for row, scores, residual in zip(rows, target_scores, residuals, strict=True):
            fused = _fused_scores(scores, residual, residual_weight=weight)
            correct += int(_prediction(fused) == int(row.answer_index))
            margins.append(_margin(fused, int(row.answer_index)))
        accuracy = correct / max(len(rows), 1)
        mean_margin = float(statistics.fmean(margins)) if margins else 0.0
        row = {"weight": float(weight), "accuracy": float(accuracy), "mean_margin": mean_margin}
        if best is None or (row["accuracy"], row["mean_margin"], -abs(row["weight"])) > (
            best["accuracy"],
            best["mean_margin"],
            -abs(best["weight"]),
        ):
            best = row
    if best is None:
        raise ValueError("could not choose residual weight")
    return best


def _score_rows_with_prompt_builder(
    rows: Sequence[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    local_files_only: bool,
    normalization: str,
    prompt_builder: Callable[[arc_gate.ArcRow], str],
    attn_implementation: str | None,
) -> tuple[list[list[float]], dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if normalization not in {"mean", "sum"}:
        raise ValueError(f"unknown normalization {normalization!r}")
    resolved_device = "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "torch_dtype": arc_gate._torch_dtype(dtype),
    }
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(resolved_device)
    model.eval()

    def score_texts(texts: list[str], *, prompt_len: int) -> list[float]:
        padding_mode: bool | str = True
        tokenizer_max_length = int(max_length)
        if str(resolved_device).startswith("mps"):
            raw = tokenizer(texts, padding=False, truncation=True, max_length=max_length)["input_ids"]
            row_max = max(len(ids) for ids in raw)
            tokenizer_max_length = min(max_length, int(math.ceil(row_max / 32.0) * 32))
            padding_mode = "max_length"
        encoded = tokenizer(
            texts,
            padding=padding_mode,
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=False)
        logits = output.logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        attention = encoded["attention_mask"][:, 1:].bool()
        out: list[float] = []
        for batch_index in range(len(texts)):
            valid = attention[batch_index].clone()
            valid[: max(0, prompt_len - 1)] = False
            values = token_logp[batch_index][valid]
            if values.numel() == 0:
                out.append(float("-inf"))
            elif normalization == "sum":
                out.append(float(values.sum().detach().cpu()))
            else:
                out.append(float(values.mean().detach().cpu()))
        return out

    scores: list[list[float]] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            prompt = prompt_builder(row)
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            scores.append(score_texts([prompt + " " + choice for choice in row.choices], prompt_len=prompt_len))
    del model, tokenizer
    preflight._release_torch_model_memory(device=resolved_device)
    gc.collect()
    return scores, {
        "kind": "local_causal_lm_choice_loglikelihood_with_prompt_builder",
        "model_path": model_path,
        "device": resolved_device,
        "dtype": dtype,
        "max_length": int(max_length),
        "normalization": normalization,
        "attn_implementation": attn_implementation or "auto",
        "latency_s": float(time.perf_counter() - start),
    }


def _paired_bootstrap(deltas: Sequence[float], *, seed: int, samples: int) -> dict[str, float]:
    values = [float(value) for value in deltas]
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(values)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _condition_metrics(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for condition in REPORT_CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        correct = sum(1 for row in subset if row["correct"])
        metrics[condition] = {
            "n": len(subset),
            "correct": int(correct),
            "accuracy": float(correct / len(subset)) if subset else 0.0,
            "mean_margin": float(statistics.fmean(float(row["margin"]) for row in subset)) if subset else 0.0,
        }
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    for control in CONTROL_CONDITIONS:
        correct_deltas = [
            float(group[MATCHED_CONDITION]["correct"]) - float(group[control]["correct"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and control in group
        ]
        margin_deltas = [
            float(group[MATCHED_CONDITION]["margin"]) - float(group[control]["margin"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and control in group
        ]
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{control}"] = _paired_bootstrap(
            correct_deltas,
            seed=seed + len(control),
            samples=bootstrap_samples,
        )
        metrics[MATCHED_CONDITION][f"mean_margin_delta_vs_{control}"] = (
            float(statistics.fmean(margin_deltas)) if margin_deltas else 0.0
        )
    return metrics


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["strict_headline"]
    lines = [
        "# ARC-Challenge Behavior-Residual Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/test disagreement rows: `{payload['train_rows']}` / `{payload['test_rows']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.6f}`",
        f"- best required control: `{headline['best_required_control']}`",
        f"- best required control accuracy: `{headline['best_required_control_accuracy']:.6f}`",
        f"- worst required paired CI95 low: `{headline['worst_required_ci95_low']:.6f}`",
        f"- packet bytes/row: `{payload['systems_packet_sideband']['packet_bytes_per_row']:.3f}`",
        "",
        "## Strict Controls",
        "",
        "| Control | Accuracy | Delta | CI95 low |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["strict_control_metrics"].items():
        lines.append(
            f"| `{name}` | {row['control_accuracy']:.6f} | {row['delta_accuracy']:.6f} | "
            f"{row['ci95_low']:.6f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    source_family_gate_dir: pathlib.Path,
    tiny_validation_score_cache: pathlib.Path,
    tiny_test_score_cache: pathlib.Path,
    qwen_validation_score_cache: pathlib.Path,
    qwen_test_score_cache: pathlib.Path,
    train_disagreement_limit: int,
    test_disagreement_limit: int,
    source_model: str,
    target_model: str,
    source_device: str,
    target_device: str,
    target_attn_implementation: str | None,
    dtype: str,
    source_max_length: int,
    target_max_length: int,
    source_hidden_layer: int,
    source_feature_dim: int,
    ridge: float,
    packet_top_k: int,
    packet_bits: int,
    residual_weight: float | None,
    local_files_only: bool,
    bootstrap_samples: int,
    same_byte_budget: int,
    min_accuracy_gap: float,
    min_ci_low: float,
    seed: int,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "strict_inputs"
    agreement_path = _resolve(source_family_gate_dir) / "source_cache_agreement.csv"

    validation_rows_all = arc_gate._load_rows(_resolve(validation_path))
    test_rows_all = arc_gate._load_rows(_resolve(test_path))
    train_ids = soft_gate._read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="validation",
        limit=train_disagreement_limit,
    )
    test_ids = soft_gate._read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="test",
        limit=test_disagreement_limit,
    )
    train_rows = soft_gate._filter_rows_by_content_ids(validation_rows_all, train_ids)
    test_rows = soft_gate._filter_rows_by_content_ids(test_rows_all, test_ids)
    overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in test_rows})
    if overlap:
        raise ValueError(f"train/test content overlap: {overlap[:3]}")
    rows = [*train_rows, *test_rows]
    fit_row_count = len(train_rows)
    fit_candidate_indices = preflight._flat_candidate_indices_for_rows(rows, list(range(fit_row_count)))

    _write_jsonl(input_dir / "arc_challenge_validation_train_plus_test_disagreement.jsonl", [soft_gate._arc_row_payload(row) for row in rows])
    _write_jsonl(
        input_dir / "tinyllama_source_score_cache.jsonl",
        [
            *soft_gate._select_cache_rows(cache_path=tiny_validation_score_cache, rows=train_rows),
            *soft_gate._select_cache_rows(cache_path=tiny_test_score_cache, rows=test_rows),
        ],
    )
    _write_jsonl(
        input_dir / "qwen_source_score_cache.jsonl",
        [
            *soft_gate._select_cache_rows(cache_path=qwen_validation_score_cache, rows=train_rows),
            *soft_gate._select_cache_rows(cache_path=qwen_test_score_cache, rows=test_rows),
        ],
    )
    tiny_cache = _load_score_rows(input_dir / "tinyllama_source_score_cache.jsonl")
    qwen_cache = _load_score_rows(input_dir / "qwen_source_score_cache.jsonl")

    target_scores, target_score_meta = _score_rows_with_prompt_builder(
        rows,
        model_path=target_model,
        device=target_device,
        dtype=dtype,
        max_length=target_max_length,
        local_files_only=local_files_only,
        normalization="mean",
        prompt_builder=preflight._mcq_prompt,
        attn_implementation=target_attn_implementation,
    )
    target_predictions = [_prediction(scores) for scores in target_scores]
    gc.collect()
    preflight._release_torch_model_memory(device=target_device)

    def same_byte_prompt(row: arc_gate.ArcRow) -> str:
        selected = _source_prediction_for_row(row, tiny_cache)
        hint = row.choices[selected].encode("utf-8")[:same_byte_budget].decode("utf-8", errors="ignore")
        choices = "\n".join(f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True))
        return (
            "Answer the science question with the best answer.\n"
            f"Question: {row.question}\n"
            f"Choices:\n{choices}\n"
            f"Source model selected this visible hint: {hint}\n"
            "Answer:"
        )

    same_byte_scores, same_byte_meta = _score_rows_with_prompt_builder(
        test_rows,
        model_path=target_model,
        device=target_device,
        dtype=dtype,
        max_length=target_max_length,
        local_files_only=local_files_only,
        normalization="mean",
        prompt_builder=same_byte_prompt,
        attn_implementation=target_attn_implementation,
    )

    flat_hidden, hidden_meta = preflight._hf_choice_hidden_features(
        rows,
        model_path=source_model,
        device=source_device,
        dtype=dtype,
        max_length=source_max_length,
        local_files_only=local_files_only,
        hidden_layer=source_hidden_layer,
    )
    public_flat, public_meta = preflight._public_candidate_hashed_features(rows, feature_dim=source_feature_dim)
    flat_hidden, innovation_meta = preflight._public_candidate_innovation_features(
        flat_hidden,
        public_flat,
        fit_flat_indices=fit_candidate_indices,
        ridge=ridge,
    )
    behavior_targets = _candidate_targets(rows, target_scores)
    source_map = _fit_ridge_scalar_map(
        flat_hidden,
        behavior_targets,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    source_predicted = source_map.predict(flat_hidden)
    source_row_residuals = _rows_from_candidate_values(rows, source_predicted)
    decoded_source_residuals, packet_meta = _quantize_sparse_row_packets(
        source_row_residuals,
        fit_row_count=fit_row_count,
        top_k=packet_top_k,
        quant_bits=packet_bits,
    )

    target_features = _target_score_features(rows, target_scores)
    target_map = _fit_ridge_scalar_map(
        target_features,
        behavior_targets,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_predicted = target_map.predict(target_features)
    target_row_residuals = _rows_from_candidate_values(rows, target_predicted)
    decoded_target_residuals, target_packet_meta = _quantize_sparse_row_packets(
        target_row_residuals,
        fit_row_count=fit_row_count,
        top_k=packet_top_k,
        quant_bits=packet_bits,
    )

    if residual_weight is None:
        selected_weight = _choose_residual_weight(
            train_rows,
            target_scores[:fit_row_count],
            decoded_source_residuals[:fit_row_count],
        )
        source_weight = float(selected_weight["weight"])
    else:
        selected_weight = {"weight": float(residual_weight), "accuracy": None, "mean_margin": None}
        source_weight = float(residual_weight)
    selected_target_weight = _choose_residual_weight(
        train_rows,
        target_scores[:fit_row_count],
        decoded_target_residuals[:fit_row_count],
    )
    target_weight = float(selected_target_weight["weight"])

    prediction_rows: list[dict[str, Any]] = []
    eval_offset = fit_row_count
    for eval_position, row in enumerate(test_rows):
        row_index = eval_offset + eval_position
        target = [float(score) for score in target_scores[row_index]]
        source_residual = decoded_source_residuals[row_index]
        target_residual = decoded_target_residuals[row_index]
        shuffled_index = eval_offset + ((eval_position + 1) % len(test_rows))
        raw_source_scores = _source_scores_for_row(row, tiny_cache)
        source_selected = _source_prediction_for_row(row, tiny_cache)
        qwen_selected = _source_prediction_for_row(row, qwen_cache)
        condition_scores = {
            MATCHED_CONDITION: _fused_scores(target, source_residual, residual_weight=source_weight),
            "target_only": target,
            "target_derived_packet": _fused_scores(target, target_residual, residual_weight=target_weight),
            "zero_source": _fused_scores(target, np.zeros_like(source_residual), residual_weight=source_weight),
            "source_row_shuffle": _fused_scores(
                target,
                decoded_source_residuals[shuffled_index],
                residual_weight=source_weight,
            ),
            "atom_shuffle": _fused_scores(target, np.roll(source_residual, 1), residual_weight=source_weight),
            "coefficient_shuffle": _fused_scores(target, source_residual[::-1], residual_weight=source_weight),
            "top_atom_knockout": _fused_scores(
                target,
                _top_atom_knockout(source_residual),
                residual_weight=source_weight,
            ),
            "candidate_roll": _fused_scores(target, np.roll(source_residual, 1), residual_weight=source_weight),
            "candidate_derangement": list(np.roll(_fused_scores(target, source_residual, residual_weight=source_weight), 1)),
            "packet_only_source_index": _source_index_scores(len(row.choices), source_selected),
            "source_rank_control": _source_rank_scores(raw_source_scores),
            "source_score_control": _centered_source_score_control(raw_source_scores),
            "same_byte_visible_text": same_byte_scores[eval_position],
            "qwen_substituted_packet": _source_index_scores(len(row.choices), qwen_selected),
        }
        for condition in REPORT_CONDITIONS:
            scores = [float(score) for score in condition_scores[condition]]
            pred = _prediction(scores)
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "condition": condition,
                    "answer_index": int(row.answer_index),
                    "answer_label": row.answer_label,
                    "prediction_index": int(pred),
                    "prediction_label": row.choice_labels[pred],
                    "correct": bool(pred == int(row.answer_index)),
                    "scores": scores,
                    "margin": float(_margin(scores, row.answer_index)),
                    "entropy": float(_entropy(scores)),
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "qwen_substituted_index": int(qwen_selected),
                    "qwen_substituted_label": row.choice_labels[qwen_selected],
                    "source_scores": [float(score) for score in raw_source_scores],
                    "source_rank_by_candidate": [
                        int(rank)
                        for rank in np.argsort(np.argsort(-np.asarray(raw_source_scores, dtype=np.float64)))
                    ],
                    "source_score_margin": float(
                        sorted(raw_source_scores, reverse=True)[0] - sorted(raw_source_scores, reverse=True)[1]
                    )
                    if len(raw_source_scores) > 1
                    else 0.0,
                    "packet_residual": [float(value) for value in source_residual],
                    "packet_residual_weight": float(source_weight),
                    "control_origin": condition,
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed, bootstrap_samples=bootstrap_samples)
    matched = metrics[MATCHED_CONDITION]
    strict_control_metrics: dict[str, dict[str, float]] = {}
    for control in STRICT_REQUIRED_CONTROLS:
        paired = matched[f"paired_accuracy_vs_{control}"]
        strict_control_metrics[control] = {
            "control_accuracy": float(metrics[control]["accuracy"]),
            "delta_accuracy": float(matched["accuracy"] - metrics[control]["accuracy"]),
            "ci95_low": float(paired["ci95_low"]),
            "ci95_high": float(paired["ci95_high"]),
        }
    best_required_control = max(STRICT_REQUIRED_CONTROLS, key=lambda name: metrics[name]["accuracy"])
    worst_ci_low = min(row["ci95_low"] for row in strict_control_metrics.values())
    strict_pass = all(
        row["delta_accuracy"] >= float(min_accuracy_gap) and row["ci95_low"] > float(min_ci_low)
        for row in strict_control_metrics.values()
    )
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "source_private_arc_challenge_behavior_residual_packet_gate",
        "date": dt.date.today().isoformat(),
        "created_utc": created,
        "pass_gate": bool(strict_pass),
        "implementation_gate_only": False,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "strict_required_controls": list(STRICT_REQUIRED_CONTROLS),
        "strict_control_metrics": strict_control_metrics,
        "strict_headline": {
            "matched_accuracy": float(matched["accuracy"]),
            "best_required_control": best_required_control,
            "best_required_control_accuracy": float(metrics[best_required_control]["accuracy"]),
            "worst_required_ci95_low": float(worst_ci_low),
        },
        "condition_metrics": metrics,
        "systems_packet_sideband": {
            "source_private": True,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "native_serving_throughput_measured": False,
            "packet_bytes_per_row": float(packet_meta["packet_bytes_per_row"]),
            "framed_packet_bytes_per_row": int(packet_meta["framed_packet_bytes_per_row"]),
            "cache_line_bytes_per_row_64b": int(packet_meta["cache_line_bytes_per_row_64b"]),
            "dma_bytes_per_row_128b": int(packet_meta["dma_bytes_per_row_128b"]),
            "sparse_packet_metadata": packet_meta,
            "note": (
                "Byte counts cover the sparse behavior-residual packet sideband only. They are not native GPU "
                "throughput, HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "feature_metadata": {
            "source_hidden": hidden_meta,
            "public": public_meta,
            "source_public_innovation": innovation_meta,
            "source_behavior_map": {
                "ridge": source_map.ridge,
                "fit_mse": source_map.fit_mse,
                "fit_r2": source_map.fit_r2,
                "residual_weight_selection": selected_weight,
            },
            "target_derived_behavior_map": {
                "ridge": target_map.ridge,
                "fit_mse": target_map.fit_mse,
                "fit_r2": target_map.fit_r2,
                "residual_weight_selection": selected_target_weight,
                "sparse_packet": target_packet_meta,
            },
            "target_score_metadata": target_score_meta,
            "same_byte_score_metadata": same_byte_meta,
        },
        "inputs": {
            "validation_path": _display(validation_path),
            "test_path": _display(test_path),
            "source_family_gate_dir": _display(source_family_gate_dir),
            "agreement_path": _display(agreement_path),
            "tiny_validation_score_cache": _display(tiny_validation_score_cache),
            "tiny_test_score_cache": _display(tiny_test_score_cache),
            "qwen_validation_score_cache": _display(qwen_validation_score_cache),
            "qwen_test_score_cache": _display(qwen_test_score_cache),
            "source_model": str(source_model),
            "target_model": str(target_model),
            "train_disagreement_limit": int(train_disagreement_limit),
            "test_disagreement_limit": int(test_disagreement_limit),
            "packet_top_k": int(packet_top_k),
            "packet_bits": int(packet_bits),
            "same_byte_budget": int(same_byte_budget),
        },
        "interpretation": (
            "This gate tests whether behavior-aligned candidate-local residual packets solve the receiver/objective "
            "failure seen in PCA coordinate packets. It passes only if the sparse source packet beats target-only, "
            "target-derived, source-destroying, source-index/rank/score, same-byte text, and Qwen-substitution controls "
            "with positive paired uncertainty on frozen test disagreement rows."
        ),
    }
    json_path = output_dir / "arc_challenge_behavior_residual_packet_gate.json"
    md_path = output_dir / "arc_challenge_behavior_residual_packet_gate.md"
    audit_path = output_dir / "prediction_audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_jsonl(audit_path, prediction_rows)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {"path": _display(json_path), "sha256": _sha256_file(json_path), "bytes": json_path.stat().st_size},
            {"path": _display(md_path), "sha256": _sha256_file(md_path), "bytes": md_path.stat().st_size},
            {"path": _display(audit_path), "sha256": _sha256_file(audit_path), "bytes": audit_path.stat().st_size},
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"headline": payload["strict_headline"], "pass_gate": payload["pass_gate"]}, sort_keys=True))
    return payload


def _top_atom_knockout(values: Sequence[float]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.size:
        out[int(np.argmax(np.abs(out)))] = 0.0
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--source-family-gate-dir", type=pathlib.Path, default=DEFAULT_SOURCE_FAMILY_GATE_DIR)
    parser.add_argument("--tiny-validation-score-cache", type=pathlib.Path, default=DEFAULT_TINY_VALIDATION_SCORE_CACHE)
    parser.add_argument("--tiny-test-score-cache", type=pathlib.Path, default=DEFAULT_TINY_TEST_SCORE_CACHE)
    parser.add_argument("--qwen-validation-score-cache", type=pathlib.Path, default=DEFAULT_QWEN_VALIDATION_SCORE_CACHE)
    parser.add_argument("--qwen-test-score-cache", type=pathlib.Path, default=DEFAULT_QWEN_TEST_SCORE_CACHE)
    parser.add_argument("--train-disagreement-limit", type=int, default=8)
    parser.add_argument("--test-disagreement-limit", type=int, default=8)
    parser.add_argument("--source-model", default=str(DEFAULT_TINY_MODEL))
    parser.add_argument("--target-model", default=str(DEFAULT_QWEN3_MODEL))
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=160)
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--packet-top-k", type=int, default=2)
    parser.add_argument("--packet-bits", type=int, default=4)
    parser.add_argument("--residual-weight", type=float, default=None)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--same-byte-budget", type=int, default=4096)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_gate(
        output_dir=args.output_dir,
        validation_path=args.validation_path,
        test_path=args.test_path,
        source_family_gate_dir=args.source_family_gate_dir,
        tiny_validation_score_cache=args.tiny_validation_score_cache,
        tiny_test_score_cache=args.tiny_test_score_cache,
        qwen_validation_score_cache=args.qwen_validation_score_cache,
        qwen_test_score_cache=args.qwen_test_score_cache,
        train_disagreement_limit=int(args.train_disagreement_limit),
        test_disagreement_limit=int(args.test_disagreement_limit),
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        source_device=str(args.source_device),
        target_device=str(args.target_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        source_max_length=int(args.source_max_length),
        target_max_length=int(args.target_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        source_feature_dim=int(args.source_feature_dim),
        ridge=float(args.ridge),
        packet_top_k=int(args.packet_top_k),
        packet_bits=int(args.packet_bits),
        residual_weight=None if args.residual_weight is None else float(args.residual_weight),
        local_files_only=str(args.local_files_only).lower() == "true",
        bootstrap_samples=int(args.bootstrap_samples),
        same_byte_budget=int(args.same_byte_budget),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
