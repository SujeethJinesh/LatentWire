from __future__ import annotations

"""Run a Mac-local target-loss soft-prefix preflight for ARC/OpenBookQA.

This is an implementation gate, not a paper-positive result.  It trains only a
small connector that maps answer-key-forbidden source summaries into target
input-embedding prefix tokens, then scores multiple-choice continuations with a
frozen target LM.  The readout is whether the matched source prefix beats
target-only/static/source-destroying controls on a tiny held-out slice.
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
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_openbookqa_soft_prefix_preflight_20260503_arc_smoke")
DEFAULT_ARC_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_ARC_SOURCE_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/source_prediction_cache.jsonl"
)
DEFAULT_QWEN_SOURCE = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_QWEN_TARGET = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)

MATCHED_CONDITION = "matched_soft_prefix"
CONTROL_CONDITIONS = (
    "target_only",
    "source_free_prefix",
    "target_cache_only_prefix",
    "target_derived_prefix",
    "slots_only_prefix",
    "zero_source",
    "shuffled_source",
    "source_row_shuffle",
    "same_norm_noise",
    "train_mean_source",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "label_shuffled",
    "candidate_roll_source",
    "candidate_roll",
    "candidate_score_roll_source",
    "candidate_derangement",
    "same_byte_visible_text",
    "packet_only_source_index",
    "source_rank_control",
    "source_score_control",
    "qwen_substituted_packet",
    "source_label_copy_audit_upper_bound",
)
PASS_CONTROL_CONDITIONS = tuple(
    condition for condition in CONTROL_CONDITIONS if condition != "source_label_copy_audit_upper_bound"
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
CONTRASTIVE_CONTROL_CHOICES = (
    "zero_source",
    "shuffled_source",
    "same_norm_noise",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll_source",
    "candidate_score_roll_source",
)


@dataclass(frozen=True)
class SoftPrefixConfig:
    prefix_len: int = 4
    hidden_dim: int = 32
    epochs: int = 2
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 17
    matched_use_target: bool = False
    length_normalize: bool = True
    contrastive_weight: float = 0.0
    contrastive_margin: float = 0.05
    contrastive_loss_cap: float = 0.5
    contrastive_controls: tuple[str, ...] = ()
    sparse_packet_rank: int = 16
    sparse_packet_top_k: int = 4
    sparse_packet_bits: int = 4


class SourceSoftPrefixConnector(torch.nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        target_embed_dim: int,
        hidden_dim: int,
        prefix_len: int,
        use_source: bool,
        use_target: bool,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.target_embed_dim = int(target_embed_dim)
        self.use_source = bool(use_source)
        self.use_target = bool(use_target)
        input_dim = (int(source_dim) if self.use_source else 0) + (
            int(target_dim) if self.use_target else 0
        )
        if input_dim == 0:
            self.slots = torch.nn.Parameter(torch.randn(prefix_len, target_embed_dim) * 0.02)
            self.net = None
        else:
            self.slots = None
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, int(hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.Linear(int(hidden_dim), int(prefix_len) * int(target_embed_dim)),
            )

    def forward(self, source_summary: torch.Tensor, target_summary: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return self.slots
        parts: list[torch.Tensor] = []
        if self.use_source:
            parts.append(source_summary)
        if self.use_target:
            parts.append(target_summary)
        return self.net(torch.cat(parts, dim=-1)).view(self.prefix_len, self.target_embed_dim)


class SourceQuerySoftPrefixConnector(torch.nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        target_embed_dim: int,
        hidden_dim: int,
        prefix_len: int,
        use_target: bool,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.target_embed_dim = int(target_embed_dim)
        self.use_target = bool(use_target)
        self.source_proj = torch.nn.Linear(int(source_dim), int(hidden_dim))
        self.query = torch.nn.Parameter(torch.randn(prefix_len, int(hidden_dim)) * 0.02)
        self.target_proj = torch.nn.Linear(int(target_dim), int(hidden_dim)) if self.use_target else None
        self.out = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), int(target_embed_dim)),
        )

    def forward(self, source_summary: torch.Tensor, target_summary: torch.Tensor) -> torch.Tensor:
        if source_summary.dim() != 2:
            raise ValueError("query-pooling connector expects a [tokens, dim] source summary")
        tokens = torch.tanh(self.source_proj(source_summary))
        query = self.query
        if self.target_proj is not None:
            query = query + torch.tanh(self.target_proj(target_summary)).unsqueeze(0)
        attention = torch.softmax((query @ tokens.T) / math.sqrt(float(tokens.shape[-1])), dim=-1)
        pooled = attention @ tokens
        return self.out(pooled).view(self.prefix_len, self.target_embed_dim)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_torch_model_memory(*, device: str) -> None:
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _peak_rss_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _torch_dtype(dtype: str) -> Any:
    return arc_gate._torch_dtype(dtype)


def _read_source_cache(path: pathlib.Path) -> dict[str, int]:
    predictions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = set(row.get("forbidden_source_fields", ()))
            if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
                raise ValueError(f"source cache row {row.get('row_id')} is missing forbidden fields")
            predictions[str(row["content_id"])] = int(row["source_selected_index"])
    if not predictions:
        raise ValueError(f"{path} contained no source predictions")
    return predictions


def _read_source_score_cache(path: pathlib.Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    scores_by_content: dict[str, list[float]] = {}
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = set(row.get("forbidden_source_fields", ()))
            if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
                raise ValueError(f"source score row {row.get('row_id')} is missing forbidden fields")
            raw_scores = [float(value) for value in row.get("source_scores", ())]
            if raw_scores:
                scores_by_content[str(row["content_id"])] = raw_scores
    return scores_by_content


def _source_predictions_for_rows(
    rows: Sequence[arc_gate.ArcRow],
    source_cache: dict[str, int],
    *,
    label: str,
) -> list[int]:
    predictions: list[int] = []
    for row in rows:
        if row.content_id not in source_cache:
            raise ValueError(f"{label} source cache is missing content_id={row.content_id}")
        prediction = int(source_cache[row.content_id])
        if prediction < 0 or prediction >= len(row.choices):
            raise ValueError(
                f"{label} source cache row {row.content_id} selected invalid choice index {prediction}"
            )
        predictions.append(prediction)
    return predictions


def _select_rows_with_cache(
    rows: Sequence[arc_gate.ArcRow],
    source_cache: dict[str, int],
    *,
    row_limit: int,
) -> tuple[list[arc_gate.ArcRow], list[int]]:
    selected: list[arc_gate.ArcRow] = []
    predictions: list[int] = []
    for row in rows:
        if row.content_id not in source_cache:
            continue
        prediction = int(source_cache[row.content_id])
        if 0 <= prediction < len(row.choices):
            selected.append(row)
            predictions.append(prediction)
        if len(selected) >= row_limit:
            break
    if len(selected) < 2:
        raise ValueError("need at least two rows with source-cache predictions")
    return selected, predictions


def _mcq_prompt(row: arc_gate.ArcRow) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True)
    )
    return (
        "Answer the science multiple-choice question. Use only the listed choices.\n"
        f"Question: {row.question}\n"
        f"Choices:\n{choices}\n"
        "Best answer:"
    )


def _source_prompt(row: arc_gate.ArcRow) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True)
    )
    return (
        "Read the science question and choices. Do not reveal the answer label.\n"
        f"Question: {row.question}\nChoices:\n{choices}\n"
        "Useful evidence:"
    )


def _row_public_text(row: arc_gate.ArcRow) -> str:
    return _source_prompt(row)


def _source_choice_texts(rows: list[arc_gate.ArcRow]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        prompt = _source_prompt(row)
        for label, choice in zip(row.choice_labels, row.choices, strict=True):
            texts.append(f"{prompt}\nCandidate under consideration: {label}. {choice}")
    return texts


def _choice_token_prompt(row: arc_gate.ArcRow) -> str:
    return f"{_source_prompt(row)}\nCandidate under consideration:"


def _continuation_text(row: arc_gate.ArcRow, choice_index: int, *, mode: str) -> str:
    if mode == "label":
        return f" {row.choice_labels[choice_index]}"
    if mode == "label_and_choice":
        return f" {row.choice_labels[choice_index]}. {row.choices[choice_index]}"
    if mode == "choice":
        return f" {row.choices[choice_index]}"
    raise ValueError(f"unknown continuation mode {mode!r}")


def _encode_ids(tokenizer: Any, text: str, *, device: str, add_special_tokens: bool) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    ids = encoded.input_ids[0].to(device)
    if ids.numel() == 0:
        raise ValueError(f"zero-token text: {text!r}")
    return ids


def _standardize(matrix: torch.Tensor, train_indices: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    train = matrix[list(train_indices)]
    if train.dim() <= 2:
        mean = train.mean(dim=0, keepdim=True)
        std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    else:
        flat_train = train.reshape(-1, train.shape[-1])
        mean = flat_train.mean(dim=0).view(*([1] * (matrix.dim() - 1)), matrix.shape[-1])
        std = flat_train.std(dim=0, unbiased=False).clamp_min(1e-6).view(
            *([1] * (matrix.dim() - 1)),
            matrix.shape[-1],
        )
    return (matrix - mean) / std, {
        "mean_l2": float(mean.norm().detach().cpu()),
        "std_min": float(std.min().detach().cpu()),
        "std_max": float(std.max().detach().cpu()),
        "tensor_rank": int(matrix.dim()),
    }


def _row_indices(
    *,
    row_count: int,
    fit_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    fit_count = max(1, min(row_count - 1, int(round(row_count * fit_fraction))))
    fit = sorted(indices[:fit_count])
    eval_indices = sorted(indices[fit_count:])
    return fit, eval_indices


def _selected_choice_features(
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    *,
    source_feature_mode: str,
    feature_dim: int,
    source_model: str,
    source_device: str,
    source_dtype: str,
    source_max_length: int,
    source_hidden_layer: int,
    source_token_pool_size: int,
    local_files_only: bool,
    target_alignment_model: str | None = None,
    target_alignment_device: str = "auto_cpu",
    fit_indices: Sequence[int] | None = None,
    innovation_ridge: float = 10.0,
    sparse_packet_rank: int = 16,
    sparse_packet_top_k: int = 4,
    sparse_packet_bits: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    residualized = source_feature_mode.endswith("_residual")
    base_mode = source_feature_mode.removesuffix("_residual")
    if base_mode == "hashed_selected":
        flat = arc_gate._features(
            arc_gate._choice_pair_texts(rows),
            dim=feature_dim,
            feature_mode="hashed",
            feature_model="",
            feature_device="auto",
            feature_dtype="float32",
            feature_max_length=source_max_length,
            local_files_only=True,
        )
        metadata = {"kind": "hashed_selected_choice", "feature_dim": int(feature_dim)}
    elif base_mode == "hf_selected_hidden":
        flat, metadata = _hf_choice_hidden_features(
            rows,
            model_path=source_model,
            device=source_device,
            dtype=source_dtype,
            max_length=source_max_length,
            local_files_only=local_files_only,
            hidden_layer=source_hidden_layer,
        )
    elif base_mode == "hf_choice_token_hidden_pool":
        pooled, metadata = _hf_choice_token_hidden_pool_features(
            rows,
            model_path=source_model,
            device=source_device,
            dtype=source_dtype,
            max_length=source_max_length,
            local_files_only=local_files_only,
            hidden_layer=source_hidden_layer,
            pool_size=source_token_pool_size,
            residualized=residualized,
        )
        metadata = {
            **metadata,
            "source_feature_mode": source_feature_mode,
            "row_centered_selected_residual": False,
            "row_centered_token_residual": bool(residualized),
            "uses_source_predictions": False,
        }
        return torch.tensor(pooled.astype(np.float32)), metadata
    elif base_mode in {
        "cached_choice_score_pool",
        "hf_choice_hidden_candidate_pool",
        "hf_choice_hidden_score_candidate_pool",
        "hf_choice_hidden_public_innovation_candidate_pool",
        "hf_choice_hidden_score_public_innovation_candidate_pool",
        "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool",
    }:
        pooled, metadata = _choice_candidate_pool_features(
            rows,
            source_predictions,
            source_feature_mode=source_feature_mode,
            feature_dim=feature_dim,
            source_model=source_model,
            source_device=source_device,
            source_dtype=source_dtype,
            source_max_length=source_max_length,
            source_hidden_layer=source_hidden_layer,
            source_token_pool_size=source_token_pool_size,
            target_alignment_model=target_alignment_model,
            target_alignment_device=target_alignment_device,
            fit_indices=fit_indices,
            innovation_ridge=innovation_ridge,
            sparse_packet_rank=sparse_packet_rank,
            sparse_packet_top_k=sparse_packet_top_k,
            sparse_packet_bits=sparse_packet_bits,
            local_files_only=local_files_only,
        )
        return torch.tensor(pooled.astype(np.float32)), metadata
    else:
        raise ValueError(f"unknown source_feature_mode {source_feature_mode!r}")

    chosen: list[np.ndarray] = []
    offset = 0
    for row, selected_index in zip(rows, source_predictions, strict=True):
        row_features = flat[offset : offset + len(row.choices)]
        feature = row_features[int(selected_index)]
        if residualized:
            feature = feature - row_features.mean(axis=0)
            norm = np.linalg.norm(feature)
            feature = feature / max(norm, 1e-12)
        chosen.append(feature)
        offset += len(row.choices)
    metadata = {
        **metadata,
        "source_feature_mode": source_feature_mode,
        "row_centered_selected_residual": bool(residualized),
        "uses_source_predictions": True,
    }
    return torch.tensor(np.asarray(chosen, dtype=np.float32)), metadata


def _fixed_feature_pool(
    features: np.ndarray,
    *,
    pool_size: int,
    residualized: bool,
    normalize_rows: bool,
) -> np.ndarray:
    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a [item_count, dim] matrix")
    if values.shape[0] == 0:
        values = np.zeros((1, values.shape[1]), dtype=np.float64)
    if residualized:
        values = values - values.mean(axis=0, keepdims=True)
    if normalize_rows:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = np.divide(values, np.maximum(norms, 1e-12), out=np.zeros_like(values), where=norms > 0)
    if values.shape[0] == pool_size:
        return values
    if values.shape[0] > pool_size:
        positions = np.linspace(0, values.shape[0] - 1, pool_size).round().astype(np.int64)
        return values[positions]
    repeats = int(math.ceil(pool_size / values.shape[0]))
    return np.tile(values, (repeats, 1))[:pool_size]


def _fixed_token_pool(tokens: np.ndarray, *, pool_size: int, residualized: bool) -> np.ndarray:
    return _fixed_feature_pool(
        tokens,
        pool_size=pool_size,
        residualized=residualized,
        normalize_rows=True,
    )


def _source_selection_score_rows(
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
) -> list[np.ndarray]:
    score_rows: list[np.ndarray] = []
    for row, selected_index in zip(rows, source_predictions, strict=True):
        scores = np.zeros((len(row.choices), 1), dtype=np.float64)
        scores[int(selected_index), 0] = 1.0
        score_rows.append(scores)
    return score_rows


def _flat_candidate_indices_for_rows(rows: Sequence[arc_gate.ArcRow], row_indices: Sequence[int]) -> np.ndarray:
    offsets: list[tuple[int, int]] = []
    start = 0
    for row in rows:
        end = start + len(row.choices)
        offsets.append((start, end))
        start = end
    flat_indices: list[int] = []
    for row_index in row_indices:
        start, end = offsets[int(row_index)]
        flat_indices.extend(range(start, end))
    if not flat_indices:
        raise ValueError("need at least one fit candidate for public innovation features")
    return np.asarray(flat_indices, dtype=np.int64)


def _public_candidate_hashed_features(rows: list[arc_gate.ArcRow], *, feature_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    features = arc_gate._features(
        _source_choice_texts(rows),
        dim=feature_dim,
        feature_mode="hashed",
        feature_model="",
        feature_device="auto",
        feature_dtype="float32",
        feature_max_length=0,
        local_files_only=True,
    )
    return np.asarray(features, dtype=np.float64), {
        "kind": "public_question_choice_hashed_side_info",
        "feature_dim": int(feature_dim),
    }


def _public_candidate_innovation_features(
    source_features: np.ndarray,
    public_features: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_features, dtype=np.float64)
    public = np.asarray(public_features, dtype=np.float64)
    if source.ndim != 2 or public.ndim != 2:
        raise ValueError("source_features and public_features must be rank-2 matrices")
    if source.shape[0] != public.shape[0]:
        raise ValueError("source/public candidate feature counts must match")
    if ridge < 0.0:
        raise ValueError("innovation_ridge must be non-negative")
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")

    fit_public = public[fit]
    public_mean = fit_public.mean(axis=0, keepdims=True)
    public_std = fit_public.std(axis=0, keepdims=True).clip(min=1e-6)
    standardized_public = (public - public_mean) / public_std

    source_mean = source[fit].mean(axis=0, keepdims=True)
    centered_source = source - source_mean
    x_fit = standardized_public[fit]
    y_fit = centered_source[fit]
    xtx = x_fit.T @ x_fit
    xty = x_fit.T @ y_fit
    if ridge > 0.0:
        xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    weights = np.linalg.solve(xtx, xty)
    predicted = standardized_public @ weights
    innovation = centered_source - predicted

    fit_baseline_mse = float(np.mean(np.square(y_fit)))
    fit_residual_mse = float(np.mean(np.square(innovation[fit])))
    explained = 0.0 if fit_baseline_mse <= 1e-12 else 1.0 - fit_residual_mse / fit_baseline_mse
    return innovation.astype(np.float64, copy=False), {
        "kind": "train_only_public_candidate_ridge_innovation",
        "ridge": float(ridge),
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(source.shape[1]),
        "public_feature_dim": int(public.shape[1]),
        "fit_baseline_mse": fit_baseline_mse,
        "fit_residual_mse": fit_residual_mse,
        "fit_explained_variance_ratio": float(explained),
        "public_mean_l2": float(np.linalg.norm(public_mean)),
        "public_std_min": float(public_std.min()),
        "public_std_max": float(public_std.max()),
    }


def _sparse_topk_quantized_coordinates(
    coeffs: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    top_k: int,
    quant_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates = np.asarray(coeffs, dtype=np.float64)
    if coordinates.ndim != 2:
        raise ValueError("coeffs must be a [candidate_count, atom_count] matrix")
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if top_k < 1:
        raise ValueError("sparse_packet_top_k must be at least 1")
    if quant_bits < 1:
        raise ValueError("sparse_packet_bits must be at least 1")
    atom_count = int(coordinates.shape[1])
    if atom_count < 1:
        raise ValueError("coeffs must contain at least one atom")
    retained_top_k = int(min(top_k, atom_count))

    sparse = np.zeros_like(coordinates)
    if retained_top_k >= atom_count:
        sparse[:] = coordinates
    else:
        top_indices = np.argpartition(np.abs(coordinates), -retained_top_k, axis=1)[:, -retained_top_k:]
        row_indices = np.arange(coordinates.shape[0])[:, None]
        sparse[row_indices, top_indices] = coordinates[row_indices, top_indices]

    fit_abs = np.abs(sparse[fit])
    scale = float(fit_abs.max()) if fit_abs.size else 0.0
    signed_levels = int((2 ** (quant_bits - 1)) - 1)
    if scale <= 1e-12 or signed_levels < 1:
        quantized = np.zeros_like(sparse, dtype=np.int64)
        dequantized = np.zeros_like(sparse)
        step = 0.0
    else:
        step = scale / float(signed_levels)
        quantized = np.clip(np.rint(sparse / step), -signed_levels, signed_levels).astype(np.int64)
        dequantized = quantized.astype(np.float64) * step

    atom_id_bits = int(math.ceil(math.log2(max(atom_count, 2))))
    packet_bits_per_candidate = int(retained_top_k * (atom_id_bits + quant_bits))
    nonzero_counts = (quantized != 0).sum(axis=1)
    energy = np.square(coordinates).sum(axis=1)
    sparse_energy = np.square(dequantized).sum(axis=1)
    fit_energy_ratio = float(
        np.mean(
            np.divide(
                sparse_energy[fit],
                np.maximum(energy[fit], 1e-12),
                out=np.zeros_like(sparse_energy[fit]),
                where=energy[fit] > 1e-12,
            )
        )
    )
    return dequantized.astype(np.float64, copy=False), {
        "top_k": int(retained_top_k),
        "quant_bits": int(quant_bits),
        "atom_id_bits": int(atom_id_bits),
        "packet_bits_per_candidate": int(packet_bits_per_candidate),
        "packet_bytes_per_candidate": float(packet_bits_per_candidate / 8.0),
        "quant_step": float(step),
        "quant_scale": float(scale),
        "fit_mean_nonzero_coefficients": float(nonzero_counts[fit].mean()) if fit.size else 0.0,
        "fit_sparse_energy_ratio": float(fit_energy_ratio),
    }


def _sparse_pca_packet_features(
    candidate_features: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    rank: int,
    top_k: int,
    quant_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = np.asarray(candidate_features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("candidate_features must be a [candidate_count, dim] matrix")
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if rank < 1:
        raise ValueError("sparse_packet_rank must be at least 1")
    if top_k < 1:
        raise ValueError("sparse_packet_top_k must be at least 1")
    if quant_bits < 1:
        raise ValueError("sparse_packet_bits must be at least 1")

    fit_features = features[fit]
    mean = fit_features.mean(axis=0, keepdims=True)
    centered = features - mean
    centered_fit = centered[fit]
    _, singular_values, vt = np.linalg.svd(centered_fit, full_matrices=False)
    actual_rank = int(min(rank, vt.shape[0], vt.shape[1]))
    if actual_rank < 1:
        raise ValueError("could not fit a non-empty PCA basis")
    basis = vt[:actual_rank]
    coeffs = centered @ basis.T
    top_k = int(min(top_k, actual_rank))

    sparse = np.zeros_like(coeffs)
    if top_k >= actual_rank:
        sparse[:] = coeffs
        top_indices = np.tile(np.arange(actual_rank, dtype=np.int64), (coeffs.shape[0], 1))
    else:
        top_indices = np.argpartition(np.abs(coeffs), -top_k, axis=1)[:, -top_k:]
        row_indices = np.arange(coeffs.shape[0])[:, None]
        sparse[row_indices, top_indices] = coeffs[row_indices, top_indices]

    fit_abs = np.abs(sparse[fit])
    scale = float(fit_abs.max()) if fit_abs.size else 0.0
    signed_levels = int((2 ** (quant_bits - 1)) - 1)
    if scale <= 1e-12 or signed_levels < 1:
        quantized = np.zeros_like(sparse, dtype=np.int64)
        dequantized = np.zeros_like(sparse)
        step = 0.0
    else:
        step = scale / float(signed_levels)
        quantized = np.clip(np.rint(sparse / step), -signed_levels, signed_levels).astype(np.int64)
        dequantized = quantized.astype(np.float64) * step

    atom_id_bits = int(math.ceil(math.log2(max(actual_rank, 2))))
    packet_bits_per_candidate = int(top_k * (atom_id_bits + quant_bits))
    nonzero_counts = (quantized != 0).sum(axis=1)
    energy = np.square(coeffs).sum(axis=1)
    sparse_energy = np.square(dequantized).sum(axis=1)
    fit_energy_ratio = float(
        np.mean(
            np.divide(
                sparse_energy[fit],
                np.maximum(energy[fit], 1e-12),
                out=np.zeros_like(sparse_energy[fit]),
                where=energy[fit] > 1e-12,
            )
        )
    )
    explained = np.square(singular_values[:actual_rank])
    total = float(np.square(singular_values).sum())
    explained_ratio = float(explained.sum() / total) if total > 1e-12 else 0.0
    return dequantized.astype(np.float64, copy=False), {
        "kind": "train_fit_sparse_pca_packet_coordinates",
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(features.shape[1]),
        "packet_rank": int(actual_rank),
        "requested_packet_rank": int(rank),
        "top_k": int(top_k),
        "quant_bits": int(quant_bits),
        "atom_id_bits": int(atom_id_bits),
        "packet_bits_per_candidate": int(packet_bits_per_candidate),
        "packet_bytes_per_candidate": float(packet_bits_per_candidate / 8.0),
        "quant_step": float(step),
        "quant_scale": float(scale),
        "fit_mean_nonzero_coefficients": float(nonzero_counts[fit].mean()) if fit.size else 0.0,
        "fit_sparse_energy_ratio": float(fit_energy_ratio),
        "pca_explained_variance_ratio": float(explained_ratio),
        "singular_value_max": float(singular_values[0]) if singular_values.size else 0.0,
        "singular_value_min_retained": float(singular_values[actual_rank - 1]) if singular_values.size else 0.0,
    }


def _target_aligned_sparse_pca_packet_features(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    rank: int,
    top_k: int,
    quant_bits: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_features, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_features and target_features must be rank-2 matrices")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source/target candidate feature counts must match")
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if rank < 1:
        raise ValueError("sparse_packet_rank must be at least 1")
    if ridge < 0.0:
        raise ValueError("innovation_ridge must be non-negative")

    source_mean = source[fit].mean(axis=0, keepdims=True)
    source_std = source[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    standardized_source = (source - source_mean) / source_std

    target_mean = target[fit].mean(axis=0, keepdims=True)
    centered_target = target - target_mean
    centered_target_fit = centered_target[fit]
    _, singular_values, vt = np.linalg.svd(centered_target_fit, full_matrices=False)
    actual_rank = int(min(rank, vt.shape[0], vt.shape[1]))
    if actual_rank < 1:
        raise ValueError("could not fit a non-empty target PCA basis")
    target_basis = vt[:actual_rank]
    target_coords = centered_target @ target_basis.T

    x_fit = standardized_source[fit]
    y_fit = target_coords[fit]
    kernel = x_fit @ x_fit.T
    if ridge > 0.0:
        kernel = kernel + float(ridge) * np.eye(kernel.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(kernel, y_fit)
    weights = x_fit.T @ alpha
    predicted_coords = standardized_source @ weights
    packet, quant_meta = _sparse_topk_quantized_coordinates(
        predicted_coords,
        fit_flat_indices=fit,
        top_k=top_k,
        quant_bits=quant_bits,
    )

    fit_baseline_mse = float(np.mean(np.square(y_fit)))
    fit_residual_mse = float(np.mean(np.square(y_fit - predicted_coords[fit])))
    fit_coord_r2 = 0.0 if fit_baseline_mse <= 1e-12 else 1.0 - fit_residual_mse / fit_baseline_mse
    explained = np.square(singular_values[:actual_rank])
    total = float(np.square(singular_values).sum())
    explained_ratio = float(explained.sum() / total) if total > 1e-12 else 0.0
    return packet, {
        "kind": "train_fit_target_aligned_sparse_pca_packet_coordinates",
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(source.shape[1]),
        "target_feature_dim": int(target.shape[1]),
        "packet_rank": int(actual_rank),
        "requested_packet_rank": int(rank),
        "ridge": float(ridge),
        "fit_target_coord_baseline_mse": fit_baseline_mse,
        "fit_target_coord_residual_mse": fit_residual_mse,
        "fit_target_coord_r2": float(fit_coord_r2),
        "target_pca_explained_variance_ratio": float(explained_ratio),
        "target_singular_value_max": float(singular_values[0]) if singular_values.size else 0.0,
        "target_singular_value_min_retained": float(singular_values[actual_rank - 1])
        if singular_values.size
        else 0.0,
        "source_std_min": float(source_std.min()),
        "source_std_max": float(source_std.max()),
        **quant_meta,
    }


def _choice_candidate_pool_features(
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    *,
    source_feature_mode: str,
    feature_dim: int,
    source_model: str,
    source_device: str,
    source_dtype: str,
    source_max_length: int,
    source_hidden_layer: int,
    source_token_pool_size: int,
    local_files_only: bool,
    target_alignment_model: str | None = None,
    target_alignment_device: str = "auto_cpu",
    fit_indices: Sequence[int] | None = None,
    innovation_ridge: float = 10.0,
    sparse_packet_rank: int = 16,
    sparse_packet_top_k: int = 4,
    sparse_packet_bits: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    residualized = source_feature_mode.endswith("_residual")
    base_mode = source_feature_mode.removesuffix("_residual")
    pool_size = int(source_token_pool_size)
    if pool_size < 1:
        raise ValueError("source_token_pool_size must be at least 1")

    hidden_rows: list[np.ndarray] | None = None
    hidden_meta: dict[str, Any] = {}
    innovation_meta: dict[str, Any] = {}
    normalize_rows = base_mode != "cached_choice_score_pool"
    hidden_base_modes = {
        "hf_choice_hidden_candidate_pool",
        "hf_choice_hidden_score_candidate_pool",
        "hf_choice_hidden_public_innovation_candidate_pool",
        "hf_choice_hidden_score_public_innovation_candidate_pool",
        "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool",
    }
    innovation_base_modes = {
        "hf_choice_hidden_public_innovation_candidate_pool",
        "hf_choice_hidden_score_public_innovation_candidate_pool",
        "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool",
    }
    sparse_packet_base_modes = {
        "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool",
        "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool",
    }
    target_aligned_sparse_packet_base_modes = {
        "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool",
    }
    if base_mode in hidden_base_modes:
        flat_hidden, hidden_meta = _hf_choice_hidden_features(
            rows,
            model_path=source_model,
            device=source_device,
            dtype=source_dtype,
            max_length=source_max_length,
            local_files_only=local_files_only,
            hidden_layer=source_hidden_layer,
        )
        if base_mode in innovation_base_modes:
            if fit_indices is None:
                raise ValueError(f"{source_feature_mode!r} requires fit_indices")
            public_flat, public_meta = _public_candidate_hashed_features(rows, feature_dim=feature_dim)
            flat_hidden, innovation_meta = _public_candidate_innovation_features(
                flat_hidden,
                public_flat,
                fit_flat_indices=_flat_candidate_indices_for_rows(rows, fit_indices),
                ridge=innovation_ridge,
            )
            innovation_meta = {**innovation_meta, "public_metadata": public_meta}
            if base_mode in target_aligned_sparse_packet_base_modes:
                if not target_alignment_model:
                    raise ValueError(f"{source_feature_mode!r} requires a target_alignment_model")
                target_flat_hidden, target_hidden_meta = _hf_choice_hidden_features(
                    rows,
                    model_path=target_alignment_model,
                    device=target_alignment_device,
                    dtype=source_dtype,
                    max_length=source_max_length,
                    local_files_only=local_files_only,
                    hidden_layer=source_hidden_layer,
                )
                target_flat_hidden, target_innovation_meta = _public_candidate_innovation_features(
                    target_flat_hidden,
                    public_flat,
                    fit_flat_indices=_flat_candidate_indices_for_rows(rows, fit_indices),
                    ridge=innovation_ridge,
                )
                sparse_packet_flat, sparse_packet_meta = _target_aligned_sparse_pca_packet_features(
                    flat_hidden,
                    target_flat_hidden,
                    fit_flat_indices=_flat_candidate_indices_for_rows(rows, fit_indices),
                    rank=sparse_packet_rank,
                    top_k=sparse_packet_top_k,
                    quant_bits=sparse_packet_bits,
                    ridge=innovation_ridge,
                )
                flat_hidden = sparse_packet_flat
                hidden_meta = {
                    "source": hidden_meta,
                    "target_alignment": target_hidden_meta,
                }
                innovation_meta = {
                    **innovation_meta,
                    "target_public_innovation_metadata": target_innovation_meta,
                    "sparse_packet_metadata": sparse_packet_meta,
                }
            elif base_mode in sparse_packet_base_modes:
                sparse_packet_flat, sparse_packet_meta = _sparse_pca_packet_features(
                    flat_hidden,
                    fit_flat_indices=_flat_candidate_indices_for_rows(rows, fit_indices),
                    rank=sparse_packet_rank,
                    top_k=sparse_packet_top_k,
                    quant_bits=sparse_packet_bits,
                )
                flat_hidden = sparse_packet_flat
                innovation_meta = {**innovation_meta, "sparse_packet_metadata": sparse_packet_meta}
        hidden_rows = []
        offset = 0
        for row in rows:
            hidden_rows.append(np.asarray(flat_hidden[offset : offset + len(row.choices)], dtype=np.float64))
            offset += len(row.choices)

    pooled_rows: list[np.ndarray] = []
    feature_kind: str
    for row_index, row in enumerate(rows):
        if base_mode == "cached_choice_score_pool":
            score_row = _source_selection_score_rows([row], [source_predictions[row_index]])[0]
            row_features = score_row
            feature_kind = "cached_source_selection_score_candidate_pool"
        elif base_mode == "hf_choice_hidden_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            row_features = hidden_rows[row_index]
            feature_kind = "hf_choice_hidden_candidate_pool"
        elif base_mode == "hf_choice_hidden_score_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            score_row = _source_selection_score_rows([row], [source_predictions[row_index]])[0]
            row_features = np.concatenate([hidden_rows[row_index], score_row], axis=1)
            feature_kind = "hf_choice_hidden_score_candidate_pool"
        elif base_mode == "hf_choice_hidden_public_innovation_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            row_features = hidden_rows[row_index]
            feature_kind = "hf_choice_hidden_public_innovation_candidate_pool"
        elif base_mode == "hf_choice_hidden_score_public_innovation_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            score_row = _source_selection_score_rows([row], [source_predictions[row_index]])[0]
            row_features = np.concatenate([hidden_rows[row_index], score_row], axis=1)
            feature_kind = "hf_choice_hidden_score_public_innovation_candidate_pool"
        elif base_mode == "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            row_features = hidden_rows[row_index]
            feature_kind = "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool"
        elif base_mode == "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            score_row = _source_selection_score_rows([row], [source_predictions[row_index]])[0]
            row_features = np.concatenate([hidden_rows[row_index], score_row], axis=1)
            feature_kind = "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool"
        elif base_mode == "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool":
            if hidden_rows is None:
                raise ValueError("hidden rows were not loaded")
            row_features = hidden_rows[row_index]
            feature_kind = "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool"
        else:
            raise ValueError(f"unknown candidate pool mode {source_feature_mode!r}")
        pooled_rows.append(
            _fixed_feature_pool(
                row_features,
                pool_size=pool_size,
                residualized=residualized,
                normalize_rows=normalize_rows,
            )
        )

    choice_counts = [len(row.choices) for row in rows]
    return np.asarray(pooled_rows, dtype=np.float64), {
        "kind": feature_kind,
        "source_feature_mode": source_feature_mode,
        "pool_size": pool_size,
        "choice_count_min": int(min(choice_counts)) if choice_counts else 0,
        "choice_count_max": int(max(choice_counts)) if choice_counts else 0,
        "choice_count_mean": float(statistics.fmean(choice_counts)) if choice_counts else 0.0,
        "row_centered_candidate_residual": bool(residualized),
        "uses_source_predictions": base_mode
        in {
            "cached_choice_score_pool",
            "hf_choice_hidden_score_candidate_pool",
            "hf_choice_hidden_score_public_innovation_candidate_pool",
            "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool",
        },
        "hidden_metadata": hidden_meta,
        "innovation_metadata": innovation_meta,
        "feature_dim": int(pooled_rows[0].shape[-1]) if pooled_rows else int(feature_dim),
        "sparse_packet": innovation_meta.get("sparse_packet_metadata", {}),
    }


def _hf_choice_hidden_features(
    rows: list[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    local_files_only: bool,
    hidden_layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(dtype),
        attn_implementation="eager",
    ).to(resolved_device)
    model.eval()

    features: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for text in _source_choice_texts(rows):
            encoded = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[hidden_layer][0]
            mask = encoded["attention_mask"][0].bool()
            values = hidden[mask]
            feature = values.mean(dim=0).detach().cpu().numpy().astype(np.float64)
            norm = np.linalg.norm(feature)
            features.append(feature / max(norm, 1e-12))
    feature_array = np.asarray(features, dtype=np.float64)
    metadata = {
        "kind": "answer_key_forbidden_hf_choice_hidden",
        "model_path": model_path,
        "device": resolved_device,
        "dtype": dtype,
        "attn_implementation": "eager",
        "max_length": int(max_length),
        "hidden_layer": int(hidden_layer),
        "latency_s": float(time.perf_counter() - start),
    }
    del model, tokenizer
    _release_torch_model_memory(device=resolved_device)
    return feature_array, metadata


def _hf_choice_token_hidden_pool_features(
    rows: list[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    local_files_only: bool,
    hidden_layer: int,
    pool_size: int,
    residualized: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(dtype),
        attn_implementation="eager",
    ).to(resolved_device)
    model.eval()

    pooled_rows: list[np.ndarray] = []
    token_counts: list[int] = []
    suffix_fallback_rows = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            prompt = _choice_token_prompt(row)
            prompt_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.shape[1]
            texts = [
                f"{prompt} {label}. {choice}"
                for label, choice in zip(row.choice_labels, row.choices, strict=True)
            ]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[hidden_layer]
            attention = encoded["attention_mask"].bool()
            row_tokens: list[np.ndarray] = []
            for choice_index in range(len(row.choices)):
                mask = attention[choice_index].clone()
                mask[: min(prompt_len, mask.shape[0])] = False
                if not bool(mask.any()):
                    suffix_fallback_rows += 1
                    valid = torch.where(attention[choice_index])[0]
                    mask = torch.zeros_like(attention[choice_index])
                    suffix_count = max(1, min(8, int(valid.numel())))
                    mask[valid[-suffix_count:]] = True
                values = hidden[choice_index][mask].detach().cpu().numpy().astype(np.float64)
                row_tokens.extend(values)
            if not row_tokens:
                row_tokens = [np.zeros(int(hidden.shape[-1]), dtype=np.float64)]
            tokens = np.asarray(row_tokens, dtype=np.float64)
            token_counts.append(int(tokens.shape[0]))
            pooled_rows.append(
                _fixed_token_pool(tokens, pool_size=pool_size, residualized=residualized)
            )
    feature_array = np.asarray(pooled_rows, dtype=np.float64)
    metadata = {
        "kind": "answer_key_forbidden_hf_choice_token_hidden_pool",
        "model_path": model_path,
        "device": resolved_device,
        "dtype": dtype,
        "attn_implementation": "eager",
        "max_length": int(max_length),
        "hidden_layer": int(hidden_layer),
        "pool_size": int(pool_size),
        "row_centered_token_residual": bool(residualized),
        "token_count_min": int(min(token_counts)) if token_counts else 0,
        "token_count_max": int(max(token_counts)) if token_counts else 0,
        "token_count_mean": float(statistics.fmean(token_counts)) if token_counts else 0.0,
        "suffix_fallback_choice_count": int(suffix_fallback_rows),
        "latency_s": float(time.perf_counter() - start),
    }
    del model, tokenizer
    _release_torch_model_memory(device=resolved_device)
    return feature_array, metadata


def _target_public_features(rows: list[arc_gate.ArcRow], *, feature_dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    features = arc_gate._hashed_features([_row_public_text(row) for row in rows], dim=feature_dim)
    return torch.tensor(features.astype(np.float32)), {
        "kind": "hashed_public_question_choices",
        "feature_dim": int(feature_dim),
    }


def _continuation_logprob(
    *,
    target_model: Any,
    embed_tokens: Any,
    prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    length_normalize: bool,
) -> torch.Tensor:
    device = prefix.device if prefix.numel() else prompt_ids.device
    prompt_embeds = embed_tokens(prompt_ids.to(device)).detach()
    continuation_embeds = embed_tokens(continuation_ids.to(device)).detach()
    prefix = prefix.to(device=device, dtype=prompt_embeds.dtype)
    if continuation_embeds.shape[0] > 1:
        inputs = torch.cat([prefix, prompt_embeds, continuation_embeds[:-1]], dim=0)
    else:
        inputs = torch.cat([prefix, prompt_embeds], dim=0)
    attention_mask = torch.ones((1, inputs.shape[0]), dtype=torch.long, device=device)
    out = target_model(inputs_embeds=inputs.unsqueeze(0), attention_mask=attention_mask, use_cache=False)
    logits = out.logits[0]
    start = int(prefix.shape[0] + prompt_embeds.shape[0] - 1)
    token_logits = logits[start : start + continuation_ids.shape[0]]
    logprobs = torch.log_softmax(token_logits.float(), dim=-1)
    score = logprobs.gather(1, continuation_ids.to(device)[:, None]).sum()
    if length_normalize:
        return score / max(int(continuation_ids.shape[0]), 1)
    return score


def _choice_scores(
    *,
    target_model: Any,
    embed_tokens: Any,
    prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    length_normalize: bool,
) -> torch.Tensor:
    if not continuation_ids:
        raise ValueError("continuation_ids must not be empty")
    device = prefix.device if prefix.numel() else prompt_ids.device
    if str(device).startswith("mps"):
        # MPSGraph can fail shape inference for padded variable-length batches
        # with prefix inputs_embeds. The unbatched scorer uses the same objective
        # and is slower but reliable for Mac-local gates.
        return torch.stack(
            [
                _continuation_logprob(
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    prefix=prefix,
                    prompt_ids=prompt_ids,
                    continuation_ids=ids,
                    length_normalize=length_normalize,
                )
                for ids in continuation_ids
            ]
        )
    prompt_embeds = embed_tokens(prompt_ids.to(device)).detach()
    prefix = prefix.to(device=device, dtype=prompt_embeds.dtype)

    batched_inputs: list[torch.Tensor] = []
    input_lengths: list[int] = []
    continuation_tensors: list[torch.Tensor] = []
    for ids in continuation_ids:
        ids = ids.to(device)
        continuation_tensors.append(ids)
        continuation_embeds = embed_tokens(ids).detach()
        if continuation_embeds.shape[0] > 1:
            inputs = torch.cat([prefix, prompt_embeds, continuation_embeds[:-1]], dim=0)
        else:
            inputs = torch.cat([prefix, prompt_embeds], dim=0)
        batched_inputs.append(inputs)
        input_lengths.append(int(inputs.shape[0]))

    inputs_embeds = torch.nn.utils.rnn.pad_sequence(
        batched_inputs,
        batch_first=True,
        padding_value=0.0,
    )
    lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    positions = torch.arange(inputs_embeds.shape[1], dtype=torch.long, device=device)
    attention_mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).long()
    out = target_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
    logits = out.logits
    start = int(prefix.shape[0] + prompt_embeds.shape[0] - 1)
    scores: list[torch.Tensor] = []
    for choice_index, ids in enumerate(continuation_tensors):
        token_logits = logits[choice_index, start : start + ids.shape[0]]
        logprobs = torch.log_softmax(token_logits.float(), dim=-1)
        score = logprobs.gather(1, ids[:, None]).sum()
        if length_normalize:
            score = score / max(int(ids.shape[0]), 1)
        scores.append(score)
    return torch.stack(scores)


def _gold_margin_tensor(scores: torch.Tensor, answer_index: int) -> torch.Tensor:
    answer_index = int(answer_index)
    gold = scores[answer_index]
    if scores.numel() <= 1:
        return gold
    mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    mask[answer_index] = False
    return gold - scores[mask].max()


def _contrastive_margin_penalty(
    *,
    matched_scores: torch.Tensor,
    control_scores: torch.Tensor,
    answer_index: int,
    margin: float,
    loss_cap: float,
) -> torch.Tensor:
    matched_margin = _gold_margin_tensor(matched_scores, answer_index)
    control_margin = _gold_margin_tensor(control_scores, answer_index)
    penalty = torch.relu(torch.as_tensor(float(margin), device=matched_scores.device) - (matched_margin - control_margin))
    return penalty.clamp_max(float(loss_cap))


def _candidate_roll_source_summary(source: torch.Tensor) -> torch.Tensor | None:
    if source.numel() == 0 or source.dim() < 2:
        return None
    return torch.roll(source, shifts=1, dims=0)


def _candidate_score_roll_source_summary(source: torch.Tensor) -> torch.Tensor | None:
    if source.numel() == 0 or source.dim() != 2 or source.shape[0] < 2:
        return None
    rolled = source.clone()
    rolled[:, -1] = torch.roll(source[:, -1], shifts=1, dims=0)
    return rolled


def _atom_shuffle_source_summary(source: torch.Tensor) -> torch.Tensor | None:
    if source.numel() == 0 or source.dim() < 1 or source.shape[-1] < 2:
        return None
    return torch.roll(source, shifts=1, dims=-1)


def _coefficient_shuffle_source_summary(source: torch.Tensor) -> torch.Tensor | None:
    if source.numel() == 0 or source.dim() < 1 or source.shape[-1] < 2:
        return None
    return torch.flip(source, dims=(-1,))


def _top_atom_knockout_source_summary(source: torch.Tensor) -> torch.Tensor | None:
    if source.numel() == 0 or source.dim() < 1:
        return None
    knocked = source.clone()
    atom_index = torch.argmax(torch.abs(knocked), dim=-1, keepdim=True)
    knocked.scatter_(-1, atom_index, 0.0)
    return knocked


def _fit_source_control_variant(
    source_summary: torch.Tensor,
    *,
    fit_indices: Sequence[int],
    fit_position: int,
    row_index: int,
    control: str,
    seed: int,
    epoch: int,
    device: str,
) -> torch.Tensor | None:
    base = source_summary[int(row_index)].to(device)
    if control == "zero_source":
        return torch.zeros_like(base)
    if control == "shuffled_source":
        other = int(fit_indices[(int(fit_position) + 1) % len(fit_indices)])
        return source_summary[other].to(device)
    if control == "same_norm_noise":
        generator = torch.Generator(device="cpu").manual_seed(int(seed) * 2003 + int(row_index) * 101 + int(epoch))
        noise = torch.randn(tuple(base.shape), generator=generator).to(device=device, dtype=base.dtype)
        return noise / noise.norm().clamp_min(1e-6) * base.norm().clamp_min(1e-6)
    if control == "atom_shuffle":
        return _atom_shuffle_source_summary(base)
    if control == "coefficient_shuffle":
        return _coefficient_shuffle_source_summary(base)
    if control == "top_atom_knockout":
        return _top_atom_knockout_source_summary(base)
    if control == "candidate_roll_source":
        return _candidate_roll_source_summary(base)
    if control == "candidate_score_roll_source":
        return _candidate_score_roll_source_summary(base)
    raise ValueError(f"unknown contrastive control {control!r}")


def _fit_connector(
    *,
    connector: torch.nn.Module,
    target_model: Any,
    embed_tokens: Any,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    prompt_ids: Sequence[torch.Tensor],
    continuation_ids: Sequence[Sequence[torch.Tensor]],
    answer_indices: Sequence[int],
    fit_indices: Sequence[int],
    config: SoftPrefixConfig,
    device: str,
    label_shuffle: bool,
    use_contrastive: bool,
) -> dict[str, float]:
    connector.to(device)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    fit_indices = list(fit_indices)
    if label_shuffle:
        shifted_answers = {
            idx: int(answer_indices[fit_indices[(pos + 1) % len(fit_indices)]])
            for pos, idx in enumerate(fit_indices)
        }
    else:
        shifted_answers = {idx: int(answer_indices[idx]) for idx in fit_indices}
    losses: list[float] = []
    contrastive_losses: list[float] = []
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        total_value = 0.0
        contrastive_value = 0.0
        for fit_position, idx in enumerate(fit_indices):
            row_source = source_summary[idx].to(device)
            row_target = target_summary[idx].to(device)
            prefix = connector(row_source, row_target)
            scores = _choice_scores(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                prompt_ids=prompt_ids[idx],
                continuation_ids=continuation_ids[idx],
                length_normalize=config.length_normalize,
            )
            label = torch.tensor([shifted_answers[idx]], dtype=torch.long, device=device)
            row_loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), label)
            if use_contrastive and config.contrastive_weight > 0.0 and config.contrastive_controls:
                row_penalties: list[torch.Tensor] = []
                for control in config.contrastive_controls:
                    control_source = _fit_source_control_variant(
                        source_summary,
                        fit_indices=fit_indices,
                        fit_position=fit_position,
                        row_index=idx,
                        control=control,
                        seed=config.seed,
                        epoch=epoch,
                        device=device,
                    )
                    if control_source is None:
                        continue
                    control_prefix = connector(control_source, row_target)
                    control_scores = _choice_scores(
                        target_model=target_model,
                        embed_tokens=embed_tokens,
                        prefix=control_prefix,
                        prompt_ids=prompt_ids[idx],
                        continuation_ids=continuation_ids[idx],
                        length_normalize=config.length_normalize,
                    )
                    penalty = _contrastive_margin_penalty(
                        matched_scores=scores,
                        control_scores=control_scores,
                        answer_index=int(answer_indices[idx]),
                        margin=config.contrastive_margin,
                        loss_cap=config.contrastive_loss_cap,
                    )
                    row_penalties.append(penalty)
                if row_penalties:
                    row_penalty = torch.stack(row_penalties).mean()
                    contrastive_value += float(row_penalty.detach().cpu())
                    row_loss = row_loss + float(config.contrastive_weight) * row_penalty
            total_value += float(row_loss.detach().cpu())
            row_loss.backward()
        optimizer.step()
        losses.append(total_value)
        contrastive_losses.append(contrastive_value)
    connector.eval()
    return {
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
        "contrastive_penalty_initial": float(contrastive_losses[0]) if contrastive_losses else 0.0,
        "contrastive_penalty_final": float(contrastive_losses[-1]) if contrastive_losses else 0.0,
        "contrastive_weight": float(config.contrastive_weight if use_contrastive else 0.0),
        "contrastive_margin": float(config.contrastive_margin if use_contrastive else 0.0),
        "contrastive_loss_cap": float(config.contrastive_loss_cap if use_contrastive else 0.0),
        "contrastive_control_count": int(len(config.contrastive_controls) if use_contrastive else 0),
    }


@torch.no_grad()
def _score_connector_condition(
    *,
    connector: torch.nn.Module | None,
    target_model: Any,
    embed_tokens: Any,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    length_normalize: bool,
    device: str,
) -> list[float]:
    if connector is None:
        embed_dim = int(embed_tokens.embedding_dim)
        prefix = torch.empty((0, embed_dim), dtype=embed_tokens.weight.dtype, device=device)
    else:
        prefix = connector(source_summary.to(device), target_summary.to(device))
    scores = _choice_scores(
        target_model=target_model,
        embed_tokens=embed_tokens,
        prefix=prefix,
        prompt_ids=prompt_ids,
        continuation_ids=continuation_ids,
        length_normalize=length_normalize,
    )
    return [float(value) for value in scores.detach().cpu()]


def _margin(scores: Sequence[float], answer_index: int) -> float:
    gold = float(scores[answer_index])
    distractors = [float(score) for index, score in enumerate(scores) if index != answer_index]
    return gold - max(distractors) if distractors else gold


def _prediction(scores: Sequence[float]) -> int:
    return int(max(range(len(scores)), key=lambda index: (float(scores[index]), -index)))


def _source_index_scores(choice_count: int, selected_index: int) -> list[float]:
    if selected_index < 0 or selected_index >= choice_count:
        raise ValueError(f"selected_index {selected_index} outside choice_count {choice_count}")
    scores = [0.0 for _ in range(choice_count)]
    scores[int(selected_index)] = 1.0
    return scores


def _source_scores_for_control(
    *,
    row: arc_gate.ArcRow,
    selected_index: int,
    source_score_cache: dict[str, list[float]],
) -> list[float]:
    raw_scores = source_score_cache.get(row.content_id)
    if raw_scores is None:
        return _source_index_scores(len(row.choices), selected_index)
    if len(raw_scores) != len(row.choices):
        raise ValueError(f"source score control length mismatch for content_id={row.content_id}")
    return [float(score) for score in raw_scores]


def _source_rank_scores(raw_scores: Sequence[float]) -> list[float]:
    values = [float(score) for score in raw_scores]
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = {index: rank for rank, index in enumerate(order)}
    count = max(len(values), 1)
    return [float(count - ranks[index]) / float(count) for index in range(len(values))]


def _centered_source_score_control(raw_scores: Sequence[float]) -> list[float]:
    values = np.asarray([float(score) for score in raw_scores], dtype=np.float64)
    if values.size == 0:
        return []
    centered = values - float(values.mean())
    scale = float(centered.std())
    if not math.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return [float(value / scale) for value in centered]


def _paired_bootstrap(deltas: list[float], *, seed: int, samples: int) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _score_entropy(scores: Sequence[float]) -> float:
    values = np.asarray([float(score) for score in scores], dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = values - float(values.max())
    probs = np.exp(values)
    total = float(probs.sum())
    if total <= 0.0 or not math.isfinite(total):
        return 0.0
    probs = probs / total
    return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())


@torch.no_grad()
def _prefix_stats_for_condition(
    *,
    connector: torch.nn.Module | None,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    embed_dim: int,
    device: str,
) -> dict[str, Any]:
    if connector is None:
        return {"prefix_l2": 0.0, "prefix_rms": 0.0, "prefix_len": 0, "embed_dim": int(embed_dim)}
    prefix = connector(source_summary.to(device), target_summary.to(device))
    return {
        "prefix_l2": float(prefix.float().norm().detach().cpu()),
        "prefix_rms": float(prefix.float().pow(2).mean().sqrt().detach().cpu()),
        "prefix_len": int(prefix.shape[0]),
        "embed_dim": int(prefix.shape[-1]) if prefix.dim() >= 2 else int(embed_dim),
    }


def _summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "correct": 0}
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": int(correct),
        "accuracy": float(correct / len(rows)),
        "mean_margin": float(statistics.fmean(float(row["margin"]) for row in rows)),
    }


def _condition_metrics(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics = {
        condition: _summarize_condition([row for row in rows if row["condition"] == condition])
        for condition in REPORT_CONDITIONS
    }
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    for condition in CONTROL_CONDITIONS:
        deltas = [
            float(group[MATCHED_CONDITION]["correct"]) - float(group[condition]["correct"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and condition in group
        ]
        margin_deltas = [
            float(group[MATCHED_CONDITION]["margin"]) - float(group[condition]["margin"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and condition in group
        ]
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{condition}"] = _paired_bootstrap(
            deltas,
            seed=seed + len(condition),
            samples=bootstrap_samples,
        )
        metrics[MATCHED_CONDITION][f"mean_margin_delta_vs_{condition}"] = (
            float(statistics.fmean(margin_deltas)) if margin_deltas else 0.0
        )
    return metrics


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_id",
        "content_id",
        "condition",
        "answer_index",
        "prediction_index",
        "correct",
        "margin",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# ARC/OpenBookQA Soft-Prefix Preflight",
        "",
        f"- date: `{payload['date']}`",
        f"- benchmark: `{payload['benchmark']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- fit/eval rows: `{payload['fit_rows']}` / `{payload['eval_rows']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.3f}`",
        f"- best control by accuracy: `{headline['best_control_by_accuracy']}`",
        f"- best control accuracy: `{headline['best_control_accuracy']:.3f}`",
        f"- matched margin: `{headline['matched_mean_margin']:.6f}`",
        f"- best control margin: `{headline['best_control_mean_margin']:.6f}`",
        f"- matched minus best-control margin: `{headline['matched_minus_best_control_margin']:.6f}`",
        "",
        "## Conditions",
        "",
        "| Condition | Accuracy | Correct / N | Mean Margin |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in payload["condition_metrics"].items():
        lines.append(
            f"| `{condition}` | {metrics['accuracy']:.3f} | {metrics['correct']} / {metrics['n']} | "
            f"{metrics['mean_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden "
            "source-model summary into soft tokens prepended to the target model. The controls ask whether "
            "the soft tokens are really using the source row, or whether a static/target-only prefix can do "
            "the same thing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_contrastive_controls(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    controls = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(controls) - set(CONTRASTIVE_CONTROL_CHOICES))
    if unknown:
        raise ValueError(f"unknown contrastive controls: {unknown}")
    return controls


def run_preflight(
    *,
    output_dir: pathlib.Path,
    eval_path: pathlib.Path,
    source_cache_path: pathlib.Path,
    qwen_source_cache_path: pathlib.Path | None,
    source_score_cache_path: pathlib.Path | None,
    benchmark: str,
    row_limit: int,
    fit_fraction: float,
    fixed_fit_rows: int | None,
    source_feature_mode: str,
    source_feature_dim: int,
    target_feature_dim: int,
    source_model: str,
    target_model_path: str,
    source_device: str,
    target_device: str,
    train_device: str | None,
    target_attn_implementation: str | None,
    dtype: str,
    source_max_length: int,
    target_max_length: int,
    source_hidden_layer: int,
    source_token_pool_size: int,
    innovation_ridge: float,
    sparse_packet_rank: int,
    sparse_packet_top_k: int,
    sparse_packet_bits: int,
    local_files_only: bool,
    prefix_len: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    bootstrap_samples: int,
    continuation_mode: str,
    matched_use_target: bool,
    length_normalize: bool,
    contrastive_weight: float,
    contrastive_margin: float,
    contrastive_loss_cap: float,
    contrastive_controls: Sequence[str],
    same_byte_budget: int,
    min_accuracy_gap: float,
    min_margin_gap: float,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_all = arc_gate._load_rows(_resolve(eval_path))
    source_cache = _read_source_cache(_resolve(source_cache_path))
    rows, source_predictions = _select_rows_with_cache(rows_all, source_cache, row_limit=row_limit)
    qwen_cache_path = source_cache_path if qwen_source_cache_path is None else qwen_source_cache_path
    qwen_source_cache = _read_source_cache(_resolve(qwen_cache_path))
    qwen_predictions = _source_predictions_for_rows(rows, qwen_source_cache, label="qwen-substituted")
    source_score_cache = _read_source_score_cache(_resolve(source_score_cache_path) if source_score_cache_path else None)
    if fixed_fit_rows is None:
        fit_indices, eval_indices = _row_indices(row_count=len(rows), fit_fraction=fit_fraction, seed=seed)
    else:
        fit_count = int(fixed_fit_rows)
        if fit_count < 1 or fit_count >= len(rows):
            raise ValueError("fixed_fit_rows must leave at least one fit row and one eval row")
        fit_indices = list(range(fit_count))
        eval_indices = list(range(fit_count, len(rows)))

    source_summary, source_meta = _selected_choice_features(
        rows,
        source_predictions,
        source_feature_mode=source_feature_mode,
        feature_dim=source_feature_dim,
        source_model=source_model,
        source_device=source_device,
        source_dtype=dtype,
        source_max_length=source_max_length,
        source_hidden_layer=source_hidden_layer,
        source_token_pool_size=source_token_pool_size,
        target_alignment_model=target_model_path,
        target_alignment_device=target_device,
        fit_indices=fit_indices,
        innovation_ridge=innovation_ridge,
        sparse_packet_rank=sparse_packet_rank,
        sparse_packet_top_k=sparse_packet_top_k,
        sparse_packet_bits=sparse_packet_bits,
        local_files_only=local_files_only,
    )
    target_summary, target_meta = _target_public_features(rows, feature_dim=target_feature_dim)
    source_summary, source_standardizer = _standardize(source_summary, fit_indices)
    target_summary, target_standardizer = _standardize(target_summary, fit_indices)

    resolved_target_device = "cpu" if target_device == "auto_cpu" else arc_gate.syn._resolve_torch_device(target_device)
    resolved_train_device = (
        resolved_target_device
        if train_device is None
        else ("cpu" if train_device == "auto_cpu" else arc_gate.syn._resolve_torch_device(train_device))
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(dtype),
    }
    if target_attn_implementation and target_attn_implementation != "auto":
        model_kwargs["attn_implementation"] = target_attn_implementation
    model = AutoModelForCausalLM.from_pretrained(target_model_path, **model_kwargs).to(resolved_target_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(resolved_train_device)
    embed_tokens = model.get_input_embeddings()

    prompt_ids = [
        _encode_ids(tokenizer, _mcq_prompt(row), device=resolved_train_device, add_special_tokens=True)[-target_max_length:]
        for row in rows
    ]
    choice_ids = [
        [
            _encode_ids(
                tokenizer,
                _continuation_text(row, index, mode=continuation_mode),
                device=resolved_train_device,
                add_special_tokens=False,
            )
            for index, _ in enumerate(row.choices)
        ]
        for row in rows
    ]
    answer_indices = [int(row.answer_index) for row in rows]

    config = SoftPrefixConfig(
        prefix_len=prefix_len,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        matched_use_target=matched_use_target,
        length_normalize=length_normalize,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        contrastive_loss_cap=contrastive_loss_cap,
        contrastive_controls=tuple(contrastive_controls),
        sparse_packet_rank=sparse_packet_rank,
        sparse_packet_top_k=sparse_packet_top_k,
        sparse_packet_bits=sparse_packet_bits,
    )
    torch.manual_seed(seed)
    source_summary = source_summary.to(resolved_train_device)
    target_summary = target_summary.to(resolved_train_device)
    source_dim = int(source_summary.shape[-1])
    target_dim = int(target_summary.shape[-1])
    embed_dim = int(embed_tokens.embedding_dim)
    source_tensor_rank = int(source_summary.dim())

    def matched_connector() -> torch.nn.Module:
        if source_tensor_rank == 3:
            return SourceQuerySoftPrefixConnector(
                source_dim=source_dim,
                target_dim=target_dim,
                target_embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                prefix_len=prefix_len,
                use_target=matched_use_target,
            )
        return SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=True,
            use_target=matched_use_target,
        )

    connectors = {
        MATCHED_CONDITION: matched_connector(),
        "source_free_prefix": SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=False,
            use_target=True,
        ),
        "target_cache_only_prefix": SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=False,
            use_target=True,
        ),
        "slots_only_prefix": SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=False,
            use_target=False,
        ),
        "label_shuffled": matched_connector(),
    }
    fit_logs: dict[str, Any] = {}
    for name, connector in connectors.items():
        fit_logs[name] = _fit_connector(
            connector=connector,
            target_model=model,
            embed_tokens=embed_tokens,
            source_summary=source_summary,
            target_summary=target_summary,
            prompt_ids=prompt_ids,
            continuation_ids=choice_ids,
            answer_indices=answer_indices,
            fit_indices=fit_indices,
            config=config,
            device=resolved_train_device,
            label_shuffle=name == "label_shuffled",
            use_contrastive=name == MATCHED_CONDITION,
        )

    train_mean_source = source_summary[fit_indices].mean(dim=0)
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for eval_position, idx in enumerate(eval_indices):
            row = rows[idx]
            shuffled_idx = eval_indices[(eval_position + 1) % len(eval_indices)] if len(eval_indices) > 1 else idx
            generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + idx)
            noise_cpu = torch.randn(tuple(source_summary[idx].shape), generator=generator)
            noise = noise_cpu.to(resolved_train_device)
            noise = noise / noise.norm().clamp_min(1e-6) * source_summary[idx].norm().clamp_min(1e-6)
            candidate_roll_source = _candidate_roll_source_summary(source_summary[idx])
            candidate_score_roll_source = _candidate_score_roll_source_summary(source_summary[idx])
            atom_shuffle_source = _atom_shuffle_source_summary(source_summary[idx])
            coefficient_shuffle_source = _coefficient_shuffle_source_summary(source_summary[idx])
            top_atom_knockout_source = _top_atom_knockout_source_summary(source_summary[idx])
            source_variants = {
                MATCHED_CONDITION: source_summary[idx],
                "zero_source": torch.zeros_like(source_summary[idx]),
                "shuffled_source": source_summary[shuffled_idx],
                "source_row_shuffle": source_summary[shuffled_idx],
                "same_norm_noise": noise,
                "train_mean_source": train_mean_source,
                "atom_shuffle": atom_shuffle_source if atom_shuffle_source is not None else source_summary[idx],
                "coefficient_shuffle": (
                    coefficient_shuffle_source if coefficient_shuffle_source is not None else source_summary[idx]
                ),
                "top_atom_knockout": (
                    top_atom_knockout_source if top_atom_knockout_source is not None else source_summary[idx]
                ),
                "candidate_roll_source": (
                    candidate_roll_source if candidate_roll_source is not None else source_summary[idx]
                ),
                "candidate_roll": (
                    candidate_roll_source if candidate_roll_source is not None else source_summary[idx]
                ),
                "candidate_score_roll_source": (
                    candidate_score_roll_source
                    if candidate_score_roll_source is not None
                    else source_summary[idx]
                ),
                "source_free_prefix": source_summary[idx],
                "target_cache_only_prefix": source_summary[idx],
                "target_derived_prefix": source_summary[idx],
                "slots_only_prefix": source_summary[idx],
                "label_shuffled": source_summary[idx],
            }
            condition_scores: dict[str, list[float]] = {}
            condition_prefix_stats: dict[str, dict[str, Any]] = {}
            condition_scores["target_only"] = _score_connector_condition(
                connector=None,
                target_model=model,
                embed_tokens=embed_tokens,
                source_summary=source_summary[idx],
                target_summary=target_summary[idx],
                prompt_ids=prompt_ids[idx],
                continuation_ids=choice_ids[idx],
                length_normalize=length_normalize,
                device=resolved_train_device,
            )
            condition_prefix_stats["target_only"] = _prefix_stats_for_condition(
                connector=None,
                source_summary=source_summary[idx],
                target_summary=target_summary[idx],
                embed_dim=embed_dim,
                device=resolved_train_device,
            )
            for condition in (
                MATCHED_CONDITION,
                "zero_source",
                "shuffled_source",
                "source_row_shuffle",
                "same_norm_noise",
                "train_mean_source",
                "atom_shuffle",
                "coefficient_shuffle",
                "top_atom_knockout",
                "candidate_roll_source",
                "candidate_roll",
                "candidate_score_roll_source",
            ):
                condition_scores[condition] = _score_connector_condition(
                    connector=connectors[MATCHED_CONDITION],
                    target_model=model,
                    embed_tokens=embed_tokens,
                    source_summary=source_variants[condition],
                    target_summary=target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    continuation_ids=choice_ids[idx],
                    length_normalize=length_normalize,
                    device=resolved_train_device,
                )
                condition_prefix_stats[condition] = _prefix_stats_for_condition(
                    connector=connectors[MATCHED_CONDITION],
                    source_summary=source_variants[condition],
                    target_summary=target_summary[idx],
                    embed_dim=embed_dim,
                    device=resolved_train_device,
                )
            for condition in ("target_cache_only_prefix", "slots_only_prefix", "label_shuffled"):
                condition_scores[condition] = _score_connector_condition(
                    connector=connectors[condition],
                    target_model=model,
                    embed_tokens=embed_tokens,
                    source_summary=source_variants[condition],
                    target_summary=target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    continuation_ids=choice_ids[idx],
                    length_normalize=length_normalize,
                    device=resolved_train_device,
                )
                condition_prefix_stats[condition] = _prefix_stats_for_condition(
                    connector=connectors[condition],
                    source_summary=source_variants[condition],
                    target_summary=target_summary[idx],
                    embed_dim=embed_dim,
                    device=resolved_train_device,
                )
            condition_scores["source_free_prefix"] = _score_connector_condition(
                connector=connectors["source_free_prefix"],
                target_model=model,
                embed_tokens=embed_tokens,
                source_summary=source_variants["source_free_prefix"],
                target_summary=target_summary[idx],
                prompt_ids=prompt_ids[idx],
                continuation_ids=choice_ids[idx],
                length_normalize=length_normalize,
                device=resolved_train_device,
            )
            condition_prefix_stats["source_free_prefix"] = _prefix_stats_for_condition(
                connector=connectors["source_free_prefix"],
                source_summary=source_variants["source_free_prefix"],
                target_summary=target_summary[idx],
                embed_dim=embed_dim,
                device=resolved_train_device,
            )
            condition_scores["target_derived_prefix"] = list(condition_scores["source_free_prefix"])
            condition_prefix_stats["target_derived_prefix"] = dict(condition_prefix_stats["source_free_prefix"])
            condition_scores["candidate_derangement"] = list(np.roll(condition_scores[MATCHED_CONDITION], 1))
            condition_prefix_stats["candidate_derangement"] = dict(condition_prefix_stats[MATCHED_CONDITION])
            condition_scores["packet_only_source_index"] = _source_index_scores(
                len(row.choices),
                int(source_predictions[idx]),
            )
            condition_prefix_stats["packet_only_source_index"] = dict(condition_prefix_stats["target_only"])
            raw_source_scores = _source_scores_for_control(
                row=row,
                selected_index=int(source_predictions[idx]),
                source_score_cache=source_score_cache,
            )
            condition_scores["source_rank_control"] = _source_rank_scores(raw_source_scores)
            condition_scores["source_score_control"] = _centered_source_score_control(raw_source_scores)
            condition_prefix_stats["source_rank_control"] = dict(condition_prefix_stats["target_only"])
            condition_prefix_stats["source_score_control"] = dict(condition_prefix_stats["target_only"])
            condition_scores["qwen_substituted_packet"] = _source_index_scores(
                len(row.choices),
                int(qwen_predictions[idx]),
            )
            condition_prefix_stats["qwen_substituted_packet"] = dict(condition_prefix_stats["target_only"])
            hint = row.choices[source_predictions[idx]].encode("utf-8")[:same_byte_budget].decode(
                "utf-8", errors="ignore"
            )
            hint_prompt = _mcq_prompt(row) + f"\nVisible same-byte hint: {hint}\nBest answer:"
            hint_prompt_ids = _encode_ids(
                tokenizer,
                hint_prompt,
                device=resolved_train_device,
                add_special_tokens=True,
            )[-target_max_length:]
            condition_scores["same_byte_visible_text"] = _score_connector_condition(
                connector=None,
                target_model=model,
                embed_tokens=embed_tokens,
                source_summary=source_summary[idx],
                target_summary=target_summary[idx],
                prompt_ids=hint_prompt_ids,
                continuation_ids=choice_ids[idx],
                length_normalize=length_normalize,
                device=resolved_train_device,
            )
            condition_prefix_stats["same_byte_visible_text"] = dict(condition_prefix_stats["target_only"])
            audit_scores = [-1.0e9 for _ in row.choices]
            audit_scores[source_predictions[idx]] = 0.0
            condition_scores["source_label_copy_audit_upper_bound"] = audit_scores
            condition_prefix_stats["source_label_copy_audit_upper_bound"] = dict(condition_prefix_stats["target_only"])
            for condition in REPORT_CONDITIONS:
                scores = condition_scores[condition]
                pred = _prediction(scores)
                prefix_stats = condition_prefix_stats.get(condition, condition_prefix_stats["target_only"])
                prediction_rows.append(
                    {
                        "row_id": row.row_id,
                        "content_id": row.content_id,
                        "condition": condition,
                        "answer_index": int(row.answer_index),
                        "answer_label": row.answer_label,
                        "prediction_index": int(pred),
                        "prediction_label": row.choice_labels[pred],
                        "correct": bool(pred == row.answer_index),
                        "margin": float(_margin(scores, row.answer_index)),
                        "entropy": _score_entropy(scores),
                        "scores": [float(score) for score in scores],
                        "source_selected_index": int(source_predictions[idx]),
                        "source_selected_label": row.choice_labels[int(source_predictions[idx])],
                        "source_scores": [float(score) for score in raw_source_scores],
                        "source_score_margin": float(
                            sorted(raw_source_scores, reverse=True)[0] - sorted(raw_source_scores, reverse=True)[1]
                            if len(raw_source_scores) > 1
                            else 0.0
                        ),
                        "source_rank_by_candidate": [
                            int(rank)
                            for rank in np.argsort(np.argsort(-np.asarray(raw_source_scores, dtype=np.float64)))
                        ],
                        "qwen_substituted_index": int(qwen_predictions[idx]),
                        "qwen_substituted_label": row.choice_labels[int(qwen_predictions[idx])],
                        "control_origin": condition,
                        **prefix_stats,
                    }
                )

    metrics = _condition_metrics(prediction_rows, seed=seed + 404, bootstrap_samples=bootstrap_samples)
    matched = metrics[MATCHED_CONDITION]
    best_control_by_accuracy = max(PASS_CONTROL_CONDITIONS, key=lambda name: metrics[name]["accuracy"])
    best_control_by_margin = max(PASS_CONTROL_CONDITIONS, key=lambda name: metrics[name]["mean_margin"])
    headline = {
        "matched_accuracy": matched["accuracy"],
        "matched_mean_margin": matched["mean_margin"],
        "best_control_by_accuracy": best_control_by_accuracy,
        "best_control_accuracy": metrics[best_control_by_accuracy]["accuracy"],
        "best_control_by_margin": best_control_by_margin,
        "best_control_mean_margin": metrics[best_control_by_margin]["mean_margin"],
        "matched_minus_best_control_accuracy": matched["accuracy"] - metrics[best_control_by_accuracy]["accuracy"],
        "matched_minus_best_control_margin": matched["mean_margin"] - metrics[best_control_by_margin]["mean_margin"],
    }
    pass_gate = bool(
        headline["matched_minus_best_control_accuracy"] >= min_accuracy_gap
        and headline["matched_minus_best_control_margin"] >= min_margin_gap
        and matched[f"paired_accuracy_vs_{best_control_by_accuracy}"]["ci95_low"] > 0.0
    )
    interpretation = (
        "This preflight passes only if the matched soft-prefix uses source information that the "
        "target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final "
        "scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak."
    )
    created_utc = dt.datetime.now(dt.UTC)
    payload = {
        "gate": "source_private_arc_openbookqa_soft_prefix_preflight",
        "date": created_utc.date().isoformat(),
        "created_utc": created_utc.isoformat(),
        "benchmark": benchmark,
        "pass_gate": pass_gate,
        "implementation_gate_only": True,
        "fit_rows": len(fit_indices),
        "eval_rows": len(eval_indices),
        "row_limit": len(rows),
        "fit_indices": fit_indices,
        "eval_indices": eval_indices,
        "config": {
            "source_feature_mode": source_feature_mode,
            "source_feature_dim": source_feature_dim,
            "source_token_pool_size": source_token_pool_size,
            "innovation_ridge": innovation_ridge,
            "sparse_packet_rank": sparse_packet_rank,
            "sparse_packet_top_k": sparse_packet_top_k,
            "sparse_packet_bits": sparse_packet_bits,
            "source_tensor_rank": source_tensor_rank,
            "target_feature_dim": target_feature_dim,
            "prefix_len": prefix_len,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "matched_use_target": matched_use_target,
            "length_normalize": length_normalize,
            "contrastive_weight": contrastive_weight,
            "contrastive_margin": contrastive_margin,
            "contrastive_loss_cap": contrastive_loss_cap,
            "contrastive_controls": list(contrastive_controls),
            "continuation_mode": continuation_mode,
            "same_byte_budget": same_byte_budget,
            "fixed_fit_rows": None if fixed_fit_rows is None else int(fixed_fit_rows),
            "source_model": source_model,
            "target_model": target_model_path,
            "source_device": source_device,
            "target_device": target_device,
            "train_device": resolved_train_device,
            "target_attn_implementation": target_attn_implementation or "auto",
            "dtype": dtype,
        },
        "feature_metadata": {
            "source": source_meta,
            "target": target_meta,
            "source_standardizer": source_standardizer,
            "target_standardizer": target_standardizer,
        },
        "fit_logs": fit_logs,
        "headline": headline,
        "pass_control_conditions": list(PASS_CONTROL_CONDITIONS),
        "audit_only_conditions": ["source_label_copy_audit_upper_bound"],
        "condition_metrics": metrics,
        "interpretation": interpretation,
        "inputs": {
            "eval_path": _display(eval_path),
            "source_cache_path": _display(source_cache_path),
            "qwen_source_cache_path": _display(qwen_cache_path),
            "source_score_cache_path": _display(source_score_cache_path) if source_score_cache_path else None,
        },
        "runtime": {
            "latency_s": float(time.perf_counter() - total_start),
            "peak_rss_mib": _peak_rss_mib(),
        },
    }
    json_path = output_dir / "arc_openbookqa_soft_prefix_preflight.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "prediction_audit.jsonl", prediction_rows)
    _write_csv(output_dir / "prediction_audit.csv", prediction_rows)
    _write_markdown(output_dir / "arc_openbookqa_soft_prefix_preflight.md", payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {
                "path": _display(path),
                "sha256": _sha256_file(path),
                "bytes": _resolve(path).stat().st_size,
            }
            for path in (
                json_path,
                output_dir / "prediction_audit.jsonl",
                output_dir / "prediction_audit.csv",
                output_dir / "arc_openbookqa_soft_prefix_preflight.md",
            )
        ],
        "inputs": payload["inputs"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", default="ARC-Challenge")
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_ARC_VALIDATION)
    parser.add_argument("--source-cache-path", type=pathlib.Path, default=DEFAULT_ARC_SOURCE_CACHE)
    parser.add_argument("--qwen-source-cache-path", type=pathlib.Path, default=None)
    parser.add_argument("--source-score-cache-path", type=pathlib.Path, default=None)
    parser.add_argument("--row-limit", type=int, default=8)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument(
        "--fixed-fit-rows",
        type=int,
        default=None,
        help="Use the first N selected rows for fitting and all remaining selected rows for eval.",
    )
    parser.add_argument(
        "--source-feature-mode",
        choices=(
            "hashed_selected",
            "hashed_selected_residual",
            "hf_selected_hidden",
            "hf_selected_hidden_residual",
            "cached_choice_score_pool",
            "cached_choice_score_pool_residual",
            "hf_choice_hidden_candidate_pool",
            "hf_choice_hidden_candidate_pool_residual",
            "hf_choice_hidden_score_candidate_pool",
            "hf_choice_hidden_score_candidate_pool_residual",
            "hf_choice_hidden_public_innovation_candidate_pool",
            "hf_choice_hidden_public_innovation_candidate_pool_residual",
            "hf_choice_hidden_score_public_innovation_candidate_pool",
            "hf_choice_hidden_score_public_innovation_candidate_pool_residual",
            "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool",
            "hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool_residual",
            "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool",
            "hf_choice_hidden_score_public_innovation_sparse_pca_packet_candidate_pool_residual",
            "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool",
            "hf_choice_hidden_public_innovation_target_aligned_sparse_pca_packet_candidate_pool_residual",
            "hf_choice_token_hidden_pool",
            "hf_choice_token_hidden_pool_residual",
        ),
        default="hf_selected_hidden",
    )
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--source-token-pool-size", type=int, default=32)
    parser.add_argument("--innovation-ridge", type=float, default=10.0)
    parser.add_argument("--sparse-packet-rank", type=int, default=16)
    parser.add_argument("--sparse-packet-top-k", type=int, default=4)
    parser.add_argument("--sparse-packet-bits", type=int, default=4)
    parser.add_argument("--target-feature-dim", type=int, default=64)
    parser.add_argument("--source-model", default=DEFAULT_QWEN_SOURCE)
    parser.add_argument("--target-model", default=DEFAULT_QWEN_TARGET)
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--target-attn-implementation", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=192)
    parser.add_argument("--target-max-length", type=int, default=256)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--continuation-mode", choices=("label", "label_and_choice", "choice"), default="label")
    parser.add_argument("--matched-use-target", choices=("true", "false"), default="false")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.05)
    parser.add_argument("--contrastive-loss-cap", type=float, default=0.5)
    parser.add_argument(
        "--contrastive-controls",
        default="",
        help=(
            "Comma-separated source controls for matched-connector margin ranking. "
            f"Allowed: {','.join(CONTRASTIVE_CONTROL_CHOICES)}"
        ),
    )
    parser.add_argument("--same-byte-budget", type=int, default=12)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-margin-gap", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contrastive_controls = _parse_contrastive_controls(str(args.contrastive_controls))
    payload = run_preflight(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        source_cache_path=args.source_cache_path,
        qwen_source_cache_path=args.qwen_source_cache_path,
        source_score_cache_path=args.source_score_cache_path,
        benchmark=str(args.benchmark),
        row_limit=int(args.row_limit),
        fit_fraction=float(args.fit_fraction),
        fixed_fit_rows=None if args.fixed_fit_rows is None else int(args.fixed_fit_rows),
        source_feature_mode=str(args.source_feature_mode),
        source_feature_dim=int(args.source_feature_dim),
        target_feature_dim=int(args.target_feature_dim),
        source_model=str(args.source_model),
        target_model_path=str(args.target_model),
        source_device=str(args.source_device),
        target_device=str(args.target_device),
        train_device=None if args.train_device is None else str(args.train_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        source_max_length=int(args.source_max_length),
        target_max_length=int(args.target_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        source_token_pool_size=int(args.source_token_pool_size),
        innovation_ridge=float(args.innovation_ridge),
        sparse_packet_rank=int(args.sparse_packet_rank),
        sparse_packet_top_k=int(args.sparse_packet_top_k),
        sparse_packet_bits=int(args.sparse_packet_bits),
        local_files_only=str(args.local_files_only).lower() == "true",
        prefix_len=int(args.prefix_len),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
        continuation_mode=str(args.continuation_mode),
        matched_use_target=str(args.matched_use_target).lower() == "true",
        length_normalize=str(args.length_normalize).lower() == "true",
        contrastive_weight=float(args.contrastive_weight),
        contrastive_margin=float(args.contrastive_margin),
        contrastive_loss_cap=float(args.contrastive_loss_cap),
        contrastive_controls=contrastive_controls,
        same_byte_budget=int(args.same_byte_budget),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_margin_gap=float(args.min_margin_gap),
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "fit_rows": payload["fit_rows"],
                "eval_rows": payload["eval_rows"],
                "matched_accuracy": payload["headline"]["matched_accuracy"],
                "best_control_by_accuracy": payload["headline"]["best_control_by_accuracy"],
                "best_control_accuracy": payload["headline"]["best_control_accuracy"],
                "matched_minus_best_control_margin": payload["headline"]["matched_minus_best_control_margin"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
