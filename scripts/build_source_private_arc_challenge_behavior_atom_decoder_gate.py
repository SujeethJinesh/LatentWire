from __future__ import annotations

"""Strict ARC-Challenge behavior-atom decoder Sparse Resonance Packet gate.

This branch keeps the source-private hidden-atom transport from the previous
gate, but replaces unsupervised PCA atoms with train-fit behavior-supervised
directions.  The packet is still computed only from answer-key-forbidden source
hidden innovations at evaluation time; labels and target behavior are used only
to fit the packet basis and receiver on the training disagreement slice.
"""

import argparse
import datetime as dt
import gc
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_behavior_residual_packet_gate as behavior_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_confidence_ecoc_packet_gate as ecoc_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_hidden_atom_decoder_gate as hidden_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as soft_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_"
    "tinyllama_to_qwen3_disagreement"
)
DEFAULT_VALIDATION = soft_gate.DEFAULT_VALIDATION
DEFAULT_TEST = soft_gate.DEFAULT_TEST
DEFAULT_SOURCE_FAMILY_GATE_DIR = soft_gate.DEFAULT_SOURCE_FAMILY_GATE_DIR
DEFAULT_TINY_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_TINY_VALIDATION_SCORE_CACHE
DEFAULT_TINY_TEST_SCORE_CACHE = soft_gate.DEFAULT_TINY_TEST_SCORE_CACHE
DEFAULT_QWEN_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_QWEN_VALIDATION_SCORE_CACHE
DEFAULT_QWEN_TEST_SCORE_CACHE = soft_gate.DEFAULT_QWEN_TEST_SCORE_CACHE
DEFAULT_TINY_MODEL = behavior_gate.DEFAULT_TINY_MODEL
DEFAULT_QWEN3_MODEL = behavior_gate.DEFAULT_QWEN3_MODEL

MATCHED_CONDITION = "matched_behavior_atom_decoder_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_decoder_only",
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "same_source_choice_row_shuffle",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll",
    "candidate_derangement",
    "packet_only_source_index",
    "source_rank_control",
    "source_score_control",
    "source_score_quantized_control",
    "same_byte_visible_text",
    "qwen_substituted_packet",
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
STRICT_REQUIRED_CONTROLS = CONTROL_CONDITIONS
EVENT_GATE_CORRUPTION_CONDITIONS = (
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "same_source_choice_row_shuffle",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll",
    "candidate_derangement",
)
RECEIVER_TRAINING_MODES = ("matched_only", "corruption_noop")
PACKET_INTEGRITY_MODES = ("none", "candidate_atom")
ATOM_BASIS_MODES = ("behavior_svd", "batchtopk_behavior", "paired_batchtopk_behavior")


def _parse_condition_weight_spec(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    text = str(spec).strip()
    if not text:
        return weights
    for item in text.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "condition weight overrides must use condition=weight entries, "
                f"got {entry!r}"
            )
        condition, value = entry.split("=", 1)
        condition = condition.strip()
        if condition not in EVENT_GATE_CORRUPTION_CONDITIONS:
            raise ValueError(f"unknown corruption condition override: {condition}")
        weights[condition] = float(value)
    return weights


@dataclass(frozen=True)
class RidgeNoInterceptScalarMap:
    x_scale: np.ndarray
    weights: np.ndarray
    ridge: float
    fit_mse: float
    fit_r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        scaled = values / self.x_scale
        return scaled @ self.weights


@dataclass(frozen=True)
class EventGateRule:
    residual_weight: float
    threshold: float
    event_model: behavior_gate.RidgeScalarMap
    require_prediction_change: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PacketIntegrityRule:
    packet_to_target_map: hidden_gate.RidgeMatrixMap
    integrity_model: behavior_gate.RidgeScalarMap
    threshold: float
    atom_profile: np.ndarray
    atom_profile_scale: np.ndarray
    metadata: dict[str, Any]


def _behavior_target_matrix(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
) -> np.ndarray:
    """Build train targets for source atom directions from target residual need."""

    residual = behavior_gate._candidate_targets(rows, target_scores).reshape(-1, 1)
    target = behavior_gate._target_score_features(rows, target_scores)
    score = target[:, [0]]
    centered = target[:, [1]]
    prob = target[:, [2]]
    inv_rank = target[:, [3]]
    margin = target[:, [4]]
    return np.concatenate(
        [
            residual,
            residual * score,
            residual * centered,
            residual * prob,
            residual * inv_rank,
            residual * margin,
            residual * np.square(centered),
            residual * np.square(prob),
            residual * np.square(margin),
            residual * np.sign(centered),
        ],
        axis=1,
    )


def _fit_behavior_atom_packet_from_features(
    source_features: np.ndarray,
    behavior_targets: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    rank: int,
    top_k: int,
    quant_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_features, dtype=np.float64)
    targets = np.asarray(behavior_targets, dtype=np.float64)
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if source.ndim != 2:
        raise ValueError("source_features must be rank-2")
    if targets.ndim != 2:
        raise ValueError("behavior_targets must be rank-2")
    if source.shape[0] != targets.shape[0]:
        raise ValueError("source/target row mismatch")
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if rank < 1:
        raise ValueError("packet rank must be at least 1")

    x_mean = source[fit].mean(axis=0, keepdims=True)
    x_std = source[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    standardized_source = (source - x_mean) / x_std
    x_fit = standardized_source[fit]

    y_fit_raw = targets[fit]
    y_mean = y_fit_raw.mean(axis=0, keepdims=True)
    y_std = y_fit_raw.std(axis=0, keepdims=True).clip(min=1e-6)
    y_fit = (y_fit_raw - y_mean) / y_std

    cross_cov = x_fit.T @ y_fit / float(max(fit.size, 1))
    u, singular_values, _ = np.linalg.svd(cross_cov, full_matrices=False)
    actual_rank = int(min(rank, u.shape[1]))
    if actual_rank < 1:
        raise ValueError("could not fit a non-empty behavior atom basis")
    directions = u[:, :actual_rank]
    coeffs = standardized_source @ directions
    sparse_packet, quant_meta = preflight._sparse_topk_quantized_coordinates(
        coeffs,
        fit_flat_indices=fit,
        top_k=top_k,
        quant_bits=quant_bits,
    )

    target_map = hidden_gate._fit_ridge_matrix_map(
        sparse_packet[fit],
        y_fit_raw,
        fit_indices=np.arange(fit.size, dtype=np.int64),
        ridge=1.0,
    )
    retained = np.square(singular_values[:actual_rank])
    total = float(np.square(singular_values).sum())
    return sparse_packet, {
        "kind": "train_fit_behavior_supervised_hidden_atom_packet_coordinates",
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(source.shape[1]),
        "behavior_target_dim": int(targets.shape[1]),
        "packet_rank": int(actual_rank),
        "requested_packet_rank": int(rank),
        "behavior_cross_cov_energy_ratio": float(retained.sum() / total) if total > 1e-12 else 0.0,
        "behavior_singular_value_max": float(singular_values[0]) if singular_values.size else 0.0,
        "behavior_singular_value_min_retained": float(singular_values[actual_rank - 1])
        if singular_values.size
        else 0.0,
        "source_std_min": float(x_std.min()),
        "source_std_max": float(x_std.max()),
        "packet_to_behavior_fit_mse": float(target_map.fit_mse),
        "packet_to_behavior_fit_r2": float(target_map.fit_r2),
        **quant_meta,
    }


def _torch_batch_topk(latent: Any, top_k: int) -> Any:
    if top_k < 1:
        return latent
    import torch

    if latent.numel() == 0:
        return latent
    keep = min(int(latent.shape[0]) * int(top_k), int(latent.numel()))
    if keep >= int(latent.numel()):
        return latent
    values = latent.reshape(-1)
    threshold = torch.topk(values, k=keep, largest=True).values[-1]
    return latent * (latent >= threshold).to(latent.dtype)


def _numpy_topk_rows(values: np.ndarray, top_k: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if top_k < 1 or top_k >= matrix.shape[1]:
        return matrix.copy()
    output = np.zeros_like(matrix, dtype=np.float64)
    top_ids = np.argsort(matrix, axis=1)[:, -int(top_k) :]
    rows = np.arange(matrix.shape[0])[:, None]
    output[rows, top_ids] = matrix[rows, top_ids]
    return output


def _fit_batchtopk_behavior_atom_packet_from_features(
    source_features: np.ndarray,
    behavior_targets: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    rank: int,
    top_k: int,
    quant_bits: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    reconstruction_weight: float,
    l1_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local venv.
        raise RuntimeError("batchtopk_behavior atom basis requires torch in the repo-local venv") from exc

    source = np.asarray(source_features, dtype=np.float32)
    targets = np.asarray(behavior_targets, dtype=np.float32)
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if source.ndim != 2:
        raise ValueError("source_features must be rank-2")
    if targets.ndim != 2:
        raise ValueError("behavior_targets must be rank-2")
    if source.shape[0] != targets.shape[0]:
        raise ValueError("source/target row mismatch")
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if rank < 1:
        raise ValueError("packet rank must be at least 1")

    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    x_mean = source[fit].mean(axis=0, keepdims=True)
    x_std = source[fit].std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    standardized_source = (source - x_mean) / x_std

    y_mean = targets[fit].mean(axis=0, keepdims=True)
    y_std = targets[fit].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    standardized_targets = (targets - y_mean) / y_std

    tensor_x = torch.from_numpy(standardized_source)
    tensor_y = torch.from_numpy(standardized_targets)
    fit_tensor = torch.from_numpy(fit.astype(np.int64))
    dim = int(tensor_x.shape[1])
    target_dim = int(tensor_y.shape[1])
    latent_dim = int(rank)

    encoder = torch.nn.Linear(dim, latent_dim)
    decoder = torch.nn.Linear(latent_dim, dim)
    head = torch.nn.Linear(latent_dim, target_dim)
    torch.nn.init.xavier_uniform_(encoder.weight)
    torch.nn.init.zeros_(encoder.bias)
    torch.nn.init.xavier_uniform_(decoder.weight)
    torch.nn.init.zeros_(decoder.bias)
    torch.nn.init.xavier_uniform_(head.weight)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(head.parameters()),
        lr=float(learning_rate),
    )
    rng = np.random.default_rng(int(seed) + 4099)
    actual_batch = max(4, int(batch_size))
    final_behavior_loss = 0.0
    final_reconstruction_loss = 0.0
    final_l1_loss = 0.0
    epochs = max(1, int(epochs))
    for _ in range(epochs):
        order = rng.permutation(fit)
        for start in range(0, order.size, actual_batch):
            batch_ids = torch.from_numpy(order[start : start + actual_batch].astype(np.int64))
            batch_x = tensor_x[batch_ids]
            batch_y = tensor_y[batch_ids]
            latent = torch.relu(encoder(batch_x))
            sparse_latent = _torch_batch_topk(latent, top_k=top_k)
            pred_y = head(sparse_latent)
            reconstruction = decoder(sparse_latent)
            behavior_loss = torch.mean((pred_y - batch_y) ** 2)
            reconstruction_loss = torch.mean((reconstruction - batch_x) ** 2)
            l1_loss = torch.mean(latent)
            loss = (
                behavior_loss
                + float(reconstruction_weight) * reconstruction_loss
                + float(l1_weight) * l1_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_behavior_loss = float(behavior_loss.detach().cpu())
            final_reconstruction_loss = float(reconstruction_loss.detach().cpu())
            final_l1_loss = float(l1_loss.detach().cpu())

    with torch.no_grad():
        latent = torch.relu(encoder(tensor_x)).detach().cpu().numpy().astype(np.float64)
        sparse_coeffs = _numpy_topk_rows(latent, top_k=top_k)
        train_latent = torch.from_numpy(sparse_coeffs[fit].astype(np.float32))
        pred_fit = head(train_latent).detach().cpu().numpy().astype(np.float64)
        y_fit = standardized_targets[fit].astype(np.float64)
    fit_mse = float(np.mean(np.square(y_fit - pred_fit)))
    baseline = float(np.mean(np.square(y_fit - y_fit.mean(axis=0, keepdims=True))))
    fit_r2 = 0.0 if baseline <= 1e-12 else 1.0 - fit_mse / baseline
    active_counts = np.count_nonzero(sparse_coeffs[fit] > 1e-8, axis=1)

    sparse_packet, quant_meta = preflight._sparse_topk_quantized_coordinates(
        sparse_coeffs,
        fit_flat_indices=fit,
        top_k=top_k,
        quant_bits=quant_bits,
    )
    return sparse_packet, {
        "kind": "train_fit_batchtopk_behavior_hidden_atom_packet_coordinates",
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(source.shape[1]),
        "behavior_target_dim": int(targets.shape[1]),
        "packet_rank": int(latent_dim),
        "requested_packet_rank": int(rank),
        "batchtopk_top_k": int(top_k),
        "batchtopk_epochs": int(epochs),
        "batchtopk_batch_size": int(actual_batch),
        "batchtopk_learning_rate": float(learning_rate),
        "batchtopk_reconstruction_weight": float(reconstruction_weight),
        "batchtopk_l1_weight": float(l1_weight),
        "batchtopk_final_behavior_loss": float(final_behavior_loss),
        "batchtopk_final_reconstruction_loss": float(final_reconstruction_loss),
        "batchtopk_final_l1_loss": float(final_l1_loss),
        "batchtopk_behavior_fit_mse": float(fit_mse),
        "batchtopk_behavior_fit_r2": float(fit_r2),
        "batchtopk_active_atoms_mean": float(active_counts.mean()) if active_counts.size else 0.0,
        "batchtopk_active_atoms_max": int(active_counts.max()) if active_counts.size else 0,
        "batchtopk_dead_atom_count": int(np.sum(np.max(sparse_coeffs[fit], axis=0) <= 1e-8)),
        "source_std_min": float(x_std.min()),
        "source_std_max": float(x_std.max()),
        **quant_meta,
    }


def _fit_paired_batchtopk_behavior_atom_packet_from_features(
    source_features: np.ndarray,
    target_features: np.ndarray,
    behavior_targets: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    rank: int,
    top_k: int,
    quant_bits: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    reconstruction_weight: float,
    l1_weight: float,
    alignment_weight: float,
    target_behavior_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local venv.
        raise RuntimeError("paired_batchtopk_behavior atom basis requires torch in the repo-local venv") from exc

    source = np.asarray(source_features, dtype=np.float32)
    target = np.asarray(target_features, dtype=np.float32)
    behavior = np.asarray(behavior_targets, dtype=np.float32)
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if source.ndim != 2 or target.ndim != 2 or behavior.ndim != 2:
        raise ValueError("source_features, target_fit_features, and behavior_targets must be rank-2")
    if source.shape[0] != behavior.shape[0]:
        raise ValueError("source/behavior row mismatch")
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if target.shape[0] != fit.size:
        raise ValueError("target_fit_features must contain exactly the fit candidates")
    if rank < 1:
        raise ValueError("packet rank must be at least 1")

    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    source_mean = source[fit].mean(axis=0, keepdims=True)
    source_std = source[fit].std(axis=0, keepdims=True)
    source_std = np.where(source_std < 1e-6, 1.0, source_std).astype(np.float32)
    source_scaled = (source - source_mean) / source_std

    target_mean = target.mean(axis=0, keepdims=True)
    target_std = target.std(axis=0, keepdims=True)
    target_std = np.where(target_std < 1e-6, 1.0, target_std).astype(np.float32)
    target_scaled = (target - target_mean) / target_std

    behavior_mean = behavior[fit].mean(axis=0, keepdims=True)
    behavior_std = behavior[fit].std(axis=0, keepdims=True)
    behavior_std = np.where(behavior_std < 1e-6, 1.0, behavior_std).astype(np.float32)
    behavior_scaled = (behavior - behavior_mean) / behavior_std

    tensor_source = torch.from_numpy(source_scaled)
    tensor_target = torch.from_numpy(target_scaled)
    tensor_behavior = torch.from_numpy(behavior_scaled)
    source_dim = int(tensor_source.shape[1])
    target_dim = int(tensor_target.shape[1])
    behavior_dim = int(tensor_behavior.shape[1])
    latent_dim = int(rank)

    source_encoder = torch.nn.Linear(source_dim, latent_dim)
    target_encoder = torch.nn.Linear(target_dim, latent_dim)
    source_decoder = torch.nn.Linear(latent_dim, source_dim)
    target_decoder = torch.nn.Linear(latent_dim, target_dim)
    head = torch.nn.Linear(latent_dim, behavior_dim)
    for module in (source_encoder, target_encoder, source_decoder, target_decoder, head):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)
    opt = torch.optim.Adam(
        list(source_encoder.parameters())
        + list(target_encoder.parameters())
        + list(source_decoder.parameters())
        + list(target_decoder.parameters())
        + list(head.parameters()),
        lr=float(learning_rate),
    )
    rng = np.random.default_rng(int(seed) + 6151)
    actual_batch = max(4, int(batch_size))
    epochs = max(1, int(epochs))
    final_source_behavior_loss = 0.0
    final_target_behavior_loss = 0.0
    final_alignment_loss = 0.0
    final_reconstruction_loss = 0.0
    final_l1_loss = 0.0
    for _ in range(epochs):
        order = rng.permutation(fit.size)
        for start in range(0, order.size, actual_batch):
            batch_positions_np = order[start : start + actual_batch].astype(np.int64)
            batch_positions = torch.from_numpy(batch_positions_np)
            batch_source_ids = torch.from_numpy(fit[batch_positions_np].astype(np.int64))
            batch_source = tensor_source[batch_source_ids]
            batch_target = tensor_target[batch_positions]
            batch_behavior = tensor_behavior[batch_source_ids]
            source_latent = _torch_batch_topk(torch.relu(source_encoder(batch_source)), top_k=top_k)
            target_latent = _torch_batch_topk(torch.relu(target_encoder(batch_target)), top_k=top_k)
            source_behavior = head(source_latent)
            target_behavior = head(target_latent)
            source_behavior_loss = torch.mean((source_behavior - batch_behavior) ** 2)
            target_behavior_loss = torch.mean((target_behavior - batch_behavior) ** 2)
            alignment_loss = torch.mean((source_latent - target_latent.detach()) ** 2)
            reconstruction_loss = torch.mean((source_decoder(source_latent) - batch_source) ** 2) + torch.mean(
                (target_decoder(target_latent) - batch_target) ** 2
            )
            l1_loss = torch.mean(source_latent) + torch.mean(target_latent)
            loss = (
                source_behavior_loss
                + float(target_behavior_weight) * target_behavior_loss
                + float(alignment_weight) * alignment_loss
                + float(reconstruction_weight) * reconstruction_loss
                + float(l1_weight) * l1_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_source_behavior_loss = float(source_behavior_loss.detach().cpu())
            final_target_behavior_loss = float(target_behavior_loss.detach().cpu())
            final_alignment_loss = float(alignment_loss.detach().cpu())
            final_reconstruction_loss = float(reconstruction_loss.detach().cpu())
            final_l1_loss = float(l1_loss.detach().cpu())

    with torch.no_grad():
        source_latent = torch.relu(source_encoder(tensor_source)).detach().cpu().numpy().astype(np.float64)
        target_latent = torch.relu(target_encoder(tensor_target)).detach().cpu().numpy().astype(np.float64)
        source_sparse = _numpy_topk_rows(source_latent, top_k=top_k)
        target_sparse = _numpy_topk_rows(target_latent, top_k=top_k)
        train_source = torch.from_numpy(source_sparse[fit].astype(np.float32))
        train_target = torch.from_numpy(target_sparse.astype(np.float32))
        pred_source_fit = head(train_source).detach().cpu().numpy().astype(np.float64)
        pred_target_fit = head(train_target).detach().cpu().numpy().astype(np.float64)
        y_fit = behavior_scaled[fit].astype(np.float64)
    source_mse = float(np.mean(np.square(y_fit - pred_source_fit)))
    target_mse = float(np.mean(np.square(y_fit - pred_target_fit)))
    baseline = float(np.mean(np.square(y_fit - y_fit.mean(axis=0, keepdims=True))))
    source_r2 = 0.0 if baseline <= 1e-12 else 1.0 - source_mse / baseline
    target_r2 = 0.0 if baseline <= 1e-12 else 1.0 - target_mse / baseline
    alignment_mse = float(np.mean(np.square(source_sparse[fit] - target_sparse)))
    source_active = np.count_nonzero(source_sparse[fit] > 1e-8, axis=1)
    target_active = np.count_nonzero(target_sparse[fit] > 1e-8, axis=1)

    sparse_packet, quant_meta = preflight._sparse_topk_quantized_coordinates(
        source_sparse,
        fit_flat_indices=fit,
        top_k=top_k,
        quant_bits=quant_bits,
    )
    return sparse_packet, {
        "kind": "train_fit_paired_batchtopk_behavior_hidden_atom_packet_coordinates",
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(source.shape[1]),
        "target_feature_dim": int(target.shape[1]),
        "target_hidden_fit_candidates_only": True,
        "behavior_target_dim": int(behavior.shape[1]),
        "packet_rank": int(latent_dim),
        "requested_packet_rank": int(rank),
        "batchtopk_top_k": int(top_k),
        "batchtopk_epochs": int(epochs),
        "batchtopk_batch_size": int(actual_batch),
        "batchtopk_learning_rate": float(learning_rate),
        "batchtopk_reconstruction_weight": float(reconstruction_weight),
        "batchtopk_l1_weight": float(l1_weight),
        "paired_alignment_weight": float(alignment_weight),
        "paired_target_behavior_weight": float(target_behavior_weight),
        "paired_final_source_behavior_loss": float(final_source_behavior_loss),
        "paired_final_target_behavior_loss": float(final_target_behavior_loss),
        "paired_final_alignment_loss": float(final_alignment_loss),
        "paired_final_reconstruction_loss": float(final_reconstruction_loss),
        "paired_final_l1_loss": float(final_l1_loss),
        "paired_source_behavior_fit_mse": float(source_mse),
        "paired_target_behavior_fit_mse": float(target_mse),
        "paired_source_behavior_fit_r2": float(source_r2),
        "paired_target_behavior_fit_r2": float(target_r2),
        "paired_source_target_latent_fit_mse": float(alignment_mse),
        "paired_source_active_atoms_mean": float(source_active.mean()) if source_active.size else 0.0,
        "paired_source_active_atoms_max": int(source_active.max()) if source_active.size else 0,
        "paired_target_active_atoms_mean": float(target_active.mean()) if target_active.size else 0.0,
        "paired_target_active_atoms_max": int(target_active.max()) if target_active.size else 0,
        "paired_dead_source_atom_count": int(np.sum(np.max(source_sparse[fit], axis=0) <= 1e-8)),
        "paired_dead_target_atom_count": int(np.sum(np.max(target_sparse[fit], axis=0) <= 1e-8)),
        "source_std_min": float(source_std.min()),
        "source_std_max": float(source_std.max()),
        "target_std_min": float(target_std.min()),
        "target_std_max": float(target_std.max()),
        **quant_meta,
    }


def _fit_source_packet(
    rows: Sequence[arc_gate.ArcRow],
    *,
    target_scores: Sequence[Sequence[float]],
    source_model: str,
    target_model: str,
    source_device: str,
    target_device: str,
    dtype: str,
    source_max_length: int,
    target_max_length: int,
    source_hidden_layer: int,
    target_hidden_layer: int,
    source_feature_dim: int,
    fit_candidate_indices: np.ndarray,
    fit_row_count: int,
    ridge: float,
    packet_rank: int,
    packet_top_k: int,
    packet_bits: int,
    atom_basis_mode: str,
    batchtopk_epochs: int,
    batchtopk_learning_rate: float,
    batchtopk_batch_size: int,
    batchtopk_reconstruction_weight: float,
    batchtopk_l1_weight: float,
    paired_alignment_weight: float,
    paired_target_behavior_weight: float,
    seed: int,
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    flat_hidden, hidden_meta = preflight._hf_choice_hidden_features(
        list(rows),
        model_path=source_model,
        device=source_device,
        dtype=dtype,
        max_length=source_max_length,
        local_files_only=local_files_only,
        hidden_layer=source_hidden_layer,
    )
    public_flat, public_meta = preflight._public_candidate_hashed_features(list(rows), feature_dim=source_feature_dim)
    innovation, innovation_meta = preflight._public_candidate_innovation_features(
        flat_hidden,
        public_flat,
        fit_flat_indices=fit_candidate_indices,
        ridge=ridge,
    )
    behavior_targets = _behavior_target_matrix(rows, target_scores)
    if atom_basis_mode == "behavior_svd":
        sparse_packet, sparse_meta = _fit_behavior_atom_packet_from_features(
            innovation,
            behavior_targets,
            fit_flat_indices=fit_candidate_indices,
            rank=packet_rank,
            top_k=packet_top_k,
            quant_bits=packet_bits,
        )
    elif atom_basis_mode == "batchtopk_behavior":
        sparse_packet, sparse_meta = _fit_batchtopk_behavior_atom_packet_from_features(
            innovation,
            behavior_targets,
            fit_flat_indices=fit_candidate_indices,
            rank=packet_rank,
            top_k=packet_top_k,
            quant_bits=packet_bits,
            epochs=batchtopk_epochs,
            learning_rate=batchtopk_learning_rate,
            batch_size=batchtopk_batch_size,
            reconstruction_weight=batchtopk_reconstruction_weight,
            l1_weight=batchtopk_l1_weight,
            seed=seed,
        )
    elif atom_basis_mode == "paired_batchtopk_behavior":
        target_fit_rows = list(rows[: int(fit_row_count)])
        target_flat_hidden, target_hidden_meta = preflight._hf_choice_hidden_features(
            target_fit_rows,
            model_path=target_model,
            device=target_device,
            dtype=dtype,
            max_length=target_max_length,
            local_files_only=local_files_only,
            hidden_layer=target_hidden_layer,
        )
        target_public_flat, target_public_meta = preflight._public_candidate_hashed_features(
            target_fit_rows,
            feature_dim=source_feature_dim,
        )
        target_fit_indices = np.arange(target_flat_hidden.shape[0], dtype=np.int64)
        target_innovation, target_innovation_meta = preflight._public_candidate_innovation_features(
            target_flat_hidden,
            target_public_flat,
            fit_flat_indices=target_fit_indices,
            ridge=ridge,
        )
        sparse_packet, sparse_meta = _fit_paired_batchtopk_behavior_atom_packet_from_features(
            innovation,
            target_innovation,
            behavior_targets,
            fit_flat_indices=fit_candidate_indices,
            rank=packet_rank,
            top_k=packet_top_k,
            quant_bits=packet_bits,
            epochs=batchtopk_epochs,
            learning_rate=batchtopk_learning_rate,
            batch_size=batchtopk_batch_size,
            reconstruction_weight=batchtopk_reconstruction_weight,
            l1_weight=batchtopk_l1_weight,
            alignment_weight=paired_alignment_weight,
            target_behavior_weight=paired_target_behavior_weight,
            seed=seed,
        )
        sparse_meta = {
            **sparse_meta,
            "target_hidden": target_hidden_meta,
            "target_public": target_public_meta,
            "target_public_innovation": target_innovation_meta,
            "target_hidden_fit_row_count": int(fit_row_count),
            "target_hidden_fit_candidate_count": int(target_flat_hidden.shape[0]),
        }
    else:
        raise ValueError(f"unknown atom_basis_mode: {atom_basis_mode}")
    return sparse_packet, {
        "source_hidden": hidden_meta,
        "public": public_meta,
        "source_public_innovation": innovation_meta,
        "atom_basis_mode": atom_basis_mode,
        "sparse_packet": sparse_meta,
    }


def _innovation_decoder_features(target_features: np.ndarray, packet_features: np.ndarray) -> np.ndarray:
    target = np.asarray(target_features, dtype=np.float64)
    packet = np.asarray(packet_features, dtype=np.float64)
    if target.ndim != 2 or packet.ndim != 2:
        raise ValueError("target_features and packet_features must be rank-2")
    if target.shape[0] != packet.shape[0]:
        raise ValueError("target/packet candidate mismatch")
    if target.shape[1] < 5:
        raise ValueError("target_features must include score, centered, prob, rank, and margin columns")
    centered = target[:, [1]]
    probability = target[:, [2]]
    margin = target[:, [4]]
    return np.concatenate(
        [
            packet,
            np.abs(packet),
            packet * centered,
            packet * probability,
            packet * margin,
            packet * np.square(centered),
        ],
        axis=1,
    )


def _fit_ridge_no_intercept_scalar_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_indices: np.ndarray,
    ridge: float,
) -> RidgeNoInterceptScalarMap:
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
    x_scale = x[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    scaled = x / x_scale
    x_fit = scaled[fit]
    y_fit = y[fit]
    xtx = x_fit.T @ x_fit
    if ridge > 0.0:
        xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x_fit.T @ y_fit
    weights = np.linalg.solve(xtx, xty)
    pred_fit = x_fit @ weights
    mse = float(np.mean(np.square(y_fit - pred_fit)))
    baseline = float(np.mean(np.square(y_fit)))
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - mse / baseline
    return RidgeNoInterceptScalarMap(
        x_scale=x_scale,
        weights=weights,
        ridge=float(ridge),
        fit_mse=mse,
        fit_r2=float(r2),
    )


def _fit_weighted_ridge_scalar_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    sample_weights: np.ndarray,
    ridge: float,
) -> behavior_gate.RidgeScalarMap:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("features must be rank-2 and targets rank-1")
    if x.shape[0] != y.shape[0] or x.shape[0] != weights.shape[0]:
        raise ValueError("feature/target/weight row mismatch")
    if x.shape[0] == 0:
        raise ValueError("features must not be empty")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    weights = np.maximum(weights, 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 1e-12:
        raise ValueError("sample weights must have positive total mass")
    norm_weights = weights / total_weight
    weights = weights * (float(weights.size) / total_weight)
    total_weight = float(weights.sum())
    x_mean = (norm_weights.reshape(-1, 1) * x).sum(axis=0, keepdims=True)
    x_var = (norm_weights.reshape(-1, 1) * np.square(x - x_mean)).sum(axis=0, keepdims=True)
    x_std = np.sqrt(x_var).clip(min=1e-6)
    y_mean = float(np.dot(norm_weights, y))
    x_scaled = (x - x_mean) / x_std
    y_centered = y - y_mean
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    x_weighted = x_scaled * sqrt_w
    y_weighted = y_centered * sqrt_w.reshape(-1)
    xtx = x_weighted.T @ x_weighted
    if ridge > 0.0:
        xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x_weighted.T @ y_weighted
    solved_weights = np.linalg.solve(xtx, xty)
    pred = x_scaled @ solved_weights + y_mean
    mse = float(np.sum(weights * np.square(y - pred)) / total_weight)
    baseline = float(np.sum(weights * np.square(y - y_mean)) / total_weight)
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - mse / baseline
    return behavior_gate.RidgeScalarMap(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        weights=solved_weights,
        ridge=float(ridge),
        fit_mse=mse,
        fit_r2=float(r2),
    )


def _fit_weighted_ridge_no_intercept_scalar_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    sample_weights: np.ndarray,
    ridge: float,
) -> RidgeNoInterceptScalarMap:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("features must be rank-2 and targets rank-1")
    if x.shape[0] != y.shape[0] or x.shape[0] != weights.shape[0]:
        raise ValueError("feature/target/weight row mismatch")
    if x.shape[0] == 0:
        raise ValueError("features must not be empty")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    weights = np.maximum(weights, 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 1e-12:
        raise ValueError("sample weights must have positive total mass")
    norm_weights = weights / total_weight
    weights = weights * (float(weights.size) / total_weight)
    total_weight = float(weights.sum())
    x_var = (norm_weights.reshape(-1, 1) * np.square(x)).sum(axis=0, keepdims=True)
    x_scale = np.sqrt(x_var).clip(min=1e-6)
    scaled = x / x_scale
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    x_weighted = scaled * sqrt_w
    y_weighted = y * sqrt_w.reshape(-1)
    xtx = x_weighted.T @ x_weighted
    if ridge > 0.0:
        xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x_weighted.T @ y_weighted
    solved_weights = np.linalg.solve(xtx, xty)
    pred = scaled @ solved_weights
    mse = float(np.sum(weights * np.square(y - pred)) / total_weight)
    baseline = float(np.sum(weights * np.square(y)) / total_weight)
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - mse / baseline
    return RidgeNoInterceptScalarMap(
        x_scale=x_scale,
        weights=solved_weights,
        ridge=float(ridge),
        fit_mse=mse,
        fit_r2=float(r2),
    )


def _same_choice_shuffle_index(
    *,
    row_index: int,
    eval_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
    source_selected: Sequence[int],
) -> int:
    row = rows[row_index]
    same_source = [
        index
        for index in eval_indices
        if index != row_index
        and len(rows[index].choices) == len(row.choices)
        and int(source_selected[index]) == int(source_selected[row_index])
    ]
    if same_source:
        return same_source[0]
    same_shape = [index for index in eval_indices if index != row_index and len(rows[index].choices) == len(row.choices)]
    if same_shape:
        return same_shape[0]
    return row_index


def _same_shape_shuffle_index(
    *,
    row_index: int,
    eval_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
) -> int:
    row = rows[row_index]
    same_shape = [index for index in eval_indices if index != row_index and len(rows[index].choices) == len(row.choices)]
    if same_shape:
        return same_shape[0]
    other = [index for index in eval_indices if index != row_index]
    if other:
        return other[0]
    return row_index


def _decode_one_row(
    row: arc_gate.ArcRow,
    *,
    rows: Sequence[arc_gate.ArcRow],
    row_index: int,
    target_features: np.ndarray,
    packet: np.ndarray,
    decoder: Any,
    decoder_mode: str,
) -> np.ndarray:
    start, end = hidden_gate._row_offsets(rows)[row_index]
    return _decode_packet_residual_rows(
        [row],
        target_features=target_features[start:end],
        packet_features=np.asarray(packet, dtype=np.float64),
        decoder=decoder,
        decoder_mode=decoder_mode,
    )[0]


def _decode_packet_residual_rows(
    rows: Sequence[arc_gate.ArcRow],
    *,
    target_features: np.ndarray,
    packet_features: np.ndarray,
    decoder: Any,
    decoder_mode: str,
) -> list[np.ndarray]:
    if decoder_mode == "target_conditioned":
        return hidden_gate._decode_residual_rows(
            rows,
            target_features=target_features,
            packet_features=packet_features,
            decoder=decoder,
        )
    if decoder_mode == "packet_innovation":
        flat_pred = decoder.predict(_innovation_decoder_features(target_features, packet_features))
        return hidden_gate._rows_from_candidate_values(rows, flat_pred)
    raise ValueError(f"unknown decoder_mode: {decoder_mode}")


def _subtract_row_baseline(residuals: Sequence[np.ndarray], baselines: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [
        np.asarray(residual, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
        for residual, baseline in zip(residuals, baselines, strict=True)
    ]


def _safe_l2(values: np.ndarray) -> float:
    return float(np.sqrt(float(np.square(values).sum())))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(left.mean()) if left.size else left
    right_centered = right - float(right.mean()) if right.size else right
    denom = _safe_l2(left_centered) * _safe_l2(right_centered)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def _event_gate_features(
    target_scores: Sequence[float],
    residual: Sequence[float],
    packet: np.ndarray,
    *,
    residual_weight: float,
) -> np.ndarray:
    target = np.asarray(target_scores, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    packet_values = np.asarray(packet, dtype=np.float64)
    if target.ndim != 1 or correction.ndim != 1:
        raise ValueError("target_scores and residual must be rank-1")
    if target.shape != correction.shape:
        raise ValueError("target/residual shape mismatch")
    weighted = target + float(residual_weight) * correction
    target_pred = behavior_gate._prediction(target)
    fused_pred = behavior_gate._prediction(weighted)
    target_probs = behavior_gate._softmax(target)
    fused_probs = behavior_gate._softmax(weighted)
    flat_packet = packet_values.reshape(-1) if packet_values.size else np.zeros(1, dtype=np.float64)
    target_margin = hidden_gate._target_margin(target)
    fused_margin = hidden_gate._target_margin(weighted)
    correction_abs = np.abs(correction)
    packet_abs = np.abs(flat_packet)
    changed = 1.0 if fused_pred != target_pred else 0.0
    return np.asarray(
        [
            1.0,
            float(target.size),
            float(target_margin),
            float(behavior_gate._entropy(target)),
            float(target_probs[target_pred]) if target_probs.size else 0.0,
            float(fused_margin),
            float(behavior_gate._entropy(weighted)),
            float(fused_probs[fused_pred]) if fused_probs.size else 0.0,
            changed,
            float(correction_abs.sum()),
            _safe_l2(correction),
            float(correction_abs.max()) if correction_abs.size else 0.0,
            float(correction.std()) if correction.size else 0.0,
            float(correction[target_pred]) if correction.size else 0.0,
            float(correction[fused_pred]) if correction.size else 0.0,
            float(correction[fused_pred] - correction[target_pred]) if correction.size else 0.0,
            _safe_corr(target, correction),
            float(packet_abs.sum()),
            _safe_l2(flat_packet),
            float(packet_abs.max()) if packet_abs.size else 0.0,
            float(np.count_nonzero(flat_packet) / max(flat_packet.size, 1)),
            float(flat_packet.mean()) if flat_packet.size else 0.0,
            float(packet_abs.mean()) if packet_abs.size else 0.0,
        ],
        dtype=np.float64,
    )


def _event_triggered_fused_scores(
    target_scores: Sequence[float],
    residual: Sequence[float],
    packet: np.ndarray,
    *,
    rule: EventGateRule,
) -> tuple[list[float], bool, float]:
    target = np.asarray(target_scores, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    fused = target + float(rule.residual_weight) * correction
    event_features = _event_gate_features(
        target,
        correction,
        packet,
        residual_weight=float(rule.residual_weight),
    ).reshape(1, -1)
    event_score = float(rule.event_model.predict(event_features)[0])
    fires = event_score >= float(rule.threshold)
    if bool(rule.require_prediction_change):
        fires = fires and behavior_gate._prediction(fused) != behavior_gate._prediction(target)
    if not fires:
        return [float(score) for score in target], False, event_score
    return [float(score) for score in fused], True, event_score


def _fuse_with_gate(
    target_scores: Sequence[float],
    residual: Sequence[float],
    packet: np.ndarray,
    *,
    gate_rule: dict[str, Any] | EventGateRule,
    gate_mode: str,
) -> tuple[list[float], bool, float | None]:
    if gate_mode == "residual_threshold":
        fused, fired = hidden_gate._fused_scores(target_scores, residual, rule=gate_rule)  # type: ignore[arg-type]
        return fused, fired, None
    if gate_mode == "event_triggered":
        fused, fired, event_score = _event_triggered_fused_scores(
            target_scores,
            residual,
            packet,
            rule=gate_rule,  # type: ignore[arg-type]
        )
        return fused, fired, event_score
    raise ValueError(f"unknown gate_mode: {gate_mode}")


def _public_gate_rule(gate_rule: dict[str, Any] | EventGateRule) -> dict[str, Any]:
    if isinstance(gate_rule, EventGateRule):
        return dict(gate_rule.metadata)
    return dict(gate_rule)


def _decode_control_residual_for_row(
    row: arc_gate.ArcRow,
    *,
    rows: Sequence[arc_gate.ArcRow],
    row_index: int,
    target_features: np.ndarray,
    packet: np.ndarray,
    decoder: Any,
    decoder_mode: str,
    subtract_zero_packet_baseline: bool,
    zero_baseline_residuals: Sequence[np.ndarray],
) -> np.ndarray:
    residual = _decode_one_row(
        row,
        rows=rows,
        row_index=row_index,
        target_features=target_features,
        packet=packet,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    if subtract_zero_packet_baseline:
        residual = np.asarray(residual, dtype=np.float64) - np.asarray(
            zero_baseline_residuals[row_index],
            dtype=np.float64,
        )
    return np.asarray(residual, dtype=np.float64)


def _event_training_packets(
    *,
    row_index: int,
    train_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
    row_packets: Sequence[np.ndarray],
    target_derived_row_packets: Sequence[np.ndarray],
    source_selected: Sequence[int],
) -> dict[str, np.ndarray]:
    shuffled_index = _same_shape_shuffle_index(row_index=row_index, eval_indices=train_indices, rows=rows)
    same_choice_index = _same_choice_shuffle_index(
        row_index=row_index,
        eval_indices=train_indices,
        rows=rows,
        source_selected=source_selected,
    )
    return {
        "target_derived_packet": target_derived_row_packets[row_index],
        "zero_source": np.zeros_like(row_packets[row_index]),
        "source_row_shuffle": row_packets[shuffled_index],
        "same_source_choice_row_shuffle": row_packets[same_choice_index],
        "atom_shuffle": hidden_gate._atom_shuffle_packet(row_packets[row_index]),
        "coefficient_shuffle": hidden_gate._coefficient_shuffle_packet(row_packets[row_index]),
        "top_atom_knockout": hidden_gate._top_atom_knockout_packet(row_packets[row_index]),
        "candidate_roll": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=1),
        "candidate_derangement": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=-1),
    }


def _cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    lhs_norm = np.linalg.norm(lhs, axis=1, keepdims=True).clip(min=1e-12)
    rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True).clip(min=1e-12)
    return (lhs / lhs_norm) @ (rhs / rhs_norm).T


def _packet_atom_profile(packet: np.ndarray) -> np.ndarray:
    values = np.asarray(packet, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("packet must be rank-2")
    return np.mean(np.abs(values), axis=0)


def _packet_integrity_features(
    *,
    packet: np.ndarray,
    target_features: np.ndarray,
    rule: PacketIntegrityRule,
) -> np.ndarray:
    packet_values = np.asarray(packet, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    if packet_values.ndim != 2 or target.ndim != 2:
        raise ValueError("packet and target_features must be rank-2")
    if packet_values.shape[0] != target.shape[0]:
        raise ValueError("packet/target row mismatch")
    pred_target = rule.packet_to_target_map.predict(packet_values)
    sim = _cosine_similarity_matrix(pred_target, target)
    diag = np.diag(sim) if sim.size else np.zeros(0, dtype=np.float64)
    if sim.shape[0] > 1:
        offdiag = sim.copy()
        np.fill_diagonal(offdiag, -1e9)
        offdiag_max = offdiag.max(axis=1)
        alignment_rate = float(np.mean(np.argmax(sim, axis=1) == np.arange(sim.shape[0])))
    else:
        offdiag_max = np.full_like(diag, -1.0)
        alignment_rate = 1.0 if diag.size else 0.0
    diag_gap = diag - offdiag_max
    flat_packet = packet_values.reshape(-1)
    abs_packet = np.abs(flat_packet)
    energy = np.square(flat_packet)
    total_energy = float(energy.sum())
    top_energy_share = float(energy.max() / total_energy) if total_energy > 1e-12 else 0.0
    atom_profile = _packet_atom_profile(packet_values)
    atom_z = np.abs((atom_profile - rule.atom_profile) / rule.atom_profile_scale)
    pred_mse = float(np.mean(np.square(pred_target - target))) if target.size else 0.0
    return np.asarray(
        [
            1.0,
            float(packet_values.shape[0]),
            float(diag.mean()) if diag.size else 0.0,
            float(diag.min()) if diag.size else 0.0,
            float(diag_gap.mean()) if diag_gap.size else 0.0,
            float(diag_gap.min()) if diag_gap.size else 0.0,
            alignment_rate,
            pred_mse,
            float(abs_packet.sum()),
            _safe_l2(flat_packet),
            float(abs_packet.max()) if abs_packet.size else 0.0,
            float(np.count_nonzero(flat_packet) / max(flat_packet.size, 1)),
            top_energy_share,
            float(atom_z.mean()) if atom_z.size else 0.0,
            float(atom_z.max()) if atom_z.size else 0.0,
            float(np.count_nonzero(atom_profile) / max(atom_profile.size, 1)),
        ],
        dtype=np.float64,
    )


def _choose_packet_integrity_rule(
    *,
    rows: Sequence[arc_gate.ArcRow],
    train_indices: Sequence[int],
    target_features: np.ndarray,
    source_packet_flat: np.ndarray,
    target_derived_packet_flat: np.ndarray,
    fit_candidate_indices: np.ndarray,
    source_selected: Sequence[int],
    ridge: float,
) -> PacketIntegrityRule:
    train = [int(index) for index in train_indices]
    packet_to_target_map = hidden_gate._fit_ridge_matrix_map(
        source_packet_flat,
        target_features,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    row_packets = hidden_gate._row_packet_arrays(rows, source_packet_flat)
    target_derived_row_packets = hidden_gate._row_packet_arrays(rows, target_derived_packet_flat)
    offsets = hidden_gate._row_offsets(rows)
    train_profiles = [_packet_atom_profile(row_packets[index]) for index in train]
    if not train_profiles:
        raise ValueError("packet integrity needs at least one train row")
    profile_matrix = np.vstack(train_profiles)
    atom_profile = profile_matrix.mean(axis=0)
    atom_profile_scale = profile_matrix.std(axis=0).clip(min=1e-6)
    template_rule = PacketIntegrityRule(
        packet_to_target_map=packet_to_target_map,
        integrity_model=behavior_gate.RidgeScalarMap(
            x_mean=np.zeros((1, 1), dtype=np.float64),
            x_std=np.ones((1, 1), dtype=np.float64),
            y_mean=0.0,
            weights=np.zeros(1, dtype=np.float64),
            ridge=float(ridge),
            fit_mse=0.0,
            fit_r2=0.0,
        ),
        threshold=0.0,
        atom_profile=atom_profile,
        atom_profile_scale=atom_profile_scale,
        metadata={},
    )
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    condition_counts: dict[str, int] = {MATCHED_CONDITION: 0}
    for row_index in train:
        start, end = offsets[row_index]
        row_target_features = target_features[start:end]
        feature_rows.append(
            _packet_integrity_features(
                packet=row_packets[row_index],
                target_features=row_target_features,
                rule=template_rule,
            )
        )
        labels.append(1.0)
        condition_counts[MATCHED_CONDITION] += 1
        corruption_packets = _event_training_packets(
            row_index=row_index,
            train_indices=train,
            rows=rows,
            row_packets=row_packets,
            target_derived_row_packets=target_derived_row_packets,
            source_selected=source_selected,
        )
        for condition, packet in corruption_packets.items():
            feature_rows.append(
                _packet_integrity_features(
                    packet=packet,
                    target_features=row_target_features,
                    rule=template_rule,
                )
            )
            labels.append(0.0)
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
    features = np.vstack(feature_rows)
    label_values = np.asarray(labels, dtype=np.float64)
    model = behavior_gate._fit_ridge_scalar_map(
        features,
        label_values,
        fit_indices=np.arange(features.shape[0], dtype=np.int64),
        ridge=ridge,
    )
    scores = np.asarray(model.predict(features), dtype=np.float64)
    thresholds = sorted(
        {
            float(scores.min() - 1e-6),
            float(scores.max() + 1e-6),
            *[float(value) for value in np.percentile(scores, [50, 60, 70, 80, 90])],
            *[float(value) for value in scores],
        }
    )
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    is_matched = label_values > 0.5
    for threshold in thresholds:
        accepted = scores >= float(threshold)
        matched_accept = int(np.sum(accepted & is_matched))
        corrupt_accept = int(np.sum(accepted & ~is_matched))
        matched_total = int(np.sum(is_matched))
        corrupt_total = int(np.sum(~is_matched))
        matched_rate = matched_accept / max(matched_total, 1)
        corrupt_rate = corrupt_accept / max(corrupt_total, 1)
        key = (
            matched_rate - corrupt_rate,
            -corrupt_accept,
            matched_accept,
            float(threshold),
        )
        if best is None or best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "train_matched_accept": int(matched_accept),
                "train_corrupt_accept": int(corrupt_accept),
                "train_matched_accept_rate": float(matched_rate),
                "train_corrupt_accept_rate": float(corrupt_rate),
            }
    if best is None:
        raise ValueError("could not select packet integrity threshold")
    metadata = {
        "packet_integrity_mode": "candidate_atom",
        "ridge": float(ridge),
        "threshold": float(best["threshold"]),
        "train_examples": int(features.shape[0]),
        "train_matched_examples": int(np.sum(is_matched)),
        "train_corruption_examples": int(np.sum(~is_matched)),
        "train_matched_accept": int(best["train_matched_accept"]),
        "train_corrupt_accept": int(best["train_corrupt_accept"]),
        "train_matched_accept_rate": float(best["train_matched_accept_rate"]),
        "train_corrupt_accept_rate": float(best["train_corrupt_accept_rate"]),
        "condition_counts": {condition: int(count) for condition, count in sorted(condition_counts.items())},
        "integrity_model_fit_mse": float(model.fit_mse),
        "integrity_model_fit_r2": float(model.fit_r2),
        "packet_to_target_fit_mse": float(packet_to_target_map.fit_mse),
        "packet_to_target_fit_r2": float(packet_to_target_map.fit_r2),
        "feature_dim": int(features.shape[1]),
    }
    return PacketIntegrityRule(
        packet_to_target_map=packet_to_target_map,
        integrity_model=model,
        threshold=float(best["threshold"]),
        atom_profile=atom_profile,
        atom_profile_scale=atom_profile_scale,
        metadata=metadata,
    )


def _packet_integrity_accept(
    *,
    packet: np.ndarray,
    target_features: np.ndarray,
    rule: PacketIntegrityRule | None,
) -> tuple[bool, float | None]:
    if rule is None:
        return True, None
    features = _packet_integrity_features(packet=packet, target_features=target_features, rule=rule).reshape(1, -1)
    score = float(rule.integrity_model.predict(features)[0])
    return score >= float(rule.threshold) - 1e-12, score


def _decoder_feature_rows(
    target_features: np.ndarray,
    packet_features: np.ndarray,
    *,
    decoder_mode: str,
) -> np.ndarray:
    if decoder_mode == "target_conditioned":
        return hidden_gate._decoder_features(target_features, packet_features)
    if decoder_mode == "packet_innovation":
        return _innovation_decoder_features(target_features, packet_features)
    raise ValueError(f"unknown decoder_mode: {decoder_mode}")


def _fit_decoder_from_features(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    decoder_mode: str,
    ridge: float,
) -> Any:
    fit_indices = np.arange(features.shape[0], dtype=np.int64)
    if sample_weights is not None:
        if decoder_mode == "target_conditioned":
            return _fit_weighted_ridge_scalar_map(
                features,
                targets,
                sample_weights=sample_weights,
                ridge=ridge,
            )
        if decoder_mode == "packet_innovation":
            return _fit_weighted_ridge_no_intercept_scalar_map(
                features,
                targets,
                sample_weights=sample_weights,
                ridge=ridge,
            )
        raise ValueError(f"unknown decoder_mode: {decoder_mode}")
    if decoder_mode == "target_conditioned":
        return behavior_gate._fit_ridge_scalar_map(
            features,
            targets,
            fit_indices=fit_indices,
            ridge=ridge,
        )
    if decoder_mode == "packet_innovation":
        return _fit_ridge_no_intercept_scalar_map(
            features,
            targets,
            fit_indices=fit_indices,
            ridge=ridge,
        )
    raise ValueError(f"unknown decoder_mode: {decoder_mode}")


def _fit_matched_only_decoder(
    *,
    target_features: np.ndarray,
    source_packet_flat: np.ndarray,
    behavior_targets: np.ndarray,
    fit_candidate_indices: np.ndarray,
    decoder_mode: str,
    ridge: float,
) -> tuple[Any, dict[str, Any]]:
    features = _decoder_feature_rows(
        target_features,
        source_packet_flat,
        decoder_mode=decoder_mode,
    )
    if decoder_mode == "target_conditioned":
        decoder = behavior_gate._fit_ridge_scalar_map(
            features,
            behavior_targets,
            fit_indices=fit_candidate_indices,
            ridge=ridge,
        )
    elif decoder_mode == "packet_innovation":
        decoder = _fit_ridge_no_intercept_scalar_map(
            features,
            behavior_targets,
            fit_indices=fit_candidate_indices,
            ridge=ridge,
        )
    else:
        raise ValueError(f"unknown decoder_mode: {decoder_mode}")
    return decoder, {
        "receiver_training_mode": "matched_only",
        "matched_training_examples": int(fit_candidate_indices.size),
        "corruption_training_examples": 0,
        "corruption_conditions": [],
    }


def _fit_corruption_noop_decoder(
    *,
    rows: Sequence[arc_gate.ArcRow],
    train_indices: Sequence[int],
    target_features: np.ndarray,
    source_packet_flat: np.ndarray,
    target_derived_packet_flat: np.ndarray,
    behavior_targets: np.ndarray,
    source_selected: Sequence[int],
    decoder_mode: str,
    corruption_loss_weight: float,
    corruption_condition_weights: dict[str, float] | None = None,
    ridge: float,
) -> tuple[Any, dict[str, Any]]:
    if corruption_loss_weight < 0.0:
        raise ValueError("corruption_loss_weight must be non-negative")
    condition_weights = dict(corruption_condition_weights or {})
    for condition, weight in condition_weights.items():
        if condition not in EVENT_GATE_CORRUPTION_CONDITIONS:
            raise ValueError(f"unknown corruption condition weight: {condition}")
        if weight < 0.0:
            raise ValueError(f"corruption condition weight must be non-negative: {condition}")
    train = [int(index) for index in train_indices]
    row_packets = hidden_gate._row_packet_arrays(rows, source_packet_flat)
    target_derived_row_packets = hidden_gate._row_packet_arrays(rows, target_derived_packet_flat)
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    matched_examples = 0
    corruption_examples = 0
    corruption_counts = {condition: 0 for condition in EVENT_GATE_CORRUPTION_CONDITIONS}

    offsets = hidden_gate._row_offsets(rows)
    for row_index in train:
        start, end = offsets[row_index]
        row_target_features = target_features[start:end]
        matched_packet = row_packets[row_index]
        matched_target = behavior_targets[start:end]
        feature_blocks.append(
            _decoder_feature_rows(
                row_target_features,
                matched_packet,
                decoder_mode=decoder_mode,
            )
        )
        target_blocks.append(np.asarray(matched_target, dtype=np.float64))
        weight_blocks.append(np.ones(end - start, dtype=np.float64))
        matched_examples += int(end - start)

        corruption_packets = _event_training_packets(
            row_index=row_index,
            train_indices=train,
            rows=rows,
            row_packets=row_packets,
            target_derived_row_packets=target_derived_row_packets,
            source_selected=source_selected,
        )
        for condition, packet in corruption_packets.items():
            condition_weight = float(condition_weights.get(condition, corruption_loss_weight))
            feature_blocks.append(
                _decoder_feature_rows(
                    row_target_features,
                    packet,
                    decoder_mode=decoder_mode,
                )
            )
            target_blocks.append(np.zeros(end - start, dtype=np.float64))
            weight_blocks.append(np.full(end - start, condition_weight, dtype=np.float64))
            corruption_examples += int(end - start)
            corruption_counts[condition] = corruption_counts.get(condition, 0) + int(end - start)

    if not feature_blocks:
        raise ValueError("corruption-noop decoder has no training examples")
    features = np.vstack(feature_blocks)
    targets = np.concatenate(target_blocks, axis=0)
    sample_weights = np.concatenate(weight_blocks, axis=0)
    decoder = _fit_decoder_from_features(
        features,
        targets,
        sample_weights=sample_weights,
        decoder_mode=decoder_mode,
        ridge=ridge,
    )
    return decoder, {
        "receiver_training_mode": "corruption_noop",
        "matched_training_examples": int(matched_examples),
        "corruption_training_examples": int(corruption_examples),
        "total_training_examples": int(features.shape[0]),
        "matched_training_weight": float(matched_examples),
        "corruption_loss_weight": float(corruption_loss_weight),
        "corruption_condition_weights": {
            condition: float(condition_weights.get(condition, corruption_loss_weight))
            for condition in EVENT_GATE_CORRUPTION_CONDITIONS
        },
        "corruption_training_weight": float(
            sum(
                corruption_counts[condition] * float(condition_weights.get(condition, corruption_loss_weight))
                for condition in EVENT_GATE_CORRUPTION_CONDITIONS
            )
        ),
        "corruption_conditions": list(EVENT_GATE_CORRUPTION_CONDITIONS),
        "corruption_condition_example_counts": {
            condition: int(count) for condition, count in sorted(corruption_counts.items())
        },
    }


def _choose_event_gate_rule(
    *,
    rows: Sequence[arc_gate.ArcRow],
    train_indices: Sequence[int],
    target_scores: Sequence[Sequence[float]],
    source_residuals: Sequence[np.ndarray],
    target_derived_residuals: Sequence[np.ndarray],
    zero_residuals: Sequence[np.ndarray],
    row_packets: Sequence[np.ndarray],
    target_derived_row_packets: Sequence[np.ndarray],
    target_features: np.ndarray,
    decoder: Any,
    decoder_mode: str,
    subtract_zero_packet_baseline: bool,
    zero_baseline_residuals: Sequence[np.ndarray],
    source_selected: Sequence[int],
    ridge: float,
) -> EventGateRule:
    train = [int(index) for index in train_indices]
    examples: list[dict[str, Any]] = []
    for row_index in train:
        row = rows[row_index]
        target = [float(score) for score in target_scores[row_index]]
        target_pred = behavior_gate._prediction(target)
        target_correct = target_pred == int(row.answer_index)
        examples.append(
            {
                "condition": MATCHED_CONDITION,
                "row_index": row_index,
                "target": target,
                "packet": row_packets[row_index],
                "residual": np.asarray(source_residuals[row_index], dtype=np.float64),
                "target_correct": target_correct,
                "is_matched": True,
            }
        )
        corruption_packets = _event_training_packets(
            row_index=row_index,
            train_indices=train,
            rows=rows,
            row_packets=row_packets,
            target_derived_row_packets=target_derived_row_packets,
            source_selected=source_selected,
        )
        for condition, packet in corruption_packets.items():
            if condition == "target_derived_packet":
                residual = np.asarray(target_derived_residuals[row_index], dtype=np.float64)
            elif condition == "zero_source":
                residual = np.asarray(zero_residuals[row_index], dtype=np.float64)
            else:
                residual = _decode_control_residual_for_row(
                    row,
                    rows=rows,
                    row_index=row_index,
                    target_features=target_features,
                    packet=packet,
                    decoder=decoder,
                    decoder_mode=decoder_mode,
                    subtract_zero_packet_baseline=subtract_zero_packet_baseline,
                    zero_baseline_residuals=zero_baseline_residuals,
                )
            examples.append(
                {
                    "condition": condition,
                    "row_index": row_index,
                    "target": target,
                    "packet": packet,
                    "residual": np.asarray(residual, dtype=np.float64),
                    "target_correct": target_correct,
                    "is_matched": False,
                }
            )

    best_rule: EventGateRule | None = None
    best_key: tuple[Any, ...] | None = None
    for residual_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        feature_rows: list[np.ndarray] = []
        labels: list[float] = []
        for example in examples:
            feature_rows.append(
                _event_gate_features(
                    example["target"],
                    example["residual"],
                    example["packet"],
                    residual_weight=float(residual_weight),
                )
            )
            fused = np.asarray(example["target"], dtype=np.float64) + float(residual_weight) * np.asarray(
                example["residual"],
                dtype=np.float64,
            )
            fused_correct = behavior_gate._prediction(fused) == int(rows[int(example["row_index"])].answer_index)
            label = bool(example["is_matched"] and fused_correct and not bool(example["target_correct"]))
            labels.append(1.0 if label else 0.0)
        features = np.vstack(feature_rows)
        label_values = np.asarray(labels, dtype=np.float64)
        model = behavior_gate._fit_ridge_scalar_map(
            features,
            label_values,
            fit_indices=np.arange(features.shape[0], dtype=np.int64),
            ridge=float(ridge),
        )
        event_scores = np.asarray(model.predict(features), dtype=np.float64)
        thresholds = sorted(
            {
                float(event_scores.min() - 1e-6),
                float(event_scores.max() + 1e-6),
                *[float(value) for value in np.percentile(event_scores, [50, 60, 70, 80, 90])],
                *[float(value) for value in event_scores],
            }
        )
        for threshold in thresholds:
            correct = 0
            fired = 0
            helped = 0
            harmed = 0
            control_fired = 0
            margins: list[float] = []
            for example, event_score in zip(examples, event_scores, strict=True):
                row = rows[int(example["row_index"])]
                target = np.asarray(example["target"], dtype=np.float64)
                residual = np.asarray(example["residual"], dtype=np.float64)
                fused = target + float(residual_weight) * residual
                pred_changed = behavior_gate._prediction(fused) != behavior_gate._prediction(target)
                did_fire = float(event_score) >= float(threshold) and pred_changed
                if not bool(example["is_matched"]):
                    control_fired += int(did_fire)
                    continue
                selected_scores = fused if did_fire else target
                target_correct = bool(example["target_correct"])
                is_correct = behavior_gate._prediction(selected_scores) == int(row.answer_index)
                correct += int(is_correct)
                fired += int(did_fire)
                helped += int(did_fire and is_correct and not target_correct)
                harmed += int(did_fire and (not is_correct) and target_correct)
                margins.append(behavior_gate._margin(selected_scores, int(row.answer_index)))
            accuracy = correct / max(len(train), 1)
            fired_rate = fired / max(len(train), 1)
            control_count = max(len(examples) - len(train), 1)
            control_fired_rate = control_fired / control_count
            metadata = {
                "gate_mode": "event_triggered",
                "residual_weight": float(residual_weight),
                "threshold": float(threshold),
                "require_prediction_change": True,
                "train_accuracy": float(accuracy),
                "train_fired": int(fired),
                "train_fired_rate": float(fired_rate),
                "train_helped": int(helped),
                "train_harmed": int(harmed),
                "train_net_help": int(helped - harmed),
                "train_mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
                "train_event_examples": int(len(examples)),
                "train_positive_events": int(label_values.sum()),
                "train_corruption_examples": int(len(examples) - len(train)),
                "train_corruption_fired": int(control_fired),
                "train_corruption_fired_rate": float(control_fired_rate),
                "event_model_fit_mse": float(model.fit_mse),
                "event_model_fit_r2": float(model.fit_r2),
                "event_model_weight_l2": _safe_l2(np.asarray(model.weights, dtype=np.float64)),
                "event_feature_dim": int(features.shape[1]),
                "corruption_conditions": list(EVENT_GATE_CORRUPTION_CONDITIONS),
            }
            key = (
                metadata["train_accuracy"],
                metadata["train_net_help"],
                -metadata["train_harmed"],
                metadata["train_helped"],
                -metadata["train_corruption_fired"],
                -abs(metadata["train_fired_rate"] - 0.35),
                metadata["train_mean_margin"],
                -metadata["residual_weight"],
            )
            if best_rule is None or best_key is None or key > best_key:
                best_key = key
                best_rule = EventGateRule(
                    residual_weight=float(residual_weight),
                    threshold=float(threshold),
                    event_model=model,
                    require_prediction_change=True,
                    metadata=metadata,
                )
    if best_rule is None:
        raise ValueError("could not select event-triggered gate rule")
    return best_rule


def _condition_metrics(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for condition in REPORT_CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        correct = sum(1 for row in subset if row["correct"])
        fired = sum(1 for row in subset if row.get("packet_fired"))
        helped = sum(1 for row in subset if row.get("packet_helped"))
        harmed = sum(1 for row in subset if row.get("packet_harmed"))
        metrics[condition] = {
            "n": len(subset),
            "correct": int(correct),
            "accuracy": float(correct / len(subset)) if subset else 0.0,
            "mean_margin": float(statistics.fmean(float(row["margin"]) for row in subset)) if subset else 0.0,
            "packet_fired": int(fired),
            "packet_fired_rate": float(fired / len(subset)) if subset else 0.0,
            "packet_helped_vs_target": int(helped),
            "packet_harmed_vs_target": int(harmed),
            "packet_net_help_vs_target": int(helped - harmed),
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
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{control}"] = behavior_gate._paired_bootstrap(
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
        "# ARC-Challenge Behavior-Atom Decoder Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/test disagreement rows: `{payload['train_rows']}` / `{payload['test_rows']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.6f}`",
        f"- target-only accuracy: `{headline['target_only_accuracy']:.6f}`",
        f"- best required control: `{headline['best_required_control']}`",
        f"- best required control accuracy: `{headline['best_required_control_accuracy']:.6f}`",
        f"- worst required paired CI95 low: `{headline['worst_required_ci95_low']:.6f}`",
        f"- fired rows: `{headline['matched_packet_fired']}`",
        f"- helps/harms vs target: `{headline['matched_packet_helped']}` / `{headline['matched_packet_harmed']}`",
        f"- packet bytes/row: `{payload['systems_packet_sideband']['packet_bytes_per_row']:.3f}`",
        f"- same-byte visible-text budget: `{payload['systems_packet_sideband']['same_byte_visible_text_budget']}`",
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
    if "no_op_residual_diagnostics" in payload:
        lines.extend(
            [
                "",
                "## No-Op Residual Diagnostics",
                "",
                "| Condition | Mean L2 | Mean Ratio | P95 Ratio | Flips vs Target |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in payload["no_op_residual_diagnostics"].items():
            lines.append(
                f"| `{name}` | {row['mean_residual_l2']:.6f} | "
                f"{row.get('mean_residual_l2_ratio_vs_matched', 1.0):.6f} | "
                f"{row.get('p95_residual_l2_ratio_vs_matched', 1.0):.6f} | "
                f"{int(row.get('prediction_flips_vs_target_only', 0))} |"
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
    target_hidden_layer: int,
    source_feature_dim: int,
    ridge: float,
    packet_rank: int,
    packet_top_k: int,
    packet_bits: int,
    atom_basis_mode: str,
    batchtopk_epochs: int,
    batchtopk_learning_rate: float,
    batchtopk_batch_size: int,
    batchtopk_reconstruction_weight: float,
    batchtopk_l1_weight: float,
    paired_alignment_weight: float,
    paired_target_behavior_weight: float,
    local_files_only: bool,
    bootstrap_samples: int,
    same_byte_budget: int,
    subtract_zero_packet_baseline: bool,
    decoder_mode: str,
    gate_mode: str,
    receiver_training_mode: str,
    corruption_loss_weight: float,
    corruption_condition_weights: dict[str, float] | None,
    packet_integrity_mode: str,
    min_accuracy_gap: float,
    min_ci_low: float,
    seed: int,
) -> dict[str, Any]:
    if decoder_mode not in {"target_conditioned", "packet_innovation"}:
        raise ValueError(f"unknown decoder_mode: {decoder_mode}")
    if gate_mode not in {"residual_threshold", "event_triggered"}:
        raise ValueError(f"unknown gate_mode: {gate_mode}")
    if receiver_training_mode not in RECEIVER_TRAINING_MODES:
        raise ValueError(f"unknown receiver_training_mode: {receiver_training_mode}")
    if packet_integrity_mode not in PACKET_INTEGRITY_MODES:
        raise ValueError(f"unknown packet_integrity_mode: {packet_integrity_mode}")
    if atom_basis_mode not in ATOM_BASIS_MODES:
        raise ValueError(f"unknown atom_basis_mode: {atom_basis_mode}")
    output_dir = behavior_gate._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "strict_inputs"
    agreement_path = behavior_gate._resolve(source_family_gate_dir) / "source_cache_agreement.csv"

    validation_rows_all = arc_gate._load_rows(behavior_gate._resolve(validation_path))
    test_rows_all = arc_gate._load_rows(behavior_gate._resolve(test_path))
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
    eval_indices = list(range(fit_row_count, fit_row_count + len(test_rows)))
    fit_candidate_indices = preflight._flat_candidate_indices_for_rows(rows, list(range(fit_row_count)))

    behavior_gate._write_jsonl(
        input_dir / "arc_challenge_validation_train_plus_test_disagreement.jsonl",
        [soft_gate._arc_row_payload(row) for row in rows],
    )
    behavior_gate._write_jsonl(
        input_dir / "tinyllama_source_score_cache.jsonl",
        [
            *soft_gate._select_cache_rows(cache_path=tiny_validation_score_cache, rows=train_rows),
            *soft_gate._select_cache_rows(cache_path=tiny_test_score_cache, rows=test_rows),
        ],
    )
    behavior_gate._write_jsonl(
        input_dir / "qwen_source_score_cache.jsonl",
        [
            *soft_gate._select_cache_rows(cache_path=qwen_validation_score_cache, rows=train_rows),
            *soft_gate._select_cache_rows(cache_path=qwen_test_score_cache, rows=test_rows),
        ],
    )
    tiny_cache = behavior_gate._load_score_rows(input_dir / "tinyllama_source_score_cache.jsonl")
    qwen_cache = behavior_gate._load_score_rows(input_dir / "qwen_source_score_cache.jsonl")

    target_scores, target_score_meta = behavior_gate._score_rows_with_prompt_builder(
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

    source_packet_flat, source_packet_meta = _fit_source_packet(
        rows,
        target_scores=target_scores,
        source_model=source_model,
        target_model=target_model,
        source_device=source_device,
        target_device=target_device,
        dtype=dtype,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
        source_hidden_layer=source_hidden_layer,
        target_hidden_layer=target_hidden_layer,
        source_feature_dim=source_feature_dim,
        fit_candidate_indices=fit_candidate_indices,
        fit_row_count=fit_row_count,
        ridge=ridge,
        packet_rank=packet_rank,
        packet_top_k=packet_top_k,
        packet_bits=packet_bits,
        atom_basis_mode=atom_basis_mode,
        batchtopk_epochs=batchtopk_epochs,
        batchtopk_learning_rate=batchtopk_learning_rate,
        batchtopk_batch_size=batchtopk_batch_size,
        batchtopk_reconstruction_weight=batchtopk_reconstruction_weight,
        batchtopk_l1_weight=batchtopk_l1_weight,
        paired_alignment_weight=paired_alignment_weight,
        paired_target_behavior_weight=paired_target_behavior_weight,
        seed=seed,
        local_files_only=local_files_only,
    )
    sparse_meta = source_packet_meta["sparse_packet"]
    estimated_packet_bytes_per_row = float(sparse_meta["packet_bytes_per_candidate"] * max(len(row.choices) for row in rows))
    framed_packet_bytes = int(math.ceil(estimated_packet_bytes_per_row))
    same_byte_budget_used = int(same_byte_budget) if same_byte_budget > 0 else int(max(framed_packet_bytes, 1))

    def same_byte_prompt(row: arc_gate.ArcRow) -> str:
        selected = behavior_gate._source_prediction_for_row(row, tiny_cache)
        hint = row.choices[selected].encode("utf-8")[:same_byte_budget_used].decode("utf-8", errors="ignore")
        choices = "\n".join(f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True))
        return (
            "Answer the science question with the best answer.\n"
            f"Question: {row.question}\n"
            f"Choices:\n{choices}\n"
            f"Source model selected this visible hint: {hint}\n"
            "Answer:"
        )

    same_byte_scores, same_byte_meta = behavior_gate._score_rows_with_prompt_builder(
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

    target_features = hidden_gate._target_score_features(rows, target_scores)
    behavior_targets = behavior_gate._candidate_targets(rows, target_scores)
    target_only_decoder = behavior_gate._fit_ridge_scalar_map(
        target_features,
        behavior_targets,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_packet_map = hidden_gate._fit_ridge_matrix_map(
        target_features,
        source_packet_flat,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_derived_packet_flat = target_packet_map.predict(target_features)
    source_selected_all = [behavior_gate._source_prediction_for_row(row, tiny_cache) for row in rows]
    if receiver_training_mode == "matched_only":
        decoder, receiver_training_meta = _fit_matched_only_decoder(
            target_features=target_features,
            source_packet_flat=source_packet_flat,
            behavior_targets=behavior_targets,
            fit_candidate_indices=fit_candidate_indices,
            decoder_mode=decoder_mode,
            ridge=ridge,
        )
    elif receiver_training_mode == "corruption_noop":
        decoder, receiver_training_meta = _fit_corruption_noop_decoder(
            rows=rows,
            train_indices=list(range(fit_row_count)),
            target_features=target_features,
            source_packet_flat=source_packet_flat,
            target_derived_packet_flat=target_derived_packet_flat,
            behavior_targets=behavior_targets,
            source_selected=source_selected_all,
            decoder_mode=decoder_mode,
            corruption_loss_weight=corruption_loss_weight,
            corruption_condition_weights=corruption_condition_weights,
            ridge=ridge,
        )
    else:
        raise ValueError(f"unknown receiver_training_mode: {receiver_training_mode}")

    source_residuals = _decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=source_packet_flat,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    target_decoder_residuals = hidden_gate._rows_from_candidate_values(rows, target_only_decoder.predict(target_features))
    target_derived_residuals = _decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=target_derived_packet_flat,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    zero_packet_flat = np.zeros_like(source_packet_flat)
    zero_residuals = _decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=zero_packet_flat,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    zero_baseline_residuals = [np.asarray(residual, dtype=np.float64) for residual in zero_residuals]
    if subtract_zero_packet_baseline:
        source_residuals = _subtract_row_baseline(source_residuals, zero_baseline_residuals)
        target_derived_residuals = _subtract_row_baseline(target_derived_residuals, zero_baseline_residuals)
        zero_residuals = [np.zeros_like(residual, dtype=np.float64) for residual in zero_residuals]
    row_packets = hidden_gate._row_packet_arrays(rows, source_packet_flat)
    target_derived_row_packets = hidden_gate._row_packet_arrays(rows, target_derived_packet_flat)
    if gate_mode == "event_triggered":
        gate_rule: dict[str, Any] | EventGateRule = _choose_event_gate_rule(
            rows=rows,
            train_indices=list(range(fit_row_count)),
            target_scores=target_scores,
            source_residuals=source_residuals,
            target_derived_residuals=target_derived_residuals,
            zero_residuals=zero_residuals,
            row_packets=row_packets,
            target_derived_row_packets=target_derived_row_packets,
            target_features=target_features,
            decoder=decoder,
            decoder_mode=decoder_mode,
            subtract_zero_packet_baseline=subtract_zero_packet_baseline,
            zero_baseline_residuals=zero_baseline_residuals,
            source_selected=source_selected_all,
            ridge=ridge,
        )
    else:
        gate_rule = hidden_gate._choose_gate_rule(
            train_rows,
            target_scores[:fit_row_count],
            source_residuals[:fit_row_count],
        )
    packet_integrity_rule: PacketIntegrityRule | None = None
    if packet_integrity_mode == "candidate_atom":
        packet_integrity_rule = _choose_packet_integrity_rule(
            rows=rows,
            train_indices=list(range(fit_row_count)),
            target_features=target_features,
            source_packet_flat=source_packet_flat,
            target_derived_packet_flat=target_derived_packet_flat,
            fit_candidate_indices=fit_candidate_indices,
            source_selected=source_selected_all,
            ridge=ridge,
        )
    prediction_rows: list[dict[str, Any]] = []
    for eval_position, row in enumerate(test_rows):
        row_index = fit_row_count + eval_position
        target = [float(score) for score in target_scores[row_index]]
        shuffled_index = _same_shape_shuffle_index(
            row_index=row_index,
            eval_indices=eval_indices,
            rows=rows,
        )
        same_choice_index = _same_choice_shuffle_index(
            row_index=row_index,
            eval_indices=eval_indices,
            rows=rows,
            source_selected=source_selected_all,
        )
        raw_source_scores = behavior_gate._source_scores_for_row(row, tiny_cache)
        source_selected = int(source_selected_all[row_index])
        qwen_selected = behavior_gate._source_prediction_for_row(row, qwen_cache)
        candidate_packets = {
            MATCHED_CONDITION: row_packets[row_index],
            "target_derived_packet": target_derived_row_packets[row_index],
            "zero_source": np.zeros_like(row_packets[row_index]),
            "source_row_shuffle": row_packets[shuffled_index],
            "same_source_choice_row_shuffle": row_packets[same_choice_index],
            "atom_shuffle": hidden_gate._atom_shuffle_packet(row_packets[row_index]),
            "coefficient_shuffle": hidden_gate._coefficient_shuffle_packet(row_packets[row_index]),
            "top_atom_knockout": hidden_gate._top_atom_knockout_packet(row_packets[row_index]),
            "candidate_roll": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=1),
            "candidate_derangement": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=-1),
        }
        condition_scores: dict[str, tuple[list[float], bool, np.ndarray, float | None]] = {}
        condition_integrity: dict[str, tuple[bool, float | None]] = {}
        for condition, packet in candidate_packets.items():
            start, end = hidden_gate._row_offsets(rows)[row_index]
            integrity_accept, integrity_score = _packet_integrity_accept(
                packet=packet,
                target_features=target_features[start:end],
                rule=packet_integrity_rule,
            )
            if condition == MATCHED_CONDITION:
                residual = source_residuals[row_index]
            elif condition == "target_derived_packet":
                residual = target_derived_residuals[row_index]
            elif condition == "zero_source":
                residual = zero_residuals[row_index]
            else:
                residual = _decode_one_row(
                    row,
                    rows=rows,
                    row_index=row_index,
                    target_features=target_features,
                    packet=packet,
                    decoder=decoder,
                    decoder_mode=decoder_mode,
                )
                if subtract_zero_packet_baseline:
                    residual = np.asarray(residual, dtype=np.float64) - np.asarray(
                        zero_baseline_residuals[row_index],
                        dtype=np.float64,
                    )
            if not integrity_accept:
                condition_scores[condition] = (
                    target,
                    False,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                )
                condition_integrity[condition] = (False, integrity_score)
                continue
            fused, fired, event_score = _fuse_with_gate(
                target,
                residual,
                packet,
                gate_rule=gate_rule,
                gate_mode=gate_mode,
            )
            condition_scores[condition] = (fused, fired, np.asarray(residual, dtype=np.float64), event_score)
            condition_integrity[condition] = (True, integrity_score)
        target_decoder_scores, target_decoder_fired, target_decoder_event_score = _fuse_with_gate(
            target,
            target_decoder_residuals[row_index],
            np.zeros_like(row_packets[row_index]),
            gate_rule=gate_rule,
            gate_mode=gate_mode,
        )
        condition_scores.update(
            {
                "target_only": (target, False, np.zeros(len(row.choices), dtype=np.float64), None),
                "target_decoder_only": (
                    target_decoder_scores,
                    target_decoder_fired,
                    np.asarray(target_decoder_residuals[row_index], dtype=np.float64),
                    target_decoder_event_score,
                ),
                "packet_only_source_index": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "source_rank_control": (
                    behavior_gate._source_rank_scores(raw_source_scores),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "source_score_control": (
                    behavior_gate._centered_source_score_control(raw_source_scores),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "source_score_quantized_control": (
                    ecoc_gate._source_score_quantized_control(raw_source_scores, bits=4),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "same_byte_visible_text": (
                    same_byte_scores[eval_position],
                    False,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "qwen_substituted_packet": (
                    behavior_gate._source_index_scores(len(row.choices), qwen_selected),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
            }
        )
        for condition in (
            "target_only",
            "target_decoder_only",
            "packet_only_source_index",
            "source_rank_control",
            "source_score_control",
            "source_score_quantized_control",
            "same_byte_visible_text",
            "qwen_substituted_packet",
        ):
            condition_integrity[condition] = (True, None)
        target_correct = behavior_gate._prediction(target) == int(row.answer_index)
        for condition in REPORT_CONDITIONS:
            scores, fired, residual, event_score = condition_scores[condition]
            integrity_accept, integrity_score = condition_integrity[condition]
            pred = behavior_gate._prediction(scores)
            correct = pred == int(row.answer_index)
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "condition": condition,
                    "answer_index": int(row.answer_index),
                    "answer_label": row.answer_label,
                    "prediction_index": int(pred),
                    "prediction_label": row.choice_labels[pred],
                    "correct": bool(correct),
                    "scores": [float(score) for score in scores],
                    "margin": float(behavior_gate._margin(scores, row.answer_index)),
                    "entropy": float(behavior_gate._entropy(scores)),
                    "packet_fired": bool(fired),
                    "packet_helped": bool(fired and correct and not target_correct),
                    "packet_harmed": bool(fired and (not correct) and target_correct),
                    "packet_residual": [float(value) for value in residual],
                    "event_score": float(event_score) if event_score is not None else None,
                    "packet_integrity_accept": bool(integrity_accept),
                    "packet_integrity_score": float(integrity_score) if integrity_score is not None else None,
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "qwen_substituted_index": int(qwen_selected),
                    "qwen_substituted_label": row.choice_labels[qwen_selected],
                    "source_scores": [float(score) for score in raw_source_scores],
                    "gate_rule": _public_gate_rule(gate_rule),
                    "control_origin": condition,
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed, bootstrap_samples=bootstrap_samples)
    matched_norms = [
        _safe_l2(np.asarray(row["packet_residual"], dtype=np.float64))
        for row in prediction_rows
        if row["condition"] == MATCHED_CONDITION
    ]
    matched_mean_norm = float(statistics.fmean(matched_norms)) if matched_norms else 0.0
    matched_p95_norm = float(np.percentile(matched_norms, 95)) if matched_norms else 0.0
    no_op_diagnostics: dict[str, dict[str, Any]] = {
        MATCHED_CONDITION: {
            "mean_residual_l2": matched_mean_norm,
            "p95_residual_l2": matched_p95_norm,
        }
    }
    for condition in EVENT_GATE_CORRUPTION_CONDITIONS:
        subset = [row for row in prediction_rows if row["condition"] == condition]
        norms = [_safe_l2(np.asarray(row["packet_residual"], dtype=np.float64)) for row in subset]
        flips = 0
        for row in subset:
            target_group = next(
                (
                    other
                    for other in prediction_rows
                    if other["content_id"] == row["content_id"] and other["condition"] == "target_only"
                ),
                None,
            )
            if target_group is not None:
                flips += int(int(row["prediction_index"]) != int(target_group["prediction_index"]))
        mean_norm = float(statistics.fmean(norms)) if norms else 0.0
        p95_norm = float(np.percentile(norms, 95)) if norms else 0.0
        no_op_diagnostics[condition] = {
            "mean_residual_l2": mean_norm,
            "p95_residual_l2": p95_norm,
            "mean_residual_l2_ratio_vs_matched": float(mean_norm / max(matched_mean_norm, 1e-12)),
            "p95_residual_l2_ratio_vs_matched": float(p95_norm / max(matched_p95_norm, 1e-12)),
            "prediction_flips_vs_target_only": int(flips),
            "prediction_flip_rate_vs_target_only": float(flips / len(subset)) if subset else 0.0,
            "packet_helped_vs_target": int(metrics[condition]["packet_helped_vs_target"]),
            "packet_harmed_vs_target": int(metrics[condition]["packet_harmed_vs_target"]),
        }
    oracle = ecoc_gate._oracle_diagnostics(prediction_rows)
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
        "gate": "source_private_arc_challenge_behavior_atom_decoder_gate",
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
            "target_only_accuracy": float(metrics["target_only"]["accuracy"]),
            "best_required_control": best_required_control,
            "best_required_control_accuracy": float(metrics[best_required_control]["accuracy"]),
            "worst_required_ci95_low": float(worst_ci_low),
            "matched_packet_fired": int(matched["packet_fired"]),
            "matched_packet_fired_rate": float(matched["packet_fired_rate"]),
            "matched_packet_helped": int(matched["packet_helped_vs_target"]),
            "matched_packet_harmed": int(matched["packet_harmed_vs_target"]),
            "matched_packet_net_help": int(matched["packet_net_help_vs_target"]),
        },
        "condition_metrics": metrics,
        "no_op_residual_diagnostics": no_op_diagnostics,
        "oracle_diagnostics": oracle,
        "systems_packet_sideband": {
            "source_private": True,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_logits_or_scores_exposed": False,
            "native_serving_throughput_measured": False,
            "packet_bytes_per_row": float(estimated_packet_bytes_per_row),
            "framed_packet_bytes_per_row": int(framed_packet_bytes),
            "same_byte_visible_text_budget": int(same_byte_budget_used),
            "cache_line_bytes_per_row_64b": int(math.ceil(max(framed_packet_bytes, 1) / 64.0) * 64),
            "dma_bytes_per_row_128b": int(math.ceil(max(framed_packet_bytes, 1) / 128.0) * 128),
            "decode_flops_proxy_per_row": int(max(len(row.choices) for row in rows) * packet_rank),
            "sparse_packet_metadata": sparse_meta,
            "atom_basis_mode": atom_basis_mode,
            "paired_alignment_weight": float(paired_alignment_weight),
            "paired_target_behavior_weight": float(paired_target_behavior_weight),
            "subtract_zero_packet_baseline": bool(subtract_zero_packet_baseline),
            "decoder_mode": decoder_mode,
            "gate_mode": gate_mode,
            "receiver_training_mode": receiver_training_mode,
            "corruption_loss_weight": float(corruption_loss_weight),
            "packet_integrity_mode": packet_integrity_mode,
            "note": (
                "Byte counts cover behavior-supervised source-hidden atom IDs plus quantized coefficients only. "
                "They are not native GPU throughput, HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "feature_metadata": {
            **source_packet_meta,
            "target_score_metadata": target_score_meta,
            "same_byte_score_metadata": same_byte_meta,
            "target_conditioned_decoder": {
                "decoder_mode": decoder_mode,
                "ridge": decoder.ridge,
                "fit_mse": decoder.fit_mse,
                "fit_r2": decoder.fit_r2,
                "selected_gate_rule": _public_gate_rule(gate_rule),
                "receiver_training": receiver_training_meta,
                "packet_integrity": packet_integrity_rule.metadata if packet_integrity_rule is not None else None,
            },
            "target_only_decoder": {
                "ridge": target_only_decoder.ridge,
                "fit_mse": target_only_decoder.fit_mse,
                "fit_r2": target_only_decoder.fit_r2,
            },
            "target_derived_packet_map": {
                "ridge": target_packet_map.ridge,
                "fit_mse": target_packet_map.fit_mse,
                "fit_r2": target_packet_map.fit_r2,
            },
        },
        "inputs": {
            "validation_path": behavior_gate._display(validation_path),
            "test_path": behavior_gate._display(test_path),
            "source_family_gate_dir": behavior_gate._display(source_family_gate_dir),
            "agreement_path": behavior_gate._display(agreement_path),
            "tiny_validation_score_cache": behavior_gate._display(tiny_validation_score_cache),
            "tiny_test_score_cache": behavior_gate._display(tiny_test_score_cache),
            "qwen_validation_score_cache": behavior_gate._display(qwen_validation_score_cache),
            "qwen_test_score_cache": behavior_gate._display(qwen_test_score_cache),
            "source_model": str(source_model),
            "target_model": str(target_model),
            "source_hidden_layer": int(source_hidden_layer),
            "target_hidden_layer": int(target_hidden_layer),
            "train_disagreement_limit": int(train_disagreement_limit),
            "test_disagreement_limit": int(test_disagreement_limit),
            "packet_rank": int(packet_rank),
            "packet_top_k": int(packet_top_k),
            "packet_bits": int(packet_bits),
            "atom_basis_mode": atom_basis_mode,
            "batchtopk_epochs": int(batchtopk_epochs),
            "batchtopk_learning_rate": float(batchtopk_learning_rate),
            "batchtopk_batch_size": int(batchtopk_batch_size),
            "batchtopk_reconstruction_weight": float(batchtopk_reconstruction_weight),
            "batchtopk_l1_weight": float(batchtopk_l1_weight),
            "paired_alignment_weight": float(paired_alignment_weight),
            "paired_target_behavior_weight": float(paired_target_behavior_weight),
            "same_byte_budget": int(same_byte_budget_used),
            "subtract_zero_packet_baseline": bool(subtract_zero_packet_baseline),
            "decoder_mode": decoder_mode,
            "gate_mode": gate_mode,
            "receiver_training_mode": receiver_training_mode,
            "corruption_loss_weight": float(corruption_loss_weight),
            "corruption_condition_weights": {
                condition: float(weight) for condition, weight in sorted((corruption_condition_weights or {}).items())
            },
            "packet_integrity_mode": packet_integrity_mode,
        },
        "interpretation": (
            "This gate tests whether source-hidden innovations become more useful when the sparse atom basis is "
            "trained toward target residual behavior rather than unsupervised PCA variance. The atom basis mode is "
            f"`{atom_basis_mode}`. It passes only if the "
            "matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and "
            "generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/"
            "rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_behavior_atom_decoder_gate.json"
    md_path = output_dir / "arc_challenge_behavior_atom_decoder_gate.md"
    audit_path = output_dir / "prediction_audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    behavior_gate._write_jsonl(audit_path, prediction_rows)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {"path": behavior_gate._display(json_path), "sha256": behavior_gate._sha256_file(json_path), "bytes": json_path.stat().st_size},
            {"path": behavior_gate._display(md_path), "sha256": behavior_gate._sha256_file(md_path), "bytes": md_path.stat().st_size},
            {"path": behavior_gate._display(audit_path), "sha256": behavior_gate._sha256_file(audit_path), "bytes": audit_path.stat().st_size},
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gc.collect()
    print(json.dumps({"headline": payload["strict_headline"], "pass_gate": payload["pass_gate"]}, sort_keys=True))
    return payload


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
    parser.add_argument("--train-disagreement-limit", type=int, default=16)
    parser.add_argument("--test-disagreement-limit", type=int, default=16)
    parser.add_argument("--source-model", default=str(DEFAULT_TINY_MODEL))
    parser.add_argument("--target-model", default=str(DEFAULT_QWEN3_MODEL))
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=160)
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--target-hidden-layer", type=int, default=-1)
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--packet-rank", type=int, default=8)
    parser.add_argument("--packet-top-k", type=int, default=2)
    parser.add_argument("--packet-bits", type=int, default=4)
    parser.add_argument("--atom-basis-mode", choices=ATOM_BASIS_MODES, default="behavior_svd")
    parser.add_argument("--batchtopk-epochs", type=int, default=300)
    parser.add_argument("--batchtopk-learning-rate", type=float, default=0.01)
    parser.add_argument("--batchtopk-batch-size", type=int, default=16)
    parser.add_argument("--batchtopk-reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--batchtopk-l1-weight", type=float, default=0.001)
    parser.add_argument("--paired-alignment-weight", type=float, default=0.25)
    parser.add_argument("--paired-target-behavior-weight", type=float, default=0.25)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--same-byte-budget",
        type=int,
        default=0,
        help="Visible-text byte budget. 0 means use framed packet bytes per row.",
    )
    parser.add_argument("--subtract-zero-packet-baseline", choices=("true", "false"), default="true")
    parser.add_argument("--decoder-mode", choices=("target_conditioned", "packet_innovation"), default="target_conditioned")
    parser.add_argument("--gate-mode", choices=("residual_threshold", "event_triggered"), default="residual_threshold")
    parser.add_argument("--receiver-training-mode", choices=RECEIVER_TRAINING_MODES, default="matched_only")
    parser.add_argument(
        "--corruption-loss-weight",
        type=float,
        default=0.1,
        help="Per-example weight for no-op corruption targets when receiver-training-mode is corruption_noop.",
    )
    parser.add_argument(
        "--corruption-condition-weights",
        default="",
        help=(
            "Comma-separated per-condition no-op weights, e.g. "
            "candidate_roll=0.25,top_atom_knockout=0.25. Overrides --corruption-loss-weight."
        ),
    )
    parser.add_argument("--packet-integrity-mode", choices=PACKET_INTEGRITY_MODES, default="none")
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=37)
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
        target_hidden_layer=int(args.target_hidden_layer),
        source_feature_dim=int(args.source_feature_dim),
        ridge=float(args.ridge),
        packet_rank=int(args.packet_rank),
        packet_top_k=int(args.packet_top_k),
        packet_bits=int(args.packet_bits),
        atom_basis_mode=str(args.atom_basis_mode),
        batchtopk_epochs=int(args.batchtopk_epochs),
        batchtopk_learning_rate=float(args.batchtopk_learning_rate),
        batchtopk_batch_size=int(args.batchtopk_batch_size),
        batchtopk_reconstruction_weight=float(args.batchtopk_reconstruction_weight),
        batchtopk_l1_weight=float(args.batchtopk_l1_weight),
        paired_alignment_weight=float(args.paired_alignment_weight),
        paired_target_behavior_weight=float(args.paired_target_behavior_weight),
        local_files_only=str(args.local_files_only).lower() == "true",
        bootstrap_samples=int(args.bootstrap_samples),
        same_byte_budget=int(args.same_byte_budget),
        subtract_zero_packet_baseline=str(args.subtract_zero_packet_baseline).lower() == "true",
        decoder_mode=str(args.decoder_mode),
        gate_mode=str(args.gate_mode),
        receiver_training_mode=str(args.receiver_training_mode),
        corruption_loss_weight=float(args.corruption_loss_weight),
        corruption_condition_weights=_parse_condition_weight_spec(str(args.corruption_condition_weights)),
        packet_integrity_mode=str(args.packet_integrity_mode),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
