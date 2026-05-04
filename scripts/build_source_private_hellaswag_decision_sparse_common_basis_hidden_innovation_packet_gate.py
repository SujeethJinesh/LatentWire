from __future__ import annotations

"""Decision-supervised sparse common-basis hidden packet gate for HellaSwag."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_anchor_relative_hidden_code_scout as anchor_code  # noqa: E402
from scripts import build_source_private_hellaswag_crosscoder_hidden_code_scout as crosscode  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_code_packet_scout as hidden_code  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as hidden_summary  # noqa: E402
from scripts import build_source_private_hellaswag_learned_source_code_packet_gate as source_code  # noqa: E402
from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate_20260504_validation1024_2048"
)
DEFAULT_EVAL_FULL = hidden_code.DEFAULT_EVAL_FULL
DEFAULT_EVAL_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048/source_eval_hidden_cache.npz"
)
DEFAULT_SOURCE_MODEL = hidden_code.DEFAULT_SOURCE_MODEL
DEFAULT_PCA_DIMS = (64,)
DEFAULT_SHARED_DIMS = (8, 16)
DEFAULT_SAE_ATOMS = (64, 128)
DEFAULT_SAE_TOPKS = (2, 4)
DEFAULT_SAE_DECISION_WEIGHTS = (0.2,)
DEFAULT_SAE_L1_WEIGHTS = (0.001,)
DEFAULT_DECODER_RIDGES = crosscode.DEFAULT_DECODER_RIDGES
DEFAULT_SAE_EPOCHS = 48
DEFAULT_SAE_LEARNING_RATE = 1e-3
DEFAULT_SAE_BATCH_SIZE = 256
MAX_CODEBOOK_SIZE = 256
STRICT_DELTA = 0.010
BASELINE_DELTA = 0.010
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 1
FRAMED_PACKET_BYTES = 4
CANDIDATE_COUNT = 4
AMBIGUITY_ATOM_BITS = 4
AMBIGUITY_ATOM_SLOTS = (1 << AMBIGUITY_ATOM_BITS) - 1
AMBIGUITY_ACTION_NAMES = (
    "source_top1",
    "source_top2",
    "qwen_target",
    "qwen_mean",
    "qwen_hybrid",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return wz._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return wz._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return hidden_code._parse_int_tuple(value)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return hidden_code._parse_float_tuple(value)


def _safe_float_label(value: float) -> str:
    return f"{float(value):.4g}".replace("-", "m").replace(".", "p")


def _fit_flat_indices(fit_indices: np.ndarray) -> np.ndarray:
    return crosscode._flat_candidate_indices(fit_indices)


def _candidate_correct_labels(answers: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    labels = []
    for row_index in row_indices.astype(np.int64):
        for candidate in range(CANDIDATE_COUNT):
            labels.append(1.0 if int(candidate) == int(answers[row_index]) else 0.0)
    return np.asarray(labels, dtype=np.float32)


def _fit_decision_sae(
    *,
    train_source_shared: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    atoms: int,
    decision_weight: float,
    l1_weight: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    label_permutation_seed: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is expected in the repo venv.
        raise RuntimeError("decision-sparse common-basis gate requires torch") from exc

    flat = train_source_shared.reshape(
        train_source_shared.shape[0] * train_source_shared.shape[1],
        train_source_shared.shape[2],
    )
    fit_flat = _fit_flat_indices(fit_indices)
    x_fit = np.asarray(flat[fit_flat], dtype=np.float32)
    y_fit = _candidate_correct_labels(answers, fit_indices)
    if label_permutation_seed is not None:
        rng = np.random.default_rng(int(label_permutation_seed))
        y_fit = y_fit[rng.permutation(len(y_fit))]

    mean = np.mean(x_fit, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(x_fit, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    x_fit = (x_fit - mean[None, :]) / scale[None, :]
    y_fit = y_fit.reshape(-1, 1)

    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    tensor_x = torch.from_numpy(x_fit)
    tensor_y = torch.from_numpy(y_fit.astype(np.float32))
    dim = int(tensor_x.shape[1])
    latent_dim = max(1, int(atoms))
    encoder = torch.nn.Linear(dim, latent_dim)
    decoder = torch.nn.Linear(latent_dim, dim)
    head = torch.nn.Linear(latent_dim, 1)
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
    positive_count = max(1.0, float(np.sum(y_fit > 0.5)))
    negative_count = max(1.0, float(y_fit.shape[0] - positive_count))
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / positive_count], dtype=torch.float32)
    )
    rng = np.random.default_rng(int(seed) + 1777)
    final_reconstruction = 0.0
    final_decision = 0.0
    final_l1 = 0.0
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(tensor_x.shape[0])
        for start in range(0, tensor_x.shape[0], max(8, int(batch_size))):
            batch_ids = torch.from_numpy(order[start : start + max(8, int(batch_size))])
            batch_x = tensor_x[batch_ids]
            batch_y = tensor_y[batch_ids]
            latent = torch.relu(encoder(batch_x))
            reconstruction = decoder(latent)
            logits = head(latent)
            reconstruction_loss = torch.mean((reconstruction - batch_x) ** 2)
            decision_loss = bce(logits, batch_y)
            l1_loss = torch.mean(latent)
            loss = (
                reconstruction_loss
                + float(decision_weight) * decision_loss
                + float(l1_weight) * l1_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_reconstruction = float(reconstruction_loss.detach().cpu())
            final_decision = float(decision_loss.detach().cpu())
            final_l1 = float(l1_loss.detach().cpu())

    with torch.no_grad():
        latent = torch.relu(encoder(tensor_x))
        logits = head(latent)
        train_pred = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
        train_decision_accuracy = float(torch.mean((train_pred == tensor_y).to(torch.float32)).cpu())
        active_rate = float(torch.mean((latent > 1e-6).to(torch.float32)).cpu())
        mean_active = float(torch.mean(torch.sum(latent > 1e-6, dim=1).to(torch.float32)).cpu())

    encoder_weight = encoder.weight.detach().cpu().numpy().astype(np.float64)
    encoder_bias = encoder.bias.detach().cpu().numpy().astype(np.float64)
    head_weight = head.weight.detach().cpu().numpy()[0].astype(np.float64)
    head_bias = float(head.bias.detach().cpu().numpy()[0])
    priority = np.abs(head_weight)
    transmit_atoms = min(int(len(priority)), (MAX_CODEBOOK_SIZE // CANDIDATE_COUNT) - 1)
    transmit_atom_ids = np.argsort(-priority)[:transmit_atoms].astype(np.int64)
    atom_digest = hashlib.sha256(encoder_weight.astype(np.float32).tobytes()).hexdigest()
    return {
        "fit_kind": "decision_supervised_sae_common_basis",
        "atoms": int(atoms),
        "transmit_atoms": int(transmit_atoms),
        "transmit_atom_ids": transmit_atom_ids,
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "head_weight": head_weight,
        "head_bias": head_bias,
        "atom_priority": priority.astype(np.float64),
        "decision_weight": float(decision_weight),
        "l1_weight": float(l1_weight),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "label_permutation_seed": label_permutation_seed,
        "fit_label_mean": float(np.mean(y_fit)),
        "sae_reconstruction_loss": final_reconstruction,
        "sae_decision_loss": final_decision,
        "sae_l1_activation": final_l1,
        "sae_train_decision_accuracy": train_decision_accuracy,
        "sae_active_rate": active_rate,
        "sae_mean_active_features": mean_active,
        "atom_digest": atom_digest,
    }


def _sae_values(shared_candidates: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    flat = shared_candidates.reshape(
        shared_candidates.shape[0] * shared_candidates.shape[1],
        shared_candidates.shape[2],
    )
    standardized = (flat - params["mean"][None, :]) / params["scale"][None, :]
    raw = standardized @ params["encoder_weight"].T + params["encoder_bias"][None, :]
    values = np.maximum(raw, 0.0)
    return values.reshape(shared_candidates.shape[0], shared_candidates.shape[1], -1).astype(np.float64)


def _encode_sparse_atom_packet(
    *,
    source_shared: np.ndarray,
    packet: np.ndarray,
    params: dict[str, Any],
    sae_topk: int,
) -> dict[str, np.ndarray]:
    values = _sae_values(source_shared, params)
    atom_ids = np.asarray(params["transmit_atom_ids"], dtype=np.int64)
    selected = values[np.arange(len(packet)), packet.astype(np.int64)][:, atom_ids]
    if 0 < int(sae_topk) < selected.shape[1]:
        kept = np.zeros_like(selected)
        top_ids = np.argsort(selected, axis=1)[:, -int(sae_topk) :]
        np.put_along_axis(kept, top_ids, np.take_along_axis(selected, top_ids, axis=1), axis=1)
        selected = kept
    best_rank = np.argmax(selected, axis=1).astype(np.int64)
    active = np.max(selected, axis=1) > 1e-8
    slot = np.where(active, best_rank + 1, 0).astype(np.int64)
    code = slot * CANDIDATE_COUNT + packet.astype(np.int64)
    full_atom = np.where(active, atom_ids[best_rank], -1).astype(np.int64)
    return {
        "code": code.astype(np.int64),
        "atom_slot": slot.astype(np.int64),
        "full_atom_id": full_atom.astype(np.int64),
        "atom_value": np.max(selected, axis=1).astype(np.float64),
    }


def _standardize_target_values(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit_flat = train_values[fit_indices].reshape(
        len(fit_indices) * train_values.shape[1],
        train_values.shape[2],
    )
    mean = np.mean(fit_flat, axis=0)
    scale = np.std(fit_flat, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (
        ((train_values - mean[None, None, :]) / scale[None, None, :]).astype(np.float64),
        ((eval_values - mean[None, None, :]) / scale[None, None, :]).astype(np.float64),
        {
            "target_atom_mean": [float(item) for item in mean],
            "target_atom_scale": [float(item) for item in scale],
        },
    )


def _target_atom_values(
    target_shared: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    values = _sae_values(target_shared, params)
    return values[:, :, np.asarray(params["transmit_atom_ids"], dtype=np.int64)].astype(np.float64)


def _candidate_decoder_features_with_sae(
    *,
    qwen_scores: np.ndarray,
    qwen_target: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_hybrid: np.ndarray,
    source_code_values: np.ndarray,
    codebook_size: int,
    target_atom_values: np.ndarray,
) -> np.ndarray:
    base = source_code._candidate_decoder_features(
        qwen_scores=qwen_scores,
        qwen_target=qwen_target,
        qwen_mean=qwen_mean,
        qwen_hybrid=qwen_hybrid,
        source_code=source_code_values,
        codebook_size=codebook_size,
    )
    source_code_values = source_code_values.astype(np.int64)
    slot = source_code_values // CANDIDATE_COUNT
    selected = np.zeros((len(source_code_values), CANDIDATE_COUNT, 1), dtype=np.float64)
    nonzero = (slot > 0).astype(np.float64)[:, None, None]
    for candidate in range(CANDIDATE_COUNT):
        local = slot > 0
        if np.any(local):
            selected[local, candidate, 0] = target_atom_values[
                np.where(local)[0],
                candidate,
                slot[local] - 1,
            ]
    return np.concatenate([base, target_atom_values, selected, np.repeat(nonzero, CANDIDATE_COUNT, axis=1)], axis=2)


def _top2_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1)
    return order[:, 0].astype(np.int64), order[:, 1].astype(np.int64)


def _pack_ambiguity_code(
    *,
    source_top1: np.ndarray,
    source_top2: np.ndarray,
    atom_slot: np.ndarray,
    max_atom_slots: int = AMBIGUITY_ATOM_SLOTS,
) -> np.ndarray:
    source_top1 = np.asarray(source_top1, dtype=np.int64) % CANDIDATE_COUNT
    source_top2 = np.asarray(source_top2, dtype=np.int64) % CANDIDATE_COUNT
    atom_slot = np.asarray(atom_slot, dtype=np.int64)
    atom_slot = np.where((atom_slot >= 1) & (atom_slot <= int(max_atom_slots)), atom_slot, 0)
    return (
        (atom_slot.astype(np.int64) << (2 * int(math.log2(CANDIDATE_COUNT))))
        | (source_top2.astype(np.int64) << int(math.log2(CANDIDATE_COUNT)))
        | source_top1.astype(np.int64)
    ).astype(np.int64)


def _decode_ambiguity_code(code: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = np.asarray(code, dtype=np.int64)
    source_top1 = code & 0b11
    source_top2 = (code >> 2) & 0b11
    atom_slot = code >> 4
    return source_top1.astype(np.int64), source_top2.astype(np.int64), atom_slot.astype(np.int64)


def _zscore_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.where(scale > 1e-8, scale, 1.0)


def _one_hot(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return np.stack([(values == item).astype(np.float64) for item in range(int(width))], axis=1)


def _ambiguity_action_features(
    *,
    qwen_scores: np.ndarray,
    qwen_target: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_hybrid: np.ndarray,
    ambiguity_code: np.ndarray,
    target_atom_values: np.ndarray,
    max_atom_slots: int = AMBIGUITY_ATOM_SLOTS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    qwen_scores = np.asarray(qwen_scores, dtype=np.float64)
    qwen_target = np.asarray(qwen_target, dtype=np.int64)
    qwen_mean = np.asarray(qwen_mean, dtype=np.int64)
    qwen_hybrid = np.asarray(qwen_hybrid, dtype=np.int64)
    source_top1, source_top2, atom_slot = _decode_ambiguity_code(ambiguity_code)
    atom_slot = np.minimum(atom_slot, int(max_atom_slots)).astype(np.int64)
    qwen_top1, qwen_top2 = _top2_from_scores(qwen_scores)
    qz = _zscore_rows(qwen_scores)
    row_ids = np.arange(qwen_scores.shape[0])
    q_margin = qz[row_ids, qwen_top1] - qz[row_ids, qwen_top2]
    actions = np.stack([source_top1, source_top2, qwen_target, qwen_mean, qwen_hybrid], axis=1)
    source_pair_disagrees = (source_top1 != source_top2).astype(np.float64)
    source_target_disagrees = (source_top1 != qwen_target).astype(np.float64)
    features: list[np.ndarray] = []
    for role, candidate in enumerate(actions.T):
        candidate = np.asarray(candidate, dtype=np.int64)
        selected_atom = np.zeros((qwen_scores.shape[0], 1), dtype=np.float64)
        active = atom_slot > 0
        if np.any(active):
            selected_atom[active, 0] = target_atom_values[
                np.where(active)[0],
                candidate[active],
                atom_slot[active] - 1,
            ]
        parts = [
            np.ones((qwen_scores.shape[0], 1), dtype=np.float64),
            _one_hot(np.full(qwen_scores.shape[0], role, dtype=np.int64), len(AMBIGUITY_ACTION_NAMES)),
            _one_hot(candidate, CANDIDATE_COUNT),
            _one_hot(source_top1, CANDIDATE_COUNT),
            _one_hot(source_top2, CANDIDATE_COUNT),
            _one_hot(qwen_target, CANDIDATE_COUNT),
            _one_hot(qwen_top1, CANDIDATE_COUNT),
            _one_hot(np.minimum(atom_slot, int(max_atom_slots)), int(max_atom_slots) + 1),
            qz[row_ids, candidate][:, None],
            qz[row_ids, qwen_target][:, None],
            q_margin[:, None],
            target_atom_values[row_ids, candidate, :].reshape(qwen_scores.shape[0], -1),
            selected_atom,
            (candidate == source_top1).astype(np.float64)[:, None],
            (candidate == source_top2).astype(np.float64)[:, None],
            (candidate == qwen_target).astype(np.float64)[:, None],
            (candidate == qwen_top1).astype(np.float64)[:, None],
            source_pair_disagrees[:, None],
            source_target_disagrees[:, None],
            (atom_slot > 0).astype(np.float64)[:, None],
        ]
        features.append(np.concatenate(parts, axis=1).astype(np.float64))
    diagnostics = {
        "source_top1": source_top1,
        "source_top2": source_top2,
        "atom_slot": atom_slot,
        "qwen_top1": qwen_top1,
        "qwen_top2": qwen_top2,
    }
    return np.stack(features, axis=1), actions.astype(np.int64), diagnostics


def _fit_ambiguity_router(
    *,
    train_features: np.ndarray,
    train_actions: np.ndarray,
    train_baseline: np.ndarray,
    train_answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    ridges: tuple[float, ...],
    label_permutation_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    dev_indices = np.asarray(dev_indices, dtype=np.int64)
    answers = np.asarray(train_answers, dtype=np.int64)
    if label_permutation_seed is not None:
        rng = np.random.default_rng(int(label_permutation_seed))
        answers = answers[rng.permutation(len(answers))]
    flat_x = train_features[fit_indices].reshape(-1, train_features.shape[-1])
    flat_actions = train_actions[fit_indices].reshape(-1)
    repeated_answers = np.repeat(answers[fit_indices], train_actions.shape[1])
    repeated_baseline = np.repeat(train_baseline[fit_indices], train_actions.shape[1])
    target = (flat_actions == repeated_answers).astype(np.float64) - (
        repeated_baseline == repeated_answers
    ).astype(np.float64)
    best_model: dict[str, Any] = {
        "weights": np.zeros(train_features.shape[-1], dtype=np.float64).tolist(),
        "threshold": 1.0,
        "ridge": 0.0,
        "selection": "no_op",
    }
    best_key: tuple[float, float, int, int, str] = (0.0, 0.0, 0, 0, "no_op")
    rows: list[dict[str, Any]] = []
    for ridge in ridges:
        penalty = float(ridge) * np.eye(flat_x.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        lhs = flat_x.T @ flat_x + penalty
        rhs = flat_x.T @ target
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs
        dev_scores = train_features[dev_indices] @ weights
        max_scores = np.max(dev_scores, axis=1)
        thresholds = sorted(set(float(value) for value in max_scores))
        thresholds.append(float(np.max(max_scores) + max(1e-6, abs(float(np.max(max_scores))) * 1e-6)))
        if len(thresholds) > 96:
            finite = thresholds[:-1]
            thresholds = sorted({finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 80)})
            thresholds.append(float(np.max(max_scores) + max(1e-6, abs(float(np.max(max_scores))) * 1e-6)))
        for threshold in thresholds:
            model = {"weights": weights.tolist(), "threshold": float(threshold), "ridge": float(ridge)}
            pred = _predict_ambiguity_router(
                features=train_features[dev_indices],
                actions=train_actions[dev_indices],
                baseline=train_baseline[dev_indices],
                model=model,
            )
            correct = pred == answers[dev_indices]
            base_correct = train_baseline[dev_indices] == answers[dev_indices]
            delta = float(np.mean(correct.astype(float) - base_correct.astype(float)))
            row = {
                "ridge": float(ridge),
                "threshold": float(threshold),
                "official_dev_accuracy": float(np.mean(correct)),
                "official_dev_delta_vs_packet": delta,
                "official_dev_helps": int(np.sum(correct & ~base_correct)),
                "official_dev_harms": int(np.sum(~correct & base_correct)),
                "official_dev_override_count": int(np.sum(pred != train_baseline[dev_indices])),
            }
            rows.append(row)
            key = (
                row["official_dev_accuracy"],
                row["official_dev_delta_vs_packet"],
                int(row["official_dev_helps"]) - int(row["official_dev_harms"]),
                -int(row["official_dev_override_count"]),
                json.dumps({"ridge": float(ridge), "threshold": float(threshold)}, sort_keys=True),
            )
            if key > best_key:
                best_key = key
                best_model = model
    return best_model, rows


def _predict_ambiguity_router(
    *,
    features: np.ndarray,
    actions: np.ndarray,
    baseline: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    scores = np.asarray(features, dtype=np.float64) @ weights
    best_action = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_action]
    predictions = np.asarray(baseline, dtype=np.int64).copy()
    mask = best_score > float(model["threshold"])
    predictions[mask] = actions[np.arange(actions.shape[0]), best_action][mask]
    return predictions.astype(np.int64)


def _source_pair_oracle_predictions(source_top1: np.ndarray, source_top2: np.ndarray, answers: np.ndarray) -> np.ndarray:
    return np.where(source_top1 == answers, source_top1, np.where(source_top2 == answers, source_top2, source_top1)).astype(np.int64)


def _score_predictions(
    *,
    name: str,
    predictions: np.ndarray,
    validation: dict[str, Any],
    seed: int,
    bootstrap_samples: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return wz._score_row(
        name=name,
        predictions=predictions,
        answers=validation["answers"],
        packet_predictions=validation["packet"],
        qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        extra=extra,
    )


def _fit_predict_decoder(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    train_answers: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
    label_permutation_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coef = wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=train_answers,
        fit_indices=fit_indices,
        ridge=float(ridge),
        label_permutation_seed=label_permutation_seed,
    )
    return wz._predict_candidate_decoder(train_features, coef), wz._predict_candidate_decoder(eval_features, coef)


def _evaluate_sparse_config(
    *,
    config: dict[str, Any],
    surfaces: dict[str, Any],
    train_source_shared: np.ndarray,
    eval_source_shared: np.ndarray,
    train_target_shared: np.ndarray,
    eval_target_shared: np.ndarray,
    decoder_ridges: tuple[float, ...],
    bootstrap_samples: int,
    row_seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    params = _fit_decision_sae(
        train_source_shared=train_source_shared,
        answers=calibration["answers"],
        fit_indices=fit_indices,
        atoms=int(config["sae_atoms"]),
        decision_weight=float(config["sae_decision_weight"]),
        l1_weight=float(config["sae_l1_weight"]),
        learning_rate=float(config["sae_learning_rate"]),
        epochs=int(config["sae_epochs"]),
        batch_size=int(config["sae_batch_size"]),
        seed=int(config["seed"]),
    )
    train_encoded = _encode_sparse_atom_packet(
        source_shared=train_source_shared,
        packet=calibration["tiny_packet"],
        params=params,
        sae_topk=int(config["sae_topk"]),
    )
    eval_encoded = _encode_sparse_atom_packet(
        source_shared=eval_source_shared,
        packet=validation["packet"],
        params=params,
        sae_topk=int(config["sae_topk"]),
    )
    codebook_size = (int(params["transmit_atoms"]) + 1) * CANDIDATE_COUNT
    train_target_atoms, eval_target_atoms, target_scale_audit = _standardize_target_values(
        _target_atom_values(train_target_shared, params),
        _target_atom_values(eval_target_shared, params),
        fit_indices,
    )
    train_features = _candidate_decoder_features_with_sae(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code_values=train_encoded["code"],
        codebook_size=codebook_size,
        target_atom_values=train_target_atoms,
    )
    eval_features = _candidate_decoder_features_with_sae(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code_values=eval_encoded["code"],
        codebook_size=codebook_size,
        target_atom_values=eval_target_atoms,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for ridge in decoder_ridges:
        train_predictions, eval_predictions = _fit_predict_decoder(
            train_features=train_features,
            eval_features=eval_features,
            train_answers=calibration["answers"],
            fit_indices=fit_indices,
            ridge=float(ridge),
        )
        predictions[(str(config["name"]), float(ridge))] = eval_predictions
        rows.append(
            _score_predictions(
                name="decision_sparse_common_basis_decoder",
                predictions=eval_predictions,
                validation=validation,
                seed=row_seed_offset + len(rows),
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(config["name"]),
                    "encoder_kind": "decision_sparse_common_basis_sae",
                    "pca_dims": int(config["pca_dims"]),
                    "shared_dims": int(config["shared_dims"]),
                    "sae_atoms": int(config["sae_atoms"]),
                    "sae_transmit_atoms": int(params["transmit_atoms"]),
                    "sae_topk": int(config["sae_topk"]),
                    "sae_decision_weight": float(config["sae_decision_weight"]),
                    "sae_l1_weight": float(config["sae_l1_weight"]),
                    "ridge": float(ridge),
                    "codebook_size": int(codebook_size),
                    "official_fit_accuracy": wz._accuracy(
                        train_predictions[fit_indices],
                        calibration["answers"][fit_indices],
                    ),
                    "official_dev_accuracy": wz._accuracy(
                        train_predictions[dev_indices],
                        calibration["answers"][dev_indices],
                    ),
                    "official_dev_delta_vs_packet": wz._accuracy(
                        train_predictions[dev_indices],
                        calibration["answers"][dev_indices],
                    )
                    - wz._accuracy(
                        calibration["tiny_packet"][dev_indices],
                        calibration["answers"][dev_indices],
                    ),
                    "eval_code_unique_count": int(len(np.unique(eval_encoded["code"]))),
                    "eval_active_atom_rate": float(np.mean(eval_encoded["atom_slot"] > 0)),
                },
            )
        )
    audit = {
        "config": config,
        "params": {
            key: value
            for key, value in params.items()
            if key
            not in {
                "encoder_weight",
                "encoder_bias",
                "mean",
                "scale",
                "transmit_atom_ids",
                "head_weight",
                "atom_priority",
            }
        },
        "transmit_atom_ids": [int(item) for item in params["transmit_atom_ids"]],
        "top_atom_priority": [float(params["atom_priority"][item]) for item in params["transmit_atom_ids"][:10]],
        "target_scale_audit": target_scale_audit,
        "train_code_unique_count": int(len(np.unique(train_encoded["code"]))),
        "eval_code_unique_count": int(len(np.unique(eval_encoded["code"]))),
        "train_active_atom_rate": float(np.mean(train_encoded["atom_slot"] > 0)),
        "eval_active_atom_rate": float(np.mean(eval_encoded["atom_slot"] > 0)),
    }
    encoded = {
        "config": config,
        "params": params,
        "train_code": train_encoded["code"],
        "eval_code": eval_encoded["code"],
        "eval_atom_slot": eval_encoded["atom_slot"],
        "eval_full_atom_id": eval_encoded["full_atom_id"],
        "eval_atom_value": eval_encoded["atom_value"],
        "train_target_atoms": train_target_atoms,
        "eval_target_atoms": eval_target_atoms,
        "train_source_shared": train_source_shared,
        "eval_source_shared": eval_source_shared,
        "train_target_shared": train_target_shared,
        "eval_target_shared": eval_target_shared,
        "codebook_size": codebook_size,
        "encoder_audit": audit,
    }
    return rows, predictions, encoded


def _replace_candidate(code: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return (code.astype(np.int64) // CANDIDATE_COUNT) * CANDIDATE_COUNT + candidate.astype(np.int64)


def _permute_atom_slots(code: np.ndarray, *, slot_count: int, rng: np.random.Generator) -> np.ndarray:
    slot = code.astype(np.int64) // CANDIDATE_COUNT
    candidate = code.astype(np.int64) % CANDIDATE_COUNT
    permutation = np.arange(slot_count + 1, dtype=np.int64)
    if slot_count > 1:
        permutation[1:] = rng.permutation(np.arange(1, slot_count + 1, dtype=np.int64))
    return permutation[slot] * CANDIDATE_COUNT + candidate


def _control_rows(
    *,
    selected_blob: dict[str, Any],
    selected_ridge: float,
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    params = selected_blob["params"]
    config = selected_blob["config"]
    codebook_size = int(selected_blob["codebook_size"])
    train_features = _candidate_decoder_features_with_sae(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code_values=selected_blob["train_code"],
        codebook_size=codebook_size,
        target_atom_values=selected_blob["train_target_atoms"],
    )
    coef = wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rng = np.random.default_rng(control_seed)
    source_shuffle = rng.permutation(len(validation["answers"]))
    shuffled_encoded = _encode_sparse_atom_packet(
        source_shared=selected_blob["eval_source_shared"][source_shuffle],
        packet=validation["packet"],
        params=params,
        sae_topk=int(config["sae_topk"]),
    )
    row_shuffle_code = selected_blob["eval_code"][rng.permutation(len(selected_blob["eval_code"]))]
    random_code = rng.integers(0, codebook_size, size=len(validation["answers"]), dtype=np.int64)
    candidate_roll = _replace_candidate(
        selected_blob["eval_code"],
        (validation["packet"].astype(np.int64) + 1) % CANDIDATE_COUNT,
    )
    control_specs = [
        ("row_shuffle_sparse_atom_code", row_shuffle_code),
        ("source_shared_shuffle_before_encoding", shuffled_encoded["code"]),
        (
            "atom_index_permutation_mismatch",
            _permute_atom_slots(
                selected_blob["eval_code"],
                slot_count=int(params["transmit_atoms"]),
                rng=rng,
            ),
        ),
        ("top_atom_knockout", validation["packet"].astype(np.int64)),
        ("candidate_roll_code", candidate_roll),
        ("random_same_byte_code", random_code),
        ("zero_source_code", np.zeros(len(validation["answers"]), dtype=np.int64)),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_code) in enumerate(control_specs):
        eval_features = _candidate_decoder_features_with_sae(
            qwen_scores=validation["qwen_scores"],
            qwen_target=validation["alternatives"]["qwen_target_score"],
            qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
            qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            source_code_values=np.mod(eval_code.astype(np.int64), codebook_size),
            codebook_size=codebook_size,
            target_atom_values=selected_blob["eval_target_atoms"],
        )
        rows.append(
            _score_predictions(
                name=name,
                predictions=wz._predict_candidate_decoder(eval_features, coef),
                validation=validation,
                seed=50100 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(config["name"]),
                    "ridge": float(selected_ridge),
                    "codebook_size": codebook_size,
                },
            )
        )
    for offset, (name, train_code, eval_code, baseline_codebook) in enumerate(
        [
            (
                "qwen_side_only_common_basis_decoder",
                np.zeros(len(calibration["answers"]), dtype=np.int64),
                np.zeros(len(validation["answers"]), dtype=np.int64),
                1,
            ),
            (
                "compact_candidate_common_basis_decoder",
                calibration["tiny_packet"],
                validation["packet"],
                CANDIDATE_COUNT,
            ),
        ]
    ):
        train_baseline = _candidate_decoder_features_with_sae(
            qwen_scores=calibration["qwen_scores"],
            qwen_target=calibration["qwen_target"],
            qwen_mean=calibration["qwen_mean"],
            qwen_hybrid=calibration["qwen_hybrid"],
            source_code_values=train_code.astype(np.int64),
            codebook_size=int(baseline_codebook),
            target_atom_values=selected_blob["train_target_atoms"],
        )
        eval_baseline = _candidate_decoder_features_with_sae(
            qwen_scores=validation["qwen_scores"],
            qwen_target=validation["alternatives"]["qwen_target_score"],
            qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
            qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            source_code_values=eval_code.astype(np.int64),
            codebook_size=int(baseline_codebook),
            target_atom_values=selected_blob["eval_target_atoms"],
        )
        _, predictions = _fit_predict_decoder(
            train_features=train_baseline,
            eval_features=eval_baseline,
            train_answers=calibration["answers"],
            fit_indices=fit_indices,
            ridge=float(selected_ridge),
        )
        rows.append(
            _score_predictions(
                name=name,
                predictions=predictions,
                validation=validation,
                seed=50180 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": name,
                    "ridge": float(selected_ridge),
                    "codebook_size": int(baseline_codebook),
                },
            )
        )
    eval_features = _candidate_decoder_features_with_sae(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code_values=selected_blob["eval_code"],
        codebook_size=codebook_size,
        target_atom_values=selected_blob["eval_target_atoms"],
    )
    _, label_perm_predictions = _fit_predict_decoder(
        train_features=train_features,
        eval_features=eval_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
        label_permutation_seed=control_seed + 77,
    )
    rows.append(
        _score_predictions(
            name="label_permutation_decoder",
            predictions=label_perm_predictions,
            validation=validation,
            seed=50190,
            bootstrap_samples=bootstrap_samples,
            extra={"encoder_name": str(config["name"]), "ridge": float(selected_ridge), "codebook_size": codebook_size},
        )
    )
    permuted_params = _fit_decision_sae(
        train_source_shared=selected_blob["train_source_shared"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        atoms=int(config["sae_atoms"]),
        decision_weight=float(config["sae_decision_weight"]),
        l1_weight=float(config["sae_l1_weight"]),
        learning_rate=float(config["sae_learning_rate"]),
        epochs=int(config["sae_epochs"]),
        batch_size=int(config["sae_batch_size"]),
        seed=int(config["seed"]) + 911,
        label_permutation_seed=control_seed + 911,
    )
    perm_train_code = _encode_sparse_atom_packet(
        source_shared=selected_blob["train_source_shared"],
        packet=calibration["tiny_packet"],
        params=permuted_params,
        sae_topk=int(config["sae_topk"]),
    )["code"]
    perm_eval_code = _encode_sparse_atom_packet(
        source_shared=selected_blob["eval_source_shared"],
        packet=validation["packet"],
        params=permuted_params,
        sae_topk=int(config["sae_topk"]),
    )["code"]
    perm_codebook = (int(permuted_params["transmit_atoms"]) + 1) * CANDIDATE_COUNT
    perm_train_target, perm_eval_target, _ = _standardize_target_values(
        _target_atom_values(selected_blob["train_target_shared"], permuted_params),
        _target_atom_values(selected_blob["eval_target_shared"], permuted_params),
        fit_indices,
    )
    perm_train_features = _candidate_decoder_features_with_sae(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code_values=perm_train_code,
        codebook_size=perm_codebook,
        target_atom_values=perm_train_target,
    )
    perm_eval_features = _candidate_decoder_features_with_sae(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code_values=perm_eval_code,
        codebook_size=perm_codebook,
        target_atom_values=perm_eval_target,
    )
    _, perm_predictions = _fit_predict_decoder(
        train_features=perm_train_features,
        eval_features=perm_eval_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rows.append(
        _score_predictions(
            name="sae_label_permutation_encoder",
            predictions=perm_predictions,
            validation=validation,
            seed=50191,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": str(config["name"]),
                "ridge": float(selected_ridge),
                "codebook_size": int(perm_codebook),
                "permuted_sae_train_decision_accuracy": permuted_params["sae_train_decision_accuracy"],
            },
        )
    )
    rows.append(
        _score_predictions(
            name="packet_only",
            predictions=validation["packet"],
            validation=validation,
            seed=50192,
            bootstrap_samples=bootstrap_samples,
            extra={"encoder_name": "packet_only", "ridge": 0.0, "codebook_size": CANDIDATE_COUNT},
        )
    )
    return rows


def _ambiguity_code_rows(
    *,
    selected_blob: dict[str, Any],
    selected_ridges: tuple[float, ...],
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    control_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    train_source_top1, train_source_top2 = _top2_from_scores(surfaces["tiny_train_scores"])
    eval_source_top1, eval_source_top2 = _top2_from_scores(surfaces["tiny_eval_scores"])
    train_code = _pack_ambiguity_code(
        source_top1=train_source_top1,
        source_top2=train_source_top2,
        atom_slot=selected_blob["train_code"] // CANDIDATE_COUNT,
    )
    eval_code = _pack_ambiguity_code(
        source_top1=eval_source_top1,
        source_top2=eval_source_top2,
        atom_slot=selected_blob["eval_code"] // CANDIDATE_COUNT,
    )
    train_features, train_actions, train_diag = _ambiguity_action_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        ambiguity_code=train_code,
        target_atom_values=selected_blob["train_target_atoms"][:, :, :AMBIGUITY_ATOM_SLOTS],
    )
    eval_features, eval_actions, eval_diag = _ambiguity_action_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        ambiguity_code=eval_code,
        target_atom_values=selected_blob["eval_target_atoms"][:, :, :AMBIGUITY_ATOM_SLOTS],
    )
    model, config_rows = _fit_ambiguity_router(
        train_features=train_features,
        train_actions=train_actions,
        train_baseline=calibration["tiny_packet"],
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=selected_ridges,
    )
    predictions = _predict_ambiguity_router(
        features=eval_features,
        actions=eval_actions,
        baseline=validation["packet"],
        model=model,
    )
    rows: list[dict[str, Any]] = [
        _score_predictions(
            name="receiver_calibrated_top2_ambiguity_code",
            predictions=predictions,
            validation=validation,
            seed=60100,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": str(selected_blob["config"]["name"]),
                "ridge": float(model["ridge"]),
                "threshold": float(model["threshold"]),
                "raw_payload_bytes": RAW_PACKET_BYTES,
                "framed_record_bytes": FRAMED_PACKET_BYTES,
                "atom_slots": AMBIGUITY_ATOM_SLOTS,
                "source_top2_visible": True,
            },
        )
    ]
    rng = np.random.default_rng(control_seed + 808)

    def predict_control(name: str, code: np.ndarray, *, seed: int) -> None:
        features, actions, _ = _ambiguity_action_features(
            qwen_scores=validation["qwen_scores"],
            qwen_target=validation["alternatives"]["qwen_target_score"],
            qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
            qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            ambiguity_code=np.asarray(code, dtype=np.int64),
            target_atom_values=selected_blob["eval_target_atoms"][:, :, :AMBIGUITY_ATOM_SLOTS],
        )
        rows.append(
            _score_predictions(
                name=name,
                predictions=_predict_ambiguity_router(
                    features=features,
                    actions=actions,
                    baseline=validation["packet"],
                    model=model,
                ),
                validation=validation,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(selected_blob["config"]["name"]),
                    "ridge": float(model["ridge"]),
                    "threshold": float(model["threshold"]),
                    "raw_payload_bytes": RAW_PACKET_BYTES,
                    "framed_record_bytes": FRAMED_PACKET_BYTES,
                },
            )
        )

    source_top1, source_top2, atom_slot = _decode_ambiguity_code(eval_code)
    no_atom_code = _pack_ambiguity_code(
        source_top1=source_top1,
        source_top2=source_top2,
        atom_slot=np.zeros_like(atom_slot),
    )
    predict_control("source_pair_no_atom_ambiguity_decoder", no_atom_code, seed=60101)
    qwen_top1, qwen_top2 = _top2_from_scores(validation["qwen_scores"])
    target_derived = _pack_ambiguity_code(
        source_top1=qwen_top1,
        source_top2=qwen_top2,
        atom_slot=np.zeros_like(atom_slot),
    )
    predict_control("target_derived_source_pair_ambiguity_control", target_derived, seed=60102)
    order = rng.permutation(len(eval_code))
    wrong_row = _pack_ambiguity_code(
        source_top1=source_top1[order],
        source_top2=source_top2[order],
        atom_slot=atom_slot[order],
    )
    predict_control("wrong_row_ambiguity_code_control", wrong_row, seed=60103)
    candidate_roll = _pack_ambiguity_code(
        source_top1=(source_top1 + 1) % CANDIDATE_COUNT,
        source_top2=(source_top2 + 1) % CANDIDATE_COUNT,
        atom_slot=atom_slot,
    )
    predict_control("candidate_roll_ambiguity_code_control", candidate_roll, seed=60104)
    perm = np.arange(AMBIGUITY_ATOM_SLOTS + 1, dtype=np.int64)
    perm[1:] = rng.permutation(np.arange(1, AMBIGUITY_ATOM_SLOTS + 1, dtype=np.int64))
    atom_permuted = _pack_ambiguity_code(
        source_top1=source_top1,
        source_top2=source_top2,
        atom_slot=perm[np.minimum(atom_slot, AMBIGUITY_ATOM_SLOTS)],
    )
    predict_control("atom_slot_permutation_ambiguity_control", atom_permuted, seed=60105)
    zero_source = _pack_ambiguity_code(
        source_top1=np.zeros_like(source_top1),
        source_top2=np.ones_like(source_top2),
        atom_slot=np.zeros_like(atom_slot),
    )
    predict_control("zero_source_ambiguity_control", zero_source, seed=60106)
    random_same_byte = rng.integers(0, 256, size=len(eval_code), dtype=np.int64)
    predict_control("random_same_byte_ambiguity_control", random_same_byte, seed=60107)

    label_model, label_config = _fit_ambiguity_router(
        train_features=train_features,
        train_actions=train_actions,
        train_baseline=calibration["tiny_packet"],
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=selected_ridges,
        label_permutation_seed=control_seed + 909,
    )
    rows.append(
        _score_predictions(
            name="label_permutation_ambiguity_decoder_control",
            predictions=_predict_ambiguity_router(
                features=eval_features,
                actions=eval_actions,
                baseline=validation["packet"],
                model=label_model,
            ),
            validation=validation,
            seed=60108,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": str(selected_blob["config"]["name"]),
                "ridge": float(label_model["ridge"]),
                "threshold": float(label_model["threshold"]),
            },
        )
    )
    rows.append(
        _score_predictions(
            name="source_top1_label_control",
            predictions=source_top1,
            validation=validation,
            seed=60109,
            bootstrap_samples=bootstrap_samples,
            extra={"source_rank": 1},
        )
    )
    rows.append(
        _score_predictions(
            name="source_top2_label_control",
            predictions=source_top2,
            validation=validation,
            seed=60110,
            bootstrap_samples=bootstrap_samples,
            extra={"source_rank": 2},
        )
    )
    rows.append(
        _score_predictions(
            name="source_top1_top2_oracle_diagnostic",
            predictions=_source_pair_oracle_predictions(source_top1, source_top2, validation["answers"]),
            validation=validation,
            seed=60111,
            bootstrap_samples=bootstrap_samples,
            extra={"oracle": True, "not_promotable": True},
        )
    )
    blocks = wz._block_rows(
        selected=predictions,
        packet=validation["packet"],
        answers=validation["answers"],
    )
    audit = {
        "model": {key: value for key, value in model.items() if key != "weights"},
        "label_permutation_model": {key: value for key, value in label_model.items() if key != "weights"},
        "source_pair_disagreement_rate": float(np.mean(eval_diag["source_top1"] != eval_diag["source_top2"])),
        "source_target_disagreement_rate": float(np.mean(eval_diag["source_top1"] != eval_diag["qwen_top1"])),
        "active_4bit_atom_rate": float(np.mean(eval_diag["atom_slot"] > 0)),
        "unique_ambiguity_codes": int(len(np.unique(eval_code))),
        "train_active_4bit_atom_rate": float(np.mean(train_diag["atom_slot"] > 0)),
        "label_permutation_config_rows": label_config,
    }
    return rows, config_rows, blocks, audit


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if isinstance(row[key], (str, int, float, bool)) or row[key] is None:
                if key not in keys:
                    keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_predictions_jsonl(
    path: pathlib.Path,
    *,
    row_ids: list[str],
    answers: np.ndarray,
    packet: np.ndarray,
    predictions: np.ndarray,
    selected_blob: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row_id in enumerate(row_ids):
            handle.write(
                json.dumps(
                    {
                        "row_index": int(index),
                        "row_id": str(row_id),
                        "answer_index": int(answers[index]),
                        "packet_prediction": int(packet[index]),
                        "decision_sparse_prediction": int(predictions[index]),
                        "source_code": int(selected_blob["eval_code"][index]),
                        "atom_slot": int(selected_blob["eval_atom_slot"][index]),
                        "full_atom_id": int(selected_blob["eval_full_atom_id"][index]),
                        "atom_value": float(selected_blob["eval_atom_value"][index]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Decision-Sparse Common-Basis Hidden Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval slice: `{h['eval_slice_start']}:{h['eval_slice_end_exclusive']}`",
        f"- default encoder: `{h['default_encoder_name']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- compact common-basis decoder accuracy: `{h['compact_common_basis_accuracy']:.6f}`",
        f"- Qwen-side-only common-basis accuracy: `{h['qwen_side_common_basis_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- top-atom knockout accuracy: `{h['top_atom_knockout_accuracy']:.6f}`",
        f"- top1/top2 ambiguity-code accuracy: `{h['ambiguity_code_accuracy']:.6f}`",
        f"- top1/top2 ambiguity-code delta vs packet-only: `{h['ambiguity_code_delta_vs_packet_only']:.6f}`",
        f"- top1/top2 ambiguity-code CI95 low: `{h['ambiguity_code_ci95_low_vs_packet_only']:.6f}`",
        f"- top1/top2 ambiguity-code no-atom accuracy: `{h['ambiguity_source_pair_no_atom_accuracy']:.6f}`",
        f"- top1/top2 ambiguity-code best destructive: `{h['ambiguity_best_destructive_control_name']}` (`{h['ambiguity_best_destructive_control_accuracy']:.6f}`)",
        f"- best scout accuracy: `{h['best_scout_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    eval_full_path: pathlib.Path = DEFAULT_EVAL_FULL,
    eval_slice_start: int = 1024,
    eval_slice_rows: int = 1024,
    eval_hidden_cache: pathlib.Path = DEFAULT_EVAL_HIDDEN_CACHE,
    pca_dims: tuple[int, ...] = DEFAULT_PCA_DIMS,
    shared_dims: tuple[int, ...] = DEFAULT_SHARED_DIMS,
    sae_atoms: tuple[int, ...] = DEFAULT_SAE_ATOMS,
    sae_topks: tuple[int, ...] = DEFAULT_SAE_TOPKS,
    sae_decision_weights: tuple[float, ...] = DEFAULT_SAE_DECISION_WEIGHTS,
    sae_l1_weights: tuple[float, ...] = DEFAULT_SAE_L1_WEIGHTS,
    sae_epochs: int = DEFAULT_SAE_EPOCHS,
    sae_learning_rate: float = DEFAULT_SAE_LEARNING_RATE,
    sae_batch_size: int = DEFAULT_SAE_BATCH_SIZE,
    decoder_ridges: tuple[float, ...] = DEFAULT_DECODER_RIDGES,
    bootstrap_samples: int = 500,
    control_seed: int = 5017,
    source_lm_model: str = DEFAULT_SOURCE_MODEL,
    source_lm_device: str = "mps",
    source_lm_dtype: str = "float16",
    source_lm_max_length: int = 256,
    source_lm_prompt_mode: str = "continuation",
    hidden_layers: tuple[int, ...] = (-1,),
    local_files_only: bool = True,
    run_date: str = "2026-05-04",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    slice_path = output_dir / f"hellaswag_validation_rows_{eval_slice_start}_{eval_slice_start + eval_slice_rows}.jsonl"
    slice_meta = hidden_code._slice_jsonl(
        source_path=eval_full_path,
        output_path=slice_path,
        start=eval_slice_start,
        count=eval_slice_rows,
    )
    eval_rows = arc_gate._load_rows(slice_path)
    hidden_npz = _resolve(eval_hidden_cache)
    hidden_meta = hidden_npz.with_suffix(".json")
    eval_hidden, eval_hidden_model = hidden_summary._source_hidden_features(
        eval_rows,
        npz_path=hidden_npz,
        meta_path=hidden_meta,
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        prompt_mode=source_lm_prompt_mode,
        layers=hidden_layers,
        local_files_only=local_files_only,
    )
    surfaces_full = wz._load_surfaces(
        train_path=wz.DEFAULT_TRAIN_PATH,
        tiny_train_cache_dir=wz.DEFAULT_TINY_TRAIN_CACHE_DIR,
        qwen_train_cache_dir=wz.DEFAULT_QWEN_TRAIN_CACHE_DIR,
        sample_seeds=wz.DEFAULT_SAMPLE_SEEDS,
        split_seeds=wz.DEFAULT_SPLIT_SEEDS,
        ridges=wz.DEFAULT_RIDGES,
        train_hidden_rows=512,
        dev_fraction=0.25,
        tiny_eval_packet_jsonl=wz.DEFAULT_TINY_EVAL_PACKET_JSONL,
        qwen_eval_packet_jsonl=wz.DEFAULT_QWEN_EVAL_PACKET_JSONL,
        qwen_global_artifact=wz.DEFAULT_QWEN_GLOBAL_ARTIFACT,
        tiny_eval_rows=wz.DEFAULT_TINY_EVAL_ROWS,
        tiny_eval_score_cache=wz.DEFAULT_TINY_EVAL_SCORE_CACHE,
        tiny_aggregation_policy="mean_zscore",
    )
    validation_full = surfaces_full["validation"]
    validation_row_ids = validation_full["row_ids"][eval_slice_start : eval_slice_start + eval_slice_rows]
    if [str(row.row_id) for row in eval_rows] != [str(row_id) for row_id in validation_row_ids]:
        raise ValueError("eval hidden slice rows do not align with validation packet bundle")
    validation_slice = {
        **validation_full,
        "rows": validation_full["rows"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "row_ids": validation_row_ids,
        "answers": validation_full["answers"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "packet": validation_full["packet"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "packet_margin": validation_full["packet_margin"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "qwen_scores": validation_full["qwen_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "qwen_hidden": validation_full["qwen_hidden"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "alternatives": {
            key: value[eval_slice_start : eval_slice_start + eval_slice_rows]
            for key, value in validation_full["alternatives"].items()
        },
    }
    surfaces = {**surfaces_full, "validation": validation_slice}
    calibration = surfaces["calibration"]
    train_hidden, train_hidden_audit = hidden_code._tiny_train_hidden_matrix(
        calibration_rows=calibration["rows"],
        train_path=wz.DEFAULT_TRAIN_PATH,
        tiny_train_cache_dir=wz.DEFAULT_TINY_TRAIN_CACHE_DIR,
        sample_seeds=wz.DEFAULT_SAMPLE_SEEDS,
        train_hidden_rows=512,
    )
    train_source_candidates = anchor_code._candidate_hidden_feature_tensor(
        hidden=train_hidden,
        scores=surfaces["tiny_train_scores"],
        reference_prediction=calibration["tiny_packet"],
    )
    eval_source_candidates = anchor_code._candidate_hidden_feature_tensor(
        hidden=eval_hidden,
        scores=surfaces["tiny_eval_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        reference_prediction=validation_slice["packet"],
    )
    train_target_candidates = anchor_code._candidate_hidden_feature_tensor(
        hidden=calibration["qwen_hidden"],
        scores=calibration["qwen_scores"],
        reference_prediction=calibration["qwen_target"],
    )
    eval_target_candidates = anchor_code._candidate_hidden_feature_tensor(
        hidden=validation_slice["qwen_hidden"],
        scores=validation_slice["qwen_scores"],
        reference_prediction=validation_slice["alternatives"]["qwen_target_score"],
    )
    frontier_rows: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[str, float], np.ndarray] = {}
    encoded_by_name: dict[str, dict[str, Any]] = {}
    crosscoder_audits: list[dict[str, Any]] = []
    config_counter = 0
    for pca_dim in pca_dims:
        for shared_dim in shared_dims:
            if int(shared_dim) > int(pca_dim):
                continue
            crosscoder = crosscode._fit_linear_crosscoder(
                source_candidate_features=train_source_candidates,
                target_candidate_features=train_target_candidates,
                fit_indices=surfaces["fit_indices"],
                pca_dims=int(pca_dim),
                shared_dims=int(shared_dim),
            )
            train_source_shared, train_target_shared = crosscode._apply_linear_crosscoder(
                source_candidate_features=train_source_candidates,
                target_candidate_features=train_target_candidates,
                crosscoder=crosscoder,
            )
            eval_source_shared, eval_target_shared = crosscode._apply_linear_crosscoder(
                source_candidate_features=eval_source_candidates,
                target_candidate_features=eval_target_candidates,
                crosscoder=crosscoder,
            )
            crosscoder_audits.append(
                {
                    "pca_dims": int(pca_dim),
                    "shared_dims": int(crosscoder["shared_dims"]),
                    "singular_values": [float(item) for item in crosscoder["singular_values"]],
                }
            )
            for atoms in sae_atoms:
                for topk in sae_topks:
                    for decision_weight in sae_decision_weights:
                        for l1_weight in sae_l1_weights:
                            seed = (
                                50000
                                + int(pca_dim) * 17
                                + int(shared_dim) * 31
                                + int(atoms) * 7
                                + int(topk) * 13
                                + config_counter
                            )
                            config = {
                                "name": (
                                    f"cca_pca{int(pca_dim)}_d{int(shared_dim)}_"
                                    f"sae{int(atoms)}_top{int(topk)}_"
                                    f"dw{_safe_float_label(float(decision_weight))}_"
                                    f"l1{_safe_float_label(float(l1_weight))}"
                                ),
                                "pca_dims": int(pca_dim),
                                "shared_dims": int(shared_dim),
                                "sae_atoms": int(atoms),
                                "sae_topk": int(topk),
                                "sae_decision_weight": float(decision_weight),
                                "sae_l1_weight": float(l1_weight),
                                "sae_learning_rate": float(sae_learning_rate),
                                "sae_epochs": int(sae_epochs),
                                "sae_batch_size": int(sae_batch_size),
                                "seed": int(seed),
                            }
                            rows, predictions, encoded = _evaluate_sparse_config(
                                config=config,
                                surfaces=surfaces,
                                train_source_shared=train_source_shared,
                                eval_source_shared=eval_source_shared,
                                train_target_shared=train_target_shared,
                                eval_target_shared=eval_target_shared,
                                decoder_ridges=decoder_ridges,
                                bootstrap_samples=bootstrap_samples,
                                row_seed_offset=51000 + config_counter * 100,
                            )
                            frontier_rows.extend(rows)
                            predictions_by_key.update(predictions)
                            encoded_by_name[str(config["name"])] = encoded
                            config_counter += 1
    if not frontier_rows:
        raise ValueError("no decision-sparse common-basis rows were evaluated")
    default_row = max(
        frontier_rows,
        key=lambda row: (
            row["official_dev_accuracy"],
            row["official_dev_delta_vs_packet"],
            -row["codebook_size"],
            -math.log10(float(row["ridge"])),
        ),
    )
    best_scout = max(
        frontier_rows,
        key=lambda row: (
            row["delta_vs_packet_only"],
            row["ci95_low_vs_packet_only"],
            row["accuracy"],
            row["official_dev_accuracy"],
        ),
    )
    selected_blob = encoded_by_name[str(default_row["encoder_name"])]
    default_predictions = predictions_by_key[(str(default_row["encoder_name"]), float(default_row["ridge"]))]
    default_blocks = wz._block_rows(
        selected=default_predictions,
        packet=validation_slice["packet"],
        answers=validation_slice["answers"],
    )
    control_rows = _control_rows(
        selected_blob=selected_blob,
        selected_ridge=float(default_row["ridge"]),
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    ambiguity_rows, ambiguity_config_rows, ambiguity_blocks, ambiguity_audit = _ambiguity_code_rows(
        selected_blob=selected_blob,
        selected_ridges=decoder_ridges,
        surfaces={
            **surfaces,
            "tiny_eval_scores": surfaces["tiny_eval_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        },
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_by_name = {row["name"]: row for row in control_rows}
    ambiguity_by_name = {row["name"]: row for row in ambiguity_rows}
    ambiguity_default = ambiguity_by_name["receiver_calibrated_top2_ambiguity_code"]
    ambiguity_no_atom = ambiguity_by_name["source_pair_no_atom_ambiguity_decoder"]
    ambiguity_top_atom_knockout = ambiguity_no_atom
    ambiguity_destructive_controls = [
        row
        for row in ambiguity_rows
        if row["name"].endswith("_control")
        or row["name"] in {"label_permutation_ambiguity_decoder_control"}
    ]
    ambiguity_best_destructive = max(ambiguity_destructive_controls, key=lambda row: row["accuracy"])
    ambiguity_block_stability = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in ambiguity_blocks) >= 4)
    ambiguity_control_gate = bool(
        ambiguity_default["accuracy"] - ambiguity_best_destructive["accuracy"] >= BASELINE_DELTA
        and ambiguity_best_destructive["delta_vs_packet_only"] <= CONTROL_TOLERANCE
    )
    ambiguity_atom_gate = bool(
        ambiguity_default["accuracy"] - ambiguity_top_atom_knockout["accuracy"] >= 0.005
    )
    ambiguity_no_atom_gate = bool(
        ambiguity_default["accuracy"] - ambiguity_no_atom["accuracy"] >= BASELINE_DELTA
    )
    ambiguity_pass_gate = bool(
        ambiguity_default["delta_vs_packet_only"] >= STRICT_DELTA
        and ambiguity_default["ci95_low_vs_packet_only"] > 0.0
        and ambiguity_no_atom_gate
        and ambiguity_atom_gate
        and ambiguity_block_stability
        and ambiguity_control_gate
    )
    compact_accuracy = control_by_name["compact_candidate_common_basis_decoder"]["accuracy"]
    qwen_side_accuracy = control_by_name["qwen_side_only_common_basis_decoder"]["accuracy"]
    top_atom_knockout_accuracy = control_by_name["top_atom_knockout"]["accuracy"]
    destructive_controls = [
        row
        for row in control_rows
        if row["name"]
        not in {
            "packet_only",
            "compact_candidate_common_basis_decoder",
            "qwen_side_only_common_basis_decoder",
        }
    ]
    control_max_delta = max(row["delta_vs_packet_only"] for row in destructive_controls)
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    compact_gate = bool(default_row["accuracy"] - compact_accuracy >= BASELINE_DELTA)
    qwen_side_gate = bool(default_row["accuracy"] - qwen_side_accuracy >= BASELINE_DELTA)
    top_atom_gate = bool(default_row["accuracy"] - top_atom_knockout_accuracy >= CONTROL_SEPARATION_DELTA)
    pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and compact_gate
        and qwen_side_gate
        and top_atom_gate
        and block_stability_gate
        and control_separation_gate
    )
    packet_only_accuracy = wz._accuracy(validation_slice["packet"], validation_slice["answers"])
    headline = {
        "eval_slice_start": int(eval_slice_start),
        "eval_slice_end_exclusive": int(eval_slice_start + eval_slice_rows),
        "official_train_calibration_rows": int(len(calibration["answers"])),
        "official_train_fit_rows": int(len(surfaces["fit_indices"])),
        "official_train_dev_rows": int(len(surfaces["dev_indices"])),
        "validation_rows": int(len(validation_slice["answers"])),
        "packet_only_accuracy": packet_only_accuracy,
        "qwen_target_accuracy": wz._accuracy(
            validation_slice["alternatives"]["qwen_target_score"],
            validation_slice["answers"],
        ),
        "compact_common_basis_accuracy": compact_accuracy,
        "qwen_side_common_basis_accuracy": qwen_side_accuracy,
        "top_atom_knockout_accuracy": top_atom_knockout_accuracy,
        "default_encoder_name": str(default_row["encoder_name"]),
        "default_pca_dims": int(default_row["pca_dims"]),
        "default_shared_dims": int(default_row["shared_dims"]),
        "default_sae_atoms": int(default_row["sae_atoms"]),
        "default_sae_transmit_atoms": int(default_row["sae_transmit_atoms"]),
        "default_sae_topk": int(default_row["sae_topk"]),
        "default_codebook_size": int(default_row["codebook_size"]),
        "default_eval_code_unique_count": int(default_row["eval_code_unique_count"]),
        "default_eval_active_atom_rate": float(default_row["eval_active_atom_rate"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_compact_common_basis": float(default_row["accuracy"] - compact_accuracy),
        "default_delta_vs_qwen_side_common_basis": float(default_row["accuracy"] - qwen_side_accuracy),
        "default_delta_vs_top_atom_knockout": float(default_row["accuracy"] - top_atom_knockout_accuracy),
        "default_ridge": default_row["ridge"],
        "best_scout_encoder_name": str(best_scout["encoder_name"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "compact_common_basis_gate": compact_gate,
        "qwen_side_gate": qwen_side_gate,
        "top_atom_gate": top_atom_gate,
        "raw_payload_bytes": RAW_PACKET_BYTES,
        "framed_record_bytes": FRAMED_PACKET_BYTES,
        "default_pass_gate": pass_gate,
        "scout_pass_gate": bool(
            best_scout["delta_vs_packet_only"] >= STRICT_DELTA
            and best_scout["ci95_low_vs_packet_only"] > 0.0
        ),
        "ambiguity_code_accuracy": ambiguity_default["accuracy"],
        "ambiguity_code_delta_vs_packet_only": ambiguity_default["delta_vs_packet_only"],
        "ambiguity_code_ci95_low_vs_packet_only": ambiguity_default["ci95_low_vs_packet_only"],
        "ambiguity_code_help_count": ambiguity_default["help_count"],
        "ambiguity_code_harm_count": ambiguity_default["harm_count"],
        "ambiguity_source_pair_no_atom_accuracy": ambiguity_no_atom["accuracy"],
        "ambiguity_delta_vs_no_atom_pair_code": float(
            ambiguity_default["accuracy"] - ambiguity_no_atom["accuracy"]
        ),
        "ambiguity_top_atom_knockout_accuracy": ambiguity_top_atom_knockout["accuracy"],
        "ambiguity_best_destructive_control_name": ambiguity_best_destructive["name"],
        "ambiguity_best_destructive_control_accuracy": ambiguity_best_destructive["accuracy"],
        "ambiguity_control_gate": ambiguity_control_gate,
        "ambiguity_no_atom_gate": ambiguity_no_atom_gate,
        "ambiguity_atom_gate": ambiguity_atom_gate,
        "ambiguity_block_stability_gate": ambiguity_block_stability,
        "ambiguity_pass_gate": ambiguity_pass_gate,
        "ambiguity_unique_codes": ambiguity_audit["unique_ambiguity_codes"],
        "ambiguity_active_4bit_atom_rate": ambiguity_audit["active_4bit_atom_rate"],
        "source_hidden_cache_hit": bool(eval_hidden_model.get("cache_hit")),
        "source_hidden_extraction_wall_time_s": float(eval_hidden_model.get("latency_s") or 0.0),
    }
    payload = {
        "gate": "source_private_hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Gate passes if the official-train-dev-selected decision-supervised sparse common-basis "
            "packet beats packet-only by >=0.010 with positive paired CI95 low, beats compact and "
            "Qwen-side-only common-basis decoders by >=0.010, degrades under top-atom knockout, "
            "is positive on at least 4/5 blocks, and separates from destructive atom/source controls."
        ),
        "packet_contract": {
            "packet_name": "decision_sparse_common_basis_hidden_innovation_packet",
            "raw_payload_bytes": RAW_PACKET_BYTES,
            "framed_record_bytes": FRAMED_PACKET_BYTES,
            "max_codebook_size": MAX_CODEBOOK_SIZE,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "learned_discrete_source_hidden_atom_transmitted": True,
            "candidate_low_bits_preserved": True,
            "decoder_uses_qwen_side_information": True,
            "decoder_uses_qwen_common_basis_atom_features": True,
            "receiver_calibrated_top2_ambiguity_packet": {
                "raw_payload_bytes": RAW_PACKET_BYTES,
                "framed_record_bytes": FRAMED_PACKET_BYTES,
                "layout": "2 bits source top1, 2 bits source top2, 4 bits sparse atom slot",
                "max_sparse_atom_slots": AMBIGUITY_ATOM_SLOTS,
                "receiver_actions": list(AMBIGUITY_ACTION_NAMES),
                "raw_scores_transmitted": False,
                "raw_hidden_vector_transmitted": False,
            },
        },
        "headline": headline,
        "frontier_rows": frontier_rows,
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "ambiguity_rows": ambiguity_rows,
        "ambiguity_config_rows": ambiguity_config_rows,
        "ambiguity_blocks": ambiguity_blocks,
        "ambiguity_audit": ambiguity_audit,
        "selected_encoder_audit": selected_blob["encoder_audit"],
        "crosscoder_audits": crosscoder_audits,
        "slice_metadata": slice_meta,
        "train_hidden_audit": train_hidden_audit,
        "eval_hidden_model": eval_hidden_model,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(len(validation_slice["answers"]) * RAW_PACKET_BYTES),
            "logical_validation_framed_record_bytes_total": int(len(validation_slice["answers"]) * FRAMED_PACKET_BYTES),
            "communication_object": "task_level_decision_sparse_common_basis_hidden_packet",
            "communication_objective": "downstream_candidate_decision_accuracy",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "total_wall_time_s": float(time.perf_counter() - started),
        },
        "inputs": {
            "eval_full_path": _display_path(eval_full_path),
            "eval_slice_path": _display_path(slice_path),
            "eval_hidden_cache": _display_path(hidden_npz),
            "eval_hidden_cache_sha256": _sha256_file(hidden_npz),
            "source_model": source_lm_model,
            "train_path": _display_path(wz.DEFAULT_TRAIN_PATH),
            "tiny_train_cache_dir": _display_path(wz.DEFAULT_TINY_TRAIN_CACHE_DIR),
            "qwen_global_artifact": _display_path(wz.DEFAULT_QWEN_GLOBAL_ARTIFACT),
        },
        "interpretation": (
            "This gate tests the highest-priority learned branch after unsupervised hidden-code and "
            "linear crosscoder codebooks failed: train a sparse SAE-like basis in linear CCA/common "
            "coordinates with a decision loss, transmit only a one-byte atom-plus-candidate packet, "
            "and require the atom to matter under atom shuffle, wrong-row source, and top-atom knockout "
            "controls. A pass would promote the common-basis packet branch; a fail weakens shallow SAE/"
            "linear common-basis methods and points to nonlinear resamplers or less saturated benchmarks."
        ),
    }
    json_path = output_dir / "hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.json"
    md_path = output_dir / "hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.md"
    frontier_csv = output_dir / "frontier_rows.csv"
    control_csv = output_dir / "control_rows.csv"
    block_csv = output_dir / "default_blocks.csv"
    ambiguity_csv = output_dir / "ambiguity_rows.csv"
    ambiguity_config_csv = output_dir / "ambiguity_config_rows.csv"
    ambiguity_block_csv = output_dir / "ambiguity_blocks.csv"
    predictions_jsonl = output_dir / "predictions.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(frontier_csv, frontier_rows)
    _write_csv(control_csv, control_rows)
    _write_csv(block_csv, default_blocks)
    _write_csv(ambiguity_csv, ambiguity_rows)
    _write_csv(ambiguity_config_csv, ambiguity_config_rows)
    _write_csv(ambiguity_block_csv, ambiguity_blocks)
    _write_predictions_jsonl(
        predictions_jsonl,
        row_ids=[str(row_id) for row_id in validation_slice["row_ids"]],
        answers=validation_slice["answers"],
        packet=validation_slice["packet"],
        predictions=default_predictions,
        selected_blob=selected_blob,
    )
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "inputs": payload["inputs"],
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                json_path,
                md_path,
                frontier_csv,
                control_csv,
                block_csv,
                ambiguity_csv,
                ambiguity_config_csv,
                ambiguity_block_csv,
                predictions_jsonl,
            )
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-full-path", type=pathlib.Path, default=DEFAULT_EVAL_FULL)
    parser.add_argument("--eval-slice-start", type=int, default=1024)
    parser.add_argument("--eval-slice-rows", type=int, default=1024)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--pca-dims", type=_parse_int_tuple, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--shared-dims", type=_parse_int_tuple, default=DEFAULT_SHARED_DIMS)
    parser.add_argument("--sae-atoms", type=_parse_int_tuple, default=DEFAULT_SAE_ATOMS)
    parser.add_argument("--sae-topks", type=_parse_int_tuple, default=DEFAULT_SAE_TOPKS)
    parser.add_argument("--sae-decision-weights", type=_parse_float_tuple, default=DEFAULT_SAE_DECISION_WEIGHTS)
    parser.add_argument("--sae-l1-weights", type=_parse_float_tuple, default=DEFAULT_SAE_L1_WEIGHTS)
    parser.add_argument("--sae-epochs", type=int, default=DEFAULT_SAE_EPOCHS)
    parser.add_argument("--sae-learning-rate", type=float, default=DEFAULT_SAE_LEARNING_RATE)
    parser.add_argument("--sae-batch-size", type=int, default=DEFAULT_SAE_BATCH_SIZE)
    parser.add_argument("--decoder-ridges", type=_parse_float_tuple, default=DEFAULT_DECODER_RIDGES)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=5017)
    parser.add_argument("--source-lm-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-lm-device", default="mps")
    parser.add_argument("--source-lm-dtype", default="float16")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-prompt-mode", default="continuation")
    parser.add_argument("--hidden-layers", type=_parse_int_tuple, default=(-1,))
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--run-date", default="2026-05-04")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        eval_full_path=args.eval_full_path,
        eval_slice_start=args.eval_slice_start,
        eval_slice_rows=args.eval_slice_rows,
        eval_hidden_cache=args.eval_hidden_cache,
        pca_dims=args.pca_dims,
        shared_dims=args.shared_dims,
        sae_atoms=args.sae_atoms,
        sae_topks=args.sae_topks,
        sae_decision_weights=args.sae_decision_weights,
        sae_l1_weights=args.sae_l1_weights,
        sae_epochs=args.sae_epochs,
        sae_learning_rate=args.sae_learning_rate,
        sae_batch_size=args.sae_batch_size,
        decoder_ridges=args.decoder_ridges,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        hidden_layers=args.hidden_layers,
        local_files_only=not args.allow_downloads,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
