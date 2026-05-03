from __future__ import annotations

"""ARC receiver-conditioned source-innovation packet gate.

This is a cache-only preflight gate.  It tests whether a fixed-byte packet that
describes source-vs-receiver candidate-score innovation adds held-out decision
information beyond source-label relaying and raw source-score packets.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_conditional_innovation_packet_gate_20260503")
DEFAULT_EVAL_PATH = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
STRICT_SOURCE_DELTA = 0.020
STRICT_CONTROL_DELTA = 0.010


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    path = pathlib.Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(rows: list[arc_gate.ArcRow]) -> str:
    return hashlib.sha256("\n".join(row.content_id for row in rows).encode("utf-8")).hexdigest()


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _load_score_cache(
    path: pathlib.Path | str,
    *,
    rows: list[arc_gate.ArcRow],
    role: str,
) -> tuple[list[list[float]], list[int], dict[str, Any], str]:
    resolved = _resolve(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    scores = [[float(score) for score in row] for row in payload["source_scores"]]
    predictions = [int(index) for index in payload["source_predictions"]]
    if len(scores) != len(rows) or len(predictions) != len(rows):
        raise ValueError(f"{role} score cache row count does not match eval rows")
    row_ids = [str(row_id) for row_id in payload.get("row_ids", [])]
    if row_ids and row_ids != [row.row_id for row in rows]:
        raise ValueError(f"{role} score cache row_ids do not match eval rows")
    digest = str(payload.get("content_digest", ""))
    if digest and digest != _content_digest(rows):
        raise ValueError(f"{role} score cache content digest does not match eval rows")
    for index, (row, score_row, prediction) in enumerate(zip(rows, scores, predictions, strict=True)):
        if len(score_row) != len(row.choices):
            raise ValueError(f"{role} score row {index} has {len(score_row)} scores for {len(row.choices)} choices")
        if not 0 <= prediction < len(row.choices):
            raise ValueError(f"{role} prediction row {index} is out of range")
    return scores, predictions, dict(payload.get("source_model", {})), _sha256_file(resolved)


def _ranked_indices(scores: list[float] | np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (float(scores[index]), -index), reverse=True)


def _top_predictions(scores_by_row: list[list[float]]) -> list[int]:
    return [_ranked_indices(scores)[0] for scores in scores_by_row]


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int], indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(sum(predictions[index] == rows[index].answer_index for index in indices) / len(indices))


def _row_zscores(scores: list[float] | np.ndarray) -> list[float]:
    finite = [float(score) for score in scores if math.isfinite(float(score))]
    if not finite:
        return [0.0 for _ in scores]
    mean = statistics.fmean(finite)
    variance = statistics.fmean((score - mean) ** 2 for score in finite)
    std = math.sqrt(max(variance, 1e-12))
    return [0.0 if not math.isfinite(float(score)) else (float(score) - mean) / std for score in scores]


def _zscore_rows(scores_by_row: list[list[float]]) -> list[list[float]]:
    return [_row_zscores(row) for row in scores_by_row]


def _quantize_4bit(values: list[float] | np.ndarray, *, clip: float) -> list[float]:
    if clip <= 0.0:
        raise ValueError("clip must be positive")
    quantized: list[float] = []
    for value in values:
        clipped = min(float(clip), max(-float(clip), float(value)))
        level = round((clipped + float(clip)) * 15.0 / (2.0 * float(clip)))
        level = max(0, min(15, int(level)))
        quantized.append((level * (2.0 * float(clip)) / 15.0) - float(clip))
    return quantized


def _softmax(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - float(np.max(array))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    return (exp / max(denom, 1e-12)).tolist()


def _rank_by_row(rows: list[list[float]]) -> list[list[int]]:
    ranks_by_row: list[list[int]] = []
    for scores in rows:
        ranked = _ranked_indices(scores)
        ranks = [0 for _ in scores]
        for rank, candidate_index in enumerate(ranked):
            ranks[candidate_index] = rank
        ranks_by_row.append(ranks)
    return ranks_by_row


def _packet_bits(max_choices: int) -> int:
    source_index_bits = int(math.ceil(math.log2(max(2, max_choices))))
    innovation_bits = 4 * int(max_choices)
    return source_index_bits + innovation_bits


def _condition_source_rows(
    *,
    source_zscores: list[list[float]],
    receiver_zscores: list[list[float]],
    source_predictions: list[int],
    condition: str,
    shuffle_seed: int,
) -> tuple[list[list[float]], list[int]]:
    if condition in {
        "matched_conditional_innovation_packet",
        "innovation_plus_source_index_packet",
        "quantized_source_score_packet",
        "source_index_only_decoder",
    }:
        return source_zscores, source_predictions
    if condition == "zero_innovation_control":
        return receiver_zscores, _top_predictions(receiver_zscores)
    if condition == "row_shuffle_innovation_control":
        rng = np.random.default_rng(shuffle_seed)
        permutation = list(range(len(source_zscores)))
        by_count: dict[int, list[int]] = {}
        for index, row in enumerate(source_zscores):
            by_count.setdefault(len(row), []).append(index)
        for grouped_indices in by_count.values():
            shuffled = rng.permutation(grouped_indices).tolist()
            for original_index, shuffled_index in zip(grouped_indices, shuffled, strict=True):
                permutation[original_index] = shuffled_index
        return [source_zscores[index] for index in permutation], [source_predictions[index] for index in permutation]
    if condition == "candidate_roll_innovation_control":
        rolled_scores: list[list[float]] = []
        rolled_predictions: list[int] = []
        for scores, prediction in zip(source_zscores, source_predictions, strict=True):
            count = len(scores)
            rolled_scores.append([float(scores[(candidate - 1) % count]) for candidate in range(count)])
            rolled_predictions.append((int(prediction) + 1) % count)
        return rolled_scores, rolled_predictions
    if condition == "receiver_only_decoder":
        return receiver_zscores, _top_predictions(receiver_zscores)
    raise ValueError(f"unsupported condition: {condition}")


def _feature_vector(
    *,
    row_index: int,
    candidate_index: int,
    candidate_count: int,
    max_choices: int,
    receiver_zscores: list[list[float]],
    receiver_ranks: list[list[int]],
    source_rows: list[list[float]],
    source_predictions: list[int],
    condition: str,
    clip: float,
) -> list[float]:
    receiver_row = receiver_zscores[row_index]
    receiver_probs = _softmax(receiver_row)
    receiver_rank = receiver_ranks[row_index][candidate_index] / max(1.0, float(candidate_count - 1))
    receiver_top = float(candidate_index == _ranked_indices(receiver_row)[0])
    other_receiver = [
        receiver_row[other_index]
        for other_index in range(candidate_count)
        if other_index != candidate_index
    ]
    receiver_margin = float(receiver_row[candidate_index] - max(other_receiver)) if other_receiver else 0.0
    candidate_scaled = candidate_index / max(1.0, float(max_choices - 1))
    candidate_eye = [1.0 if slot == candidate_index else 0.0 for slot in range(max_choices)]
    features = [
        1.0,
        float(receiver_row[candidate_index]),
        float(receiver_probs[candidate_index]),
        float(receiver_rank),
        receiver_top,
        receiver_margin,
        candidate_scaled,
    ]
    features.extend(candidate_eye)
    if condition == "receiver_only_decoder":
        return features

    source_row = source_rows[row_index]
    source_prediction = int(source_predictions[row_index])
    source_is_top = float(candidate_index == source_prediction)
    source_agrees_receiver = float(source_prediction == _ranked_indices(receiver_row)[0])
    if condition == "source_index_only_decoder":
        features.extend(
            [
                source_is_top,
                source_agrees_receiver,
                source_is_top * float(receiver_row[candidate_index]),
                source_is_top * receiver_margin,
            ]
        )
        return features

    if condition == "zero_innovation_control":
        innovation_q = 0.0
        reconstructed_source = float(receiver_row[candidate_index])
        reconstructed_row = list(receiver_row)
    elif condition == "quantized_source_score_packet":
        source_q_row = _quantize_4bit(source_row, clip=clip)
        innovation_q = float(source_q_row[candidate_index] - receiver_row[candidate_index])
        reconstructed_source = float(source_q_row[candidate_index])
        reconstructed_row = source_q_row
    else:
        innovation_row = [
            float(source_score) - float(receiver_score)
            for source_score, receiver_score in zip(source_row, receiver_row, strict=True)
        ]
        innovation_q_row = _quantize_4bit(innovation_row, clip=clip)
        innovation_q = float(innovation_q_row[candidate_index])
        reconstructed_source = float(receiver_row[candidate_index] + innovation_q)
        reconstructed_row = [
            float(receiver_score) + float(innovation_score)
            for receiver_score, innovation_score in zip(receiver_row, innovation_q_row, strict=True)
        ]
    reconstructed_ranks = _rank_by_row([reconstructed_row])[0]
    reconstructed_probs = _softmax(reconstructed_row)
    reconstructed_rank = reconstructed_ranks[candidate_index] / max(1.0, float(candidate_count - 1))
    reconstructed_top = float(candidate_index == _ranked_indices(reconstructed_row)[0])
    other_reconstructed = [
        reconstructed_row[other_index]
        for other_index in range(candidate_count)
        if other_index != candidate_index
    ]
    reconstructed_margin = (
        float(reconstructed_row[candidate_index] - max(other_reconstructed)) if other_reconstructed else 0.0
    )
    features.extend(
        [
            innovation_q,
            abs(innovation_q),
            1.0 if innovation_q > 0.0 else 0.0,
            1.0 if innovation_q < 0.0 else 0.0,
            float(receiver_row[candidate_index]) * innovation_q,
            float(receiver_probs[candidate_index]) * innovation_q,
            reconstructed_source,
            float(reconstructed_probs[candidate_index]),
            float(reconstructed_rank),
            reconstructed_top,
            reconstructed_margin,
        ]
    )
    if condition == "innovation_plus_source_index_packet":
        features.extend(
            [
                source_is_top,
                source_agrees_receiver,
                source_is_top * reconstructed_source,
                source_is_top * receiver_margin,
            ]
        )
    return features


def _feature_matrix(
    *,
    rows: list[arc_gate.ArcRow],
    receiver_zscores: list[list[float]],
    source_zscores: list[list[float]],
    source_predictions: list[int],
    condition: str,
    clip: float,
    shuffle_seed: int,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    max_choices = max(len(row.choices) for row in rows)
    receiver_ranks = _rank_by_row(receiver_zscores)
    conditioned_source_rows, conditioned_source_predictions = _condition_source_rows(
        source_zscores=source_zscores,
        receiver_zscores=receiver_zscores,
        source_predictions=source_predictions,
        condition=condition,
        shuffle_seed=shuffle_seed,
    )
    feature_rows: list[list[float]] = []
    labels: list[float] = []
    row_candidates: list[tuple[int, int]] = []
    for row_index in indices:
        row = rows[row_index]
        for candidate_index in range(len(row.choices)):
            feature_rows.append(
                _feature_vector(
                    row_index=row_index,
                    candidate_index=candidate_index,
                    candidate_count=len(row.choices),
                    max_choices=max_choices,
                    receiver_zscores=receiver_zscores,
                    receiver_ranks=receiver_ranks,
                    source_rows=conditioned_source_rows,
                    source_predictions=conditioned_source_predictions,
                    condition=condition,
                    clip=clip,
                )
            )
            labels.append(float(candidate_index == row.answer_index))
            row_candidates.append((row_index, candidate_index))
    return (
        np.asarray(feature_rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        row_candidates,
    )


def _fit_ridge_binary(features: np.ndarray, labels: np.ndarray, *, ridge: float) -> dict[str, Any]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = (features - mean) / scale
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    coef = np.linalg.solve(x.T @ x + reg, x.T @ labels)
    return {
        "coef": coef.astype(np.float64),
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "ridge": float(ridge),
    }


def _predict_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x = (features - model["mean"]) / model["scale"]
    return (x @ model["coef"]).astype(np.float64)


def _candidate_predictions(
    rows: list[arc_gate.ArcRow],
    scores: np.ndarray,
    row_candidates: list[tuple[int, int]],
    indices: list[int],
) -> list[int]:
    grouped: dict[int, list[tuple[int, float]]] = {row_index: [] for row_index in indices}
    for (row_index, candidate_index), score in zip(row_candidates, scores, strict=True):
        grouped[row_index].append((candidate_index, float(score)))
    predictions: list[int] = [0 for _ in rows]
    for row_index in indices:
        candidates = grouped[row_index]
        predictions[row_index] = max(candidates, key=lambda item: (item[1], -item[0]))[0]
    return predictions


def _select_condition_model(
    *,
    rows: list[arc_gate.ArcRow],
    receiver_zscores: list[list[float]],
    source_zscores: list[list[float]],
    source_predictions: list[int],
    condition: str,
    calibration_indices: list[int],
    eval_indices: list[int],
    clips: tuple[float, ...],
    ridges: tuple[float, ...],
    shuffle_seed: int,
) -> dict[str, Any]:
    clip_candidates = (1.0,) if condition in {"receiver_only_decoder", "source_index_only_decoder"} else clips
    readouts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for clip in clip_candidates:
        train_features, train_labels, train_row_candidates = _feature_matrix(
            rows=rows,
            receiver_zscores=receiver_zscores,
            source_zscores=source_zscores,
            source_predictions=source_predictions,
            condition=condition,
            clip=float(clip),
            shuffle_seed=shuffle_seed,
            indices=calibration_indices,
        )
        eval_features, _, eval_row_candidates = _feature_matrix(
            rows=rows,
            receiver_zscores=receiver_zscores,
            source_zscores=source_zscores,
            source_predictions=source_predictions,
            condition=condition,
            clip=float(clip),
            shuffle_seed=shuffle_seed,
            indices=eval_indices,
        )
        for ridge in ridges:
            model = _fit_ridge_binary(train_features, train_labels, ridge=float(ridge))
            calibration_scores = _predict_scores(train_features, model)
            eval_scores = _predict_scores(eval_features, model)
            calibration_predictions = _candidate_predictions(
                rows,
                calibration_scores,
                train_row_candidates,
                calibration_indices,
            )
            eval_predictions = _candidate_predictions(rows, eval_scores, eval_row_candidates, eval_indices)
            readout = {
                "condition": condition,
                "clip": float(clip),
                "ridge": float(ridge),
                "calibration_accuracy": _accuracy(rows, calibration_predictions, calibration_indices),
                "heldout_accuracy": _accuracy(rows, eval_predictions, eval_indices),
            }
            readouts.append(readout)
            if best is None or (
                readout["calibration_accuracy"],
                readout["heldout_accuracy"],
                -abs(float(clip) - 2.0),
                -math.log10(float(ridge)),
            ) > (
                best["calibration_accuracy"],
                best["heldout_accuracy"],
                -abs(float(best["clip"]) - 2.0),
                -math.log10(float(best["ridge"])),
            ):
                best = readout | {"model": model, "eval_predictions": eval_predictions}
    if best is None:
        raise RuntimeError(f"no model selected for {condition}")
    return {
        "selected": {key: value for key, value in best.items() if key not in {"model", "eval_predictions"}},
        "grid": readouts,
        "predictions": best["eval_predictions"],
    }


def _prediction_rows(
    *,
    rows: list[arc_gate.ArcRow],
    predictions_by_condition: dict[str, list[int]],
    indices: list[int],
) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for condition, predictions in predictions_by_condition.items():
        for row_index in indices:
            row = rows[row_index]
            prediction = int(predictions[row_index])
            prediction_rows.append(
                {
                    "condition": condition,
                    "content_id": row.content_id,
                    "row_id": row.row_id,
                    "answer_index": row.answer_index,
                    "answer_label": row.answer_label,
                    "prediction_index": prediction,
                    "prediction_label": row.choice_labels[prediction],
                    "correct": prediction == row.answer_index,
                }
            )
    return prediction_rows


def _metrics_by_condition(
    rows: list[arc_gate.ArcRow],
    predictions_by_condition: dict[str, list[int]],
    indices: list[int],
) -> dict[str, dict[str, Any]]:
    return {
        condition: {
            "heldout_accuracy": _accuracy(rows, predictions, indices),
            "heldout_correct": int(sum(predictions[index] == rows[index].answer_index for index in indices)),
            "heldout_n": len(indices),
        }
        for condition, predictions in predictions_by_condition.items()
    }


def _subgroup_metrics(
    rows: list[arc_gate.ArcRow],
    predictions_by_condition: dict[str, list[int]],
    indices: list[int],
) -> dict[str, dict[str, dict[str, Any]]]:
    source_predictions = predictions_by_condition["source_label_text"]
    receiver_predictions = predictions_by_condition["receiver_label_text"]
    groups = {
        "all": list(indices),
        "source_receiver_disagree": [
            index for index in indices if source_predictions[index] != receiver_predictions[index]
        ],
        "source_wrong": [
            index for index in indices if source_predictions[index] != rows[index].answer_index
        ],
        "source_wrong_receiver_wrong": [
            index
            for index in indices
            if source_predictions[index] != rows[index].answer_index
            and receiver_predictions[index] != rows[index].answer_index
        ],
        "source_right": [
            index for index in indices if source_predictions[index] == rows[index].answer_index
        ],
    }
    return {
        group_name: {
            condition: {
                "accuracy": _accuracy(rows, predictions, group_indices),
                "correct": int(sum(predictions[index] == rows[index].answer_index for index in group_indices)),
                "n": len(group_indices),
            }
            for condition, predictions in predictions_by_condition.items()
        }
        for group_name, group_indices in groups.items()
    }


def _matched_flip_audit(
    rows: list[arc_gate.ArcRow],
    predictions_by_condition: dict[str, list[int]],
    indices: list[int],
) -> dict[str, Any]:
    source_predictions = predictions_by_condition["source_label_text"]
    matched_predictions = predictions_by_condition["matched_conditional_innovation_packet"]
    counts = {
        "same_prediction": 0,
        "fixed_source_error": 0,
        "broke_source_correct": 0,
        "changed_wrong_to_wrong": 0,
    }
    for index in indices:
        source_prediction = source_predictions[index]
        matched_prediction = matched_predictions[index]
        answer_index = rows[index].answer_index
        if source_prediction == matched_prediction:
            counts["same_prediction"] += 1
        elif source_prediction != answer_index and matched_prediction == answer_index:
            counts["fixed_source_error"] += 1
        elif source_prediction == answer_index and matched_prediction != answer_index:
            counts["broke_source_correct"] += 1
        else:
            counts["changed_wrong_to_wrong"] += 1
    counts["net_correct_vs_source"] = counts["fixed_source_error"] - counts["broke_source_correct"]
    counts["heldout_n"] = len(indices)
    return counts


def build_gate(
    *,
    output_dir: pathlib.Path,
    eval_path: pathlib.Path,
    source_score_cache: pathlib.Path,
    receiver_score_cache: pathlib.Path,
    clips: tuple[float, ...],
    ridges: tuple[float, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    shuffle_seed: int,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    eval_path = _resolve(eval_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = arc_gate._load_rows(eval_path)
    source_scores, source_predictions, source_model, source_cache_sha256 = _load_score_cache(
        source_score_cache,
        rows=rows,
        role="source",
    )
    receiver_scores, receiver_predictions, receiver_model, receiver_cache_sha256 = _load_score_cache(
        receiver_score_cache,
        rows=rows,
        role="receiver",
    )
    source_zscores = _zscore_rows(source_scores)
    receiver_zscores = _zscore_rows(receiver_scores)
    indices = list(range(len(rows)))
    calibration_indices = [index for index in indices if index % 2 == 0]
    eval_indices = [index for index in indices if index % 2 == 1]
    conditions = [
        "receiver_only_decoder",
        "source_index_only_decoder",
        "quantized_source_score_packet",
        "matched_conditional_innovation_packet",
        "innovation_plus_source_index_packet",
        "zero_innovation_control",
        "row_shuffle_innovation_control",
        "candidate_roll_innovation_control",
    ]
    condition_models = {
        condition: _select_condition_model(
            rows=rows,
            receiver_zscores=receiver_zscores,
            source_zscores=source_zscores,
            source_predictions=source_predictions,
            condition=condition,
            calibration_indices=calibration_indices,
            eval_indices=eval_indices,
            clips=clips,
            ridges=ridges,
            shuffle_seed=shuffle_seed,
        )
        for condition in conditions
    }
    predictions_by_condition = {
        "source_label_text": source_predictions,
        "receiver_label_text": receiver_predictions,
    } | {condition: result["predictions"] for condition, result in condition_models.items()}
    prediction_rows = _prediction_rows(rows=rows, predictions_by_condition=predictions_by_condition, indices=eval_indices)
    metrics = _metrics_by_condition(rows, predictions_by_condition, eval_indices)
    subgroup_metrics = _subgroup_metrics(rows, predictions_by_condition, eval_indices)
    matched_flip_audit = _matched_flip_audit(rows, predictions_by_condition, eval_indices)
    control_conditions = [
        "receiver_only_decoder",
        "source_index_only_decoder",
        "quantized_source_score_packet",
        "zero_innovation_control",
        "row_shuffle_innovation_control",
        "candidate_roll_innovation_control",
    ]
    best_control = max(
        control_conditions,
        key=lambda condition: (metrics[condition]["heldout_accuracy"], condition != "quantized_source_score_packet"),
    )
    matched = "matched_conditional_innovation_packet"
    paired = {
        f"{matched}_minus_source_label_text": arc_gate._paired_bootstrap(
            prediction_rows,
            condition=matched,
            baseline="source_label_text",
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        f"{matched}_minus_source_index_only_decoder": arc_gate._paired_bootstrap(
            prediction_rows,
            condition=matched,
            baseline="source_index_only_decoder",
            seed=bootstrap_seed + 1,
            samples=bootstrap_samples,
        ),
        f"{matched}_minus_quantized_source_score_packet": arc_gate._paired_bootstrap(
            prediction_rows,
            condition=matched,
            baseline="quantized_source_score_packet",
            seed=bootstrap_seed + 2,
            samples=bootstrap_samples,
        ),
        f"{matched}_minus_best_control": arc_gate._paired_bootstrap(
            prediction_rows,
            condition=matched,
            baseline=best_control,
            seed=bootstrap_seed + 3,
            samples=bootstrap_samples,
        ),
    }
    raw_bits = _packet_bits(max(len(row.choices) for row in rows))
    raw_bytes = int(math.ceil(raw_bits / 8.0))
    headline = {
        "source_label_text_heldout_accuracy": metrics["source_label_text"]["heldout_accuracy"],
        "receiver_label_text_heldout_accuracy": metrics["receiver_label_text"]["heldout_accuracy"],
        "source_index_only_decoder_heldout_accuracy": metrics["source_index_only_decoder"]["heldout_accuracy"],
        "quantized_source_score_packet_heldout_accuracy": metrics["quantized_source_score_packet"]["heldout_accuracy"],
        "matched_conditional_innovation_packet_heldout_accuracy": metrics[matched]["heldout_accuracy"],
        "innovation_plus_source_index_packet_heldout_accuracy": metrics["innovation_plus_source_index_packet"][
            "heldout_accuracy"
        ],
        "best_control_condition": best_control,
        "best_control_heldout_accuracy": metrics[best_control]["heldout_accuracy"],
        "matched_minus_source_label_text": metrics[matched]["heldout_accuracy"]
        - metrics["source_label_text"]["heldout_accuracy"],
        "matched_minus_source_index_only_decoder": metrics[matched]["heldout_accuracy"]
        - metrics["source_index_only_decoder"]["heldout_accuracy"],
        "matched_minus_quantized_source_score_packet": metrics[matched]["heldout_accuracy"]
        - metrics["quantized_source_score_packet"]["heldout_accuracy"],
        "matched_minus_best_control": metrics[matched]["heldout_accuracy"] - metrics[best_control]["heldout_accuracy"],
    }
    pass_rule = {
        "matched_beats_source_label_text_by": STRICT_SOURCE_DELTA,
        "matched_beats_source_index_only_decoder_by": STRICT_CONTROL_DELTA,
        "matched_beats_quantized_source_score_packet_by": STRICT_CONTROL_DELTA,
        "matched_beats_best_control_by": STRICT_CONTROL_DELTA,
        "paired_ci95_low_must_be_positive_vs_source_index_and_best_control": True,
    }
    pass_gate = bool(
        headline["matched_minus_source_label_text"] >= STRICT_SOURCE_DELTA
        and headline["matched_minus_source_index_only_decoder"] >= STRICT_CONTROL_DELTA
        and headline["matched_minus_quantized_source_score_packet"] >= STRICT_CONTROL_DELTA
        and headline["matched_minus_best_control"] >= STRICT_CONTROL_DELTA
        and paired[f"{matched}_minus_source_index_only_decoder"]["ci95_low"] > 0.0
        and paired[f"{matched}_minus_best_control"]["ci95_low"] > 0.0
    )
    payload = {
        "gate": "source_private_arc_conditional_innovation_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "eval_rows": len(rows),
        "calibration_rows": len(calibration_indices),
        "heldout_eval_rows": len(eval_indices),
        "source_score_cache": _display_path(source_score_cache),
        "source_score_cache_sha256": source_cache_sha256,
        "receiver_score_cache": _display_path(receiver_score_cache),
        "receiver_score_cache_sha256": receiver_cache_sha256,
        "source_model": source_model
        | {
            "role": "source_sender",
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS) + ["question_concept"],
        },
        "receiver_model": receiver_model
        | {
            "role": "receiver_side_information",
            "receiver_visible_fields": ["question", "choices"],
            "forbidden_receiver_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS) + ["question_concept"],
        },
        "split": {
            "calibration": "even row indices from frozen ARC-Challenge validation cache",
            "heldout": "odd row indices from frozen ARC-Challenge validation cache",
            "claim_boundary": "preflight validation split; not a final benchmark number",
        },
        "packet_contract": {
            "packet_name": "receiver_conditioned_source_innovation",
            "raw_payload_bits": raw_bits,
            "raw_payload_bytes": raw_bytes,
            "record_bytes_with_header_crc": raw_bytes + 3,
            "candidate_count": max(len(row.choices) for row in rows),
            "source_payload": (
                "one 4-bit clipped innovation value per public candidate, where innovation is "
                "source row-zscore minus receiver row-zscore"
            ),
            "receiver_side_information": "receiver local row-zscores over the same public candidates",
            "decoder_rule": (
                "fit a candidate-wise ridge receiver on even rows; select clip/ridge on calibration accuracy; "
                "evaluate on odd rows"
            ),
            "destructive_controls": [
                "zero_innovation_control",
                "row_shuffle_innovation_control",
                "candidate_roll_innovation_control",
            ],
            "claim_boundary": (
                "Promotion requires beating source-label text, source-index decoder, quantized source-score packet, "
                "and best destructive/control decoder on heldout rows with paired uncertainty."
            ),
        },
        "condition_metrics": metrics,
        "subgroup_metrics": subgroup_metrics,
        "matched_flip_audit": matched_flip_audit,
        "condition_selection": {
            condition: result["selected"] for condition, result in condition_models.items()
        },
        "condition_grids": {
            condition: result["grid"] for condition, result in condition_models.items()
        },
        "paired_bootstrap": paired,
        "headline": headline,
        "pass_rule": pass_rule,
        "pass_gate": pass_gate,
    }
    json_path = output_dir / "arc_conditional_innovation_packet_gate.json"
    md_path = output_dir / "arc_conditional_innovation_packet_gate.md"
    predictions_path = output_dir / "arc_conditional_innovation_packet_predictions.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    lines = [
        "# ARC Conditional-Innovation Packet Gate",
        "",
        f"- pass gate: `{pass_gate}`",
        f"- source-label heldout accuracy: `{headline['source_label_text_heldout_accuracy']:.3f}`",
        f"- source-index decoder heldout accuracy: `{headline['source_index_only_decoder_heldout_accuracy']:.3f}`",
        f"- quantized source-score packet heldout accuracy: `{headline['quantized_source_score_packet_heldout_accuracy']:.3f}`",
        f"- matched conditional-innovation heldout accuracy: `{headline['matched_conditional_innovation_packet_heldout_accuracy']:.3f}`",
        f"- matched minus source-index decoder: `{headline['matched_minus_source_index_only_decoder']:.3f}`",
        f"- matched minus quantized source-score packet: `{headline['matched_minus_quantized_source_score_packet']:.3f}`",
        f"- best control: `{best_control}` at `{headline['best_control_heldout_accuracy']:.3f}`",
        f"- paired matched minus best control: `{paired[f'{matched}_minus_best_control']['mean']:.3f}` "
        f"[{paired[f'{matched}_minus_best_control']['ci95_low']:.3f}, "
        f"{paired[f'{matched}_minus_best_control']['ci95_high']:.3f}]",
        "",
        "Lay explanation: the packet sends where the source model's answer scores differ from the receiver's "
        "own scores, then a small calibrated decoder asks whether that disagreement helps choose the answer.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ARC receiver-conditioned source-innovation packet gate.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--source-score-cache", type=pathlib.Path, required=True)
    parser.add_argument("--receiver-score-cache", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=_parse_float_tuple, default=(1.5, 2.0, 3.0, 4.0))
    parser.add_argument("--ridges", type=_parse_float_tuple, default=(0.1, 1.0, 10.0, 100.0))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=731)
    parser.add_argument("--shuffle-seed", type=int, default=20260503)
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_gate(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        source_score_cache=args.source_score_cache,
        receiver_score_cache=args.receiver_score_cache,
        clips=args.clips,
        ridges=args.ridges,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        shuffle_seed=args.shuffle_seed,
        run_date=args.run_date,
    )


if __name__ == "__main__":
    main()
