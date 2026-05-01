from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl"
)
DEFAULT_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json"
)
DEFAULT_TRAIN_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/"
    "source_train_score_cache.json"
)
DEFAULT_TRAIN_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/"
    "source_train_hidden_cache.npz"
)
DEFAULT_EVAL_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/"
    "source_eval_hidden_cache.npz"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_top2_contrastive_repair_probe_20260501_qwen05_train512_validation1024"
)

STRICT_DELTA = 0.02


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(rows: list[arc_gate.ArcRow]) -> str:
    return hashlib.sha256("\n".join(row.content_id for row in rows).encode("utf-8")).hexdigest()


def _ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    return float(sum(row.answer_index == pred for row, pred in zip(rows, predictions, strict=True)) / len(rows))


def _evaluate(rows: list[arc_gate.ArcRow], predictions: list[int]) -> dict[str, Any]:
    return {
        "accuracy": _accuracy(rows, predictions),
        "correct": int(sum(row.answer_index == pred for row, pred in zip(rows, predictions, strict=True))),
        "rows": len(rows),
    }


def _select_train_rows(rows: list[arc_gate.ArcRow], *, count: int, seed: int) -> list[arc_gate.ArcRow]:
    if count >= len(rows):
        return list(rows)
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:count]


def _split_indices(n: int, *, dev_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    dev_count = max(1, min(n - 1, int(round(n * dev_fraction))))
    dev = sorted(indices[:dev_count])
    fit = sorted(indices[dev_count:])
    return fit, dev


def _take_rows(rows: list[arc_gate.ArcRow], indices: list[int]) -> list[arc_gate.ArcRow]:
    return [rows[index] for index in indices]


def _take_scores(scores: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [scores[index] for index in indices]


def _take_array(values: np.ndarray, indices: list[int]) -> np.ndarray:
    return values[np.asarray(indices, dtype=np.int64)]


def _load_hidden_cache(path: pathlib.Path, *, rows: list[arc_gate.ArcRow]) -> tuple[np.ndarray, dict[str, Any]]:
    path = _resolve(path)
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing hidden cache pair: {path} / {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_row_ids = [row.row_id for row in rows]
    if metadata.get("row_count") != len(rows):
        raise ValueError(f"hidden cache row count does not match rows: {path}")
    if metadata.get("row_ids") != expected_row_ids:
        raise ValueError(f"hidden cache row id order does not match rows: {path}")
    expected_digest = _content_digest(rows)
    if metadata.get("content_digest") not in {None, expected_digest}:
        raise ValueError(f"hidden cache content digest does not match rows: {path}")
    with np.load(path) as data:
        features = np.asarray(data["features"], dtype=np.float64)
    if features.shape[0] != len(rows):
        raise ValueError(f"hidden feature count does not match rows: {path}")
    return features, metadata | {"cache_npz": _display_path(path), "cache_meta": _display_path(meta_path)}


def _safe_softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _entropy(scores: list[float]) -> float:
    probs = _safe_softmax(scores)
    return float(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs))


def _score_features(scores: list[float]) -> np.ndarray:
    ranked = _ranked_indices(scores)
    top1 = ranked[0]
    top2 = ranked[1]
    margin12 = float(scores[top1] - scores[top2])
    margin23 = float(scores[top2] - scores[ranked[2]]) if len(ranked) > 2 else 0.0
    values = [
        margin12,
        margin23,
        _entropy(scores),
        float(scores[top1]),
        float(scores[top2]),
        float(np.std(np.asarray(scores, dtype=np.float64))),
    ]
    return np.asarray(values, dtype=np.float64)


def _public_contrast_features(rows: list[arc_gate.ArcRow], scores: list[list[float]], *, dim: int) -> np.ndarray:
    pair_features = arc_gate._features(
        arc_gate._choice_pair_texts(rows),
        dim=dim,
        feature_mode="hashed",
        feature_model="unused",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
    )
    residuals = arc_gate._candidate_residuals(rows, pair_features)
    contrasts: list[np.ndarray] = []
    for row_residuals, row_scores in zip(residuals, scores, strict=True):
        ranked = _ranked_indices(row_scores)
        contrasts.append(row_residuals[ranked[1]] - row_residuals[ranked[0]])
    return np.asarray(contrasts, dtype=np.float64)


def _hidden_contrast_features(hidden: np.ndarray, scores: list[list[float]], *, layer_index: int) -> np.ndarray:
    layer = hidden[:, :, layer_index, :]
    contrasts: list[np.ndarray] = []
    for row_index, row_scores in enumerate(scores):
        ranked = _ranked_indices(row_scores)
        contrasts.append(layer[row_index, ranked[1]] - layer[row_index, ranked[0]])
    return np.asarray(contrasts, dtype=np.float64)


def _fit_binary_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    trainable: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    if int(np.sum(trainable)) < 2:
        raise ValueError("not enough trainable top-2 rows for ridge switch model")
    x_raw = features[trainable]
    y = labels[trainable]
    mean = np.mean(x_raw, axis=0)
    scale = np.std(x_raw, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_body = (x_raw - mean) / scale
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    xtx[0, 0] -= float(ridge)
    weights = np.linalg.solve(xtx, x.T @ y)
    return {
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "ridge": float(ridge),
        "feature_dim": int(features.shape[1]),
        "trainable_rows": int(np.sum(trainable)),
    }


def _score_binary_ridge(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x_body = (features - model["mean"]) / model["scale"]
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    return np.asarray(x @ model["weights"], dtype=np.float64)


def _switch_labels(rows: list[arc_gate.ArcRow], scores: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    trainable: list[bool] = []
    for row, row_scores in zip(rows, scores, strict=True):
        ranked = _ranked_indices(row_scores)
        if row.answer_index == ranked[1]:
            labels.append(1.0)
            trainable.append(True)
        elif row.answer_index == ranked[0]:
            labels.append(-1.0)
            trainable.append(True)
        else:
            labels.append(-1.0)
            trainable.append(False)
    return np.asarray(labels, dtype=np.float64), np.asarray(trainable, dtype=bool)


def _predictions_from_switch_scores(scores: list[list[float]], switch_scores: np.ndarray, threshold: float) -> list[int]:
    predictions: list[int] = []
    for row_scores, switch_score in zip(scores, switch_scores, strict=True):
        ranked = _ranked_indices(row_scores)
        predictions.append(ranked[1] if float(switch_score) > threshold else ranked[0])
    return predictions


def _switch_rate(switch_scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(switch_scores > threshold))


def _select_threshold(
    *,
    rows: list[arc_gate.ArcRow],
    scores: list[list[float]],
    switch_scores: np.ndarray,
) -> dict[str, Any]:
    values = sorted(float(value) for value in switch_scores)
    if not values:
        raise ValueError("cannot select threshold without scores")
    candidates = {0.0, min(values) - 1e-6, max(values) + 1e-6}
    candidates.update(values)
    for left, right in zip(values, values[1:], strict=False):
        candidates.add((left + right) / 2.0)
    selected = max(
        candidates,
        key=lambda threshold: (
            _accuracy(rows, _predictions_from_switch_scores(scores, switch_scores, threshold)),
            -abs(_switch_rate(switch_scores, threshold) - 0.25),
            -abs(float(threshold)),
        ),
    )
    predictions = _predictions_from_switch_scores(scores, switch_scores, float(selected))
    return {
        "threshold": float(selected),
        "accuracy": _accuracy(rows, predictions),
        "switch_rate": _switch_rate(switch_scores, float(selected)),
    }


def _paired_ci_predictions(
    rows: list[arc_gate.ArcRow],
    candidate: list[int],
    baseline: list[int],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    deltas = [
        float(row.answer_index == cand) - float(row.answer_index == base)
        for row, cand, base in zip(rows, candidate, baseline, strict=True)
    ]
    if not deltas:
        raise ValueError("cannot bootstrap empty delta list")
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _source_label_predictions(scores: list[list[float]]) -> list[int]:
    return [_ranked_indices(row_scores)[0] for row_scores in scores]


def _topk_oracle(rows: list[arc_gate.ArcRow], scores: list[list[float]], *, k: int) -> float:
    return float(
        sum(row.answer_index in _ranked_indices(row_scores)[:k] for row, row_scores in zip(rows, scores, strict=True))
        / len(rows)
    )


def _build_feature_views(
    *,
    score_matrix: list[list[float]],
    public_contrast: np.ndarray,
    hidden_contrast: np.ndarray,
) -> dict[str, np.ndarray]:
    score = np.asarray([_score_features(row_scores) for row_scores in score_matrix], dtype=np.float64)
    return {
        "score_only": score,
        "public_contrast_only": public_contrast,
        "score_public_contrast": np.concatenate([score, public_contrast], axis=1),
        "hidden_contrast_only": hidden_contrast,
        "hidden_score_contrast": np.concatenate([hidden_contrast, score], axis=1),
        "hidden_score_public_contrast": np.concatenate([hidden_contrast, score, public_contrast], axis=1),
    }


def _fit_and_eval_view(
    *,
    view_name: str,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: list[int],
    dev_indices: list[int],
    train_rows: list[arc_gate.ArcRow],
    train_scores: list[list[float]],
    eval_rows: list[arc_gate.ArcRow],
    eval_scores: list[list[float]],
    ridge: float,
) -> dict[str, Any]:
    train_labels, trainable = _switch_labels(train_rows, train_scores)
    fit_mask = np.zeros(len(train_rows), dtype=bool)
    fit_mask[np.asarray(fit_indices, dtype=np.int64)] = True
    model = _fit_binary_ridge(train_features, train_labels, trainable & fit_mask, ridge=ridge)
    train_switch_scores = _score_binary_ridge(train_features, model)
    eval_switch_scores = _score_binary_ridge(eval_features, model)
    fit_rows = _take_rows(train_rows, fit_indices)
    dev_rows = _take_rows(train_rows, dev_indices)
    fit_scores = _take_scores(train_scores, fit_indices)
    dev_scores = _take_scores(train_scores, dev_indices)
    fit_switch_scores = train_switch_scores[np.asarray(fit_indices, dtype=np.int64)]
    dev_switch_scores = train_switch_scores[np.asarray(dev_indices, dtype=np.int64)]
    threshold = _select_threshold(rows=dev_rows, scores=dev_scores, switch_scores=dev_switch_scores)
    selected_threshold = float(threshold["threshold"])
    fit_predictions = _predictions_from_switch_scores(fit_scores, fit_switch_scores, selected_threshold)
    dev_predictions = _predictions_from_switch_scores(dev_scores, dev_switch_scores, selected_threshold)
    train_predictions = _predictions_from_switch_scores(train_scores, train_switch_scores, selected_threshold)
    eval_predictions = _predictions_from_switch_scores(eval_scores, eval_switch_scores, selected_threshold)
    return {
        "view": view_name,
        "ridge": float(ridge),
        "threshold": selected_threshold,
        "feature_dim": int(train_features.shape[1]),
        "model_trainable_rows": model["trainable_rows"],
        "fit": _evaluate(fit_rows, fit_predictions),
        "internal_dev": _evaluate(dev_rows, dev_predictions),
        "train": _evaluate(train_rows, train_predictions),
        "eval": _evaluate(eval_rows, eval_predictions),
        "fit_switch_rate": _switch_rate(fit_switch_scores, selected_threshold),
        "internal_dev_switch_rate": _switch_rate(dev_switch_scores, selected_threshold),
        "eval_switch_rate": _switch_rate(eval_switch_scores, selected_threshold),
        "eval_predictions": eval_predictions,
        "eval_switch_scores": [float(value) for value in eval_switch_scores],
    }


def build_probe(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    eval_score_cache: pathlib.Path,
    train_score_cache: pathlib.Path,
    train_hidden_cache: pathlib.Path,
    eval_hidden_cache: pathlib.Path,
    train_hidden_rows: int,
    selection_seed: int,
    dev_fraction: float,
    hidden_layer_index: int,
    public_feature_dim: int,
    ridges: tuple[float, ...],
    bootstrap_samples: int,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = _resolve(train_path)
    eval_path = _resolve(eval_path)
    eval_score_cache = _resolve(eval_score_cache)
    train_score_cache = _resolve(train_score_cache)
    train_hidden_cache = _resolve(train_hidden_cache)
    eval_hidden_cache = _resolve(eval_hidden_cache)

    all_train_rows = arc_gate._load_rows(train_path)
    train_rows = _select_train_rows(all_train_rows, count=train_hidden_rows, seed=selection_seed)
    eval_rows = arc_gate._load_rows(eval_path)
    fit_indices, dev_indices = _split_indices(len(train_rows), dev_fraction=dev_fraction, seed=selection_seed + 17)

    train_scores, _, train_source_model = headroom._load_score_cache(train_score_cache, rows=train_rows)
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
    train_hidden, train_hidden_meta = _load_hidden_cache(train_hidden_cache, rows=train_rows)
    eval_hidden, eval_hidden_meta = _load_hidden_cache(eval_hidden_cache, rows=eval_rows)
    if hidden_layer_index < 0:
        hidden_layer_index = train_hidden.shape[2] + hidden_layer_index
    if hidden_layer_index < 0 or hidden_layer_index >= train_hidden.shape[2]:
        raise ValueError(f"hidden layer index out of range: {hidden_layer_index}")

    train_public = _public_contrast_features(train_rows, train_scores, dim=public_feature_dim)
    eval_public = _public_contrast_features(eval_rows, eval_scores, dim=public_feature_dim)
    train_hidden_contrast = _hidden_contrast_features(train_hidden, train_scores, layer_index=hidden_layer_index)
    eval_hidden_contrast = _hidden_contrast_features(eval_hidden, eval_scores, layer_index=hidden_layer_index)

    train_views = _build_feature_views(
        score_matrix=train_scores,
        public_contrast=train_public,
        hidden_contrast=train_hidden_contrast,
    )
    eval_views = _build_feature_views(
        score_matrix=eval_scores,
        public_contrast=eval_public,
        hidden_contrast=eval_hidden_contrast,
    )

    candidate_readouts: list[dict[str, Any]] = []
    for view_name in sorted(train_views):
        for ridge in ridges:
            readout = _fit_and_eval_view(
                view_name=view_name,
                train_features=train_views[view_name],
                eval_features=eval_views[view_name],
                fit_indices=fit_indices,
                dev_indices=dev_indices,
                train_rows=train_rows,
                train_scores=train_scores,
                eval_rows=eval_rows,
                eval_scores=eval_scores,
                ridge=ridge,
            )
            candidate_readouts.append(readout)

    selected = max(
        candidate_readouts,
        key=lambda item: (
            item["internal_dev"]["accuracy"],
            item["fit"]["accuracy"],
            item["view"].startswith("hidden"),
            -item["ridge"],
            item["view"],
        ),
    )
    source_label_eval = _source_label_predictions(eval_scores)
    offsets = score_repair._fit_choice_bias_offsets(_take_rows(train_rows, fit_indices), _take_scores(train_scores, fit_indices))
    trained_label_eval = [score_repair._predict_calibrated_label(row_scores, offsets) for row_scores in eval_scores]
    source_label_eval_accuracy = _accuracy(eval_rows, source_label_eval)
    trained_label_eval_accuracy = _accuracy(eval_rows, trained_label_eval)
    best_label_copy_eval_accuracy = max(source_label_eval_accuracy, trained_label_eval_accuracy)

    best_score_control = max(
        (item for item in candidate_readouts if item["view"] in {"score_only", "score_public_contrast"}),
        key=lambda item: (item["internal_dev"]["accuracy"], item["eval"]["accuracy"], -item["ridge"]),
    )
    best_public_control = max(
        (item for item in candidate_readouts if item["view"] == "public_contrast_only"),
        key=lambda item: (item["internal_dev"]["accuracy"], item["eval"]["accuracy"], -item["ridge"]),
    )
    best_zero_hidden_control_accuracy = max(best_score_control["eval"]["accuracy"], best_public_control["eval"]["accuracy"])

    selected_predictions = [int(value) for value in selected["eval_predictions"]]
    selected_switch_scores = np.asarray(selected["eval_switch_scores"], dtype=np.float64)
    wrong_scores = np.roll(selected_switch_scores, 1)
    wrong_example_predictions = _predictions_from_switch_scores(eval_scores, wrong_scores, float(selected["threshold"]))
    pair_swap_predictions: list[int] = []
    for row_scores, switch_score in zip(eval_scores, selected_switch_scores, strict=True):
        ranked = _ranked_indices(row_scores)
        pair_swap_predictions.append(ranked[0] if float(switch_score) > float(selected["threshold"]) else ranked[1])

    paired_ci_label = _paired_ci_predictions(
        eval_rows,
        selected_predictions,
        source_label_eval if source_label_eval_accuracy >= trained_label_eval_accuracy else trained_label_eval,
        seed=selection_seed + 3001,
        samples=bootstrap_samples,
    )
    paired_ci_score = _paired_ci_predictions(
        eval_rows,
        selected_predictions,
        [int(value) for value in best_score_control["eval_predictions"]],
        seed=selection_seed + 3002,
        samples=bootstrap_samples,
    )

    control_readouts = {
        "source_label_copy": _evaluate(eval_rows, source_label_eval),
        "trained_choice_bias_label_copy": _evaluate(eval_rows, trained_label_eval),
        "best_score_or_score_public_control": {
            "view": best_score_control["view"],
            "ridge": best_score_control["ridge"],
            **best_score_control["eval"],
        },
        "best_public_only_control": {
            "view": best_public_control["view"],
            "ridge": best_public_control["ridge"],
            **best_public_control["eval"],
        },
        "wrong_example_packet_control": _evaluate(eval_rows, wrong_example_predictions),
        "top2_identity_pair_swap_control": _evaluate(eval_rows, pair_swap_predictions),
    }

    candidate_summary = [
        {
            "view": item["view"],
            "ridge": item["ridge"],
            "feature_dim": item["feature_dim"],
            "fit_accuracy": item["fit"]["accuracy"],
            "internal_dev_accuracy": item["internal_dev"]["accuracy"],
            "eval_accuracy": item["eval"]["accuracy"],
            "eval_switch_rate": item["eval_switch_rate"],
        }
        for item in candidate_readouts
    ]

    packet_contract = {
        "packet_name": "top2_contrastive_source_error_packet",
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "fields": [
            "source top-1 candidate id packed into 2 bits",
            "source top-2 candidate id packed into 2 bits",
            "binary source-error switch decision",
            "quantized switch confidence/debug bin",
        ],
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
        "decoder_rule": "choose source runner-up when the switch score exceeds the train-dev-selected threshold; otherwise choose source top-1",
    }

    selected_eval_accuracy = selected["eval"]["accuracy"]
    headline = {
        "selected_view": selected["view"],
        "selected_ridge": selected["ridge"],
        "selected_threshold": selected["threshold"],
        "selected_internal_dev_accuracy": selected["internal_dev"]["accuracy"],
        "selected_eval_accuracy": selected_eval_accuracy,
        "selected_eval_switch_rate": selected["eval_switch_rate"],
        "source_label_copy_eval_accuracy": source_label_eval_accuracy,
        "trained_choice_bias_label_copy_eval_accuracy": trained_label_eval_accuracy,
        "best_label_copy_eval_accuracy": best_label_copy_eval_accuracy,
        "selected_minus_source_label_copy": selected_eval_accuracy - source_label_eval_accuracy,
        "selected_minus_trained_choice_bias_label_copy": selected_eval_accuracy - trained_label_eval_accuracy,
        "selected_minus_best_label_copy": selected_eval_accuracy - best_label_copy_eval_accuracy,
        "best_zero_hidden_control_accuracy": best_zero_hidden_control_accuracy,
        "selected_minus_best_zero_hidden_control": selected_eval_accuracy - best_zero_hidden_control_accuracy,
        "wrong_example_packet_accuracy": control_readouts["wrong_example_packet_control"]["accuracy"],
        "pair_swap_control_accuracy": control_readouts["top2_identity_pair_swap_control"]["accuracy"],
        "source_top2_oracle_accuracy": _topk_oracle(eval_rows, eval_scores, k=2),
        "source_top4_oracle_accuracy": _topk_oracle(eval_rows, eval_scores, k=4),
        "paired_ci95_selected_vs_best_label_copy": paired_ci_label,
        "paired_ci95_selected_vs_best_score_control": paired_ci_score,
    }
    pass_gate = bool(
        headline["selected_minus_best_label_copy"] >= STRICT_DELTA
        and headline["paired_ci95_selected_vs_best_label_copy"]["ci95_low"] > 0.0
        and headline["selected_minus_best_zero_hidden_control"] >= 0.01
        and headline["wrong_example_packet_accuracy"] <= best_label_copy_eval_accuracy + 0.005
    )

    payload = {
        "gate": "source_private_hellaswag_top2_contrastive_repair_probe",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "train_path": _display_path(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "eval_score_cache": _display_path(eval_score_cache),
        "eval_score_cache_sha256": _sha256_file(eval_score_cache),
        "train_score_cache": _display_path(train_score_cache),
        "train_score_cache_sha256": _sha256_file(train_score_cache),
        "train_hidden_cache": _display_path(train_hidden_cache),
        "train_hidden_cache_sha256": _sha256_file(train_hidden_cache),
        "eval_hidden_cache": _display_path(eval_hidden_cache),
        "eval_hidden_cache_sha256": _sha256_file(eval_hidden_cache),
        "all_train_rows": len(all_train_rows),
        "scored_train_rows": len(train_rows),
        "internal_fit_rows": len(fit_indices),
        "internal_dev_rows": len(dev_indices),
        "eval_rows": len(eval_rows),
        "selection_seed": selection_seed,
        "dev_fraction": dev_fraction,
        "hidden_layer_index": hidden_layer_index,
        "source_model": {
            "score_train": train_source_model,
            "score_eval": eval_source_model,
            "hidden_train": train_hidden_meta,
            "hidden_eval": eval_hidden_meta,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS)
            + ["label", "activity_label", "source_id", "split", "split_type", "ind"],
        },
        "packet_contract": packet_contract,
        "headline": headline,
        "control_readouts": control_readouts,
        "candidate_readouts": candidate_summary,
        "pass_rule": {
            "selected_on_train_dev_only": True,
            "selected_must_beat_best_label_copy_by": STRICT_DELTA,
            "paired_ci95_low_vs_best_label_copy_must_be_positive": True,
            "selected_must_beat_best_zero_hidden_control_by": 0.01,
            "wrong_example_packet_must_not_exceed_best_label_copy_by_more_than": 0.005,
            "claim_boundary": (
                "This gate tests whether contrastive source evidence can identify when the source top-1 "
                "HellaSwag continuation should be replaced by its runner-up. Promotion requires beating "
                "source-label and trained-label-copy controls; otherwise the branch remains diagnostic."
            ),
        },
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }

    (output_dir / "hellaswag_top2_contrastive_repair_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "candidate_readouts.jsonl").open("w", encoding="utf-8") as handle:
        for item in candidate_summary:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, row_scores, pred, source_pred, trained_pred, switch_score in zip(
            eval_rows,
            eval_scores,
            selected_predictions,
            source_label_eval,
            trained_label_eval,
            selected_switch_scores,
            strict=True,
        ):
            ranked = _ranked_indices(row_scores)
            handle.write(
                json.dumps(
                    {
                        "row_id": row.row_id,
                        "answer_index": row.answer_index,
                        "source_top1": ranked[0],
                        "source_top2": ranked[1],
                        "selected_prediction": pred,
                        "source_label_prediction": source_pred,
                        "trained_label_prediction": trained_pred,
                        "switch_score": float(switch_score),
                        "switch": bool(float(switch_score) > float(selected["threshold"])),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    lines = [
        "# HellaSwag Top-2 Contrastive Repair Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected view: `{headline['selected_view']}`",
        f"- selected eval accuracy: `{headline['selected_eval_accuracy']:.3f}`",
        f"- source-label copy eval accuracy: `{headline['source_label_copy_eval_accuracy']:.3f}`",
        f"- trained label-copy eval accuracy: `{headline['trained_choice_bias_label_copy_eval_accuracy']:.3f}`",
        f"- selected minus best label-copy: `{headline['selected_minus_best_label_copy']:.3f}`",
        f"- selected minus best zero-hidden control: `{headline['selected_minus_best_zero_hidden_control']:.3f}`",
        f"- wrong-example packet accuracy: `{headline['wrong_example_packet_accuracy']:.3f}`",
        f"- source top-2 oracle accuracy: `{headline['source_top2_oracle_accuracy']:.3f}`",
        "",
        "## Interpretation",
        "",
        "This probe asks whether a tiny switch packet can use contrastive source evidence to repair",
        "the source model's top HellaSwag choice. A pass would require beating label-copy controls;",
        "otherwise the branch remains diagnostic and HellaSwag should stay out of the headline result.",
        "",
    ]
    (output_dir / "hellaswag_top2_contrastive_repair_probe.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HellaSwag top-2 contrastive source-error repair probe.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--train-score-cache", type=pathlib.Path, default=DEFAULT_TRAIN_SCORE_CACHE)
    parser.add_argument("--train-hidden-cache", type=pathlib.Path, default=DEFAULT_TRAIN_HIDDEN_CACHE)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-layer-index", type=int, default=-1)
    parser.add_argument("--public-feature-dim", type=int, default=256)
    parser.add_argument("--ridges", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_probe(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        train_score_cache=args.train_score_cache,
        train_hidden_cache=args.train_hidden_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        selection_seed=args.selection_seed,
        dev_fraction=args.dev_fraction,
        hidden_layer_index=args.hidden_layer_index,
        public_feature_dim=args.public_feature_dim,
        ridges=tuple(args.ridges),
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
