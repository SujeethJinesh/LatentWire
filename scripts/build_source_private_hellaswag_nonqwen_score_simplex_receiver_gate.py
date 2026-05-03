from __future__ import annotations

"""Cached TinyLlama->Phi score-simplex common-basis receiver gate.

The gate tests the cheapest current common-basis hypothesis for non-Qwen
receiver-family transfer: both source and target score the same four HellaSwag
candidate endings, so their row-centered score vectors can be represented in a
shared 3D candidate-contrast basis. No model inference or hidden states are
required.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_nonqwen_score_simplex_receiver_gate_20260503_validation1024_2048"
)
DEFAULT_SLICE_DIRS = (
    pathlib.Path(
        "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048"
    ),
)
DEFAULT_SOURCE_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "source_eval_score_cache.json"
)
DEFAULT_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
STRICT_TARGET_DELTA = 0.02
STRICT_PACKET_DELTA = 0.005
CANDIDATE_COUNT = 4


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
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


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _accuracy(predictions: np.ndarray, answers: np.ndarray, indices: np.ndarray) -> float:
    return float(np.mean(predictions[indices].astype(np.int64) == answers[indices].astype(np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    delta = (selected[indices] == answers[indices]).astype(np.float64) - (
        baseline[indices] == answers[indices]
    ).astype(np.float64)
    if int(samples) <= 0:
        mean = float(np.mean(delta))
        return {"delta": mean, "ci95_low": mean, "ci95_high": mean}
    rng = np.random.default_rng(seed)
    boot = np.mean(delta[rng.integers(0, len(delta), size=(int(samples), len(delta)))], axis=1)
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _row_zscores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.where(scale < 1e-8, 1.0, scale)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.zeros_like(order)
    for row_index in range(scores.shape[0]):
        for rank, candidate in enumerate(order[row_index]):
            ranks[row_index, candidate] = rank
    return ranks.astype(np.int64)


def _one_hot(values: np.ndarray, width: int = CANDIDATE_COUNT) -> np.ndarray:
    eye = np.eye(int(width), dtype=np.float64)
    return eye[np.mod(values.astype(np.int64), int(width))]


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores.astype(np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _contrast_basis() -> np.ndarray:
    """Return a 4x3 orthonormal Helmert/Fourier-style candidate basis."""
    return np.asarray(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
            [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
            [0.0, -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)],
            [0.0, 0.0, -3.0 / math.sqrt(12.0)],
        ],
        dtype=np.float64,
    )


def _fit_svd_basis(
    *,
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    train_indices: np.ndarray,
    dims: int,
) -> dict[str, Any]:
    basis = _contrast_basis()
    source_coeff = _row_zscores(source_scores) @ basis
    target_coeff = _row_zscores(target_scores) @ basis
    fit_source = source_coeff[train_indices]
    fit_target = target_coeff[train_indices]
    cov = (fit_source.T @ fit_target) / max(1, len(train_indices) - 1)
    u, singular_values, vt = np.linalg.svd(cov, full_matrices=False)
    use_dims = min(int(dims), u.shape[1], vt.shape[0])
    return {
        "basis": basis,
        "source_rotation": u[:, :use_dims].astype(np.float64),
        "target_rotation": vt[:use_dims].T.astype(np.float64),
        "singular_values": singular_values[:use_dims].astype(np.float64),
        "dims": int(use_dims),
    }


def _score_coefficients(
    *,
    scores: np.ndarray,
    basis_config: dict[str, Any],
    side: str,
    source_transform: str,
) -> np.ndarray:
    coeff = _row_zscores(scores) @ basis_config["basis"]
    if side == "source":
        if source_transform == "sign_flip":
            coeff = -coeff
        elif source_transform == "basis_permute":
            coeff = coeff[:, [1, 2, 0]]
        rotation = basis_config.get("source_rotation")
    else:
        rotation = basis_config.get("target_rotation")
    if rotation is not None:
        coeff = coeff @ rotation
    return coeff.astype(np.float64)


def _basis_loadings(
    *,
    basis_config: dict[str, Any],
    side: str,
    source_transform: str,
) -> np.ndarray:
    loadings = np.asarray(basis_config["basis"], dtype=np.float64)
    if side == "source":
        if source_transform == "sign_flip":
            loadings = -loadings
        elif source_transform == "basis_permute":
            loadings = loadings[:, [1, 2, 0]]
        rotation = basis_config.get("source_rotation")
    else:
        rotation = basis_config.get("target_rotation")
    if rotation is not None:
        loadings = loadings @ rotation
    return loadings.astype(np.float64)


def _candidate_features(
    *,
    target_scores: np.ndarray,
    source_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    basis_config: dict[str, Any],
    source_transform: str = "matched",
) -> np.ndarray:
    target_scores = np.asarray(target_scores, dtype=np.float64)
    source_scores = np.asarray(source_scores, dtype=np.float64)
    target_z = _row_zscores(target_scores)
    target_probs = _softmax(target_scores)
    target_ranks = _rank_positions(target_scores)
    target_predictions = np.argmax(target_scores, axis=1).astype(np.int64)
    target_margin = _top2_margin(target_z)
    source_coeff = _score_coefficients(
        scores=source_scores,
        basis_config=basis_config,
        side="source",
        source_transform=source_transform,
    )
    target_coeff = _score_coefficients(
        scores=target_scores,
        basis_config=basis_config,
        side="target",
        source_transform=source_transform,
    )
    source_loadings = _basis_loadings(
        basis_config=basis_config,
        side="source",
        source_transform=source_transform,
    )
    target_loadings = _basis_loadings(
        basis_config=basis_config,
        side="target",
        source_transform=source_transform,
    )
    common_dims = source_coeff.shape[1]
    feature_rows: list[list[list[float]]] = []
    for row_index in range(target_scores.shape[0]):
        row_features: list[list[float]] = []
        source_global = source_coeff[row_index]
        target_global = target_coeff[row_index]
        diff_global = source_global - target_global
        sign_agree = np.sign(source_global) * np.sign(target_global)
        for candidate in range(CANDIDATE_COUNT):
            source_projected = source_loadings[candidate] * source_global
            target_projected = target_loadings[candidate] * target_global
            diff_projected = source_projected - target_projected
            row_features.append(
                [
                    1.0,
                    float(target_scores[row_index, candidate]),
                    float(target_z[row_index, candidate]),
                    float(target_probs[row_index, candidate]),
                    float(target_ranks[row_index, candidate]),
                    float(candidate == int(target_predictions[row_index])),
                    float(candidate == int(packet_predictions[row_index])),
                    float(packet_margins[row_index]) if candidate == int(packet_predictions[row_index]) else 0.0,
                    float(packet_margins[row_index]),
                    float(target_margin[row_index]),
                    *[1.0 if candidate == item else 0.0 for item in range(CANDIDATE_COUNT)],
                    *[1.0 if int(target_ranks[row_index, candidate]) == item else 0.0 for item in range(CANDIDATE_COUNT)],
                    *source_global.tolist(),
                    *target_global.tolist(),
                    *diff_global.tolist(),
                    *sign_agree.tolist(),
                    *source_projected.tolist(),
                    *target_projected.tolist(),
                    *diff_projected.tolist(),
                    *source_loadings[candidate].tolist(),
                    *target_loadings[candidate].tolist(),
                ]
            )
        feature_rows.append(row_features)
    result = np.asarray(feature_rows, dtype=np.float64)
    expected_dims = 10 + CANDIDATE_COUNT + CANDIDATE_COUNT + 9 * common_dims
    if result.shape[-1] != expected_dims:
        raise AssertionError(f"unexpected feature width {result.shape[-1]} != {expected_dims}")
    return result


def _fit_receiver(
    *,
    target_scores: np.ndarray,
    source_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
    basis_config: dict[str, Any],
    ridges: tuple[float, ...],
    kind: str,
) -> dict[str, Any]:
    features = _candidate_features(
        target_scores=target_scores,
        source_scores=source_scores,
        packet_predictions=packet_predictions,
        packet_margins=packet_margins,
        basis_config=basis_config,
    )
    target_predictions = np.argmax(target_scores, axis=1).astype(np.int64)
    labels: list[float] = []
    weights: list[float] = []
    for index in train_indices:
        for candidate in range(CANDIDATE_COUNT):
            labels.append(1.0 if int(answers[index]) == candidate else -1.0)
            weights.append(
                1.0
                if candidate
                in {int(answers[index]), int(target_predictions[index]), int(packet_predictions[index])}
                else 0.5
            )
    y = np.asarray(labels, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    x_fit = features[train_indices].reshape(-1, features.shape[-1])
    mean = np.mean(x_fit, axis=0)
    scale = np.std(x_fit, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_body = (x_fit - mean) / scale
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    candidates: list[dict[str, Any]] = []
    for ridge in ridges:
        weighted_x = x * sample_weights[:, None]
        xtx = x.T @ weighted_x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        xtx[0, 0] -= float(ridge)
        beta = np.linalg.solve(xtx, weighted_x.T @ y)
        predictions = _predict_receiver(
            target_scores=target_scores,
            source_scores=source_scores,
            packet_predictions=packet_predictions,
            packet_margins=packet_margins,
            basis_config=basis_config,
            mean=mean,
            scale=scale,
            beta=beta,
        )
        candidates.append(
            {
                "ridge": float(ridge),
                "train_accuracy": _accuracy(predictions, answers, train_indices),
                "beta": beta,
            }
        )
    selected = max(candidates, key=lambda item: (item["train_accuracy"], -item["ridge"]))
    return {
        "kind": kind,
        "ridge": float(selected["ridge"]),
        "train_accuracy": float(selected["train_accuracy"]),
        "basis_dims": int(basis_config.get("dims", 3)),
        "basis_singular_values": [
            float(item) for item in np.asarray(basis_config.get("singular_values", []), dtype=np.float64)
        ],
        "basis_config": basis_config,
        "mean": mean,
        "scale": scale,
        "beta": selected["beta"],
    }


def _predict_receiver(
    *,
    target_scores: np.ndarray,
    source_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    basis_config: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
    beta: np.ndarray,
    source_transform: str = "matched",
) -> np.ndarray:
    features = _candidate_features(
        target_scores=target_scores,
        source_scores=source_scores,
        packet_predictions=packet_predictions,
        packet_margins=packet_margins,
        basis_config=basis_config,
        source_transform=source_transform,
    )
    x_body = (features.reshape(-1, features.shape[-1]) - mean) / scale
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    scores = (x @ beta).reshape(target_scores.shape[0], CANDIDATE_COUNT)
    return np.argmax(scores, axis=1).astype(np.int64)


def _load_source_score_lookup(path: pathlib.Path) -> dict[str, list[float]]:
    cache = _read_json(path)
    return {
        str(row_id): [float(value) for value in scores]
        for row_id, scores in zip(cache["row_ids"], cache["source_scores"], strict=True)
    }


def _load_slice(slice_dir: pathlib.Path, source_score_lookup: dict[str, list[float]]) -> dict[str, Any]:
    slice_dir = _resolve(slice_dir)
    packet_path = slice_dir / "tinyllama_source_packet_slice_augmented.jsonl"
    target_score_path = slice_dir / "target_score_cache.json"
    gate_path = slice_dir / "hellaswag_nonqwen_receiver_family_packet_gate.json"
    packet_rows = _read_jsonl(packet_path)
    target_cache = _read_json(target_score_path)
    row_ids = [str(row["row_id"]) for row in packet_rows]
    if row_ids != [str(item) for item in target_cache["row_ids"]]:
        raise ValueError(f"packet rows and target scores are not aligned in {slice_dir}")
    missing = [row_id for row_id in row_ids if row_id not in source_score_lookup]
    if missing:
        raise ValueError(f"missing {len(missing)} source score rows for {slice_dir}")
    return {
        "slice_dir": slice_dir,
        "packet_path": packet_path,
        "target_score_path": target_score_path,
        "gate_path": gate_path,
        "packet_rows": packet_rows,
        "row_ids": row_ids,
        "answers": np.asarray([int(row["answer_index"]) for row in packet_rows], dtype=np.int64),
        "packet_predictions": np.asarray(
            [int(row["selected_prediction"]) for row in packet_rows],
            dtype=np.int64,
        ),
        "packet_margins": np.asarray(
            [float(row.get("selected_margin", 0.0)) for row in packet_rows],
            dtype=np.float64,
        ),
        "source_scores": np.asarray([source_score_lookup[row_id] for row_id in row_ids], dtype=np.float64),
        "target_scores": np.asarray(target_cache["source_scores"], dtype=np.float64),
        "target_predictions": np.asarray(target_cache["source_predictions"], dtype=np.int64),
        "gate": _read_json(gate_path),
    }


def _baseline_readout(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": name,
        "train_accuracy": _accuracy(predictions, answers, train_indices),
        "eval_accuracy": _accuracy(predictions, answers, eval_indices),
    }


def _control_predictions(
    *,
    control: str,
    slice_data: dict[str, Any],
    selected: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    packet = slice_data["packet_predictions"].copy()
    source_scores = slice_data["source_scores"].copy()
    transform = "matched"
    packet_rows = slice_data["packet_rows"]
    if control in {
        "wrong_example_hidden_prediction",
        "candidate_roll_hidden_prediction",
        "zero_hidden_prediction",
        "source_label_prediction",
        "score_only_bagged_prediction",
        "row_shuffle_packet",
        "random_same_byte_packet",
        "target_derived_packet",
        "candidate_derangement_packet",
    }:
        packet = np.asarray([int(row[control]) for row in packet_rows], dtype=np.int64)
    elif control == "source_score_row_shuffle":
        order = rng.permutation(len(source_scores))
        source_scores = source_scores[order]
    elif control == "packet_and_source_row_shuffle":
        order = rng.permutation(len(source_scores))
        source_scores = source_scores[order]
        packet = np.asarray([int(packet_rows[index]["row_shuffle_packet"]) for index in range(len(packet_rows))])
    elif control == "candidate_roll_packet_and_source_score":
        source_scores = np.roll(source_scores, shift=1, axis=1)
        packet = np.asarray([int(row["candidate_roll_hidden_prediction"]) for row in packet_rows], dtype=np.int64)
    elif control == "target_derived_packet_and_scores":
        source_scores = slice_data["target_scores"].copy()
        packet = slice_data["target_predictions"].copy()
    elif control == "basis_sign_flip_source":
        transform = "sign_flip"
    elif control == "basis_permute_source":
        transform = "basis_permute"
    else:
        raise ValueError(f"unsupported control: {control}")
    predictions = _predict_receiver(
        target_scores=slice_data["target_scores"],
        source_scores=source_scores,
        packet_predictions=packet,
        packet_margins=slice_data["packet_margins"],
        basis_config=selected["basis_config"],
        mean=selected["mean"],
        scale=selected["scale"],
        beta=selected["beta"],
        source_transform=transform,
    )
    return predictions, {"control": control, "source_transform": transform}


def _evaluate_slice(
    *,
    slice_data: dict[str, Any],
    train_prefix_rows: int,
    bootstrap_samples: int,
    ridges: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    row_count = len(slice_data["packet_rows"])
    if not 0 < int(train_prefix_rows) < row_count:
        raise ValueError("train_prefix_rows must leave held-out eval rows")
    train_indices = np.arange(int(train_prefix_rows), dtype=np.int64)
    eval_indices = np.arange(int(train_prefix_rows), row_count, dtype=np.int64)
    answers = slice_data["answers"]
    packet_predictions = slice_data["packet_predictions"]
    target_predictions = np.argmax(slice_data["target_scores"], axis=1).astype(np.int64)
    oracle_predictions = np.where(packet_predictions == answers, packet_predictions, target_predictions)

    fixed_basis = {"basis": _contrast_basis(), "dims": 3, "source_rotation": None, "target_rotation": None}
    receivers = [
        _fit_receiver(
            target_scores=slice_data["target_scores"],
            source_scores=slice_data["source_scores"],
            packet_predictions=packet_predictions,
            packet_margins=slice_data["packet_margins"],
            answers=answers,
            train_indices=train_indices,
            basis_config=fixed_basis,
            ridges=ridges,
            kind="fixed_helmert_score_simplex",
        )
    ]
    for dims in (1, 2, 3):
        receivers.append(
            _fit_receiver(
                target_scores=slice_data["target_scores"],
                source_scores=slice_data["source_scores"],
                packet_predictions=packet_predictions,
                packet_margins=slice_data["packet_margins"],
                answers=answers,
                train_indices=train_indices,
                basis_config=_fit_svd_basis(
                    source_scores=slice_data["source_scores"],
                    target_scores=slice_data["target_scores"],
                    train_indices=train_indices,
                    dims=dims,
                ),
                ridges=ridges,
                kind=f"train_svd_score_simplex_d{dims}",
            )
        )
    for receiver in receivers:
        predictions = _predict_receiver(
            target_scores=slice_data["target_scores"],
            source_scores=slice_data["source_scores"],
            packet_predictions=packet_predictions,
            packet_margins=slice_data["packet_margins"],
            basis_config=receiver["basis_config"],
            mean=receiver["mean"],
            scale=receiver["scale"],
            beta=receiver["beta"],
        )
        receiver["predictions"] = predictions
        receiver["eval_accuracy"] = _accuracy(predictions, answers, eval_indices)
    selected = max(
        receivers,
        key=lambda item: (
            item["train_accuracy"],
            item["kind"] == "fixed_helmert_score_simplex",
            -item["basis_dims"],
            -item["ridge"],
        ),
    )
    receiver_predictions = selected["predictions"]
    ci_target = _paired_ci(
        selected=receiver_predictions,
        baseline=target_predictions,
        answers=answers,
        indices=eval_indices,
        seed=seed + 1,
        samples=bootstrap_samples,
    )
    ci_packet = _paired_ci(
        selected=receiver_predictions,
        baseline=packet_predictions,
        answers=answers,
        indices=eval_indices,
        seed=seed + 2,
        samples=bootstrap_samples,
    )
    baseline_rows = [
        _baseline_readout(
            name="target_only",
            predictions=target_predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _baseline_readout(
            name="source_score_argmax",
            predictions=np.argmax(slice_data["source_scores"], axis=1).astype(np.int64),
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _baseline_readout(
            name="packet_only",
            predictions=packet_predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _baseline_readout(
            name="receiver_selected",
            predictions=receiver_predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _baseline_readout(
            name="target_or_packet_oracle",
            predictions=oracle_predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
    ]
    for field in (
        "source_label_prediction",
        "score_only_bagged_prediction",
        "wrong_example_hidden_prediction",
        "candidate_roll_hidden_prediction",
        "zero_hidden_prediction",
        "row_shuffle_packet",
        "random_same_byte_packet",
        "target_derived_packet",
        "candidate_derangement_packet",
    ):
        if field in slice_data["packet_rows"][0]:
            values = np.asarray([int(row[field]) for row in slice_data["packet_rows"]], dtype=np.int64)
            baseline_rows.append(
                _baseline_readout(
                    name=f"packet_control_{field}",
                    predictions=values,
                    answers=answers,
                    train_indices=train_indices,
                    eval_indices=eval_indices,
                )
            )
    control_rows: list[dict[str, Any]] = []
    control_names = [
        "source_score_row_shuffle",
        "packet_and_source_row_shuffle",
        "candidate_roll_packet_and_source_score",
        "target_derived_packet_and_scores",
        "basis_sign_flip_source",
        "basis_permute_source",
        "wrong_example_hidden_prediction",
        "candidate_roll_hidden_prediction",
        "zero_hidden_prediction",
        "row_shuffle_packet",
        "random_same_byte_packet",
        "target_derived_packet",
        "candidate_derangement_packet",
    ]
    for offset, control in enumerate(control_names):
        try:
            control_pred, metadata = _control_predictions(
                control=control,
                slice_data=slice_data,
                selected=selected,
                rng=np.random.default_rng(seed + 100 + offset),
            )
        except KeyError:
            continue
        ci = _paired_ci(
            selected=receiver_predictions,
            baseline=control_pred,
            answers=answers,
            indices=eval_indices,
            seed=seed + 200 + offset,
            samples=bootstrap_samples,
        )
        control_rows.append(
            {
                **_baseline_readout(
                    name=control,
                    predictions=control_pred,
                    answers=answers,
                    train_indices=train_indices,
                    eval_indices=eval_indices,
                ),
                "receiver_minus_control": ci["delta"],
                "ci95_low_vs_control": ci["ci95_low"],
                "ci95_high_vs_control": ci["ci95_high"],
                **metadata,
            }
        )
    target_gate = bool(ci_target["delta"] >= STRICT_TARGET_DELTA and ci_target["ci95_low"] > 0.0)
    packet_gate = bool(ci_packet["delta"] >= STRICT_PACKET_DELTA and ci_packet["ci95_low"] > 0.0)
    destructive_controls_pass = bool(
        control_rows
        and all(row["receiver_minus_control"] >= STRICT_TARGET_DELTA for row in control_rows)
        and all(row["ci95_low_vs_control"] > 0.0 for row in control_rows)
    )
    slice_start = slice_data["gate"]["headline"]["slice_start"]
    slice_end = slice_data["gate"]["headline"]["slice_end_exclusive"]
    headline = {
        "slice_start": int(slice_start),
        "slice_end_exclusive": int(slice_end),
        "row_count": int(row_count),
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "target_only_eval_accuracy": _accuracy(target_predictions, answers, eval_indices),
        "packet_only_eval_accuracy": _accuracy(packet_predictions, answers, eval_indices),
        "receiver_eval_accuracy": _accuracy(receiver_predictions, answers, eval_indices),
        "target_or_packet_oracle_eval_accuracy": _accuracy(oracle_predictions, answers, eval_indices),
        "receiver_minus_target_only": ci_target["delta"],
        "receiver_ci95_low_vs_target_only": ci_target["ci95_low"],
        "receiver_ci95_high_vs_target_only": ci_target["ci95_high"],
        "receiver_minus_packet_only": ci_packet["delta"],
        "receiver_ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "receiver_ci95_high_vs_packet_only": ci_packet["ci95_high"],
        "target_transfer_gate": target_gate,
        "packet_improvement_gate": packet_gate,
        "destructive_controls_pass": destructive_controls_pass,
        "selected_receiver_kind": selected["kind"],
        "selected_receiver_train_accuracy": selected["train_accuracy"],
        "selected_receiver_basis_dims": selected["basis_dims"],
        "selected_receiver_ridge": selected["ridge"],
    }
    return {
        "headline": headline,
        "baseline_rows": baseline_rows,
        "control_rows": control_rows,
        "receiver_candidates": [
            {
                "kind": item["kind"],
                "ridge": item["ridge"],
                "basis_dims": item["basis_dims"],
                "train_accuracy": item["train_accuracy"],
                "eval_accuracy": item["eval_accuracy"],
                "basis_singular_values": item["basis_singular_values"],
            }
            for item in receivers
        ],
        "pass_gate": bool(target_gate and packet_gate and destructive_controls_pass),
        "source_files": {
            "slice_dir": _display_path(slice_data["slice_dir"]),
            "packet_path": _display_path(slice_data["packet_path"]),
            "packet_sha256": _sha256_file(slice_data["packet_path"]),
            "target_score_path": _display_path(slice_data["target_score_path"]),
            "target_score_sha256": _sha256_file(slice_data["target_score_path"]),
            "gate_path": _display_path(slice_data["gate_path"]),
            "gate_sha256": _sha256_file(slice_data["gate_path"]),
        },
    }


def _weighted(slice_payloads: list[dict[str, Any]], key: str) -> float:
    total = sum(int(item["headline"]["eval_rows"]) for item in slice_payloads)
    if total <= 0:
        return 0.0
    return sum(float(item["headline"][key]) * int(item["headline"]["eval_rows"]) for item in slice_payloads) / total


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Non-Qwen Score-Simplex Receiver Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- slices: `{h['slice_count']}`",
        f"- range: `{h['range_start']}:{h['range_end_exclusive']}`",
        f"- total eval rows: `{h['total_eval_rows']}`",
        f"- target-only accuracy: `{h['weighted_target_only_eval_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['weighted_packet_only_eval_accuracy']:.6f}`",
        f"- receiver accuracy: `{h['weighted_receiver_eval_accuracy']:.6f}`",
        f"- oracle accuracy: `{h['weighted_target_or_packet_oracle_eval_accuracy']:.6f}`",
        f"- receiver minus packet-only: `{h['weighted_receiver_minus_packet_only']:.6f}`",
        f"- packet-improvement slices: `{h['packet_improvement_slice_count']}/{h['slice_count']}`",
        f"- destructive-control slices: `{h['destructive_control_slice_count']}/{h['slice_count']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    slice_dirs: tuple[pathlib.Path, ...] = DEFAULT_SLICE_DIRS,
    source_score_cache: pathlib.Path = DEFAULT_SOURCE_SCORE_CACHE,
    train_prefix_rows: int = 128,
    bootstrap_samples: int = 500,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    run_date: str = "2026-05-03",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_lookup = _load_source_score_lookup(_resolve(source_score_cache))
    slice_payloads = [
        _evaluate_slice(
            slice_data=_load_slice(slice_dir, source_lookup),
            train_prefix_rows=train_prefix_rows,
            bootstrap_samples=bootstrap_samples,
            ridges=ridges,
            seed=20260503 + index * 1000,
        )
        for index, slice_dir in enumerate(slice_dirs)
    ]
    slice_rows = [item["headline"] for item in slice_payloads]
    weighted_target = _weighted(slice_payloads, "target_only_eval_accuracy")
    weighted_packet = _weighted(slice_payloads, "packet_only_eval_accuracy")
    weighted_receiver = _weighted(slice_payloads, "receiver_eval_accuracy")
    weighted_oracle = _weighted(slice_payloads, "target_or_packet_oracle_eval_accuracy")
    headline = {
        "slice_count": len(slice_payloads),
        "range_start": min(item["headline"]["slice_start"] for item in slice_payloads),
        "range_end_exclusive": max(item["headline"]["slice_end_exclusive"] for item in slice_payloads),
        "total_rows": sum(int(item["headline"]["row_count"]) for item in slice_payloads),
        "total_train_rows": sum(int(item["headline"]["train_rows"]) for item in slice_payloads),
        "total_eval_rows": sum(int(item["headline"]["eval_rows"]) for item in slice_payloads),
        "weighted_target_only_eval_accuracy": weighted_target,
        "weighted_packet_only_eval_accuracy": weighted_packet,
        "weighted_receiver_eval_accuracy": weighted_receiver,
        "weighted_target_or_packet_oracle_eval_accuracy": weighted_oracle,
        "weighted_packet_minus_target_only": weighted_packet - weighted_target,
        "weighted_receiver_minus_target_only": weighted_receiver - weighted_target,
        "weighted_receiver_minus_packet_only": weighted_receiver - weighted_packet,
        "weighted_oracle_minus_packet_only": weighted_oracle - weighted_packet,
        "target_transfer_slice_count": sum(
            1 for item in slice_payloads if item["headline"]["target_transfer_gate"]
        ),
        "packet_improvement_slice_count": sum(
            1 for item in slice_payloads if item["headline"]["packet_improvement_gate"]
        ),
        "destructive_control_slice_count": sum(
            1 for item in slice_payloads if item["headline"]["destructive_controls_pass"]
        ),
        "min_receiver_ci95_low_vs_packet_only": min(
            float(item["headline"]["receiver_ci95_low_vs_packet_only"]) for item in slice_payloads
        ),
        "strict_target_delta_required": STRICT_TARGET_DELTA,
        "strict_packet_delta_required": STRICT_PACKET_DELTA,
    }
    pass_gate = bool(
        slice_payloads
        and all(item["headline"]["target_transfer_gate"] for item in slice_payloads)
        and all(item["headline"]["packet_improvement_gate"] for item in slice_payloads)
        and all(item["headline"]["destructive_controls_pass"] for item in slice_payloads)
    )
    interpretation = (
        "This cached gate tests the practical common-basis hypothesis suggested by the "
        "receiver-family failures: TinyLlama and Phi already share the four HellaSwag "
        "candidate slots, so row-centered score vectors are projected into a shared "
        "candidate-contrast basis and used by a target-side receiver. A pass requires "
        "beating packet-only, not only target-only."
    )
    payload = {
        "gate": "source_private_hellaswag_nonqwen_score_simplex_receiver_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires every slice to beat target-only by >=0.02 with positive paired CI, "
            "beat packet-only by >=0.005 with positive paired CI, and keep destructive source "
            "score/basis/packet controls below the selected receiver by >=0.02 with positive CI."
        ),
        "headline": headline,
        "slice_rows": slice_rows,
        "slice_payloads": slice_payloads,
        "source_score_cache": {
            "path": _display_path(source_score_cache),
            "sha256": _sha256_file(source_score_cache),
        },
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_nonqwen_score_simplex_receiver_gate.json"
    md_path = output_dir / "hellaswag_nonqwen_score_simplex_receiver_gate.md"
    slice_csv = output_dir / "slice_rows.csv"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(slice_csv, slice_rows)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path, slice_csv)
            if _resolve(path).exists()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slice-dirs", type=pathlib.Path, nargs="+", default=list(DEFAULT_SLICE_DIRS))
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--train-prefix-rows", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--run-date", default="2026-05-03")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        slice_dirs=tuple(args.slice_dirs),
        source_score_cache=args.source_score_cache,
        train_prefix_rows=args.train_prefix_rows,
        bootstrap_samples=args.bootstrap_samples,
        ridges=args.ridges,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
