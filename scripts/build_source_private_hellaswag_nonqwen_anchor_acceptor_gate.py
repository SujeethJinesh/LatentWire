from __future__ import annotations

"""Packet-preserving anchor/quantile acceptor gate for TinyLlama->Phi HellaSwag.

This gate is stricter than the score-simplex receiver diagnostic: the receiver
does not see full source scores. The source may encode only a tiny discrete
code derived from its score surface plus the packet choice. The target-side
acceptor defaults to packet-only and may switch to Phi's own top candidate only
if a fit/select split inside the train prefix supports the override.
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

from scripts import build_source_private_hellaswag_nonqwen_score_simplex_receiver_gate as simplex  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_nonqwen_anchor_acceptor_gate_20260503_validation1024_2048"
)
DEFAULT_SLICE_DIRS = simplex.DEFAULT_SLICE_DIRS
DEFAULT_SOURCE_SCORE_CACHE = simplex.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_BINS = (2, 4, 8, 16, 32, 64)
DEFAULT_ANCHORS = (4, 8, 16, 32, 64)
CANDIDATE_COUNT = 4
STRICT_TARGET_DELTA = 0.02
STRICT_PACKET_DELTA = 0.005
SELECT_PACKET_DELTA = 0.005
MIN_SELECT_HELP_COUNT = 2
MIN_SELECT_NET_HELP = 2


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return simplex._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return simplex._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    return simplex._sha256_file(path)


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return simplex._read_json(path)


def _accuracy(predictions: np.ndarray, answers: np.ndarray, indices: np.ndarray) -> float:
    return simplex._accuracy(predictions, answers, indices)


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    return simplex._paired_ci(
        selected=selected,
        baseline=baseline,
        answers=answers,
        indices=indices,
        seed=seed,
        samples=samples,
    )


def _row_zscores(scores: np.ndarray) -> np.ndarray:
    return simplex._row_zscores(scores)


def _softmax(scores: np.ndarray) -> np.ndarray:
    return simplex._softmax(scores)


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    return simplex._top2_margin(_row_zscores(scores))


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    return simplex._rank_positions(scores)


def _one_hot(values: np.ndarray, width: int) -> np.ndarray:
    eye = np.eye(int(width), dtype=np.float64)
    return eye[np.mod(values.astype(np.int64), int(width))]


def _candidate_only_config() -> dict[str, Any]:
    return {"name": "candidate_only_no_override", "kind": "candidate_only", "codebook_size": 1}


def _code_configs(
    *,
    source_scores: np.ndarray,
    packet_predictions: np.ndarray,
    fit_indices: np.ndarray,
    bins: tuple[int, ...],
    anchors: tuple[int, ...],
) -> list[dict[str, Any]]:
    zscores = _row_zscores(source_scores)
    probs = _softmax(source_scores)
    source_top = np.argmax(source_scores, axis=1).astype(np.int64)
    row_index = np.arange(len(packet_predictions), dtype=np.int64)
    scalar_features = {
        "packet_z": zscores[row_index, packet_predictions],
        "packet_prob": probs[row_index, packet_predictions],
        "source_margin": _top2_margin(source_scores),
        "source_entropy": _entropy(probs),
        "top_agrees_packet": (source_top == packet_predictions).astype(np.float64),
    }
    configs: list[dict[str, Any]] = [_candidate_only_config()]
    for feature_name, values in scalar_features.items():
        for bin_count in bins:
            if int(bin_count) * CANDIDATE_COUNT > 256:
                continue
            edges = np.quantile(values[fit_indices], np.linspace(0.0, 1.0, int(bin_count) + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            configs.append(
                {
                    "name": f"{feature_name}_q{int(bin_count)}",
                    "kind": "quantile",
                    "feature_name": feature_name,
                    "codebook_size": int(bin_count),
                    "edges": edges.astype(np.float64),
                }
            )
    contrast = zscores @ simplex._contrast_basis()
    contrast_norm = np.linalg.norm(contrast, axis=1, keepdims=True)
    contrast = contrast / np.where(contrast_norm < 1e-8, 1.0, contrast_norm)
    for anchor_count in anchors:
        if int(anchor_count) * CANDIDATE_COUNT > 256 or int(anchor_count) > len(fit_indices):
            continue
        anchor_positions = np.linspace(0, len(fit_indices) - 1, int(anchor_count), dtype=np.int64)
        anchor_indices = fit_indices[anchor_positions]
        configs.append(
            {
                "name": f"anchor_relative_k{int(anchor_count)}",
                "kind": "anchor_relative",
                "feature_name": "score_contrast_cosine",
                "codebook_size": int(anchor_count),
                "anchor_indices": [int(item) for item in anchor_indices],
                "anchors": contrast[anchor_indices].astype(np.float64),
            }
        )
    return configs


def _encode_codes(
    *,
    config: dict[str, Any],
    source_scores: np.ndarray,
    packet_predictions: np.ndarray,
) -> np.ndarray:
    if config["kind"] == "candidate_only":
        return np.zeros(len(packet_predictions), dtype=np.int64)
    zscores = _row_zscores(source_scores)
    probs = _softmax(source_scores)
    row_index = np.arange(len(packet_predictions), dtype=np.int64)
    if config["kind"] == "quantile":
        feature_name = config["feature_name"]
        if feature_name == "packet_z":
            values = zscores[row_index, packet_predictions]
        elif feature_name == "packet_prob":
            values = probs[row_index, packet_predictions]
        elif feature_name == "source_margin":
            values = _top2_margin(source_scores)
        elif feature_name == "source_entropy":
            values = _entropy(probs)
        elif feature_name == "top_agrees_packet":
            values = (
                np.argmax(source_scores, axis=1).astype(np.int64) == packet_predictions
            ).astype(np.float64)
        else:
            raise ValueError(f"unsupported quantile feature: {feature_name}")
        return np.clip(
            np.searchsorted(config["edges"], values, side="right") - 1,
            0,
            int(config["codebook_size"]) - 1,
        ).astype(np.int64)
    if config["kind"] == "anchor_relative":
        contrast = zscores @ simplex._contrast_basis()
        contrast_norm = np.linalg.norm(contrast, axis=1, keepdims=True)
        contrast = contrast / np.where(contrast_norm < 1e-8, 1.0, contrast_norm)
        return np.argmax(contrast @ np.asarray(config["anchors"], dtype=np.float64).T, axis=1).astype(np.int64)
    raise ValueError(f"unsupported code config kind: {config['kind']}")


def _raw_packet_bytes(codebook_size: int) -> int:
    states = int(codebook_size) * CANDIDATE_COUNT
    return max(1, int(math.ceil(math.log2(max(2, states)) / 8.0)))


def _framed_packet_bytes(raw_bytes: int) -> int:
    return int(raw_bytes) + 3


def _target_features(target_scores: np.ndarray, packet: np.ndarray, code: np.ndarray, codebook_size: int) -> np.ndarray:
    zscores = _row_zscores(target_scores)
    probs = _softmax(target_scores)
    ranks = _rank_positions(target_scores)
    target_top = np.argmax(target_scores, axis=1).astype(np.int64)
    row_index = np.arange(len(packet), dtype=np.int64)
    codebook_size = int(codebook_size)
    parts = [
        np.ones((len(packet), 1), dtype=np.float64),
        zscores,
        probs,
        _one_hot(packet, CANDIDATE_COUNT),
        _one_hot(target_top, CANDIDATE_COUNT),
        _one_hot(code, codebook_size),
        (target_top == packet).astype(np.float64)[:, None],
        zscores[row_index, packet][:, None],
        zscores[row_index, target_top][:, None],
        (zscores[row_index, target_top] - zscores[row_index, packet])[:, None],
        probs[row_index, packet][:, None],
        probs[row_index, target_top][:, None],
        _top2_margin(target_scores)[:, None],
        _entropy(probs)[:, None],
        (ranks[row_index, packet].astype(np.float64) / float(CANDIDATE_COUNT - 1))[:, None],
        (code.astype(np.float64) / max(1.0, float(codebook_size - 1)))[:, None],
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


def _standardize_fit(features: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features[fit_indices], axis=0)
    scale = np.std(features[fit_indices], axis=0)
    return mean.astype(np.float64), np.where(scale < 1e-8, 1.0, scale).astype(np.float64)


def _fit_ridge_scores(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    mean, scale = _standardize_fit(features, fit_indices)
    standardized = (features - mean) / scale
    x = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), standardized], axis=1)
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x[fit_indices].T @ x[fit_indices] + reg, x[fit_indices].T @ targets[fit_indices])
    return {"mean": mean, "scale": scale, "beta": beta.astype(np.float64), "ridge": float(ridge)}


def _apply_acceptor_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    x = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), standardized], axis=1)
    return (x @ model["beta"]).astype(np.float64)


def _predict_from_scores(
    *,
    score: np.ndarray,
    threshold: float,
    packet: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    accept = (target.astype(np.int64) != packet.astype(np.int64)) & (score.astype(np.float64) >= float(threshold))
    return np.where(accept, target, packet).astype(np.int64)


def _help_harm(predictions: np.ndarray, packet: np.ndarray, answers: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    selected_correct = predictions[indices] == answers[indices]
    packet_correct = packet[indices] == answers[indices]
    help_count = int(np.sum(selected_correct & ~packet_correct))
    harm_count = int(np.sum(~selected_correct & packet_correct))
    return {
        "help_count": help_count,
        "harm_count": harm_count,
        "net_help": help_count - harm_count,
        "override_count": int(np.sum(predictions[indices] != packet[indices])),
        "override_rate": float(np.mean(predictions[indices] != packet[indices])),
    }


def _candidate_row(
    *,
    config: dict[str, Any],
    ridge: float | None,
    threshold: float,
    predictions: np.ndarray,
    answers: np.ndarray,
    packet: np.ndarray,
    fit_indices: np.ndarray,
    select_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, Any]:
    select_help = _help_harm(predictions, packet, answers, select_indices)
    eval_help = _help_harm(predictions, packet, answers, eval_indices)
    packet_select = _accuracy(packet, answers, select_indices)
    return {
        "config_name": config["name"],
        "codebook_size": int(config["codebook_size"]),
        "raw_packet_bytes": _raw_packet_bytes(int(config["codebook_size"])),
        "framed_packet_bytes": _framed_packet_bytes(_raw_packet_bytes(int(config["codebook_size"]))),
        "ridge": None if ridge is None else float(ridge),
        "threshold": float(threshold),
        "fit_accuracy": _accuracy(predictions, answers, fit_indices),
        "select_accuracy": _accuracy(predictions, answers, select_indices),
        "select_packet_accuracy": packet_select,
        "select_delta_vs_packet": _accuracy(predictions, answers, select_indices) - packet_select,
        "eval_accuracy": _accuracy(predictions, answers, eval_indices),
        "eval_delta_vs_packet": _accuracy(predictions, answers, eval_indices) - _accuracy(packet, answers, eval_indices),
        "select_help_count": select_help["help_count"],
        "select_harm_count": select_help["harm_count"],
        "select_override_rate": select_help["override_rate"],
        "eval_help_count": eval_help["help_count"],
        "eval_harm_count": eval_help["harm_count"],
        "eval_override_rate": eval_help["override_rate"],
    }


def _select_acceptor(
    *,
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    packet: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    select_indices: np.ndarray,
    eval_indices: np.ndarray,
    ridges: tuple[float, ...],
    bins: tuple[int, ...],
    anchors: tuple[int, ...],
) -> dict[str, Any]:
    target_top = np.argmax(target_scores, axis=1).astype(np.int64)
    benefit = ((target_top == answers) & (packet != answers)).astype(np.float64) - 2.0 * (
        (target_top != answers) & (packet == answers)
    ).astype(np.float64)
    candidates: list[dict[str, Any]] = []
    no_override_score = np.full(len(packet), np.inf, dtype=np.float64)
    no_override_pred = packet.copy()
    no_override_config = _candidate_only_config()
    candidates.append(
        {
            **_candidate_row(
                config=no_override_config,
                ridge=None,
                threshold=np.inf,
                predictions=no_override_pred,
                answers=answers,
                packet=packet,
                fit_indices=fit_indices,
                select_indices=select_indices,
                eval_indices=eval_indices,
            ),
            "config": no_override_config,
            "codes": np.zeros(len(packet), dtype=np.int64),
            "model": None,
            "score": no_override_score,
            "predictions": no_override_pred,
        }
    )
    for config in _code_configs(
        source_scores=source_scores,
        packet_predictions=packet,
        fit_indices=fit_indices,
        bins=bins,
        anchors=anchors,
    ):
        if config["kind"] == "candidate_only":
            continue
        codes = _encode_codes(config=config, source_scores=source_scores, packet_predictions=packet)
        features = _target_features(target_scores, packet, codes, int(config["codebook_size"]))
        for ridge in ridges:
            model = _fit_ridge_scores(
                features=features,
                targets=benefit,
                fit_indices=fit_indices,
                ridge=float(ridge),
            )
            score = _apply_acceptor_scores(features, model)
            thresholds = sorted(set(np.quantile(score[fit_indices], np.linspace(0.0, 1.0, 9)).tolist() + np.quantile(score[select_indices], np.linspace(0.0, 1.0, 9)).tolist() + [0.0]))
            for threshold in thresholds:
                predictions = _predict_from_scores(
                    score=score,
                    threshold=float(threshold),
                    packet=packet,
                    target=target_top,
                )
                row = _candidate_row(
                    config=config,
                    ridge=float(ridge),
                    threshold=float(threshold),
                    predictions=predictions,
                    answers=answers,
                    packet=packet,
                    fit_indices=fit_indices,
                    select_indices=select_indices,
                    eval_indices=eval_indices,
                )
                row.update(
                    {
                        "config": config,
                        "codes": codes,
                        "model": model,
                        "score": score,
                        "predictions": predictions,
                    }
                )
                candidates.append(row)
    viable = [
        row
        for row in candidates
        if row["select_delta_vs_packet"] >= SELECT_PACKET_DELTA
        and row["select_help_count"] >= MIN_SELECT_HELP_COUNT
        and row["select_help_count"] - row["select_harm_count"] >= MIN_SELECT_NET_HELP
    ]
    selected = max(
        viable or [candidates[0]],
        key=lambda row: (
            row["select_accuracy"],
            row["select_delta_vs_packet"],
            -row["select_harm_count"],
            -row["raw_packet_bytes"],
            -row["select_override_rate"],
        ),
    )
    selected["candidate_rows"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"config", "codes", "model", "score", "predictions"}
        }
        for row in sorted(
            candidates,
            key=lambda row: (row["select_accuracy"], row["select_delta_vs_packet"], -row["select_harm_count"]),
            reverse=True,
        )[:40]
    ]
    return selected


def _predict_selected(
    *,
    selected: dict[str, Any],
    target_scores: np.ndarray,
    packet: np.ndarray,
    source_scores: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if selected["model"] is None:
        return packet.astype(np.int64)
    codes = _encode_codes(
        config=selected["config"],
        source_scores=source_scores,
        packet_predictions=packet.astype(np.int64),
    )
    if rng is not None and selected["config"]["codebook_size"] > 1:
        codes = rng.integers(0, int(selected["config"]["codebook_size"]), size=len(packet), dtype=np.int64)
    features = _target_features(target_scores, packet.astype(np.int64), codes, int(selected["config"]["codebook_size"]))
    score = _apply_acceptor_scores(features, selected["model"])
    target_top = np.argmax(target_scores, axis=1).astype(np.int64)
    return _predict_from_scores(
        score=score,
        threshold=float(selected["threshold"]),
        packet=packet.astype(np.int64),
        target=target_top,
    )


def _baseline_row(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    select_indices: np.ndarray,
    eval_indices: np.ndarray,
    packet: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ci = _paired_ci(
        selected=predictions,
        baseline=packet,
        answers=answers,
        indices=eval_indices,
        seed=seed,
        samples=bootstrap_samples,
    )
    return {
        "name": name,
        "fit_accuracy": _accuracy(predictions, answers, fit_indices),
        "select_accuracy": _accuracy(predictions, answers, select_indices),
        "eval_accuracy": _accuracy(predictions, answers, eval_indices),
        "delta_vs_packet_only": ci["delta"],
        "ci95_low_vs_packet_only": ci["ci95_low"],
        "ci95_high_vs_packet_only": ci["ci95_high"],
        **_help_harm(predictions, packet, answers, eval_indices),
    }


def _evaluate_slice(
    *,
    slice_data: dict[str, Any],
    train_prefix_rows: int,
    bootstrap_samples: int,
    ridges: tuple[float, ...],
    bins: tuple[int, ...],
    anchors: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    row_count = len(slice_data["packet_rows"])
    if int(train_prefix_rows) < 4 or int(train_prefix_rows) >= row_count:
        raise ValueError("train_prefix_rows must leave fit/select/eval rows")
    fit_rows = int(train_prefix_rows) // 2
    select_rows = int(train_prefix_rows) - fit_rows
    fit_indices = np.arange(0, fit_rows, dtype=np.int64)
    select_indices = np.arange(fit_rows, fit_rows + select_rows, dtype=np.int64)
    eval_indices = np.arange(int(train_prefix_rows), row_count, dtype=np.int64)
    answers = slice_data["answers"]
    packet = slice_data["packet_predictions"].astype(np.int64)
    target_scores = slice_data["target_scores"].astype(np.float64)
    source_scores = slice_data["source_scores"].astype(np.float64)
    target_top = np.argmax(target_scores, axis=1).astype(np.int64)
    oracle = np.where(packet == answers, packet, target_top).astype(np.int64)
    selected = _select_acceptor(
        source_scores=source_scores,
        target_scores=target_scores,
        packet=packet,
        answers=answers,
        fit_indices=fit_indices,
        select_indices=select_indices,
        eval_indices=eval_indices,
        ridges=ridges,
        bins=bins,
        anchors=anchors,
    )
    predictions = selected["predictions"].astype(np.int64)
    ci_target = _paired_ci(
        selected=predictions,
        baseline=target_top,
        answers=answers,
        indices=eval_indices,
        seed=seed + 1,
        samples=bootstrap_samples,
    )
    ci_packet = _paired_ci(
        selected=predictions,
        baseline=packet,
        answers=answers,
        indices=eval_indices,
        seed=seed + 2,
        samples=bootstrap_samples,
    )
    baseline_rows = [
        _baseline_row(
            name="target_only",
            predictions=target_top,
            answers=answers,
            fit_indices=fit_indices,
            select_indices=select_indices,
            eval_indices=eval_indices,
            packet=packet,
            seed=seed + 10,
            bootstrap_samples=bootstrap_samples,
        ),
        _baseline_row(
            name="packet_only",
            predictions=packet,
            answers=answers,
            fit_indices=fit_indices,
            select_indices=select_indices,
            eval_indices=eval_indices,
            packet=packet,
            seed=seed + 11,
            bootstrap_samples=bootstrap_samples,
        ),
        _baseline_row(
            name="target_or_packet_oracle",
            predictions=oracle,
            answers=answers,
            fit_indices=fit_indices,
            select_indices=select_indices,
            eval_indices=eval_indices,
            packet=packet,
            seed=seed + 12,
            bootstrap_samples=bootstrap_samples,
        ),
        _baseline_row(
            name="anchor_acceptor_selected",
            predictions=predictions,
            answers=answers,
            fit_indices=fit_indices,
            select_indices=select_indices,
            eval_indices=eval_indices,
            packet=packet,
            seed=seed + 13,
            bootstrap_samples=bootstrap_samples,
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
                _baseline_row(
                    name=f"packet_control_{field}",
                    predictions=values,
                    answers=answers,
                    fit_indices=fit_indices,
                    select_indices=select_indices,
                    eval_indices=eval_indices,
                    packet=packet,
                    seed=seed + 20,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    control_specs = {
        "source_code_row_shuffle": (packet, source_scores[np.random.default_rng(seed + 100).permutation(len(source_scores))]),
        "packet_and_source_code_row_shuffle": (
            np.asarray([int(row["row_shuffle_packet"]) for row in slice_data["packet_rows"]], dtype=np.int64),
            source_scores[np.random.default_rng(seed + 101).permutation(len(source_scores))],
        ),
        "candidate_roll_packet_and_source_code": (
            np.asarray([int(row["candidate_roll_hidden_prediction"]) for row in slice_data["packet_rows"]], dtype=np.int64),
            np.roll(source_scores, shift=1, axis=1),
        ),
        "target_derived_packet_and_code": (target_top, target_scores),
        "basis_permute_source_code": (packet, source_scores[:, [1, 2, 3, 0]]),
    }
    control_rows: list[dict[str, Any]] = []
    for offset, (name, (control_packet, control_source_scores)) in enumerate(control_specs.items()):
        control_predictions = _predict_selected(
            selected=selected,
            target_scores=target_scores,
            packet=control_packet,
            source_scores=control_source_scores,
        )
        ci = _paired_ci(
            selected=predictions,
            baseline=control_predictions,
            answers=answers,
            indices=eval_indices,
            seed=seed + 200 + offset,
            samples=bootstrap_samples,
        )
        control_rows.append(
            {
                **_baseline_row(
                    name=name,
                    predictions=control_predictions,
                    answers=answers,
                    fit_indices=fit_indices,
                    select_indices=select_indices,
                    eval_indices=eval_indices,
                    packet=packet,
                    seed=seed + 300 + offset,
                    bootstrap_samples=bootstrap_samples,
                ),
                "receiver_minus_control": ci["delta"],
                "ci95_low_vs_control": ci["ci95_low"],
                "ci95_high_vs_control": ci["ci95_high"],
            }
        )
    raw_bytes = int(selected["raw_packet_bytes"])
    framed_bytes = int(selected["framed_packet_bytes"])
    target_gate = bool(ci_target["delta"] >= STRICT_TARGET_DELTA and ci_target["ci95_low"] > 0.0)
    packet_gate = bool(ci_packet["delta"] >= STRICT_PACKET_DELTA and ci_packet["ci95_low"] > 0.0)
    preserve_gate = bool(abs(ci_packet["delta"]) < 1e-12 and raw_bytes <= 1)
    slice_start = int(slice_data["gate"]["headline"]["slice_start"])
    slice_end = int(slice_data["gate"]["headline"]["slice_end_exclusive"])
    headline = {
        "slice_start": slice_start,
        "slice_end_exclusive": slice_end,
        "row_count": int(row_count),
        "fit_rows": int(len(fit_indices)),
        "select_rows": int(len(select_indices)),
        "eval_rows": int(len(eval_indices)),
        "target_only_eval_accuracy": _accuracy(target_top, answers, eval_indices),
        "packet_only_eval_accuracy": _accuracy(packet, answers, eval_indices),
        "anchor_acceptor_eval_accuracy": _accuracy(predictions, answers, eval_indices),
        "target_or_packet_oracle_eval_accuracy": _accuracy(oracle, answers, eval_indices),
        "anchor_acceptor_minus_target_only": ci_target["delta"],
        "anchor_acceptor_ci95_low_vs_target_only": ci_target["ci95_low"],
        "anchor_acceptor_ci95_high_vs_target_only": ci_target["ci95_high"],
        "anchor_acceptor_minus_packet_only": ci_packet["delta"],
        "anchor_acceptor_ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "anchor_acceptor_ci95_high_vs_packet_only": ci_packet["ci95_high"],
        "target_transfer_gate": target_gate,
        "packet_improvement_gate": packet_gate,
        "candidate_only_preservation_gate": preserve_gate,
        "selected_config_name": selected["config_name"],
        "selected_codebook_size": int(selected["codebook_size"]),
        "selected_raw_packet_bytes": raw_bytes,
        "selected_framed_packet_bytes": framed_bytes,
        "selected_select_delta_vs_packet": float(selected["select_delta_vs_packet"]),
        "selected_eval_override_rate": float(selected["eval_override_rate"]),
        "selected_eval_help_count": int(selected["eval_help_count"]),
        "selected_eval_harm_count": int(selected["eval_harm_count"]),
    }
    return {
        "headline": headline,
        "baseline_rows": baseline_rows,
        "control_rows": control_rows,
        "candidate_rows": selected["candidate_rows"],
        "pass_gate": bool(target_gate and packet_gate),
        "preservation_gate": preserve_gate,
        "source_files": {
            "slice_dir": _display_path(slice_data["slice_dir"]),
            "packet_path": _display_path(slice_data["packet_path"]),
            "packet_sha256": _sha256_file(slice_data["packet_path"]),
            "target_score_path": _display_path(slice_data["target_score_path"]),
            "target_score_sha256": _sha256_file(slice_data["target_score_path"]),
        },
    }


def _weighted(slice_payloads: list[dict[str, Any]], key: str) -> float:
    total = sum(int(item["headline"]["eval_rows"]) for item in slice_payloads)
    return sum(float(item["headline"][key]) * int(item["headline"]["eval_rows"]) for item in slice_payloads) / total


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Non-Qwen Anchor Acceptor Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- preservation gate: `{payload['preservation_gate']}`",
        f"- slices: `{h['slice_count']}`",
        f"- range: `{h['range_start']}:{h['range_end_exclusive']}`",
        f"- total eval rows: `{h['total_eval_rows']}`",
        f"- target-only accuracy: `{h['weighted_target_only_eval_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['weighted_packet_only_eval_accuracy']:.6f}`",
        f"- anchor acceptor accuracy: `{h['weighted_anchor_acceptor_eval_accuracy']:.6f}`",
        f"- oracle accuracy: `{h['weighted_target_or_packet_oracle_eval_accuracy']:.6f}`",
        f"- acceptor minus packet-only: `{h['weighted_anchor_acceptor_minus_packet_only']:.6f}`",
        f"- max selected raw bytes: `{h['max_selected_raw_packet_bytes']}`",
        f"- selected configs: `{', '.join(h['selected_config_names'])}`",
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
    bins: tuple[int, ...] = DEFAULT_BINS,
    anchors: tuple[int, ...] = DEFAULT_ANCHORS,
    run_date: str = "2026-05-03",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_lookup = simplex._load_source_score_lookup(_resolve(source_score_cache))
    slice_payloads = [
        _evaluate_slice(
            slice_data=simplex._load_slice(slice_dir, source_lookup),
            train_prefix_rows=train_prefix_rows,
            bootstrap_samples=bootstrap_samples,
            ridges=ridges,
            bins=bins,
            anchors=anchors,
            seed=20260503 + index * 1000,
        )
        for index, slice_dir in enumerate(slice_dirs)
    ]
    slice_rows = [item["headline"] for item in slice_payloads]
    headline = {
        "slice_count": len(slice_payloads),
        "range_start": min(item["headline"]["slice_start"] for item in slice_payloads),
        "range_end_exclusive": max(item["headline"]["slice_end_exclusive"] for item in slice_payloads),
        "total_eval_rows": sum(int(item["headline"]["eval_rows"]) for item in slice_payloads),
        "weighted_target_only_eval_accuracy": _weighted(slice_payloads, "target_only_eval_accuracy"),
        "weighted_packet_only_eval_accuracy": _weighted(slice_payloads, "packet_only_eval_accuracy"),
        "weighted_anchor_acceptor_eval_accuracy": _weighted(slice_payloads, "anchor_acceptor_eval_accuracy"),
        "weighted_target_or_packet_oracle_eval_accuracy": _weighted(slice_payloads, "target_or_packet_oracle_eval_accuracy"),
        "weighted_anchor_acceptor_minus_target_only": _weighted(slice_payloads, "anchor_acceptor_minus_target_only"),
        "weighted_anchor_acceptor_minus_packet_only": _weighted(slice_payloads, "anchor_acceptor_minus_packet_only"),
        "weighted_oracle_minus_packet_only": _weighted(slice_payloads, "target_or_packet_oracle_eval_accuracy") - _weighted(slice_payloads, "packet_only_eval_accuracy"),
        "target_transfer_slice_count": sum(1 for item in slice_payloads if item["headline"]["target_transfer_gate"]),
        "packet_improvement_slice_count": sum(1 for item in slice_payloads if item["headline"]["packet_improvement_gate"]),
        "candidate_only_preservation_slice_count": sum(1 for item in slice_payloads if item["headline"]["candidate_only_preservation_gate"]),
        "max_selected_raw_packet_bytes": max(int(item["headline"]["selected_raw_packet_bytes"]) for item in slice_payloads),
        "max_selected_framed_packet_bytes": max(int(item["headline"]["selected_framed_packet_bytes"]) for item in slice_payloads),
        "selected_config_names": [item["headline"]["selected_config_name"] for item in slice_payloads],
        "min_anchor_acceptor_ci95_low_vs_packet_only": min(float(item["headline"]["anchor_acceptor_ci95_low_vs_packet_only"]) for item in slice_payloads),
    }
    pass_gate = bool(
        slice_payloads
        and all(item["headline"]["target_transfer_gate"] for item in slice_payloads)
        and all(item["headline"]["packet_improvement_gate"] for item in slice_payloads)
    )
    preservation_gate = bool(
        slice_payloads
        and all(item["headline"]["candidate_only_preservation_gate"] for item in slice_payloads)
    )
    interpretation = (
        "This gate tests a packet-preserving acceptor. Unlike the failed score-simplex receiver, "
        "the receiver only sees a discrete source code plus the packet choice; if the fit/select "
        "split cannot justify overrides, the selected policy falls back to candidate-only packet."
    )
    payload = {
        "gate": "source_private_hellaswag_nonqwen_anchor_acceptor_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "preservation_gate": preservation_gate,
        "pass_rule": (
            "Pass requires each slice to beat target-only by >=0.02 with positive paired CI "
            "and beat packet-only by >=0.005 with positive paired CI. Preservation records "
            "whether the cautious policy keeps packet-only accuracy with <=1 raw byte."
        ),
        "headline": headline,
        "slice_rows": slice_rows,
        "slice_payloads": slice_payloads,
        "packet_contract": {
            "receiver_visible_source_fields": ["packet_candidate", "discrete_source_code"],
            "forbidden_source_fields": ["source_text", "source_kv", "raw_hidden", "raw_scores", "source_logits"],
            "max_raw_packet_bytes": headline["max_selected_raw_packet_bytes"],
            "max_framed_packet_bytes": headline["max_selected_framed_packet_bytes"],
        },
        "source_score_cache": {
            "path": _display_path(source_score_cache),
            "sha256": _sha256_file(source_score_cache),
        },
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_nonqwen_anchor_acceptor_gate.json"
    md_path = output_dir / "hellaswag_nonqwen_anchor_acceptor_gate.md"
    csv_path = output_dir / "slice_rows.csv"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(csv_path, slice_rows)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path, csv_path)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slice-dirs", type=pathlib.Path, nargs="+", default=list(DEFAULT_SLICE_DIRS))
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--train-prefix-rows", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--bins", type=_parse_int_tuple, default=DEFAULT_BINS)
    parser.add_argument("--anchors", type=_parse_int_tuple, default=DEFAULT_ANCHORS)
    parser.add_argument("--run-date", default="2026-05-03")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        slice_dirs=tuple(args.slice_dirs),
        source_score_cache=args.source_score_cache,
        train_prefix_rows=args.train_prefix_rows,
        bootstrap_samples=args.bootstrap_samples,
        ridges=args.ridges,
        bins=args.bins,
        anchors=args.anchors,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
