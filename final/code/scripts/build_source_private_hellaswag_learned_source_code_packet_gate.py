from __future__ import annotations

"""Train-only learned discrete source-code packet gate for HellaSwag."""

import argparse
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

from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_learned_source_code_packet_gate_20260502")
DEFAULT_MAX_CODES = 256
STRICT_DELTA = 0.010
STRICT_PRIOR_SCOUT_DELTA = 0.005
BEST_PRIOR_RECEIVER_SCOUT_ACCURACY = 0.620594
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 1
FRAMED_PACKET_BYTES = 4
CANDIDATE_COUNT = 4
DEFAULT_QUANTILE_BINS = (2, 4, 8, 16, 32, 64)
DEFAULT_KMEANS_COUNTS = (4, 8, 16, 32, 64)
DEFAULT_DECODER_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


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
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _packet_bytes_for_codebook(codebook_size: int) -> int:
    if codebook_size < 1:
        raise ValueError("codebook_size must be positive")
    return max(1, int(math.ceil(math.log2(codebook_size) / 8.0)))


def _source_feature_matrix(scores: np.ndarray, packet: np.ndarray) -> np.ndarray:
    zscores = wz._row_zscores(scores)
    packet = packet.astype(np.int64)
    order = np.argsort(-zscores, axis=1)
    raw_top = order[:, 0]
    top = zscores[np.arange(len(zscores)), order[:, 0]]
    runner_up = zscores[np.arange(len(zscores)), order[:, 1]]
    packet_z = zscores[np.arange(len(zscores)), packet]
    packet_rank = np.zeros(len(zscores), dtype=np.float64)
    for row_index, row_order in enumerate(order):
        packet_rank[row_index] = float(np.where(row_order == packet[row_index])[0][0])
    return np.column_stack(
        [
            zscores,
            packet_z,
            top - runner_up,
            packet_rank / float(CANDIDATE_COUNT - 1),
            (packet == raw_top).astype(np.float64),
            packet.astype(np.float64) / float(CANDIDATE_COUNT - 1),
        ]
    ).astype(np.float64)


def _standardize_from_fit(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_features[fit_indices], axis=0)
    scale = np.std(train_features[fit_indices], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train_features - mean) / scale, (eval_features - mean) / scale, mean, scale


def _fit_quantile_edges(values: np.ndarray, fit_indices: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("quantile encoder requires at least two bins")
    edges = np.quantile(values[fit_indices], np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges.astype(np.float64)


def _apply_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2).astype(
        np.int64
    )


def _fit_kmeans(
    features: np.ndarray,
    fit_indices: np.ndarray,
    clusters: int,
    *,
    seed: int,
    iterations: int,
) -> np.ndarray:
    if clusters < 2:
        raise ValueError("kmeans encoder requires at least two clusters")
    fit_features = features[fit_indices]
    rng = np.random.default_rng(seed)
    if len(fit_features) < clusters:
        raise ValueError("more clusters than fit rows")
    centers = fit_features[rng.choice(len(fit_features), size=int(clusters), replace=False)].copy()
    for _ in range(int(iterations)):
        distances = np.sum((fit_features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        for cluster_index in range(int(clusters)):
            local = labels == cluster_index
            if np.any(local):
                centers[cluster_index] = np.mean(fit_features[local], axis=0)
    return centers.astype(np.float64)


def _nearest_center_codes(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)


def _encode_source_codes(
    *,
    config: dict[str, Any],
    train_source_features: np.ndarray,
    eval_source_features: np.ndarray,
    train_packet: np.ndarray,
    eval_packet: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    kind = str(config["kind"])
    if kind == "candidate_only":
        train_code = train_packet.astype(np.int64)
        eval_code = eval_packet.astype(np.int64)
        return {
            "train_code": train_code,
            "eval_code": eval_code,
            "codebook_size": CANDIDATE_COUNT,
            "encoder_audit": {"kind": kind},
        }
    if kind == "quantile":
        feature_name = str(config["feature_name"])
        feature_index = int(config["feature_index"])
        bins = int(config["bins"])
        edges = _fit_quantile_edges(train_source_features[:, feature_index], fit_indices, bins)
        train_bin = _apply_edges(train_source_features[:, feature_index], edges)
        eval_bin = _apply_edges(eval_source_features[:, feature_index], edges)
        return {
            "train_code": (train_bin * CANDIDATE_COUNT + train_packet).astype(np.int64),
            "eval_code": (eval_bin * CANDIDATE_COUNT + eval_packet).astype(np.int64),
            "codebook_size": int(bins * CANDIDATE_COUNT),
            "encoder_audit": {
                "kind": kind,
                "feature_name": feature_name,
                "feature_index": feature_index,
                "bins": bins,
                "edges": [float(item) for item in edges],
            },
        }
    if kind == "kmeans":
        clusters = int(config["clusters"])
        seed = int(config["seed"])
        train_std, eval_std, mean, scale = _standardize_from_fit(
            train_source_features,
            eval_source_features,
            fit_indices,
        )
        centers = _fit_kmeans(
            train_std,
            fit_indices,
            clusters,
            seed=seed,
            iterations=int(config.get("iterations", 25)),
        )
        train_cluster = _nearest_center_codes(train_std, centers)
        eval_cluster = _nearest_center_codes(eval_std, centers)
        return {
            "train_code": (train_cluster * CANDIDATE_COUNT + train_packet).astype(np.int64),
            "eval_code": (eval_cluster * CANDIDATE_COUNT + eval_packet).astype(np.int64),
            "codebook_size": int(clusters * CANDIDATE_COUNT),
            "encoder_audit": {
                "kind": kind,
                "clusters": clusters,
                "seed": seed,
                "iterations": int(config.get("iterations", 25)),
                "feature_mean": [float(item) for item in mean],
                "feature_scale": [float(item) for item in scale],
                "centers_shape": list(centers.shape),
            },
        }
    raise ValueError(f"unsupported encoder kind: {kind}")


def _candidate_decoder_features(
    *,
    qwen_scores: np.ndarray,
    qwen_target: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_hybrid: np.ndarray,
    source_code: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    row_count = len(source_code)
    qwen_z = wz._row_zscores(qwen_scores)
    candidate_eye = np.eye(CANDIDATE_COUNT, dtype=np.float64)
    source_code = source_code.astype(np.int64)
    source_packet = np.mod(source_code, CANDIDATE_COUNT).astype(np.int64)
    code_eye = np.eye(int(codebook_size), dtype=np.float64)
    code_one_hot = code_eye[source_code]
    features: list[np.ndarray] = []
    for candidate in range(CANDIDATE_COUNT):
        code_candidate_interaction = np.zeros(
            (row_count, int(codebook_size) * CANDIDATE_COUNT),
            dtype=np.float64,
        )
        start = candidate * int(codebook_size)
        code_candidate_interaction[:, start : start + int(codebook_size)] = code_one_hot
        parts = [
            qwen_z[:, candidate : candidate + 1],
            (qwen_target.astype(np.int64) == candidate)[:, None].astype(np.float64),
            (qwen_mean.astype(np.int64) == candidate)[:, None].astype(np.float64),
            (qwen_hybrid.astype(np.int64) == candidate)[:, None].astype(np.float64),
            np.full((row_count, 1), candidate / float(CANDIDATE_COUNT - 1), dtype=np.float64),
            np.repeat(candidate_eye[candidate][None, :], row_count, axis=0),
            (source_packet == candidate)[:, None].astype(np.float64),
            code_candidate_interaction,
        ]
        features.append(np.concatenate(parts, axis=1))
    return np.stack(features, axis=1).astype(np.float64)


def _fit_candidate_decoder(
    *,
    train_features: np.ndarray,
    train_answers: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
    label_permutation_seed: int | None = None,
) -> np.ndarray:
    return wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=train_answers,
        fit_indices=fit_indices,
        ridge=ridge,
        label_permutation_seed=label_permutation_seed,
    )


def _predict_candidate_decoder(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return wz._predict_candidate_decoder(features, coef)


def _score_predictions(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    packet_predictions: np.ndarray,
    qwen_target_predictions: np.ndarray,
    seed: int,
    bootstrap_samples: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return wz._score_row(
        name=name,
        predictions=predictions,
        answers=answers,
        packet_predictions=packet_predictions,
        qwen_target_predictions=qwen_target_predictions,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        extra=extra,
    )


def _encoder_configs(
    *,
    quantile_bins: tuple[int, ...],
    kmeans_counts: tuple[int, ...],
    kmeans_seed: int,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{"name": "candidate_only", "kind": "candidate_only"}]
    quantile_features = [
        ("packet_z", 4),
        ("top2_margin", 5),
        ("packet_rank", 6),
    ]
    for feature_name, feature_index in quantile_features:
        for bins in quantile_bins:
            if int(bins) * CANDIDATE_COUNT <= DEFAULT_MAX_CODES:
                configs.append(
                    {
                        "name": f"{feature_name}_quantile_{int(bins)}",
                        "kind": "quantile",
                        "feature_name": feature_name,
                        "feature_index": int(feature_index),
                        "bins": int(bins),
                    }
                )
    for clusters in kmeans_counts:
        if int(clusters) * CANDIDATE_COUNT <= DEFAULT_MAX_CODES:
            configs.append(
                {
                    "name": f"source_feature_kmeans_{int(clusters)}",
                    "kind": "kmeans",
                    "clusters": int(clusters),
                    "seed": int(kmeans_seed),
                    "iterations": 25,
                }
            )
    return configs


def _evaluate_config(
    *,
    config: dict[str, Any],
    surfaces: dict[str, Any],
    train_source_features: np.ndarray,
    eval_source_features: np.ndarray,
    decoder_ridges: tuple[float, ...],
    bootstrap_samples: int,
    row_seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    encoded = _encode_source_codes(
        config=config,
        train_source_features=train_source_features,
        eval_source_features=eval_source_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        fit_indices=fit_indices,
    )
    codebook_size = int(encoded["codebook_size"])
    train_features = _candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=encoded["train_code"],
        codebook_size=codebook_size,
    )
    eval_features = _candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=encoded["eval_code"],
        codebook_size=codebook_size,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for ridge in decoder_ridges:
        coef = _fit_candidate_decoder(
            train_features=train_features,
            train_answers=calibration["answers"],
            fit_indices=fit_indices,
            ridge=float(ridge),
        )
        train_predictions = _predict_candidate_decoder(train_features, coef)
        eval_predictions = _predict_candidate_decoder(eval_features, coef)
        predictions[(str(config["name"]), float(ridge))] = eval_predictions
        rows.append(
            _score_predictions(
                name="learned_source_code_decoder",
                predictions=eval_predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=row_seed_offset + len(rows),
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(config["name"]),
                    "encoder_kind": str(config["kind"]),
                    "ridge": float(ridge),
                    "codebook_size": codebook_size,
                    "raw_payload_bytes": _packet_bytes_for_codebook(codebook_size),
                    "framed_record_bytes": _packet_bytes_for_codebook(codebook_size) + 3,
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
                    "eval_code_unique_count": int(len(np.unique(encoded["eval_code"]))),
                },
            )
        )
    return rows, predictions, encoded


def _control_rows(
    *,
    selected_config: dict[str, Any],
    selected_ridge: float,
    surfaces: dict[str, Any],
    train_source_features: np.ndarray,
    eval_source_features: np.ndarray,
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    encoded = _encode_source_codes(
        config=selected_config,
        train_source_features=train_source_features,
        eval_source_features=eval_source_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        fit_indices=fit_indices,
    )
    codebook_size = int(encoded["codebook_size"])
    train_features = _candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=encoded["train_code"],
        codebook_size=codebook_size,
    )
    coef = _fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rng = np.random.default_rng(control_seed)
    qwen_source_features = _source_feature_matrix(
        validation["qwen_scores"],
        validation["alternatives"]["qwen_target_score"],
    )
    qwen_encoded = _encode_source_codes(
        config=selected_config,
        train_source_features=train_source_features,
        eval_source_features=qwen_source_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["alternatives"]["qwen_target_score"],
        fit_indices=fit_indices,
    )
    random_code = rng.integers(0, codebook_size, size=len(validation["answers"]), dtype=np.int64)
    random_bin_preserve_packet = (
        rng.integers(0, max(1, codebook_size // CANDIDATE_COUNT), size=len(validation["answers"]))
        * CANDIDATE_COUNT
        + validation["packet"]
    ).astype(np.int64)
    shuffled_source_order = rng.permutation(len(validation["answers"]))
    shuffled_source_encoded = _encode_source_codes(
        config=selected_config,
        train_source_features=train_source_features,
        eval_source_features=eval_source_features[shuffled_source_order],
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"][shuffled_source_order],
        fit_indices=fit_indices,
    )
    code_permutation = rng.permutation(codebook_size).astype(np.int64)
    control_specs = [
        ("row_shuffle_source_code", encoded["eval_code"][rng.permutation(len(encoded["eval_code"]))]),
        ("source_feature_shuffle_before_encoding", np.mod(shuffled_source_encoded["eval_code"], codebook_size)),
        ("codebook_permutation_mismatch", code_permutation[encoded["eval_code"]]),
        ("random_same_byte_code", random_code),
        ("random_subcode_preserve_packet", np.mod(random_bin_preserve_packet, codebook_size)),
        ("qwen_derived_code", np.mod(qwen_encoded["eval_code"], codebook_size)),
        ("candidate_only_code", validation["packet"].astype(np.int64)),
        ("zero_source_code", np.zeros(len(validation["answers"]), dtype=np.int64)),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_code) in enumerate(control_specs):
        eval_features = _candidate_decoder_features(
            qwen_scores=validation["qwen_scores"],
            qwen_target=validation["alternatives"]["qwen_target_score"],
            qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
            qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            source_code=eval_code.astype(np.int64),
            codebook_size=codebook_size,
        )
        predictions = _predict_candidate_decoder(eval_features, coef)
        rows.append(
            _score_predictions(
                name=name,
                predictions=predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=17000 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(selected_config["name"]),
                    "ridge": float(selected_ridge),
                    "codebook_size": codebook_size,
                },
            )
        )
    qwen_only_train_features = _candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=np.zeros(len(calibration["answers"]), dtype=np.int64),
        codebook_size=1,
    )
    qwen_only_eval_features = _candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=np.zeros(len(validation["answers"]), dtype=np.int64),
        codebook_size=1,
    )
    qwen_only_coef = _fit_candidate_decoder(
        train_features=qwen_only_train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rows.append(
        _score_predictions(
            name="qwen_side_only_decoder",
            predictions=_predict_candidate_decoder(qwen_only_eval_features, qwen_only_coef),
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=17080,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": "qwen_side_only",
                "ridge": float(selected_ridge),
                "codebook_size": 1,
            },
        )
    )
    candidate_only_train_features = _candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=calibration["tiny_packet"],
        codebook_size=CANDIDATE_COUNT,
    )
    candidate_only_eval_features = _candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=validation["packet"],
        codebook_size=CANDIDATE_COUNT,
    )
    candidate_only_coef = _fit_candidate_decoder(
        train_features=candidate_only_train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rows.append(
        _score_predictions(
            name="compact_candidate_only_decoder",
            predictions=_predict_candidate_decoder(candidate_only_eval_features, candidate_only_coef),
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=17081,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": "compact_candidate_only",
                "ridge": float(selected_ridge),
                "codebook_size": CANDIDATE_COUNT,
            },
        )
    )
    permuted_coef = _fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
        label_permutation_seed=control_seed + 77,
    )
    eval_features = _candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=encoded["eval_code"],
        codebook_size=codebook_size,
    )
    rows.append(
        _score_predictions(
            name="label_permutation_decoder",
            predictions=_predict_candidate_decoder(eval_features, permuted_coef),
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=17099,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": str(selected_config["name"]),
                "ridge": float(selected_ridge),
                "codebook_size": codebook_size,
            },
        )
    )
    rows.append(
        _score_predictions(
            name="packet_only",
            predictions=validation["packet"],
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=17100,
            bootstrap_samples=bootstrap_samples,
            extra={"encoder_name": "baseline", "ridge": 0.0, "codebook_size": CANDIDATE_COUNT},
        )
    )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Learned Source-Code Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- default encoder: `{h['default_encoder_name']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
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
    decoder_ridges: tuple[float, ...] = DEFAULT_DECODER_RIDGES,
    quantile_bins: tuple[int, ...] = DEFAULT_QUANTILE_BINS,
    kmeans_counts: tuple[int, ...] = DEFAULT_KMEANS_COUNTS,
    kmeans_seed: int = 7,
    bootstrap_samples: int = 500,
    control_seed: int = 19001,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces = wz._load_surfaces(
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
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    train_source_features = _source_feature_matrix(
        surfaces["tiny_train_scores"],
        calibration["tiny_packet"],
    )
    eval_source_features = _source_feature_matrix(
        surfaces["tiny_eval_scores"],
        validation["packet"],
    )
    configs = _encoder_configs(
        quantile_bins=quantile_bins,
        kmeans_counts=kmeans_counts,
        kmeans_seed=kmeans_seed,
    )
    frontier_rows: list[dict[str, Any]] = []
    encoded_by_name: dict[str, dict[str, Any]] = {}
    predictions_by_key: dict[tuple[str, float], np.ndarray] = {}
    for config_index, config in enumerate(configs):
        rows, predictions, encoded = _evaluate_config(
            config=config,
            surfaces=surfaces,
            train_source_features=train_source_features,
            eval_source_features=eval_source_features,
            decoder_ridges=decoder_ridges,
            bootstrap_samples=bootstrap_samples,
            row_seed_offset=18000 + config_index * 100,
        )
        frontier_rows.extend(rows)
        predictions_by_key.update(predictions)
        encoded_by_name[str(config["name"])] = encoded | {"config": config}
    learned_rows = [row for row in frontier_rows if row["encoder_kind"] != "candidate_only"]
    if not learned_rows:
        raise ValueError("no learned encoder rows were evaluated")
    default_row = max(
        learned_rows,
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
    packet_decoder_baseline = max(
        [row for row in frontier_rows if row["encoder_kind"] == "candidate_only"],
        key=lambda row: (row["official_dev_accuracy"], row["accuracy"]),
    )
    default_predictions = predictions_by_key[(str(default_row["encoder_name"]), float(default_row["ridge"]))]
    default_blocks = wz._block_rows(
        selected=default_predictions,
        packet=validation["packet"],
        answers=validation["answers"],
    )
    selected_config = encoded_by_name[str(default_row["encoder_name"])]["config"]
    control_rows = _control_rows(
        selected_config=selected_config,
        selected_ridge=float(default_row["ridge"]),
        surfaces=surfaces,
        train_source_features=train_source_features,
        eval_source_features=eval_source_features,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_max_delta = max(
        row["delta_vs_packet_only"] for row in control_rows if row["name"] != "packet_only"
    )
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    prior_receiver_scout_gate = bool(
        default_row["accuracy"] - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY >= STRICT_PRIOR_SCOUT_DELTA
    )
    default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and block_stability_gate
        and control_separation_gate
        and prior_receiver_scout_gate
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    encoded_default = encoded_by_name[str(default_row["encoder_name"])]
    packet_only_accuracy = wz._accuracy(validation["packet"], validation["answers"])
    qwen_target_accuracy = wz._accuracy(
        validation["alternatives"]["qwen_target_score"],
        validation["answers"],
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["answers"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "validation_rows": int(len(validation["answers"])),
        "packet_only_accuracy": packet_only_accuracy,
        "qwen_target_accuracy": qwen_target_accuracy,
        "packet_decoder_baseline_accuracy": packet_decoder_baseline["accuracy"],
        "packet_decoder_baseline_delta_vs_packet_only": packet_decoder_baseline["delta_vs_packet_only"],
        "default_encoder_name": str(default_row["encoder_name"]),
        "default_encoder_kind": str(default_row["encoder_kind"]),
        "default_codebook_size": int(default_row["codebook_size"]),
        "default_eval_code_unique_count": int(default_row["eval_code_unique_count"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_ridge": default_row["ridge"],
        "best_scout_encoder_name": str(best_scout["encoder_name"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_prior_receiver_scout_accuracy": BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "default_delta_vs_best_prior_receiver_scout": default_row["accuracy"]
        - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "prior_receiver_scout_gate": prior_receiver_scout_gate,
        "default_pass_gate": default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "raw_payload_bytes": RAW_PACKET_BYTES,
        "framed_record_bytes": FRAMED_PACKET_BYTES,
        "strict_delta_required": STRICT_DELTA,
        "strict_prior_receiver_scout_delta_required": STRICT_PRIOR_SCOUT_DELTA,
    }
    payload = {
        "gate": "source_private_hellaswag_learned_source_code_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(default_pass_gate),
        "pass_rule": (
            "Pass if the official-train-dev-selected learned source-code encoder beats compact "
            "TinyLlama packet-only by >=0.010 with positive paired CI95 low, beats the best prior "
            "official-train receiver scout by >=0.005, is positive on at least 4/5 contiguous "
            "blocks, and destructive controls remain within +0.002 of packet-only while separated "
            "from the real default by >=0.003."
        ),
        "packet_contract": {
            "packet_name": "learned_discrete_source_code_packet",
            "raw_payload_bytes": RAW_PACKET_BYTES,
            "framed_record_bytes": FRAMED_PACKET_BYTES,
            "max_codebook_size": DEFAULT_MAX_CODES,
            "selected_codebook_size": int(default_row["codebook_size"]),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "learned_discrete_source_code_transmitted": True,
            "decoder_uses_qwen_side_information": True,
        },
        "headline": headline,
        "frontier_rows": frontier_rows,
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "selected_encoder_audit": encoded_default["encoder_audit"],
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(len(validation["answers"]) * RAW_PACKET_BYTES),
            "logical_validation_framed_record_bytes_total": int(
                len(validation["answers"]) * FRAMED_PACKET_BYTES
            ),
            "communication_object": "task_level_source_private_learned_discrete_packet",
            "communication_objective": "downstream_candidate_decision_accuracy",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "calibration_wall_time_s": surfaces["calibration_wall_time_s"],
            "total_wall_time_s": float(time.perf_counter() - started),
        },
        "inputs": {
            "train_path": _display_path(wz.DEFAULT_TRAIN_PATH),
            "train_sha256": _sha256_file(wz.DEFAULT_TRAIN_PATH),
            "tiny_eval_packet_jsonl": _display_path(wz.DEFAULT_TINY_EVAL_PACKET_JSONL),
            "tiny_eval_packet_jsonl_sha256": _sha256_file(wz.DEFAULT_TINY_EVAL_PACKET_JSONL),
            "tiny_eval_score_cache": _display_path(wz.DEFAULT_TINY_EVAL_SCORE_CACHE),
            "tiny_eval_score_cache_sha256": _sha256_file(wz.DEFAULT_TINY_EVAL_SCORE_CACHE),
            "qwen_eval_packet_jsonl": _display_path(wz.DEFAULT_QWEN_EVAL_PACKET_JSONL),
            "qwen_eval_packet_jsonl_sha256": _sha256_file(wz.DEFAULT_QWEN_EVAL_PACKET_JSONL),
            "qwen_global_artifact": _display_path(wz.DEFAULT_QWEN_GLOBAL_ARTIFACT),
            "qwen_global_artifact_sha256": _sha256_file(wz.DEFAULT_QWEN_GLOBAL_ARTIFACT),
        },
        "audit": surfaces["audit"],
        "interpretation": (
            "This gate tests whether changing the source encoder to a train-only learned discrete "
            "code can recover the Tiny/Qwen oracle headroom after receiver-only and residual-logit "
            "branches saturated. A pass would promote a learned source-code packet under the compact "
            "one-byte contract; a fail means source-score-derived discrete codes also do not beat "
            "packet-only on the current HellaSwag calibration surface."
        ),
    }
    json_path = output_dir / "hellaswag_learned_source_code_packet_gate.json"
    md_path = output_dir / "hellaswag_learned_source_code_packet_gate.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "inputs": payload["inputs"],
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decoder-ridges", type=_parse_float_tuple, default=DEFAULT_DECODER_RIDGES)
    parser.add_argument("--quantile-bins", type=_parse_int_tuple, default=DEFAULT_QUANTILE_BINS)
    parser.add_argument("--kmeans-counts", type=_parse_int_tuple, default=DEFAULT_KMEANS_COUNTS)
    parser.add_argument("--kmeans-seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=19001)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        decoder_ridges=args.decoder_ridges,
        quantile_bins=args.quantile_bins,
        kmeans_counts=args.kmeans_counts,
        kmeans_seed=args.kmeans_seed,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
