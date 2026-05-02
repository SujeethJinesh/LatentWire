from __future__ import annotations

"""Decoder-conditioned discrete query innovation codec gate for HellaSwag."""

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

from scripts import build_source_private_hellaswag_learned_source_code_packet_gate as source_code  # noqa: E402
from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_discrete_query_innovation_codec_gate_20260502"
)
DEFAULT_QUERY_COUNTS = (2, 4, 8)
DEFAULT_QUERY_SEEDS = (17, 29)
DEFAULT_QUERY_TEMPERATURES = (0.75, 1.5)
DEFAULT_ENCODER_RIDGES = (1.0, 10.0)
DEFAULT_CLUSTER_COUNTS = (4, 16, 32)
DEFAULT_DECODER_RIDGES = (1.0, 10.0, 100.0)
DEFAULT_TARGET_PRIORS = ("qwen_prob", "qwen_hybrid_onehot")
DEFAULT_KMEANS_ITERATIONS = 35
STRICT_DELTA = 0.010
STRICT_TARGET_DELTA = 0.020
STRICT_PRIOR_SCOUT_DELTA = 0.005
BEST_PRIOR_RECEIVER_SCOUT_ACCURACY = 0.620594
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 1
FRAMED_PACKET_BYTES = 4
CANDIDATE_COUNT = 4
DEFAULT_MAX_CODES = 256
FP16_ONE_TOKEN_KV_FLOOR_BYTES = 12_288.0
KVCOMM30_FP16_ONE_TOKEN_KV_FLOOR_BYTES = 3_686.4
QJL_1BIT_ONE_TOKEN_KV_FLOOR_BYTES = 768.0
TURBOQUANT_35BIT_ONE_TOKEN_KV_FLOOR_BYTES = 2_688.0
KIVI_2BIT_ONE_TOKEN_KV_FLOOR_BYTES = 1_536.0
KVQUANT_3BIT_ONE_TOKEN_KV_FLOOR_BYTES = 2_304.0


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


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one string is required")
    return result


def _packet_bytes_for_codebook(codebook_size: int) -> int:
    if codebook_size < 1:
        raise ValueError("codebook_size must be positive")
    return max(1, int(math.ceil(math.log2(codebook_size) / 8.0)))


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _candidate_ranks(zscores: np.ndarray) -> np.ndarray:
    order = np.argsort(-zscores, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    for row_index, row_order in enumerate(order):
        ranks[row_index, row_order] = np.arange(CANDIDATE_COUNT, dtype=np.float64)
    return ranks


def _candidate_token_features(scores: np.ndarray, packet: np.ndarray) -> np.ndarray:
    """Return source-only candidate tokens for a fixed-query bottleneck."""

    zscores = wz._row_zscores(scores)
    probs = _softmax_rows(zscores)
    ranks = _candidate_ranks(zscores) / float(CANDIDATE_COUNT - 1)
    packet = packet.astype(np.int64)
    row_count = len(packet)
    candidate_eye = np.eye(CANDIDATE_COUNT, dtype=np.float64)
    top = np.argmax(zscores, axis=1).astype(np.int64)
    token_parts: list[np.ndarray] = []
    for candidate in range(CANDIDATE_COUNT):
        other = np.delete(zscores, candidate, axis=1)
        other_max = np.max(other, axis=1)
        is_packet = (packet == candidate).astype(np.float64)
        is_top = (top == candidate).astype(np.float64)
        parts = [
            zscores[:, candidate : candidate + 1],
            probs[:, candidate : candidate + 1],
            ranks[:, candidate : candidate + 1],
            is_packet[:, None],
            is_top[:, None],
            (zscores[:, candidate] - other_max)[:, None],
            np.full((row_count, 1), candidate / float(CANDIDATE_COUNT - 1), dtype=np.float64),
            np.repeat(candidate_eye[candidate][None, :], row_count, axis=0),
            (is_packet * zscores[:, candidate])[:, None],
            (is_packet * probs[:, candidate])[:, None],
            (is_top * probs[:, candidate])[:, None],
        ]
        token_parts.append(np.concatenate(parts, axis=1))
    return np.stack(token_parts, axis=1).astype(np.float64)


def _standardize_tokens_from_fit(
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit = train_tokens[fit_indices].reshape(-1, train_tokens.shape[-1])
    mean = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train_tokens - mean) / scale, (eval_tokens - mean) / scale, mean, scale


def _fixed_query_parameters(feature_dim: int, *, query_count: int, seed: int) -> np.ndarray:
    if query_count < 1:
        raise ValueError("query_count must be positive")
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(int(query_count), int(feature_dim))).astype(np.float64)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    return queries / np.where(norms < 1e-9, 1.0, norms)


def _query_summaries(
    tokens: np.ndarray,
    query_parameters: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = np.einsum("ncd,qd->nqc", tokens, query_parameters)
    logits = logits / (math.sqrt(tokens.shape[-1]) * float(temperature))
    logits = logits - np.max(logits, axis=2, keepdims=True)
    attention = np.exp(logits)
    attention = attention / np.sum(attention, axis=2, keepdims=True)
    weighted = np.einsum("nqc,ncd->nqd", attention, tokens).reshape(tokens.shape[0], -1)
    pooled = np.concatenate(
        [
            np.mean(tokens, axis=1),
            np.max(tokens, axis=1),
            np.min(tokens, axis=1),
            attention.reshape(tokens.shape[0], -1),
        ],
        axis=1,
    )
    return np.concatenate([weighted, pooled], axis=1).astype(np.float64)


def _one_hot(indices: np.ndarray, width: int = CANDIDATE_COUNT) -> np.ndarray:
    indices = indices.astype(np.int64)
    result = np.zeros((len(indices), int(width)), dtype=np.float64)
    result[np.arange(len(indices)), indices] = 1.0
    return result


def _innovation_targets(
    *,
    answers: np.ndarray,
    qwen_scores: np.ndarray,
    qwen_target: np.ndarray,
    qwen_hybrid: np.ndarray,
    prior: str,
) -> np.ndarray:
    answer_one_hot = _one_hot(answers)
    if prior == "qwen_prob":
        baseline = _softmax_rows(wz._row_zscores(qwen_scores))
    elif prior == "qwen_target_onehot":
        baseline = _one_hot(qwen_target)
    elif prior == "qwen_hybrid_onehot":
        baseline = _one_hot(qwen_hybrid)
    elif prior == "uniform":
        baseline = np.full_like(answer_one_hot, 1.0 / float(CANDIDATE_COUNT))
    else:
        raise ValueError(f"unsupported target prior: {prior}")
    return (answer_one_hot - baseline).astype(np.float64)


def _fit_multi_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    fit_indices: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    fit_features = features[fit_indices]
    mean = np.mean(fit_features, axis=0)
    scale = np.std(fit_features, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x = np.concatenate(
        [np.ones((len(fit_indices), 1), dtype=np.float64), (fit_features - mean) / scale],
        axis=1,
    )
    y = targets[fit_indices]
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return {
        "coef": coef.astype(np.float64),
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "ridge": float(ridge),
    }


def _predict_multi_ridge(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    body = (features - model["mean"]) / model["scale"]
    x = np.concatenate([np.ones((len(features), 1), dtype=np.float64), body], axis=1)
    return (x @ model["coef"]).astype(np.float64)


def _fit_query_residual_encoder(
    *,
    train_tokens: np.ndarray,
    train_packet: np.ndarray,
    train_targets: np.ndarray,
    fit_indices: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    train_std, _, token_mean, token_scale = _standardize_tokens_from_fit(
        train_tokens,
        train_tokens,
        fit_indices,
    )
    query_parameters = _fixed_query_parameters(
        train_tokens.shape[-1],
        query_count=int(config["query_count"]),
        seed=int(config["query_seed"]),
    )
    train_summary = _query_summaries(
        train_std,
        query_parameters,
        temperature=float(config["query_temperature"]),
    )
    residual_model = _fit_multi_ridge(
        train_summary,
        train_targets,
        fit_indices,
        ridge=float(config["encoder_ridge"]),
    )
    train_residual = _predict_multi_ridge(train_summary, residual_model)
    train_residual_std, _, residual_mean, residual_scale = source_code._standardize_from_fit(
        train_residual,
        train_residual,
        fit_indices,
    )
    centers = source_code._fit_kmeans(
        train_residual_std,
        fit_indices,
        int(config["clusters"]),
        seed=int(config["cluster_seed"]),
        iterations=int(config.get("kmeans_iterations", DEFAULT_KMEANS_ITERATIONS)),
    )
    train_cluster = source_code._nearest_center_codes(train_residual_std, centers)
    train_code = (train_cluster * CANDIDATE_COUNT + train_packet.astype(np.int64)).astype(np.int64)
    return {
        "config": dict(config),
        "token_mean": token_mean.astype(np.float64),
        "token_scale": token_scale.astype(np.float64),
        "query_parameters": query_parameters.astype(np.float64),
        "residual_model": residual_model,
        "residual_mean": residual_mean.astype(np.float64),
        "residual_scale": residual_scale.astype(np.float64),
        "centers": centers.astype(np.float64),
        "train_residual_prediction_mean": [float(item) for item in np.mean(train_residual, axis=0)],
        "train_residual_prediction_std": [float(item) for item in np.std(train_residual, axis=0)],
        "train_code": train_code,
    }


def _apply_query_residual_encoder(
    *,
    artifacts: dict[str, Any],
    tokens: np.ndarray,
    packet: np.ndarray,
) -> dict[str, Any]:
    token_std = (tokens - artifacts["token_mean"]) / artifacts["token_scale"]
    summary = _query_summaries(
        token_std,
        artifacts["query_parameters"],
        temperature=float(artifacts["config"]["query_temperature"]),
    )
    residual = _predict_multi_ridge(summary, artifacts["residual_model"])
    residual_std = (residual - artifacts["residual_mean"]) / artifacts["residual_scale"]
    cluster = source_code._nearest_center_codes(residual_std, artifacts["centers"])
    code = (cluster * CANDIDATE_COUNT + packet.astype(np.int64)).astype(np.int64)
    return {
        "code": code,
        "cluster": cluster,
        "residual_prediction": residual,
        "summary": summary,
    }


def _encode_query_residual_codes(
    *,
    config: dict[str, Any],
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    train_packet: np.ndarray,
    eval_packet: np.ndarray,
    train_targets: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    clusters = int(config["clusters"])
    if clusters * CANDIDATE_COUNT > DEFAULT_MAX_CODES:
        raise ValueError("codebook exceeds one-byte budget")
    artifacts = _fit_query_residual_encoder(
        train_tokens=train_tokens,
        train_packet=train_packet,
        train_targets=train_targets,
        fit_indices=fit_indices,
        config=config,
    )
    eval_encoded = _apply_query_residual_encoder(
        artifacts=artifacts,
        tokens=eval_tokens,
        packet=eval_packet,
    )
    return {
        "train_code": artifacts["train_code"].astype(np.int64),
        "eval_code": eval_encoded["code"].astype(np.int64),
        "codebook_size": int(clusters * CANDIDATE_COUNT),
        "artifacts": artifacts,
        "eval_cluster": eval_encoded["cluster"].astype(np.int64),
        "eval_residual_prediction_mean": [
            float(item) for item in np.mean(eval_encoded["residual_prediction"], axis=0)
        ],
        "eval_residual_prediction_std": [
            float(item) for item in np.std(eval_encoded["residual_prediction"], axis=0)
        ],
    }


def _encoder_configs(
    *,
    query_counts: tuple[int, ...],
    query_seeds: tuple[int, ...],
    query_temperatures: tuple[float, ...],
    encoder_ridges: tuple[float, ...],
    cluster_counts: tuple[int, ...],
    target_priors: tuple[str, ...],
    cluster_seed: int,
    kmeans_iterations: int,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for target_prior in target_priors:
        for query_count in query_counts:
            for query_seed in query_seeds:
                for temperature in query_temperatures:
                    for encoder_ridge in encoder_ridges:
                        for clusters in cluster_counts:
                            if int(clusters) * CANDIDATE_COUNT <= DEFAULT_MAX_CODES:
                                configs.append(
                                    {
                                        "name": (
                                            f"prior_{target_prior}_q{int(query_count)}"
                                            f"_seed{int(query_seed)}_temp{float(temperature):g}"
                                            f"_er{float(encoder_ridge):g}_k{int(clusters)}"
                                        ),
                                        "kind": "query_residual_vq",
                                        "target_prior": str(target_prior),
                                        "query_count": int(query_count),
                                        "query_seed": int(query_seed),
                                        "query_temperature": float(temperature),
                                        "encoder_ridge": float(encoder_ridge),
                                        "clusters": int(clusters),
                                        "cluster_seed": int(cluster_seed),
                                        "kmeans_iterations": int(kmeans_iterations),
                                    }
                                )
    return configs


def _decoder_feature_bundle(
    *,
    surfaces: dict[str, Any],
    train_code: np.ndarray,
    eval_code: np.ndarray,
    codebook_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    train_features = source_code._candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=train_code,
        codebook_size=codebook_size,
    )
    eval_features = source_code._candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=eval_code,
        codebook_size=codebook_size,
    )
    return train_features, eval_features


def _evaluate_config(
    *,
    config: dict[str, Any],
    surfaces: dict[str, Any],
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    decoder_ridges: tuple[float, ...],
    bootstrap_samples: int,
    row_seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    targets = _innovation_targets(
        answers=calibration["answers"],
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_hybrid=calibration["qwen_hybrid"],
        prior=str(config["target_prior"]),
    )
    encoded = _encode_query_residual_codes(
        config=config,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        train_targets=targets,
        fit_indices=fit_indices,
    )
    codebook_size = int(encoded["codebook_size"])
    train_features, eval_features = _decoder_feature_bundle(
        surfaces=surfaces,
        train_code=encoded["train_code"],
        eval_code=encoded["eval_code"],
        codebook_size=codebook_size,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for ridge in decoder_ridges:
        coef = wz._fit_candidate_decoder(
            train_features=train_features,
            train_answers=calibration["answers"],
            fit_indices=fit_indices,
            ridge=float(ridge),
        )
        train_predictions = wz._predict_candidate_decoder(train_features, coef)
        eval_predictions = wz._predict_candidate_decoder(eval_features, coef)
        predictions[(str(config["name"]), float(ridge))] = eval_predictions
        rows.append(
            wz._score_row(
                name="discrete_query_innovation_codec",
                predictions=eval_predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=row_seed_offset + len(rows),
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(config["name"]),
                    "encoder_kind": str(config["kind"]),
                    "target_prior": str(config["target_prior"]),
                    "query_count": int(config["query_count"]),
                    "query_seed": int(config["query_seed"]),
                    "query_temperature": float(config["query_temperature"]),
                    "encoder_ridge": float(config["encoder_ridge"]),
                    "clusters": int(config["clusters"]),
                    "decoder_ridge": float(ridge),
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


def _score_control(
    *,
    name: str,
    eval_code: np.ndarray,
    train_code: np.ndarray,
    codebook_size: int,
    decoder_ridge: float,
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
    label_permutation_seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    train_features, eval_features = _decoder_feature_bundle(
        surfaces=surfaces,
        train_code=train_code,
        eval_code=np.mod(eval_code.astype(np.int64), int(codebook_size)),
        codebook_size=codebook_size,
    )
    coef = wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=surfaces["fit_indices"],
        ridge=float(decoder_ridge),
        label_permutation_seed=label_permutation_seed,
    )
    predictions = wz._predict_candidate_decoder(eval_features, coef)
    row_extra = {
        "decoder_ridge": float(decoder_ridge),
        "codebook_size": int(codebook_size),
    }
    if extra:
        row_extra.update(extra)
    return wz._score_row(
        name=name,
        predictions=predictions,
        answers=validation["answers"],
        packet_predictions=validation["packet"],
        qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        extra=row_extra,
    )


def _rescore_frontier_row(
    *,
    row: dict[str, Any],
    predictions: np.ndarray,
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metric_keys = {
        "name",
        "accuracy",
        "packet_only_accuracy",
        "qwen_target_accuracy",
        "delta_vs_packet_only",
        "ci95_low_vs_packet_only",
        "ci95_high_vs_packet_only",
        "delta_vs_qwen_target",
        "ci95_low_vs_qwen_target",
        "ci95_high_vs_qwen_target",
        "help_count",
        "harm_count",
        "net_help",
        "override_rate_vs_packet",
    }
    extra = {key: value for key, value in row.items() if key not in metric_keys}
    validation = surfaces["validation"]
    return wz._score_row(
        name=str(row["name"]),
        predictions=predictions,
        answers=validation["answers"],
        packet_predictions=validation["packet"],
        qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        extra=extra,
    )


def _control_rows(
    *,
    selected_config: dict[str, Any],
    selected_ridge: float,
    selected_encoded: dict[str, Any],
    surfaces: dict[str, Any],
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    rng = np.random.default_rng(control_seed)
    codebook_size = int(selected_encoded["codebook_size"])
    train_code = selected_encoded["train_code"].astype(np.int64)
    eval_code = selected_encoded["eval_code"].astype(np.int64)
    artifacts = selected_encoded["artifacts"]
    rows: list[dict[str, Any]] = []

    shuffled_rows = rng.permutation(len(eval_code))
    shuffled_encoded = _apply_query_residual_encoder(
        artifacts=artifacts,
        tokens=eval_tokens[shuffled_rows],
        packet=validation["packet"][shuffled_rows],
    )["code"]
    qwen_tokens = _candidate_token_features(
        validation["qwen_scores"],
        validation["alternatives"]["qwen_target_score"],
    )
    qwen_encoded = _apply_query_residual_encoder(
        artifacts=artifacts,
        tokens=qwen_tokens,
        packet=validation["alternatives"]["qwen_target_score"],
    )["code"]
    clusters = int(selected_config["clusters"])
    eval_cluster = eval_code // CANDIDATE_COUNT
    eval_packet = np.mod(eval_code, CANDIDATE_COUNT)
    cluster_permutation = rng.permutation(clusters).astype(np.int64)
    random_cluster_code = (
        rng.integers(0, clusters, size=len(eval_code), dtype=np.int64) * CANDIDATE_COUNT
        + validation["packet"]
    ).astype(np.int64)
    control_specs = [
        ("row_shuffle_source_code", eval_code[rng.permutation(len(eval_code))]),
        ("source_feature_shuffle_before_encoding", shuffled_encoded),
        ("codebook_permutation_mismatch", cluster_permutation[eval_cluster] * CANDIDATE_COUNT + eval_packet),
        ("random_same_byte_code", rng.integers(0, codebook_size, size=len(eval_code), dtype=np.int64)),
        ("random_cluster_preserve_packet", random_cluster_code),
        ("qwen_derived_code", qwen_encoded),
        ("candidate_only_code", validation["packet"].astype(np.int64)),
        ("zero_source_code", np.zeros(len(eval_code), dtype=np.int64)),
    ]
    for offset, (name, eval_control_code) in enumerate(control_specs):
        rows.append(
            _score_control(
                name=name,
                eval_code=eval_control_code,
                train_code=train_code,
                codebook_size=codebook_size,
                decoder_ridge=selected_ridge,
                surfaces=surfaces,
                bootstrap_samples=bootstrap_samples,
                seed=22000 + offset,
                extra={
                    "encoder_name": str(selected_config["name"]),
                    "control_type": "source_destroying_or_mismatch",
                },
            )
        )

    rows.append(
        _score_control(
            name="label_permutation_decoder",
            eval_code=eval_code,
            train_code=train_code,
            codebook_size=codebook_size,
            decoder_ridge=selected_ridge,
            surfaces=surfaces,
            bootstrap_samples=bootstrap_samples,
            seed=22080,
            label_permutation_seed=control_seed + 77,
            extra={"encoder_name": str(selected_config["name"]), "control_type": "decoder_label_permutation"},
        )
    )

    qwen_only_train = np.zeros(len(calibration["answers"]), dtype=np.int64)
    qwen_only_eval = np.zeros(len(validation["answers"]), dtype=np.int64)
    rows.append(
        _score_control(
            name="qwen_side_only_decoder",
            eval_code=qwen_only_eval,
            train_code=qwen_only_train,
            codebook_size=1,
            decoder_ridge=selected_ridge,
            surfaces=surfaces,
            bootstrap_samples=bootstrap_samples,
            seed=22090,
            extra={"encoder_name": "qwen_side_only", "control_type": "no_source"},
        )
    )

    rows.append(
        _score_control(
            name="compact_candidate_only_decoder",
            eval_code=validation["packet"].astype(np.int64),
            train_code=calibration["tiny_packet"].astype(np.int64),
            codebook_size=CANDIDATE_COUNT,
            decoder_ridge=selected_ridge,
            surfaces=surfaces,
            bootstrap_samples=bootstrap_samples,
            seed=22091,
            extra={"encoder_name": "compact_candidate_only", "control_type": "candidate_only"},
        )
    )
    rows.append(
        wz._score_row(
            name="packet_only",
            predictions=validation["packet"],
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=22099,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": "baseline",
                "decoder_ridge": 0.0,
                "codebook_size": CANDIDATE_COUNT,
                "control_type": "baseline",
            },
        )
    )
    return rows


def _oracle_capture(
    *,
    predictions: np.ndarray,
    packet: np.ndarray,
    qwen_target: np.ndarray,
    answers: np.ndarray,
) -> dict[str, float]:
    packet_correct = packet == answers
    qwen_correct = qwen_target == answers
    oracle_accuracy = float(np.mean(packet_correct | qwen_correct))
    packet_accuracy = wz._accuracy(packet, answers)
    prediction_accuracy = wz._accuracy(predictions, answers)
    headroom = oracle_accuracy - packet_accuracy
    return {
        "packet_or_qwen_target_oracle_accuracy": oracle_accuracy,
        "packet_or_qwen_target_oracle_delta_vs_packet": headroom,
        "capture_fraction_of_packet_or_qwen_target_oracle_delta": float(
            0.0 if headroom <= 0.0 else (prediction_accuracy - packet_accuracy) / headroom
        ),
    }


def _source_state_floor_ratios(framed_record_bytes: int) -> dict[str, Any]:
    framed = float(framed_record_bytes)
    floors = {
        "fp16_one_token_kv_floor_bytes": FP16_ONE_TOKEN_KV_FLOOR_BYTES,
        "kvcomm30_fp16_one_token_kv_floor_bytes": KVCOMM30_FP16_ONE_TOKEN_KV_FLOOR_BYTES,
        "qjl_1bit_one_token_kv_floor_bytes": QJL_1BIT_ONE_TOKEN_KV_FLOOR_BYTES,
        "turboquant_35bit_one_token_kv_floor_bytes": TURBOQUANT_35BIT_ONE_TOKEN_KV_FLOOR_BYTES,
        "kivi_2bit_one_token_kv_floor_bytes": KIVI_2BIT_ONE_TOKEN_KV_FLOOR_BYTES,
        "kvquant_3bit_one_token_kv_floor_bytes": KVQUANT_3BIT_ONE_TOKEN_KV_FLOOR_BYTES,
    }
    return {
        **floors,
        **{f"{name}_ratio_vs_framed_packet": float(value / framed) for name, value in floors.items()},
        "byte_floor_claim_boundary": (
            "These are conservative byte/exposure floors for source state, not native throughput, "
            "quality, TTFT, TPOT, HBM, or goodput measurements."
        ),
    }


def _microbench_selected(
    *,
    selected_encoded: dict[str, Any],
    decoder_coef: np.ndarray,
    eval_tokens: np.ndarray,
    eval_packet: np.ndarray,
    validation: dict[str, Any],
    batch_sizes: tuple[int, ...] = (1, 32, 256),
    repeats: int = 75,
) -> list[dict[str, Any]]:
    artifacts = selected_encoded["artifacts"]
    codebook_size = int(selected_encoded["codebook_size"])
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        n = min(int(batch_size), len(eval_packet))
        local_tokens = eval_tokens[:n]
        local_packet = eval_packet[:n]
        timings_encode: list[float] = []
        timings_decode: list[float] = []
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            encoded = _apply_query_residual_encoder(
                artifacts=artifacts,
                tokens=local_tokens,
                packet=local_packet,
            )
            t1 = time.perf_counter()
            features = source_code._candidate_decoder_features(
                qwen_scores=validation["qwen_scores"][:n],
                qwen_target=validation["alternatives"]["qwen_target_score"][:n],
                qwen_mean=validation["alternatives"]["mean_zscore_prediction"][:n],
                qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"][:n],
                source_code=encoded["code"],
                codebook_size=codebook_size,
            )
            _ = wz._predict_candidate_decoder(features, decoder_coef)
            t2 = time.perf_counter()
            timings_encode.append((t1 - t0) * 1_000_000.0 / float(n))
            timings_decode.append((t2 - t1) * 1_000_000.0 / float(n))
        rows.append(
            {
                "batch_size": int(n),
                "encode_p50_us_per_request": float(np.median(timings_encode)),
                "encode_p95_us_per_request": float(np.quantile(timings_encode, 0.95)),
                "decode_p50_us_per_request": float(np.median(timings_decode)),
                "decode_p95_us_per_request": float(np.quantile(timings_decode, 0.95)),
            }
        )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Discrete Query Innovation Codec Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- default encoder: `{h['default_encoder_name']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- default oracle-headroom capture: `{h['default_oracle_capture_fraction']:.6f}`",
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
    query_counts: tuple[int, ...] = DEFAULT_QUERY_COUNTS,
    query_seeds: tuple[int, ...] = DEFAULT_QUERY_SEEDS,
    query_temperatures: tuple[float, ...] = DEFAULT_QUERY_TEMPERATURES,
    encoder_ridges: tuple[float, ...] = DEFAULT_ENCODER_RIDGES,
    cluster_counts: tuple[int, ...] = DEFAULT_CLUSTER_COUNTS,
    decoder_ridges: tuple[float, ...] = DEFAULT_DECODER_RIDGES,
    target_priors: tuple[str, ...] = DEFAULT_TARGET_PRIORS,
    cluster_seed: int = 31,
    kmeans_iterations: int = DEFAULT_KMEANS_ITERATIONS,
    frontier_bootstrap_samples: int = 0,
    bootstrap_samples: int = 500,
    control_seed: int = 23017,
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
    train_tokens = _candidate_token_features(surfaces["tiny_train_scores"], calibration["tiny_packet"])
    eval_tokens = _candidate_token_features(surfaces["tiny_eval_scores"], validation["packet"])
    configs = _encoder_configs(
        query_counts=query_counts,
        query_seeds=query_seeds,
        query_temperatures=query_temperatures,
        encoder_ridges=encoder_ridges,
        cluster_counts=cluster_counts,
        target_priors=target_priors,
        cluster_seed=cluster_seed,
        kmeans_iterations=kmeans_iterations,
    )
    frontier_rows: list[dict[str, Any]] = []
    encoded_by_name: dict[str, dict[str, Any]] = {}
    predictions_by_key: dict[tuple[str, float], np.ndarray] = {}
    for config_index, config in enumerate(configs):
        rows, predictions, encoded = _evaluate_config(
            config=config,
            surfaces=surfaces,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            decoder_ridges=decoder_ridges,
            bootstrap_samples=frontier_bootstrap_samples,
            row_seed_offset=21000 + config_index * 100,
        )
        frontier_rows.extend(rows)
        predictions_by_key.update(predictions)
        encoded_by_name[str(config["name"])] = encoded | {"config": config}

    if not frontier_rows:
        raise ValueError("no discrete query innovation codec rows were evaluated")
    default_row = max(
        frontier_rows,
        key=lambda row: (
            row["official_dev_accuracy"],
            row["official_dev_delta_vs_packet"],
            row["delta_vs_packet_only"],
            -row["codebook_size"],
            -math.log10(float(row["decoder_ridge"])),
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
    default_predictions = predictions_by_key[
        (str(default_row["encoder_name"]), float(default_row["decoder_ridge"]))
    ]
    best_scout_predictions = predictions_by_key[
        (str(best_scout["encoder_name"]), float(best_scout["decoder_ridge"]))
    ]
    default_row = _rescore_frontier_row(
        row=default_row,
        predictions=default_predictions,
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        seed=21801,
    )
    best_scout = _rescore_frontier_row(
        row=best_scout,
        predictions=best_scout_predictions,
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        seed=21802,
    )
    selected_encoded = encoded_by_name[str(default_row["encoder_name"])]
    selected_config = selected_encoded["config"]
    train_features, _ = _decoder_feature_bundle(
        surfaces=surfaces,
        train_code=selected_encoded["train_code"],
        eval_code=selected_encoded["eval_code"],
        codebook_size=int(selected_encoded["codebook_size"]),
    )
    default_coef = wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(default_row["decoder_ridge"]),
    )
    default_blocks = wz._block_rows(
        selected=default_predictions,
        packet=validation["packet"],
        answers=validation["answers"],
    )
    control_rows = _control_rows(
        selected_config=selected_config,
        selected_ridge=float(default_row["decoder_ridge"]),
        selected_encoded=selected_encoded,
        surfaces=surfaces,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_max_delta = max(
        row["delta_vs_packet_only"] for row in control_rows if row["name"] != "packet_only"
    )
    destructive_control_max_delta = max(
        row["delta_vs_packet_only"]
        for row in control_rows
        if row.get("control_type") == "source_destroying_or_mismatch"
    )
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    prior_receiver_scout_gate = bool(
        default_row["accuracy"] - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY >= STRICT_PRIOR_SCOUT_DELTA
    )
    oracle = _oracle_capture(
        predictions=default_predictions,
        packet=validation["packet"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        answers=validation["answers"],
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and default_row["delta_vs_qwen_target"] >= STRICT_TARGET_DELTA
        and block_stability_gate
        and control_separation_gate
        and prior_receiver_scout_gate
    )
    packet_only_accuracy = wz._accuracy(validation["packet"], validation["answers"])
    qwen_target_accuracy = wz._accuracy(
        validation["alternatives"]["qwen_target_score"],
        validation["answers"],
    )
    timing_rows = _microbench_selected(
        selected_encoded=selected_encoded,
        decoder_coef=default_coef,
        eval_tokens=eval_tokens,
        eval_packet=validation["packet"],
        validation=validation,
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
        "default_encoder_name": str(default_row["encoder_name"]),
        "default_target_prior": str(default_row["target_prior"]),
        "default_query_count": int(default_row["query_count"]),
        "default_query_seed": int(default_row["query_seed"]),
        "default_query_temperature": float(default_row["query_temperature"]),
        "default_encoder_ridge": float(default_row["encoder_ridge"]),
        "default_clusters": int(default_row["clusters"]),
        "default_codebook_size": int(default_row["codebook_size"]),
        "default_decoder_ridge": float(default_row["decoder_ridge"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_qwen_target": default_row["delta_vs_qwen_target"],
        "default_delta_vs_best_prior_receiver_scout": default_row["accuracy"]
        - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "default_oracle_capture_fraction": oracle[
            "capture_fraction_of_packet_or_qwen_target_oracle_delta"
        ],
        "best_prior_receiver_scout_accuracy": BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "best_scout_encoder_name": str(best_scout["encoder_name"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "control_max_delta_vs_packet_only": control_max_delta,
        "destructive_control_max_delta_vs_packet_only": destructive_control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "prior_receiver_scout_gate": prior_receiver_scout_gate,
        "default_pass_gate": default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "raw_payload_bytes": RAW_PACKET_BYTES,
        "framed_record_bytes": FRAMED_PACKET_BYTES,
        "strict_delta_required": STRICT_DELTA,
        "strict_target_delta_required": STRICT_TARGET_DELTA,
        "strict_prior_receiver_scout_delta_required": STRICT_PRIOR_SCOUT_DELTA,
        **oracle,
    }
    payload = {
        "gate": "source_private_hellaswag_discrete_query_innovation_codec_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(default_pass_gate),
        "pass_rule": (
            "Pass if the official-train-dev-selected decoder-conditioned discrete query innovation "
            "codec beats compact TinyLlama packet-only by >=0.010 with positive paired CI95 low, "
            "beats Qwen target-only by >=0.020, beats the best prior official-train receiver scout "
            "by >=0.005, is positive on at least 4/5 contiguous blocks, and destructive/no-source "
            "controls remain within +0.002 of packet-only while separated from the real default by "
            ">=0.003."
        ),
        "packet_contract": {
            "packet_name": "discrete_query_innovation_code",
            "raw_payload_bytes": RAW_PACKET_BYTES,
            "framed_record_bytes": FRAMED_PACKET_BYTES,
            "max_codebook_size": DEFAULT_MAX_CODES,
            "selected_codebook_size": int(default_row["codebook_size"]),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "query_vectors_transmitted": False,
            "learned_discrete_source_code_transmitted": True,
            "decoder_uses_qwen_side_information": True,
            "training_uses_target_conditioned_residual_objective": True,
        },
        "headline": headline,
        "frontier_rows": frontier_rows,
        "frontier_bootstrap_samples": int(frontier_bootstrap_samples),
        "selected_bootstrap_samples": int(bootstrap_samples),
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "selected_encoder_audit": {
            "config": selected_config,
            "codebook_size": int(selected_encoded["codebook_size"]),
            "eval_code_unique_count": int(len(np.unique(selected_encoded["eval_code"]))),
            "eval_cluster_unique_count": int(len(np.unique(selected_encoded["eval_cluster"]))),
            "query_parameter_shape": list(selected_encoded["artifacts"]["query_parameters"].shape),
            "center_shape": list(selected_encoded["artifacts"]["centers"].shape),
            "train_residual_prediction_mean": selected_encoded["artifacts"][
                "train_residual_prediction_mean"
            ],
            "train_residual_prediction_std": selected_encoded["artifacts"][
                "train_residual_prediction_std"
            ],
            "eval_residual_prediction_mean": selected_encoded["eval_residual_prediction_mean"],
            "eval_residual_prediction_std": selected_encoded["eval_residual_prediction_std"],
        },
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(len(validation["answers"]) * RAW_PACKET_BYTES),
            "logical_validation_framed_record_bytes_total": int(
                len(validation["answers"]) * FRAMED_PACKET_BYTES
            ),
            "communication_object": "task_level_source_private_discrete_query_innovation_code",
            "communication_objective": "downstream_candidate_decision_accuracy",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "cached_codec_timing": timing_rows,
            "source_state_byte_floors": _source_state_floor_ratios(FRAMED_PACKET_BYTES),
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
            "This gate is the bounded Mac-local test of the next materially different source-score "
            "branch after linear, nonlinear, and switch-observability selectors saturated. The source "
            "encoder uses fixed query summaries over source candidate tokens and is trained on an "
            "official-train target-conditioned residual objective, but inference transmits only a "
            "one-byte discrete code whose low bits preserve the source candidate packet. A pass would "
            "promote a decoder-conditioned innovation-code contribution; a fail means the current "
            "cached score surface still cannot convert HellaSwag complementarity into a positive "
            "learned method without stronger source representations or native connector training."
        ),
    }
    json_path = output_dir / "hellaswag_discrete_query_innovation_codec_gate.json"
    md_path = output_dir / "hellaswag_discrete_query_innovation_codec_gate.md"
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
    parser.add_argument("--query-counts", type=_parse_int_tuple, default=DEFAULT_QUERY_COUNTS)
    parser.add_argument("--query-seeds", type=_parse_int_tuple, default=DEFAULT_QUERY_SEEDS)
    parser.add_argument("--query-temperatures", type=_parse_float_tuple, default=DEFAULT_QUERY_TEMPERATURES)
    parser.add_argument("--encoder-ridges", type=_parse_float_tuple, default=DEFAULT_ENCODER_RIDGES)
    parser.add_argument("--cluster-counts", type=_parse_int_tuple, default=DEFAULT_CLUSTER_COUNTS)
    parser.add_argument("--decoder-ridges", type=_parse_float_tuple, default=DEFAULT_DECODER_RIDGES)
    parser.add_argument("--target-priors", type=_parse_str_tuple, default=DEFAULT_TARGET_PRIORS)
    parser.add_argument("--cluster-seed", type=int, default=31)
    parser.add_argument("--kmeans-iterations", type=int, default=DEFAULT_KMEANS_ITERATIONS)
    parser.add_argument("--frontier-bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=23017)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        query_counts=args.query_counts,
        query_seeds=args.query_seeds,
        query_temperatures=args.query_temperatures,
        encoder_ridges=args.encoder_ridges,
        cluster_counts=args.cluster_counts,
        decoder_ridges=args.decoder_ridges,
        target_priors=args.target_priors,
        cluster_seed=args.cluster_seed,
        kmeans_iterations=args.kmeans_iterations,
        frontier_bootstrap_samples=args.frontier_bootstrap_samples,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
