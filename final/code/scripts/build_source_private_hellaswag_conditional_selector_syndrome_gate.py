from __future__ import annotations

"""Train-only conditional selector/syndrome gate for HellaSwag packets."""

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


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_conditional_selector_syndrome_gate_20260502"
)
DEFAULT_DECODER_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_QUANTILE_BINS = (2, 4, 8, 16, 32, 64)
DEFAULT_THRESHOLDS = tuple(np.linspace(-0.3, 0.3, 31).round(6).tolist())
STRICT_DELTA = 0.020
SCOUT_DELTA = 0.020
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 1
FRAMED_PACKET_BYTES = 4
CANDIDATE_COUNT = 4
FP16_ONE_TOKEN_KV_FLOOR_BYTES = 12_288.0
KVCOMM30_FP16_ONE_TOKEN_KV_FLOOR_BYTES = 3_686.4
QJL_1BIT_ONE_TOKEN_KV_FLOOR_BYTES = 768.0
TURBOQUANT_35BIT_ONE_TOKEN_KV_FLOOR_BYTES = 2_688.0
KIVI_2BIT_ONE_TOKEN_KV_FLOOR_BYTES = 1_536.0
KVQUANT_3BIT_ONE_TOKEN_KV_FLOOR_BYTES = 2_304.0


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


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one float is required")
    return parsed


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == answers.astype(np.int64)))


def _packet_bytes_for_codebook(codebook_size: int) -> int:
    if codebook_size < 1:
        raise ValueError("codebook_size must be positive")
    return max(1, int(math.ceil(math.log2(codebook_size) / 8.0)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    delta = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    if int(samples) <= 0:
        mean = float(np.mean(delta))
        return {"delta": mean, "ci95_low": mean, "ci95_high": mean}
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    boot = np.mean(delta[boot_indices], axis=1)
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _fit_ridge(features: np.ndarray, targets: np.ndarray, fit_indices: np.ndarray, ridge: float) -> np.ndarray:
    x_fit = features[fit_indices].astype(np.float64)
    y_fit = targets[fit_indices].astype(np.float64)
    reg = float(ridge) * np.eye(x_fit.shape[1], dtype=np.float64)
    return np.linalg.solve(x_fit.T @ x_fit + reg, x_fit.T @ y_fit)


def _predict_score(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return features.astype(np.float64) @ coef.astype(np.float64)


def _safe_divide(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values / np.where(np.abs(scale) < 1e-6, 1.0, scale)


def _row_zscores(scores: np.ndarray) -> np.ndarray:
    centered = scores.astype(np.float64) - np.mean(scores.astype(np.float64), axis=1, keepdims=True)
    return _safe_divide(centered, np.std(centered, axis=1, keepdims=True))


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores.astype(np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _candidate_rank(zscores: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    order = np.argsort(-zscores, axis=1)
    ranks = np.zeros(len(candidate), dtype=np.float64)
    for row_index, row_order in enumerate(order):
        ranks[row_index] = float(np.where(row_order == int(candidate[row_index]))[0][0])
    return ranks


def _quantile_edges(values: np.ndarray, fit_indices: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(values[fit_indices], np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges.astype(np.float64)


def _apply_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2).astype(
        np.int64
    )


def _fit_source_code(
    *,
    kind: str,
    feature_name: str,
    bins: int,
    train_scores: np.ndarray,
    eval_scores: np.ndarray,
    train_packet: np.ndarray,
    eval_packet: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    if kind == "candidate_only":
        return {
            "train_code": train_packet.astype(np.int64),
            "eval_code": eval_packet.astype(np.int64),
            "codebook_size": CANDIDATE_COUNT,
            "audit": {"kind": kind, "feature_name": "candidate_id", "bins": 1},
        }
    train_z = _row_zscores(train_scores)
    eval_z = _row_zscores(eval_scores)
    if feature_name == "packet_z":
        train_values = train_z[np.arange(len(train_packet)), train_packet.astype(np.int64)]
        eval_values = eval_z[np.arange(len(eval_packet)), eval_packet.astype(np.int64)]
    elif feature_name == "top2_margin":
        train_values = _top2_margin(train_z)
        eval_values = _top2_margin(eval_z)
    elif feature_name == "packet_rank":
        train_values = _candidate_rank(train_z, train_packet) / float(CANDIDATE_COUNT - 1)
        eval_values = _candidate_rank(eval_z, eval_packet) / float(CANDIDATE_COUNT - 1)
    else:
        raise ValueError(f"unsupported source-code feature: {feature_name}")
    edges = _quantile_edges(train_values, fit_indices, int(bins))
    train_bin = _apply_edges(train_values, edges)
    eval_bin = _apply_edges(eval_values, edges)
    return {
        "train_code": (train_bin * CANDIDATE_COUNT + train_packet).astype(np.int64),
        "eval_code": (eval_bin * CANDIDATE_COUNT + eval_packet).astype(np.int64),
        "codebook_size": int(bins * CANDIDATE_COUNT),
        "audit": {
            "kind": kind,
            "feature_name": feature_name,
            "bins": int(bins),
            "edges": [float(item) for item in edges],
        },
    }


def _source_code_configs(quantile_bins: tuple[int, ...]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {"name": "candidate_only", "kind": "candidate_only", "feature_name": "candidate_id", "bins": 1}
    ]
    for feature_name in ("packet_z", "top2_margin", "packet_rank"):
        for bins in quantile_bins:
            if int(bins) * CANDIDATE_COUNT <= 256:
                configs.append(
                    {
                        "name": f"{feature_name}_q{int(bins)}",
                        "kind": "quantile",
                        "feature_name": feature_name,
                        "bins": int(bins),
                    }
                )
    return configs


def _alternative_predictions(
    *,
    train: dict[str, Any],
    eval_bundle: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "qwen_target_score": (
            train["qwen_target"].astype(np.int64),
            eval_bundle["alternatives"]["qwen_target_score"].astype(np.int64),
        ),
        "qwen_mean_zscore": (
            train["qwen_mean"].astype(np.int64),
            eval_bundle["alternatives"]["mean_zscore_prediction"].astype(np.int64),
        ),
        "qwen_hybrid": (
            train["qwen_hybrid"].astype(np.int64),
            eval_bundle["alternatives"]["hybrid_vote_on_score_agreement_prediction"].astype(np.int64),
        ),
    }


def _selector_feature_matrix(
    *,
    qwen_scores: np.ndarray,
    packet: np.ndarray,
    alternative: np.ndarray,
    source_code: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    qwen_z = _row_zscores(qwen_scores)
    packet = packet.astype(np.int64)
    alternative = alternative.astype(np.int64)
    source_code = np.mod(source_code.astype(np.int64), int(codebook_size))
    source_candidate = np.mod(source_code, CANDIDATE_COUNT)
    row_index = np.arange(len(packet))
    qwen_top = np.argmax(qwen_z, axis=1).astype(np.int64)
    source_eye = np.eye(int(codebook_size), dtype=np.float64)[source_code]
    candidate_eye = np.eye(CANDIDATE_COUNT, dtype=np.float64)
    packet_eye = candidate_eye[packet]
    alt_eye = candidate_eye[alternative]
    qwen_top_eye = candidate_eye[qwen_top]
    packet_z = qwen_z[row_index, packet]
    alt_z = qwen_z[row_index, alternative]
    features = [
        np.ones((len(packet), 1), dtype=np.float64),
        qwen_z,
        packet_z[:, None],
        alt_z[:, None],
        (alt_z - packet_z)[:, None],
        _top2_margin(qwen_z)[:, None],
        (packet == alternative)[:, None].astype(np.float64),
        (alternative == qwen_top)[:, None].astype(np.float64),
        (packet == qwen_top)[:, None].astype(np.float64),
        (source_candidate == packet)[:, None].astype(np.float64),
        (source_candidate == alternative)[:, None].astype(np.float64),
        packet_eye,
        alt_eye,
        qwen_top_eye,
        source_eye,
    ]
    return np.concatenate(features, axis=1).astype(np.float64)


def _threshold_predictions(
    *,
    packet: np.ndarray,
    alternative: np.ndarray,
    benefit_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    use_alternative = (benefit_scores > float(threshold)) & (
        alternative.astype(np.int64) != packet.astype(np.int64)
    )
    return np.where(use_alternative, alternative, packet).astype(np.int64)


def _score_row(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    packet: np.ndarray,
    target: np.ndarray,
    seed: int,
    bootstrap_samples: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ci_packet = _paired_ci(
        selected=predictions,
        baseline=packet,
        answers=answers,
        seed=seed,
        samples=bootstrap_samples,
    )
    ci_target = _paired_ci(
        selected=predictions,
        baseline=target,
        answers=answers,
        seed=seed + 1000,
        samples=bootstrap_samples,
    )
    selected_correct = predictions == answers
    packet_correct = packet == answers
    row = {
        "name": name,
        "accuracy": _accuracy(predictions, answers),
        "packet_only_accuracy": _accuracy(packet, answers),
        "target_side_accuracy": _accuracy(target, answers),
        "delta_vs_packet_only": ci_packet["delta"],
        "ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "ci95_high_vs_packet_only": ci_packet["ci95_high"],
        "delta_vs_target_side": ci_target["delta"],
        "ci95_low_vs_target_side": ci_target["ci95_low"],
        "ci95_high_vs_target_side": ci_target["ci95_high"],
        "help_count": int(np.sum(selected_correct & ~packet_correct)),
        "harm_count": int(np.sum(~selected_correct & packet_correct)),
        "net_help": int(np.sum(selected_correct & ~packet_correct) - np.sum(~selected_correct & packet_correct)),
        "override_rate_vs_packet": float(np.mean(predictions != packet)),
    }
    if extra:
        row.update(extra)
    return row


def _block_rows(selected: np.ndarray, packet: np.ndarray, answers: np.ndarray, blocks: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(np.arange(len(answers), dtype=np.int64), int(blocks))):
        selected_acc = _accuracy(selected[indices], answers[indices])
        packet_acc = _accuracy(packet[indices], answers[indices])
        rows.append(
            {
                "block": int(index),
                "rows": int(len(indices)),
                "selected_accuracy": selected_acc,
                "packet_only_accuracy": packet_acc,
                "delta_vs_packet_only": selected_acc - packet_acc,
            }
        )
    return rows


def _evaluate_config(
    *,
    surfaces: dict[str, Any],
    code_config: dict[str, Any],
    alternative_name: str,
    train_alternative: np.ndarray,
    eval_alternative: np.ndarray,
    decoder_ridges: tuple[float, ...],
    thresholds: tuple[float, ...],
    bootstrap_samples: int,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, float, float], np.ndarray], dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    encoded = _fit_source_code(
        kind=str(code_config["kind"]),
        feature_name=str(code_config["feature_name"]),
        bins=int(code_config["bins"]),
        train_scores=surfaces["tiny_train_scores"],
        eval_scores=surfaces["tiny_eval_scores"],
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        fit_indices=fit_indices,
    )
    train_features = _selector_feature_matrix(
        qwen_scores=calibration["qwen_scores"],
        packet=calibration["tiny_packet"],
        alternative=train_alternative,
        source_code=encoded["train_code"],
        codebook_size=int(encoded["codebook_size"]),
    )
    eval_features = _selector_feature_matrix(
        qwen_scores=validation["qwen_scores"],
        packet=validation["packet"],
        alternative=eval_alternative,
        source_code=encoded["eval_code"],
        codebook_size=int(encoded["codebook_size"]),
    )
    targets = (
        (train_alternative == calibration["answers"]).astype(np.float64)
        - (calibration["tiny_packet"] == calibration["answers"]).astype(np.float64)
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, float, float], np.ndarray] = {}
    for ridge in decoder_ridges:
        coef = _fit_ridge(train_features, targets, fit_indices, float(ridge))
        train_scores = _predict_score(train_features, coef)
        eval_scores = _predict_score(eval_features, coef)
        for threshold in thresholds:
            train_predictions = _threshold_predictions(
                packet=calibration["tiny_packet"],
                alternative=train_alternative,
                benefit_scores=train_scores,
                threshold=float(threshold),
            )
            eval_predictions = _threshold_predictions(
                packet=validation["packet"],
                alternative=eval_alternative,
                benefit_scores=eval_scores,
                threshold=float(threshold),
            )
            key = (str(code_config["name"]), alternative_name, float(ridge), float(threshold))
            predictions[key] = eval_predictions
            rows.append(
                _score_row(
                    name="conditional_selector_syndrome",
                    predictions=eval_predictions,
                    answers=validation["answers"],
                    packet=validation["packet"],
                    target=validation["alternatives"]["qwen_target_score"],
                    seed=seed_offset + len(rows),
                    bootstrap_samples=bootstrap_samples,
                    extra={
                        "source_code_name": str(code_config["name"]),
                        "source_code_kind": str(code_config["kind"]),
                        "source_code_feature": str(code_config["feature_name"]),
                        "source_code_bins": int(code_config["bins"]),
                        "codebook_size": int(encoded["codebook_size"]),
                        "raw_payload_bytes": _packet_bytes_for_codebook(int(encoded["codebook_size"])),
                        "framed_record_bytes": _packet_bytes_for_codebook(int(encoded["codebook_size"])) + 3,
                        "alternative": alternative_name,
                        "ridge": float(ridge),
                        "threshold": float(threshold),
                        "official_fit_accuracy": _accuracy(train_predictions[fit_indices], calibration["answers"][fit_indices]),
                        "official_dev_accuracy": _accuracy(train_predictions[dev_indices], calibration["answers"][dev_indices]),
                        "official_dev_delta_vs_packet": _accuracy(
                            train_predictions[dev_indices],
                            calibration["answers"][dev_indices],
                        )
                        - _accuracy(calibration["tiny_packet"][dev_indices], calibration["answers"][dev_indices]),
                        "official_dev_override_rate_vs_packet": float(
                            np.mean(train_predictions[dev_indices] != calibration["tiny_packet"][dev_indices])
                        ),
                    },
                )
            )
    return rows, predictions, encoded


def _baseline_rows(
    *,
    surfaces: dict[str, Any],
    alternatives: dict[str, tuple[np.ndarray, np.ndarray]],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    validation = surfaces["validation"]
    rows = [
        _score_row(
            name="packet_only",
            predictions=validation["packet"],
            answers=validation["answers"],
            packet=validation["packet"],
            target=validation["alternatives"]["qwen_target_score"],
            seed=25000,
            bootstrap_samples=bootstrap_samples,
            extra={"raw_payload_bytes": RAW_PACKET_BYTES, "framed_record_bytes": FRAMED_PACKET_BYTES},
        )
    ]
    for offset, (name, (_, eval_pred)) in enumerate(alternatives.items()):
        rows.append(
            _score_row(
                name=name,
                predictions=eval_pred,
                answers=validation["answers"],
                packet=validation["packet"],
                target=validation["alternatives"]["qwen_target_score"],
                seed=25010 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={"raw_payload_bytes": 0, "framed_record_bytes": 0},
            )
        )
    oracle_correct = validation["packet"] == validation["answers"]
    for _, eval_pred in alternatives.values():
        oracle_correct |= eval_pred == validation["answers"]
    oracle_predictions = np.where(oracle_correct, validation["answers"], validation["packet"]).astype(
        np.int64
    )
    rows.append(
        _score_row(
            name="packet_or_any_qwen_oracle",
            predictions=oracle_predictions,
            answers=validation["answers"],
            packet=validation["packet"],
            target=validation["alternatives"]["qwen_target_score"],
            seed=25020,
            bootstrap_samples=bootstrap_samples,
            extra={"not_promotable": True},
        )
    )
    return rows


def _control_rows(
    *,
    surfaces: dict[str, Any],
    selected_row: dict[str, Any],
    selected_config: dict[str, Any],
    selected_encoded: dict[str, Any],
    alternatives: dict[str, tuple[np.ndarray, np.ndarray]],
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    alternative_name = str(selected_row["alternative"])
    train_alt, eval_alt = alternatives[alternative_name]
    codebook_size = int(selected_encoded["codebook_size"])
    train_features = _selector_feature_matrix(
        qwen_scores=calibration["qwen_scores"],
        packet=calibration["tiny_packet"],
        alternative=train_alt,
        source_code=selected_encoded["train_code"],
        codebook_size=codebook_size,
    )
    targets = (
        (train_alt == calibration["answers"]).astype(np.float64)
        - (calibration["tiny_packet"] == calibration["answers"]).astype(np.float64)
    )
    coef = _fit_ridge(train_features, targets, fit_indices, float(selected_row["ridge"]))
    rng = np.random.default_rng(control_seed)
    row_count = len(validation["answers"])
    qwen_eval_code = _fit_source_code(
        kind=str(selected_config["kind"]),
        feature_name=str(selected_config["feature_name"]),
        bins=int(selected_config["bins"]),
        train_scores=surfaces["tiny_train_scores"],
        eval_scores=validation["qwen_scores"],
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["alternatives"]["qwen_target_score"],
        fit_indices=fit_indices,
    )["eval_code"]
    random_codes = rng.integers(0, codebook_size, size=row_count, dtype=np.int64)
    random_preserve_candidate = (
        rng.integers(0, max(1, codebook_size // CANDIDATE_COUNT), size=row_count) * CANDIDATE_COUNT
        + validation["packet"]
    ).astype(np.int64)
    candidate_roll = (validation["packet"].astype(np.int64) + 1) % CANDIDATE_COUNT
    controls = [
        ("row_shuffle_source_code", selected_encoded["eval_code"][rng.permutation(row_count)], eval_alt),
        ("qwen_derived_source_code", np.mod(qwen_eval_code, codebook_size), eval_alt),
        ("random_same_byte_code", random_codes, eval_alt),
        ("random_subcode_preserve_packet", np.mod(random_preserve_candidate, codebook_size), eval_alt),
        ("candidate_derangement_packet_code", candidate_roll, eval_alt),
        ("wrong_alternative_roll", selected_encoded["eval_code"], (eval_alt + 1) % CANDIDATE_COUNT),
        ("packet_only_candidate_code", validation["packet"].astype(np.int64), eval_alt),
        ("zero_source_code", np.zeros(row_count, dtype=np.int64), eval_alt),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_code, control_alt) in enumerate(controls):
        eval_features = _selector_feature_matrix(
            qwen_scores=validation["qwen_scores"],
            packet=validation["packet"],
            alternative=control_alt.astype(np.int64),
            source_code=eval_code.astype(np.int64),
            codebook_size=codebook_size,
        )
        predictions = _threshold_predictions(
            packet=validation["packet"],
            alternative=control_alt.astype(np.int64),
            benefit_scores=_predict_score(eval_features, coef),
            threshold=float(selected_row["threshold"]),
        )
        rows.append(
            _score_row(
                name=name,
                predictions=predictions,
                answers=validation["answers"],
                packet=validation["packet"],
                target=validation["alternatives"]["qwen_target_score"],
                seed=26000 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "source_code_name": str(selected_row["source_code_name"]),
                    "alternative": alternative_name,
                    "ridge": float(selected_row["ridge"]),
                    "threshold": float(selected_row["threshold"]),
                    "codebook_size": codebook_size,
                },
            )
        )
    permuted_targets = targets[rng.permutation(len(targets))]
    permuted_coef = _fit_ridge(train_features, permuted_targets, fit_indices, float(selected_row["ridge"]))
    eval_features = _selector_feature_matrix(
        qwen_scores=validation["qwen_scores"],
        packet=validation["packet"],
        alternative=eval_alt,
        source_code=selected_encoded["eval_code"],
        codebook_size=codebook_size,
    )
    rows.append(
        _score_row(
            name="label_permutation_benefit_decoder",
            predictions=_threshold_predictions(
                packet=validation["packet"],
                alternative=eval_alt,
                benefit_scores=_predict_score(eval_features, permuted_coef),
                threshold=float(selected_row["threshold"]),
            ),
            answers=validation["answers"],
            packet=validation["packet"],
            target=validation["alternatives"]["qwen_target_score"],
            seed=26099,
            bootstrap_samples=bootstrap_samples,
            extra={
                "source_code_name": str(selected_row["source_code_name"]),
                "alternative": alternative_name,
                "ridge": float(selected_row["ridge"]),
                "threshold": float(selected_row["threshold"]),
                "codebook_size": codebook_size,
            },
        )
    )
    return rows


def _strip_large(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "predictions"}


def _top_rows(rows: list[dict[str, Any]], *, key: str, limit: int) -> list[dict[str, Any]]:
    return [_strip_large(row) for row in sorted(rows, key=lambda item: item[key], reverse=True)[:limit]]


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
            "These are conservative source-state byte/exposure floors only, not native throughput, "
            "quality, TTFT, TPOT, HBM, or goodput measurements."
        ),
    }


def _rescore_selected_row(
    *,
    row: dict[str, Any],
    predictions: np.ndarray,
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    extra = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "name",
            "accuracy",
            "packet_only_accuracy",
            "target_side_accuracy",
            "delta_vs_packet_only",
            "ci95_low_vs_packet_only",
            "ci95_high_vs_packet_only",
            "delta_vs_target_side",
            "ci95_low_vs_target_side",
            "ci95_high_vs_target_side",
            "help_count",
            "harm_count",
            "net_help",
            "override_rate_vs_packet",
        }
    }
    return _score_row(
        name=str(row["name"]),
        predictions=predictions,
        answers=surfaces["validation"]["answers"],
        packet=surfaces["validation"]["packet"],
        target=surfaces["validation"]["alternatives"]["qwen_target_score"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        extra=extra,
    )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Conditional Selector/Syndrome Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- default source code: `{h['default_source_code_name']}`",
        f"- default alternative: `{h['default_alternative']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- best scout accuracy: `{h['best_scout_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
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
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    bootstrap_samples: int = 500,
    control_seed: int = 27001,
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
    alternatives = _alternative_predictions(train=calibration, eval_bundle=validation)
    configs = _source_code_configs(quantile_bins)
    frontier_rows: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[str, str, float, float], np.ndarray] = {}
    encoded_by_name: dict[str, dict[str, Any]] = {}
    config_by_name: dict[str, dict[str, Any]] = {}
    for config_index, config in enumerate(configs):
        config_by_name[str(config["name"])] = config
        for alt_index, (alt_name, (train_alt, eval_alt)) in enumerate(alternatives.items()):
            rows, predictions, encoded = _evaluate_config(
                surfaces=surfaces,
                code_config=config,
                alternative_name=alt_name,
                train_alternative=train_alt,
                eval_alternative=eval_alt,
                decoder_ridges=decoder_ridges,
                thresholds=thresholds,
                bootstrap_samples=0,
                seed_offset=24000 + config_index * 1000 + alt_index * 200,
            )
            frontier_rows.extend(rows)
            predictions_by_key.update(predictions)
            encoded_by_name[str(config["name"])] = encoded | {"config": config}
    default_row = max(
        frontier_rows,
        key=lambda row: (
            row["official_dev_accuracy"],
            row["official_dev_delta_vs_packet"],
            -row["official_dev_override_rate_vs_packet"],
            row["delta_vs_packet_only"],
            -row["codebook_size"],
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
    default_key = (
        str(default_row["source_code_name"]),
        str(default_row["alternative"]),
        float(default_row["ridge"]),
        float(default_row["threshold"]),
    )
    default_predictions = predictions_by_key[default_key]
    default_row = _rescore_selected_row(
        row=default_row,
        predictions=default_predictions,
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        seed=24501,
    )
    best_key = (
        str(best_scout["source_code_name"]),
        str(best_scout["alternative"]),
        float(best_scout["ridge"]),
        float(best_scout["threshold"]),
    )
    best_scout = _rescore_selected_row(
        row=best_scout,
        predictions=predictions_by_key[best_key],
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        seed=24502,
    )
    default_blocks = _block_rows(default_predictions, validation["packet"], validation["answers"])
    control_rows = _control_rows(
        surfaces=surfaces,
        selected_row=default_row,
        selected_config=config_by_name[str(default_row["source_code_name"])],
        selected_encoded=encoded_by_name[str(default_row["source_code_name"])],
        alternatives=alternatives,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_max_delta = max(row["delta_vs_packet_only"] for row in control_rows)
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and block_stability_gate
        and control_separation_gate
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= SCOUT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    baselines = _baseline_rows(
        surfaces=surfaces,
        alternatives=alternatives,
        bootstrap_samples=bootstrap_samples,
    )
    oracle_delta = next(
        row["delta_vs_packet_only"] for row in baselines if row["name"] == "packet_or_any_qwen_oracle"
    )
    packet_only_accuracy = _accuracy(validation["packet"], validation["answers"])
    qwen_hybrid_accuracy = _accuracy(
        validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        validation["answers"],
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["answers"])),
        "official_train_fit_rows": int(len(surfaces["fit_indices"])),
        "official_train_dev_rows": int(len(surfaces["dev_indices"])),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "validation_rows": int(len(validation["answers"])),
        "packet_only_accuracy": packet_only_accuracy,
        "qwen_hybrid_accuracy": qwen_hybrid_accuracy,
        "default_source_code_name": str(default_row["source_code_name"]),
        "default_alternative": str(default_row["alternative"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_ci95_high_vs_packet_only": default_row["ci95_high_vs_packet_only"],
        "default_help_count": int(default_row["help_count"]),
        "default_harm_count": int(default_row["harm_count"]),
        "default_oracle_headroom_capture": float(
            default_row["delta_vs_packet_only"] / oracle_delta if oracle_delta > 0.0 else 0.0
        ),
        "default_override_rate_vs_packet": default_row["override_rate_vs_packet"],
        "default_official_dev_accuracy": default_row["official_dev_accuracy"],
        "default_official_dev_delta_vs_packet": default_row["official_dev_delta_vs_packet"],
        "best_scout_source_code_name": str(best_scout["source_code_name"]),
        "best_scout_alternative": str(best_scout["alternative"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_scout_oracle_headroom_capture": float(
            best_scout["delta_vs_packet_only"] / oracle_delta if oracle_delta > 0.0 else 0.0
        ),
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "default_pass_gate": default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "raw_payload_bytes": int(default_row["raw_payload_bytes"]),
        "framed_record_bytes": int(default_row["framed_record_bytes"]),
        "strict_delta_required": STRICT_DELTA,
        "scout_delta_required": SCOUT_DELTA,
    }
    lay_explanation = (
        "This experiment trains a tiny referee. The source still sends only a compact byte packet. "
        "When the source packet and Qwen disagree, the referee predicts whether switching to Qwen's "
        "candidate is likely to help or harm, using only official-train calibration labels."
    )
    interpretation = (
        "This is the first post-headroom method gate aimed directly at the Tiny/Qwen disagreement "
        "surface. A pass would promote a source-private conditional selector/syndrome method under a "
        "one-byte packet contract. A fail means the measured oracle headroom is not recovered by "
        "linear train-only benefit prediction over packet id, source confidence bins, and Qwen score "
        "features; the next method would need a nonlinear resampler/cross-attention connector or a "
        "different benchmark surface."
    )
    payload = {
        "gate": "source_private_hellaswag_conditional_selector_syndrome_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(default_pass_gate),
        "pass_rule": (
            "Pass if the official-train-dev-selected selector beats packet-only by >=0.020 with "
            "positive paired CI95 low, captures a predeclared nontrivial fraction of oracle "
            "headroom, improves at least 4/5 contiguous blocks, and destructive controls stay "
            "within +0.002 of packet-only while separated from the selected row by >=0.003. "
            "Best-scout rows are diagnostic unless also train-dev selected."
        ),
        "packet_contract": {
            "packet_name": "conditional_selector_syndrome_packet",
            "raw_payload_bytes": int(default_row["raw_payload_bytes"]),
            "framed_record_bytes": int(default_row["framed_record_bytes"]),
            "max_codebook_size": 256,
            "selected_codebook_size": int(default_row["codebook_size"]),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "learned_discrete_source_code_transmitted": True,
            "receiver_uses_qwen_side_information": True,
        },
        "headline": headline,
        "baselines": baselines,
        "frontier_rows_top_by_dev": _top_rows(frontier_rows, key="official_dev_accuracy", limit=30),
        "frontier_rows_top_by_eval_diagnostic": _top_rows(frontier_rows, key="delta_vs_packet_only", limit=30),
        "default_row": _strip_large(default_row),
        "best_scout_row": _strip_large(best_scout),
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "selected_source_code_audit": encoded_by_name[str(default_row["source_code_name"])]["audit"],
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": int(default_row["raw_payload_bytes"]),
            "framed_record_bytes_per_request": int(default_row["framed_record_bytes"]),
            "logical_validation_raw_payload_bytes_total": int(
                len(validation["answers"]) * int(default_row["raw_payload_bytes"])
            ),
            "logical_validation_framed_record_bytes_total": int(
                len(validation["answers"]) * int(default_row["framed_record_bytes"])
            ),
            "communication_object": "task_level_source_private_conditional_selector_syndrome",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "single_request_cacheline_bytes": 64,
            "batch64_framed_bytes_per_request": int(default_row["framed_record_bytes"]),
            "batch64_cacheline_amortized_bytes_per_request": 64.0 / 64.0,
            "source_state_byte_floors": _source_state_floor_ratios(int(default_row["framed_record_bytes"])),
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
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_conditional_selector_syndrome_gate.json"
    md_path = output_dir / "hellaswag_conditional_selector_syndrome_gate.md"
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
    parser.add_argument("--thresholds", type=_parse_float_tuple, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=27001)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        decoder_ridges=args.decoder_ridges,
        quantile_bins=args.quantile_bins,
        thresholds=args.thresholds,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
