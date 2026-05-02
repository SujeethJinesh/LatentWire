from __future__ import annotations

"""Nonlinear train-only selector/syndrome gate for HellaSwag packets.

This is a bounded Mac-local scout. It keeps the compact source-private packet
contract from the linear conditional selector gate, but replaces the linear
benefit predictor with deterministic random Fourier feature (RFF) kernels.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_conditional_selector_syndrome_gate as linear  # noqa: E402
from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_nonlinear_selector_syndrome_gate_20260502"
)
DEFAULT_DECODER_RIDGES = (1.0, 10.0, 100.0)
DEFAULT_QUANTILE_BINS = (4, 16, 32)
DEFAULT_THRESHOLDS = tuple(np.linspace(-0.2, 0.3, 21).round(6).tolist())
DEFAULT_RFF_COMPONENTS = (16, 64)
DEFAULT_RFF_GAMMAS = (0.3, 1.0, 3.0)
DEFAULT_RFF_SEEDS = (7, 19)
DEFAULT_HARM_WEIGHTS = (1.0, 2.0)
STRICT_DELTA = 0.020
SCOUT_DELTA = 0.015
MIN_ORACLE_HEADROOM_CAPTURE = 0.20
SCOUT_MIN_ORACLE_HEADROOM_CAPTURE = 0.10
CONTROL_TOLERANCE = 0.005
CONTROL_SEPARATION_DELTA = 0.010


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


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _config_hash(row: dict[str, Any]) -> str:
    keys = {
        key: row[key]
        for key in (
            "source_code_name",
            "alternative",
            "rff_components",
            "rff_gamma",
            "rff_seed",
            "harm_weight",
            "ridge",
            "threshold",
        )
        if key in row
    }
    encoded = json.dumps(keys, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _standardize_fit_eval(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fit = train_features[fit_indices].astype(np.float64)
    center = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return (
        (train_features.astype(np.float64) - center) / scale,
        (eval_features.astype(np.float64) - center) / scale,
        {"center": center.astype(np.float64), "scale": scale.astype(np.float64)},
    )


def _standardize_apply(features: np.ndarray, standardizer: dict[str, np.ndarray]) -> np.ndarray:
    return (features.astype(np.float64) - standardizer["center"]) / standardizer["scale"]


def _rff_parameters(
    input_dim: int,
    *,
    components: int,
    gamma: float,
    seed: int,
) -> dict[str, np.ndarray | int | float]:
    if int(components) <= 0:
        raise ValueError("components must be positive")
    rng = np.random.default_rng(int(seed))
    weight_scale = math.sqrt(2.0 * float(gamma))
    weights = rng.normal(
        loc=0.0,
        scale=weight_scale,
        size=(int(input_dim), int(components)),
    ).astype(np.float64)
    bias = rng.uniform(0.0, 2.0 * math.pi, size=int(components)).astype(np.float64)
    return {
        "weights": weights,
        "bias": bias,
        "components": int(components),
        "gamma": float(gamma),
        "seed": int(seed),
    }


def _rff_transform(standardized_features: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    projected = standardized_features.astype(np.float64) @ params["weights"] + params["bias"]
    scale = math.sqrt(2.0 / float(params["components"]))
    return (scale * np.cos(projected)).astype(np.float64)


def _nonlinear_feature_matrix(
    standardized_features: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    rff = _rff_transform(standardized_features, params)
    return np.concatenate(
        [
            np.ones((standardized_features.shape[0], 1), dtype=np.float64),
            standardized_features.astype(np.float64),
            rff,
        ],
        axis=1,
    )


def _benefit_targets(
    *,
    alternative: np.ndarray,
    packet: np.ndarray,
    answers: np.ndarray,
    harm_weight: float,
) -> np.ndarray:
    helps = (alternative.astype(np.int64) == answers.astype(np.int64)) & (
        packet.astype(np.int64) != answers.astype(np.int64)
    )
    harms = (alternative.astype(np.int64) != answers.astype(np.int64)) & (
        packet.astype(np.int64) == answers.astype(np.int64)
    )
    return helps.astype(np.float64) - float(harm_weight) * harms.astype(np.float64)


def _help_harm_counts(
    predictions: np.ndarray,
    packet: np.ndarray,
    answers: np.ndarray,
) -> dict[str, int]:
    selected_correct = predictions.astype(np.int64) == answers.astype(np.int64)
    packet_correct = packet.astype(np.int64) == answers.astype(np.int64)
    help_count = int(np.sum(selected_correct & ~packet_correct))
    harm_count = int(np.sum(~selected_correct & packet_correct))
    return {
        "help_count": help_count,
        "harm_count": harm_count,
        "net_help": help_count - harm_count,
    }


def _model_resident_bytes(
    *,
    standardizer: dict[str, np.ndarray],
    params: dict[str, Any],
    coef: np.ndarray,
) -> int:
    return int(
        standardizer["center"].nbytes
        + standardizer["scale"].nbytes
        + params["weights"].nbytes
        + params["bias"].nbytes
        + coef.nbytes
    )


def _source_code_configs(quantile_bins: tuple[int, ...]) -> list[dict[str, Any]]:
    return linear._source_code_configs(quantile_bins)


def _fit_source_code_for_config(
    *,
    surfaces: dict[str, Any],
    code_config: dict[str, Any],
) -> dict[str, Any]:
    return linear._fit_source_code(
        kind=str(code_config["kind"]),
        feature_name=str(code_config["feature_name"]),
        bins=int(code_config["bins"]),
        train_scores=surfaces["tiny_train_scores"],
        eval_scores=surfaces["tiny_eval_scores"],
        train_packet=surfaces["calibration"]["tiny_packet"],
        eval_packet=surfaces["validation"]["packet"],
        fit_indices=surfaces["fit_indices"],
    )


def _base_features(
    *,
    qwen_scores: np.ndarray,
    packet: np.ndarray,
    alternative: np.ndarray,
    source_code: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    return linear._selector_feature_matrix(
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
        source_code=source_code,
        codebook_size=codebook_size,
    )


def _predict_from_scores(
    *,
    packet: np.ndarray,
    alternative: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return linear._threshold_predictions(
        packet=packet,
        alternative=alternative,
        benefit_scores=scores,
        threshold=float(threshold),
    )


def _row_extra(
    *,
    code_config: dict[str, Any],
    encoded: dict[str, Any],
    alternative_name: str,
    components: int,
    gamma: float,
    rff_seed: int,
    harm_weight: float,
    ridge: float,
    threshold: float,
    standardizer: dict[str, np.ndarray],
    params: dict[str, Any],
    coef: np.ndarray,
    train_predictions: np.ndarray,
    surfaces: dict[str, Any],
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> dict[str, Any]:
    calibration = surfaces["calibration"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    fit_counts = _help_harm_counts(
        train_predictions[fit_indices],
        calibration["tiny_packet"][fit_indices],
        calibration["answers"][fit_indices],
    )
    dev_counts = _help_harm_counts(
        train_predictions[dev_indices],
        calibration["tiny_packet"][dev_indices],
        calibration["answers"][dev_indices],
    )
    raw_payload_bytes = linear._packet_bytes_for_codebook(int(encoded["codebook_size"]))
    model_bytes = _model_resident_bytes(standardizer=standardizer, params=params, coef=coef)
    return {
        "source_code_name": str(code_config["name"]),
        "source_code_kind": str(code_config["kind"]),
        "source_code_feature": str(code_config["feature_name"]),
        "source_code_bins": int(code_config["bins"]),
        "codebook_size": int(encoded["codebook_size"]),
        "raw_payload_bytes": raw_payload_bytes,
        "framed_record_bytes": raw_payload_bytes + 3,
        "theoretical_bits": int(math.ceil(math.log2(int(encoded["codebook_size"])))),
        "alternative": alternative_name,
        "rff_components": int(components),
        "rff_gamma": float(gamma),
        "rff_seed": int(rff_seed),
        "harm_weight": float(harm_weight),
        "ridge": float(ridge),
        "threshold": float(threshold),
        "base_feature_dim": int(train_features.shape[1] - int(components) - 1),
        "nonlinear_feature_dim": int(train_features.shape[1]),
        "model_public_resident_bytes": model_bytes,
        "public_preloaded_state_bytes": model_bytes,
        "transmitted_state_bytes": raw_payload_bytes,
        "official_fit_accuracy": linear._accuracy(
            train_predictions[fit_indices],
            calibration["answers"][fit_indices],
        ),
        "official_dev_accuracy": linear._accuracy(
            train_predictions[dev_indices],
            calibration["answers"][dev_indices],
        ),
        "official_fit_delta_vs_packet": linear._accuracy(
            train_predictions[fit_indices],
            calibration["answers"][fit_indices],
        )
        - linear._accuracy(calibration["tiny_packet"][fit_indices], calibration["answers"][fit_indices]),
        "official_dev_delta_vs_packet": linear._accuracy(
            train_predictions[dev_indices],
            calibration["answers"][dev_indices],
        )
        - linear._accuracy(calibration["tiny_packet"][dev_indices], calibration["answers"][dev_indices]),
        "official_fit_help_count": fit_counts["help_count"],
        "official_fit_harm_count": fit_counts["harm_count"],
        "official_fit_net_help": fit_counts["net_help"],
        "official_dev_help_count": dev_counts["help_count"],
        "official_dev_harm_count": dev_counts["harm_count"],
        "official_dev_net_help": dev_counts["net_help"],
        "official_dev_utility": float(dev_counts["help_count"] - float(harm_weight) * dev_counts["harm_count"]),
        "official_dev_override_rate_vs_packet": float(
            np.mean(train_predictions[dev_indices] != calibration["tiny_packet"][dev_indices])
        ),
    }


def _evaluate_config(
    *,
    surfaces: dict[str, Any],
    code_config: dict[str, Any],
    alternative_name: str,
    train_alternative: np.ndarray,
    eval_alternative: np.ndarray,
    decoder_ridges: tuple[float, ...],
    thresholds: tuple[float, ...],
    rff_components: tuple[int, ...],
    rff_gammas: tuple[float, ...],
    rff_seeds: tuple[int, ...],
    harm_weights: tuple[float, ...],
    seed_offset: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    encoded = _fit_source_code_for_config(surfaces=surfaces, code_config=code_config)
    train_base = _base_features(
        qwen_scores=calibration["qwen_scores"],
        packet=calibration["tiny_packet"],
        alternative=train_alternative,
        source_code=encoded["train_code"],
        codebook_size=int(encoded["codebook_size"]),
    )
    eval_base = _base_features(
        qwen_scores=validation["qwen_scores"],
        packet=validation["packet"],
        alternative=eval_alternative,
        source_code=encoded["eval_code"],
        codebook_size=int(encoded["codebook_size"]),
    )
    train_std, eval_std, standardizer = _standardize_fit_eval(train_base, eval_base, fit_indices)
    rows: list[dict[str, Any]] = []
    row_index = 0
    for components in rff_components:
        for gamma in rff_gammas:
            for rff_seed in rff_seeds:
                params = _rff_parameters(
                    train_std.shape[1],
                    components=int(components),
                    gamma=float(gamma),
                    seed=int(rff_seed),
                )
                train_features = _nonlinear_feature_matrix(train_std, params)
                eval_features = _nonlinear_feature_matrix(eval_std, params)
                for harm_weight in harm_weights:
                    targets = _benefit_targets(
                        alternative=train_alternative,
                        packet=calibration["tiny_packet"],
                        answers=calibration["answers"],
                        harm_weight=float(harm_weight),
                    )
                    for ridge in decoder_ridges:
                        coef = linear._fit_ridge(
                            train_features,
                            targets,
                            fit_indices,
                            float(ridge),
                        )
                        train_scores = linear._predict_score(train_features, coef)
                        eval_scores = linear._predict_score(eval_features, coef)
                        for threshold in thresholds:
                            train_predictions = _predict_from_scores(
                                packet=calibration["tiny_packet"],
                                alternative=train_alternative,
                                scores=train_scores,
                                threshold=float(threshold),
                            )
                            eval_predictions = _predict_from_scores(
                                packet=validation["packet"],
                                alternative=eval_alternative,
                                scores=eval_scores,
                                threshold=float(threshold),
                            )
                            rows.append(
                                linear._score_row(
                                    name="nonlinear_selector_syndrome",
                                    predictions=eval_predictions,
                                    answers=validation["answers"],
                                    packet=validation["packet"],
                                    target=validation["alternatives"]["qwen_target_score"],
                                    seed=seed_offset + row_index,
                                    bootstrap_samples=0,
                                    extra=_row_extra(
                                        code_config=code_config,
                                        encoded=encoded,
                                        alternative_name=alternative_name,
                                        components=int(components),
                                        gamma=float(gamma),
                                        rff_seed=int(rff_seed),
                                        harm_weight=float(harm_weight),
                                        ridge=float(ridge),
                                        threshold=float(threshold),
                                        standardizer=standardizer,
                                        params=params,
                                        coef=coef,
                                        train_predictions=train_predictions,
                                        surfaces=surfaces,
                                        train_features=train_features,
                                        eval_features=eval_features,
                                    ),
                                )
                            )
                            row_index += 1
    return rows


def _fit_state_for_row(
    *,
    surfaces: dict[str, Any],
    code_config: dict[str, Any],
    alternatives: dict[str, tuple[np.ndarray, np.ndarray]],
    row: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    train_alt, eval_alt = alternatives[str(row["alternative"])]
    encoded = _fit_source_code_for_config(surfaces=surfaces, code_config=code_config)
    train_base = _base_features(
        qwen_scores=calibration["qwen_scores"],
        packet=calibration["tiny_packet"],
        alternative=train_alt,
        source_code=encoded["train_code"],
        codebook_size=int(encoded["codebook_size"]),
    )
    eval_base = _base_features(
        qwen_scores=validation["qwen_scores"],
        packet=validation["packet"],
        alternative=eval_alt,
        source_code=encoded["eval_code"],
        codebook_size=int(encoded["codebook_size"]),
    )
    train_std, eval_std, standardizer = _standardize_fit_eval(train_base, eval_base, fit_indices)
    params = _rff_parameters(
        train_std.shape[1],
        components=int(row["rff_components"]),
        gamma=float(row["rff_gamma"]),
        seed=int(row["rff_seed"]),
    )
    train_features = _nonlinear_feature_matrix(train_std, params)
    eval_features = _nonlinear_feature_matrix(eval_std, params)
    targets = _benefit_targets(
        alternative=train_alt,
        packet=calibration["tiny_packet"],
        answers=calibration["answers"],
        harm_weight=float(row["harm_weight"]),
    )
    coef = linear._fit_ridge(train_features, targets, fit_indices, float(row["ridge"]))
    predictions = _predict_from_scores(
        packet=validation["packet"],
        alternative=eval_alt,
        scores=linear._predict_score(eval_features, coef),
        threshold=float(row["threshold"]),
    )
    state = {
        "encoded": encoded,
        "code_config": code_config,
        "train_alternative": train_alt,
        "eval_alternative": eval_alt,
        "standardizer": standardizer,
        "rff_params": params,
        "train_features": train_features,
        "eval_features": eval_features,
        "targets": targets,
        "coef": coef,
    }
    return predictions, state


def _eval_features_for_control(
    *,
    surfaces: dict[str, Any],
    state: dict[str, Any],
    eval_code: np.ndarray,
    eval_alternative: np.ndarray,
    rff_params: dict[str, Any] | None = None,
) -> np.ndarray:
    validation = surfaces["validation"]
    encoded = state["encoded"]
    base = _base_features(
        qwen_scores=validation["qwen_scores"],
        packet=validation["packet"],
        alternative=eval_alternative,
        source_code=np.mod(eval_code.astype(np.int64), int(encoded["codebook_size"])),
        codebook_size=int(encoded["codebook_size"]),
    )
    std = _standardize_apply(base, state["standardizer"])
    return _nonlinear_feature_matrix(std, rff_params or state["rff_params"])


def _control_rows(
    *,
    surfaces: dict[str, Any],
    selected_row: dict[str, Any],
    selected_config: dict[str, Any],
    state: dict[str, Any],
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    encoded = state["encoded"]
    eval_alt = state["eval_alternative"].astype(np.int64)
    rng = np.random.default_rng(control_seed)
    row_count = len(validation["answers"])
    codebook_size = int(encoded["codebook_size"])
    qwen_eval_code = linear._fit_source_code(
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
        rng.integers(0, max(1, codebook_size // linear.CANDIDATE_COUNT), size=row_count)
        * linear.CANDIDATE_COUNT
        + validation["packet"]
    ).astype(np.int64)
    candidate_roll = (validation["packet"].astype(np.int64) + 1) % linear.CANDIDATE_COUNT
    controls = [
        ("row_shuffle_source_code", encoded["eval_code"][rng.permutation(row_count)], eval_alt, None),
        ("qwen_derived_source_code", np.mod(qwen_eval_code, codebook_size), eval_alt, None),
        ("random_same_byte_code", random_codes, eval_alt, None),
        (
            "random_subcode_preserve_packet",
            np.mod(random_preserve_candidate, codebook_size),
            eval_alt,
            None,
        ),
        ("candidate_derangement_packet_code", candidate_roll, eval_alt, None),
        ("wrong_alternative_roll", encoded["eval_code"], (eval_alt + 1) % linear.CANDIDATE_COUNT, None),
        ("packet_only_candidate_code", validation["packet"].astype(np.int64), eval_alt, None),
        ("zero_source_code", np.zeros(row_count, dtype=np.int64), eval_alt, None),
        (
            "rff_projection_seed_control",
            encoded["eval_code"],
            eval_alt,
            _rff_parameters(
                state["standardizer"]["center"].shape[0],
                components=int(selected_row["rff_components"]),
                gamma=float(selected_row["rff_gamma"]),
                seed=int(selected_row["rff_seed"]) + 100003,
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_code, control_alt, rff_params) in enumerate(controls):
        features = _eval_features_for_control(
            surfaces=surfaces,
            state=state,
            eval_code=eval_code,
            eval_alternative=control_alt.astype(np.int64),
            rff_params=rff_params,
        )
        predictions = _predict_from_scores(
            packet=validation["packet"],
            alternative=control_alt.astype(np.int64),
            scores=linear._predict_score(features, state["coef"]),
            threshold=float(selected_row["threshold"]),
        )
        rows.append(
            linear._score_row(
                name=name,
                predictions=predictions,
                answers=validation["answers"],
                packet=validation["packet"],
                target=validation["alternatives"]["qwen_target_score"],
                seed=30000 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "source_code_name": str(selected_row["source_code_name"]),
                    "alternative": str(selected_row["alternative"]),
                    "rff_components": int(selected_row["rff_components"]),
                    "rff_gamma": float(selected_row["rff_gamma"]),
                    "rff_seed": int(selected_row["rff_seed"]),
                    "harm_weight": float(selected_row["harm_weight"]),
                    "ridge": float(selected_row["ridge"]),
                    "threshold": float(selected_row["threshold"]),
                    "codebook_size": codebook_size,
                },
            )
        )
    permuted_targets = state["targets"][rng.permutation(len(state["targets"]))]
    permuted_coef = linear._fit_ridge(
        state["train_features"],
        permuted_targets,
        fit_indices,
        float(selected_row["ridge"]),
    )
    predictions = _predict_from_scores(
        packet=validation["packet"],
        alternative=eval_alt,
        scores=linear._predict_score(state["eval_features"], permuted_coef),
        threshold=float(selected_row["threshold"]),
    )
    rows.append(
        linear._score_row(
            name="label_permutation_benefit_decoder",
            predictions=predictions,
            answers=validation["answers"],
            packet=validation["packet"],
            target=validation["alternatives"]["qwen_target_score"],
            seed=30099,
            bootstrap_samples=bootstrap_samples,
            extra={
                "source_code_name": str(selected_row["source_code_name"]),
                "alternative": str(selected_row["alternative"]),
                "rff_components": int(selected_row["rff_components"]),
                "rff_gamma": float(selected_row["rff_gamma"]),
                "rff_seed": int(selected_row["rff_seed"]),
                "harm_weight": float(selected_row["harm_weight"]),
                "ridge": float(selected_row["ridge"]),
                "threshold": float(selected_row["threshold"]),
                "codebook_size": codebook_size,
            },
        )
    )
    return rows


def _rescore_selected_row(
    *,
    row: dict[str, Any],
    predictions: np.ndarray,
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    excluded = {
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
    return linear._score_row(
        name=str(row["name"]),
        predictions=predictions,
        answers=surfaces["validation"]["answers"],
        packet=surfaces["validation"]["packet"],
        target=surfaces["validation"]["alternatives"]["qwen_target_score"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        extra={key: value for key, value in row.items() if key not in excluded},
    )


def _top_rows(rows: list[dict[str, Any]], *, key: str, limit: int) -> list[dict[str, Any]]:
    return [linear._strip_large(row) for row in sorted(rows, key=lambda item: item[key], reverse=True)[:limit]]


def _darwin_safe_peak_rss_mib() -> float:
    maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _percentiles_us(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _microbench_selected(
    *,
    surfaces: dict[str, Any],
    state: dict[str, Any],
    selected_row: dict[str, Any],
    repetitions: int = 40,
) -> dict[str, Any]:
    validation = surfaces["validation"]
    row_count = len(validation["answers"])
    batch_rows: list[dict[str, Any]] = []
    for batch_size in (1, 4, 16, 64, 256):
        count = min(batch_size, row_count)
        indices = np.arange(count, dtype=np.int64)
        base_us: list[float] = []
        rff_us: list[float] = []
        decision_us: list[float] = []
        end_to_end_us: list[float] = []
        for _ in range(int(repetitions)):
            start_total = time.perf_counter()
            start = time.perf_counter()
            base = _base_features(
                qwen_scores=validation["qwen_scores"][indices],
                packet=validation["packet"][indices],
                alternative=state["eval_alternative"][indices],
                source_code=state["encoded"]["eval_code"][indices],
                codebook_size=int(state["encoded"]["codebook_size"]),
            )
            std = _standardize_apply(base, state["standardizer"])
            base_us.append((time.perf_counter() - start) * 1_000_000.0 / count)
            start = time.perf_counter()
            rff = _rff_transform(std, state["rff_params"])
            rff_us.append((time.perf_counter() - start) * 1_000_000.0 / count)
            start = time.perf_counter()
            features = np.concatenate(
                [np.ones((std.shape[0], 1), dtype=np.float64), std, rff],
                axis=1,
            )
            scores = linear._predict_score(features, state["coef"])
            _predict_from_scores(
                packet=validation["packet"][indices],
                alternative=state["eval_alternative"][indices],
                scores=scores,
                threshold=float(selected_row["threshold"]),
            )
            decision_us.append((time.perf_counter() - start) * 1_000_000.0 / count)
            end_to_end_us.append((time.perf_counter() - start_total) * 1_000_000.0 / count)
        batch_rows.append(
            {
                "batch_size": int(count),
                "packet_decode_us_per_request": _percentiles_us(base_us),
                "resampler_forward_us_per_request": _percentiles_us(rff_us),
                "selector_decision_us_per_request": _percentiles_us(decision_us),
                "end_to_end_cached_packet_us_per_request": _percentiles_us(end_to_end_us),
                "records_per_second_p50": float(1_000_000.0 / max(_percentiles_us(end_to_end_us)["p50"], 1e-9)),
            }
        )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "platform": sys.platform,
        "repetitions": int(repetitions),
        "batch_rows": batch_rows,
        "peak_rss_mib": _darwin_safe_peak_rss_mib(),
        "cpu_user_s": float(usage.ru_utime),
        "cpu_sys_s": float(usage.ru_stime),
        "source_feature_extract_ms_per_request": None,
        "source_feature_extract_note": "Mac-local microbench covers cached packet decode/selector only; model forward extraction is not measured.",
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Nonlinear Selector/Syndrome Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- default source code: `{h['default_source_code_name']}`",
        f"- default alternative: `{h['default_alternative']}`",
        f"- default RFF: `{h['default_rff_components']}` components, gamma `{h['default_rff_gamma']}`, seed `{h['default_rff_seed']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- default oracle-headroom capture: `{h['default_oracle_headroom_capture']:.6f}`",
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
    rff_components: tuple[int, ...] = DEFAULT_RFF_COMPONENTS,
    rff_gammas: tuple[float, ...] = DEFAULT_RFF_GAMMAS,
    rff_seeds: tuple[int, ...] = DEFAULT_RFF_SEEDS,
    harm_weights: tuple[float, ...] = DEFAULT_HARM_WEIGHTS,
    bootstrap_samples: int = 500,
    control_seed: int = 31001,
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
    alternatives = linear._alternative_predictions(train=calibration, eval_bundle=validation)
    configs = _source_code_configs(quantile_bins)
    config_by_name = {str(config["name"]): config for config in configs}
    frontier_rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        for alt_index, (alt_name, (train_alt, eval_alt)) in enumerate(alternatives.items()):
            frontier_rows.extend(
                _evaluate_config(
                    surfaces=surfaces,
                    code_config=config,
                    alternative_name=alt_name,
                    train_alternative=train_alt,
                    eval_alternative=eval_alt,
                    decoder_ridges=decoder_ridges,
                    thresholds=thresholds,
                    rff_components=rff_components,
                    rff_gammas=rff_gammas,
                    rff_seeds=rff_seeds,
                    harm_weights=harm_weights,
                    seed_offset=28000 + config_index * 2000 + alt_index * 400,
                )
            )
    default_row = max(
        frontier_rows,
        key=lambda row: (
            row["official_dev_delta_vs_packet"],
            row["official_dev_utility"],
            row["official_dev_net_help"],
            row["official_dev_accuracy"],
            -row["official_dev_override_rate_vs_packet"],
            -row["model_public_resident_bytes"],
            -row["codebook_size"],
        ),
    )
    best_scout = max(
        frontier_rows,
        key=lambda row: (
            row["delta_vs_packet_only"],
            row["ci95_low_vs_packet_only"],
            row["accuracy"],
            row["official_dev_delta_vs_packet"],
        ),
    )
    default_predictions, default_state = _fit_state_for_row(
        surfaces=surfaces,
        code_config=config_by_name[str(default_row["source_code_name"])],
        alternatives=alternatives,
        row=default_row,
    )
    best_predictions, _ = _fit_state_for_row(
        surfaces=surfaces,
        code_config=config_by_name[str(best_scout["source_code_name"])],
        alternatives=alternatives,
        row=best_scout,
    )
    default_row = _rescore_selected_row(
        row=default_row,
        predictions=default_predictions,
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        seed=28501,
    )
    best_scout = _rescore_selected_row(
        row=best_scout,
        predictions=best_predictions,
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        seed=28502,
    )
    baselines = linear._baseline_rows(
        surfaces=surfaces,
        alternatives=alternatives,
        bootstrap_samples=bootstrap_samples,
    )
    oracle_delta = next(
        row["delta_vs_packet_only"] for row in baselines if row["name"] == "packet_or_any_qwen_oracle"
    )
    default_oracle_capture = float(default_row["delta_vs_packet_only"] / oracle_delta if oracle_delta > 0 else 0.0)
    best_oracle_capture = float(best_scout["delta_vs_packet_only"] / oracle_delta if oracle_delta > 0 else 0.0)
    default_blocks = linear._block_rows(default_predictions, validation["packet"], validation["answers"])
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_rows = _control_rows(
        surfaces=surfaces,
        selected_row=default_row,
        selected_config=config_by_name[str(default_row["source_code_name"])],
        state=default_state,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_max_delta = max(row["delta_vs_packet_only"] for row in control_rows)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and default_oracle_capture >= MIN_ORACLE_HEADROOM_CAPTURE
        and block_stability_gate
        and control_separation_gate
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= SCOUT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
        and best_oracle_capture >= SCOUT_MIN_ORACLE_HEADROOM_CAPTURE
    )
    packet_only_accuracy = linear._accuracy(validation["packet"], validation["answers"])
    qwen_hybrid_accuracy = linear._accuracy(
        validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        validation["answers"],
    )
    microbench = _microbench_selected(
        surfaces=surfaces,
        state=default_state,
        selected_row=default_row,
        repetitions=40,
    )
    config_hash = _config_hash(default_row)
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
        "default_rff_components": int(default_row["rff_components"]),
        "default_rff_gamma": float(default_row["rff_gamma"]),
        "default_rff_seed": int(default_row["rff_seed"]),
        "default_harm_weight": float(default_row["harm_weight"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_ci95_high_vs_packet_only": default_row["ci95_high_vs_packet_only"],
        "default_help_count": int(default_row["help_count"]),
        "default_harm_count": int(default_row["harm_count"]),
        "default_oracle_headroom_capture": default_oracle_capture,
        "default_override_rate_vs_packet": default_row["override_rate_vs_packet"],
        "default_official_dev_accuracy": default_row["official_dev_accuracy"],
        "default_official_dev_delta_vs_packet": default_row["official_dev_delta_vs_packet"],
        "best_scout_source_code_name": str(best_scout["source_code_name"]),
        "best_scout_alternative": str(best_scout["alternative"]),
        "best_scout_rff_components": int(best_scout["rff_components"]),
        "best_scout_rff_gamma": float(best_scout["rff_gamma"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_scout_oracle_headroom_capture": best_oracle_capture,
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "default_pass_gate": default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "raw_payload_bytes": int(default_row["raw_payload_bytes"]),
        "framed_record_bytes": int(default_row["framed_record_bytes"]),
        "theoretical_bits": int(default_row["theoretical_bits"]),
        "model_public_resident_bytes": int(default_row["model_public_resident_bytes"]),
        "strict_delta_required": STRICT_DELTA,
        "min_oracle_headroom_capture_required": MIN_ORACLE_HEADROOM_CAPTURE,
    }
    lay_explanation = (
        "This experiment asks whether the previous failure was just too linear. TinyLlama still "
        "sends only a tiny byte packet. The receiver uses a small nonlinear referee, trained only "
        "on official HellaSwag train examples, to decide whether to trust TinyLlama's packet or "
        "switch to Qwen's candidate."
    )
    interpretation = (
        "A pass would promote a fixed-byte nonlinear conditional syndrome method. A fail means the "
        "HellaSwag oracle headroom is not recovered by a bounded nonlinear train-only receiver over "
        "the current packet/syndrome and Qwen score features; the next live method must change the "
        "source information itself, move to a true joint connector on GPU, or cut the HellaSwag "
        "receiver-improvement claim."
    )
    payload = {
        "gate": "source_private_hellaswag_nonlinear_selector_syndrome_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_id": f"nonlinear_selector_syndrome:{config_hash}",
        "method_family": "nonlinear_syndrome_resampler",
        "commit_hash": _git_head(),
        "config_hash": config_hash,
        "pass_gate": bool(default_pass_gate),
        "promotion_status": "promoted" if default_pass_gate else "not_promoted",
        "predeclared_default": True,
        "selected_on_train_dev_only": True,
        "pass_rule": (
            "Pass if the train-dev-selected nonlinear selector beats packet-only by >=0.020, "
            "has positive paired CI95 low, captures >=20% of oracle headroom, improves at least "
            "4/5 contiguous blocks, and separates from destructive controls."
        ),
        "packet_contract": {
            "packet_name": "nonlinear_selector_syndrome_packet",
            "theoretical_bits": int(default_row["theoretical_bits"]),
            "raw_payload_bytes": int(default_row["raw_payload_bytes"]),
            "framed_record_bytes": int(default_row["framed_record_bytes"]),
            "batch64_framed_bytes": int(default_row["framed_record_bytes"]) * 64,
            "single_request_cacheline_bytes": 64,
            "max_codebook_size": 256,
            "selected_codebook_size": int(default_row["codebook_size"]),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "raw_syndrome_vector_exposed": False,
            "learned_discrete_source_code_transmitted": True,
            "receiver_uses_qwen_side_information": True,
        },
        "model_contract": {
            "resampler_type": "random_fourier_feature_rbf_ridge",
            "hidden_dim": int(default_row["nonlinear_feature_dim"]),
            "query_count": 0,
            "active_codes": int(default_row["codebook_size"]),
            "codebook_size": int(default_row["codebook_size"]),
            "codebook_bytes": int(default_row["codebook_size"]),
            "params_bytes": int(default_row["model_public_resident_bytes"]),
            "public_preloaded_state_bytes": int(default_row["public_preloaded_state_bytes"]),
            "transmitted_state_bytes": int(default_row["transmitted_state_bytes"]),
        },
        "headline": headline,
        "baselines": baselines,
        "frontier_rows_top_by_dev": _top_rows(frontier_rows, key="official_dev_delta_vs_packet", limit=30),
        "frontier_rows_top_by_eval_diagnostic": _top_rows(frontier_rows, key="delta_vs_packet_only", limit=30),
        "default_row": linear._strip_large(default_row),
        "best_scout_row": linear._strip_large(best_scout),
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "selected_source_code_audit": default_state["encoded"]["audit"],
        "mac_microbench": microbench,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": int(default_row["raw_payload_bytes"]),
            "framed_record_bytes_per_request": int(default_row["framed_record_bytes"]),
            "logical_validation_raw_payload_bytes_total": int(
                len(validation["answers"]) * int(default_row["raw_payload_bytes"])
            ),
            "logical_validation_framed_record_bytes_total": int(
                len(validation["answers"]) * int(default_row["framed_record_bytes"])
            ),
            "communication_object": "task_level_source_private_nonlinear_selector_syndrome",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "source_state_byte_floors": linear._source_state_floor_ratios(int(default_row["framed_record_bytes"])),
            "native_gpu_claims_allowed": False,
            "native_status": "pending",
            "pending_native_metrics": ["TTFT", "TPOT", "goodput", "HBM", "GPU_memory"],
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
    json_path = output_dir / "hellaswag_nonlinear_selector_syndrome_gate.json"
    md_path = output_dir / "hellaswag_nonlinear_selector_syndrome_gate.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "row_id": payload["row_id"],
        "commit_hash": payload["commit_hash"],
        "config_hash": payload["config_hash"],
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
    parser.add_argument("--rff-components", type=_parse_int_tuple, default=DEFAULT_RFF_COMPONENTS)
    parser.add_argument("--rff-gammas", type=_parse_float_tuple, default=DEFAULT_RFF_GAMMAS)
    parser.add_argument("--rff-seeds", type=_parse_int_tuple, default=DEFAULT_RFF_SEEDS)
    parser.add_argument("--harm-weights", type=_parse_float_tuple, default=DEFAULT_HARM_WEIGHTS)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=31001)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        decoder_ridges=args.decoder_ridges,
        quantile_bins=args.quantile_bins,
        thresholds=args.thresholds,
        rff_components=args.rff_components,
        rff_gammas=args.rff_gammas,
        rff_seeds=args.rff_seeds,
        harm_weights=args.harm_weights,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
