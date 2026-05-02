from __future__ import annotations

"""Wyner-Ziv-style residual-logit packet gate for HellaSwag."""

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

from scripts import build_source_private_hellaswag_disagreement_prototype_receiver as proto  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_acceptance_gate as accept  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_headroom_decomposition as decomp  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_wyner_ziv_residual_packet_gate_20260502")
DEFAULT_TRAIN_PATH = official.DEFAULT_TRAIN_PATH
DEFAULT_TINY_TRAIN_CACHE_DIR = official.DEFAULT_TINY_TRAIN_CACHE_DIR
DEFAULT_QWEN_TRAIN_CACHE_DIR = official.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_TINY_EVAL_PACKET_JSONL = official.DEFAULT_TINY_EVAL_PACKET_JSONL
DEFAULT_TINY_EVAL_ARTIFACT = official.DEFAULT_TINY_EVAL_ARTIFACT
DEFAULT_TINY_EVAL_ROWS = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "hellaswag_validation_rows_0_10042.jsonl"
)
DEFAULT_TINY_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "source_eval_score_cache.json"
)
DEFAULT_QWEN_EVAL_PACKET_JSONL = official.DEFAULT_QWEN_EVAL_PACKET_JSONL
DEFAULT_QWEN_GLOBAL_ARTIFACT = official.DEFAULT_QWEN_GLOBAL_ARTIFACT
DEFAULT_SAMPLE_SEEDS = official.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = official.DEFAULT_SPLIT_SEEDS
DEFAULT_RIDGES = official.DEFAULT_RIDGES
DEFAULT_DECODER_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
DEFAULT_QUANTIZER_BINS = (2, 4, 8)
STRICT_DELTA = 0.010
STRICT_TARGET_DELTA = 0.02
BEST_PRIOR_RECEIVER_SCOUT_ACCURACY = 0.620594
STRICT_PRIOR_SCOUT_DELTA = 0.005
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 2
FRAMED_PACKET_BYTES = 5
CANDIDATE_COUNT = 4


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return official._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return official._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    return official._read_jsonl(path)


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


def _row_zscores(scores: np.ndarray) -> np.ndarray:
    centered = scores.astype(np.float64) - np.mean(scores.astype(np.float64), axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.where(scale < 1e-6, 1.0, scale)


def _one_hot(indices: np.ndarray, width: int = CANDIDATE_COUNT) -> np.ndarray:
    indices = indices.astype(np.int64)
    out = np.zeros((len(indices), width), dtype=np.float64)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def _packet_bits_per_request(*, candidate_count: int, quantizer_bins: int) -> int:
    return int(math.ceil(math.log2(candidate_count))) + int(candidate_count) * int(
        math.ceil(math.log2(quantizer_bins))
    )


def _quantizer_from_fit(
    *,
    train_source_z: np.ndarray,
    fit_indices: np.ndarray,
    bins: int,
) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("quantizer must have at least two bins")
    values = train_source_z[fit_indices].reshape(-1)
    edges = np.quantile(values, np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "bins": int(bins),
        "edges": edges.astype(np.float64),
        "centers": centers.astype(np.float64),
    }


def _apply_quantizer(values: np.ndarray, quantizer: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(quantizer["edges"], dtype=np.float64)
    centers = np.asarray(quantizer["centers"], dtype=np.float64)
    codes = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(centers) - 1).astype(
        np.int64
    )
    decoded = centers[codes].astype(np.float64)
    return codes, decoded


def _candidate_feature_tensor(
    *,
    source_packet: np.ndarray,
    source_score_sketch: np.ndarray,
    qwen_scores: np.ndarray,
    qwen_target: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_hybrid: np.ndarray,
    feature_view: str,
) -> np.ndarray:
    row_count = len(source_packet)
    qwen_z = _row_zscores(qwen_scores)
    candidate_eye = np.eye(CANDIDATE_COUNT, dtype=np.float64)
    source_packet = source_packet.astype(np.int64)
    qwen_target = qwen_target.astype(np.int64)
    qwen_mean = qwen_mean.astype(np.int64)
    qwen_hybrid = qwen_hybrid.astype(np.int64)
    features: list[np.ndarray] = []
    for candidate in range(CANDIDATE_COUNT):
        parts: list[np.ndarray] = [
            qwen_z[:, candidate : candidate + 1],
            (qwen_target == candidate)[:, None].astype(np.float64),
            (qwen_mean == candidate)[:, None].astype(np.float64),
            (qwen_hybrid == candidate)[:, None].astype(np.float64),
            np.full((row_count, 1), candidate / float(CANDIDATE_COUNT - 1), dtype=np.float64),
            np.repeat(candidate_eye[candidate][None, :], row_count, axis=0),
        ]
        if feature_view in {"full", "packet_only"}:
            parts.extend(
                [
                    (source_packet == candidate)[:, None].astype(np.float64),
                    _one_hot(source_packet),
                ]
            )
        if feature_view in {"full", "sketch_only"}:
            parts.extend(
                [
                    source_score_sketch[:, candidate : candidate + 1],
                    (source_score_sketch[:, candidate : candidate + 1] * qwen_z[:, candidate : candidate + 1]),
                ]
            )
        if feature_view == "target_side_only":
            pass
        elif feature_view not in {"full", "packet_only", "sketch_only"}:
            raise ValueError(f"unsupported feature_view: {feature_view}")
        features.append(np.concatenate(parts, axis=1))
    return np.stack(features, axis=1).astype(np.float64)


def _candidate_feature_names(feature_view: str) -> list[str]:
    names = [
        "qwen_candidate_zscore",
        "is_qwen_target_score_candidate",
        "is_qwen_mean_zscore_candidate",
        "is_qwen_hybrid_candidate",
        "candidate_index_scaled",
        "candidate_one_hot_0",
        "candidate_one_hot_1",
        "candidate_one_hot_2",
        "candidate_one_hot_3",
    ]
    if feature_view in {"full", "packet_only"}:
        names.extend(
            [
                "is_source_packet_candidate",
                "source_packet_one_hot_0",
                "source_packet_one_hot_1",
                "source_packet_one_hot_2",
                "source_packet_one_hot_3",
            ]
        )
    if feature_view in {"full", "sketch_only"}:
        names.extend(
            [
                "source_quantized_score_residual",
                "qwen_zscore_x_source_quantized_score_residual",
            ]
        )
    if feature_view == "target_side_only":
        return names
    if feature_view not in {"full", "packet_only", "sketch_only"}:
        raise ValueError(f"unsupported feature_view: {feature_view}")
    return names


def _fit_candidate_decoder(
    *,
    train_features: np.ndarray,
    train_answers: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
    label_permutation_seed: int | None = None,
) -> np.ndarray:
    row_count, candidate_count, feature_dim = train_features.shape
    features = train_features.reshape(row_count * candidate_count, feature_dim)
    target_answers = train_answers.astype(np.int64).copy()
    if label_permutation_seed is not None:
        rng = np.random.default_rng(label_permutation_seed)
        target_answers = target_answers[rng.permutation(len(target_answers))]
    targets = np.zeros(row_count * candidate_count, dtype=np.float64)
    targets[np.arange(row_count) * candidate_count + target_answers] = 1.0
    flat_indices = np.concatenate(
        [np.arange(int(index) * candidate_count, int(index) * candidate_count + candidate_count) for index in fit_indices]
    )
    x_fit = features[flat_indices]
    y_fit = targets[flat_indices]
    reg = float(ridge) * np.eye(feature_dim, dtype=np.float64)
    return np.linalg.solve(x_fit.T @ x_fit + reg, x_fit.T @ y_fit)


def _predict_candidate_decoder(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    scores = features.reshape(features.shape[0] * features.shape[1], features.shape[2]) @ coef
    return np.argmax(scores.reshape(features.shape[0], features.shape[1]), axis=1).astype(np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == answers.astype(np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    delta = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    if samples <= 0:
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


def _block_rows(
    *,
    selected: np.ndarray,
    packet: np.ndarray,
    answers: np.ndarray,
    block_count: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = np.array_split(np.arange(len(answers), dtype=np.int64), block_count)
    for index, indices in enumerate(splits):
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


def _score_row(
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
    ci_packet = _paired_ci(
        selected=predictions,
        baseline=packet_predictions,
        answers=answers,
        seed=seed,
        samples=bootstrap_samples,
    )
    ci_target = _paired_ci(
        selected=predictions,
        baseline=qwen_target_predictions,
        answers=answers,
        seed=seed + 1000,
        samples=bootstrap_samples,
    )
    packet_correct = packet_predictions == answers
    pred_correct = predictions == answers
    row = {
        "name": name,
        "accuracy": _accuracy(predictions, answers),
        "packet_only_accuracy": _accuracy(packet_predictions, answers),
        "qwen_target_accuracy": _accuracy(qwen_target_predictions, answers),
        "delta_vs_packet_only": ci_packet["delta"],
        "ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "ci95_high_vs_packet_only": ci_packet["ci95_high"],
        "delta_vs_qwen_target": ci_target["delta"],
        "ci95_low_vs_qwen_target": ci_target["ci95_low"],
        "help_count": int(np.sum(pred_correct & ~packet_correct)),
        "harm_count": int(np.sum(~pred_correct & packet_correct)),
        "net_help": int(np.sum(pred_correct & ~packet_correct) - np.sum(~pred_correct & packet_correct)),
        "override_rate_vs_packet": float(np.mean(predictions != packet_predictions)),
    }
    if extra:
        row.update(extra)
    return row


def _tiny_score_matrix_for_calibration(
    *,
    calibration_rows: list[dict[str, Any]],
    train_path: pathlib.Path,
    tiny_train_cache_dir: pathlib.Path,
    sample_seeds: tuple[int, ...],
    train_hidden_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_train_rows = official.arc_gate._load_rows(_resolve(train_path))
    score_by_row_id: dict[str, np.ndarray] = {}
    sample_audit: list[dict[str, Any]] = []
    for seed in sample_seeds:
        sample = official._load_sample_cache(
            cache_dir=tiny_train_cache_dir,
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for row_id, scores in zip(sample["row_ids"], sample["scores"], strict=True):
            score_by_row_id.setdefault(str(row_id), np.asarray(scores, dtype=np.float64))
        sample_audit.append(
            {
                "sample_seed": int(seed),
                "row_count": int(len(sample["row_ids"])),
                "score_cache": sample["score_path"],
                "score_cache_sha256": sample["score_sha256"],
            }
        )
    missing = [str(row["row_id"]) for row in calibration_rows if str(row["row_id"]) not in score_by_row_id]
    if missing:
        raise ValueError(f"missing TinyLlama scores for {len(missing)} calibration rows")
    return (
        np.vstack([score_by_row_id[str(row["row_id"])] for row in calibration_rows]).astype(np.float64),
        sample_audit,
    )


def _load_surfaces(
    *,
    train_path: pathlib.Path,
    tiny_train_cache_dir: pathlib.Path,
    qwen_train_cache_dir: pathlib.Path,
    sample_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    ridges: tuple[float, ...],
    train_hidden_rows: int,
    dev_fraction: float,
    tiny_eval_packet_jsonl: pathlib.Path,
    qwen_eval_packet_jsonl: pathlib.Path,
    qwen_global_artifact: pathlib.Path,
    tiny_eval_rows: pathlib.Path,
    tiny_eval_score_cache: pathlib.Path,
    tiny_aggregation_policy: str,
) -> dict[str, Any]:
    calibration_started = time.perf_counter()
    calibration, fit_indices, dev_indices, audit = proto._load_official_train_calibration(
        train_path=train_path,
        tiny_train_cache_dir=tiny_train_cache_dir,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        ridges=ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
        tiny_aggregation_policy=tiny_aggregation_policy,
    )
    tiny_train_scores, tiny_score_audit = _tiny_score_matrix_for_calibration(
        calibration_rows=calibration["rows"],
        train_path=train_path,
        tiny_train_cache_dir=tiny_train_cache_dir,
        sample_seeds=sample_seeds,
        train_hidden_rows=train_hidden_rows,
    )
    validation = proto._load_validation_bundle(
        tiny_eval_packet_jsonl=tiny_eval_packet_jsonl,
        qwen_eval_packet_jsonl=qwen_eval_packet_jsonl,
        qwen_global_artifact=qwen_global_artifact,
    )
    tiny_eval_arc_rows = arc_gate._load_rows(_resolve(tiny_eval_rows))
    tiny_eval_scores, _, tiny_eval_score_model = headroom._load_score_cache(
        _resolve(tiny_eval_score_cache),
        rows=tiny_eval_arc_rows,
    )
    tiny_eval_scores_np = np.asarray(tiny_eval_scores, dtype=np.float64)
    if len(tiny_eval_scores_np) != len(validation["answers"]):
        raise ValueError("TinyLlama eval score cache is not aligned with validation packet rows")
    return {
        "calibration": calibration,
        "fit_indices": fit_indices,
        "dev_indices": dev_indices,
        "validation": validation,
        "tiny_train_scores": tiny_train_scores,
        "tiny_eval_scores": tiny_eval_scores_np,
        "tiny_eval_score_model": tiny_eval_score_model,
        "audit": {**audit, "tiny_score_sample_cache_rows": tiny_score_audit},
        "calibration_wall_time_s": float(time.perf_counter() - calibration_started),
    }


def _make_feature_surfaces(
    *,
    calibration: dict[str, Any],
    validation: dict[str, Any],
    tiny_train_scores: np.ndarray,
    tiny_eval_scores: np.ndarray,
    fit_indices: np.ndarray,
    quantizer_bins: int,
    feature_view: str,
) -> dict[str, Any]:
    train_source_z = _row_zscores(tiny_train_scores)
    eval_source_z = _row_zscores(tiny_eval_scores)
    quantizer = _quantizer_from_fit(
        train_source_z=train_source_z,
        fit_indices=fit_indices,
        bins=quantizer_bins,
    )
    train_codes, train_sketch = _apply_quantizer(train_source_z, quantizer)
    eval_codes, eval_sketch = _apply_quantizer(eval_source_z, quantizer)
    train_features = _candidate_feature_tensor(
        source_packet=calibration["tiny_packet"],
        source_score_sketch=train_sketch,
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        feature_view=feature_view,
    )
    eval_features = _candidate_feature_tensor(
        source_packet=validation["packet"],
        source_score_sketch=eval_sketch,
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        feature_view=feature_view,
    )
    return {
        "train_features": train_features,
        "eval_features": eval_features,
        "train_codes": train_codes,
        "eval_codes": eval_codes,
        "train_sketch": train_sketch,
        "eval_sketch": eval_sketch,
        "quantizer": quantizer,
    }


def _decoder_predictions(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
    label_permutation_seed: int | None = None,
) -> np.ndarray:
    coef = _fit_candidate_decoder(
        train_features=train_features,
        train_answers=answers,
        fit_indices=fit_indices,
        ridge=ridge,
        label_permutation_seed=label_permutation_seed,
    )
    return _predict_candidate_decoder(eval_features, coef)


def _control_rows(
    *,
    selected_config: dict[str, Any],
    surfaces: dict[str, Any],
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    feature_view = selected_config["feature_view"]
    quantizer_bins = int(selected_config["quantizer_bins"])
    ridge = float(selected_config["ridge"])
    feature_surfaces = _make_feature_surfaces(
        calibration=calibration,
        validation=validation,
        tiny_train_scores=surfaces["tiny_train_scores"],
        tiny_eval_scores=surfaces["tiny_eval_scores"],
        fit_indices=fit_indices,
        quantizer_bins=quantizer_bins,
        feature_view=feature_view,
    )
    eval_sketch = feature_surfaces["eval_sketch"]
    rng = np.random.default_rng(control_seed)
    qwen_eval_z = _row_zscores(validation["qwen_scores"])
    _, qwen_eval_sketch = _apply_quantizer(qwen_eval_z, feature_surfaces["quantizer"])
    random_eval_packet = rng.integers(0, CANDIDATE_COUNT, size=len(validation["answers"]))
    random_eval_sketch = rng.choice(
        feature_surfaces["quantizer"]["centers"],
        size=eval_sketch.shape,
        replace=True,
    )

    full_packet_shuffle = rng.permutation(len(eval_sketch))
    score_shuffle = rng.permutation(len(eval_sketch))
    candidate_roll_packet = (validation["packet"].astype(np.int64) + 1) % CANDIDATE_COUNT
    score_roll_sketch = np.roll(eval_sketch, 1, axis=1)
    qwen_derived_packet = np.argmax(qwen_eval_z, axis=1).astype(np.int64)

    control_specs = [
        ("candidate_and_score_roll_packet", candidate_roll_packet, score_roll_sketch),
        ("score_bin_roll_only", validation["packet"], score_roll_sketch),
        ("candidate_label_roll_only", candidate_roll_packet, eval_sketch),
        (
            "row_shuffle_full_tiny_packet",
            validation["packet"][full_packet_shuffle],
            eval_sketch[full_packet_shuffle],
        ),
        ("row_shuffle_score_sketch", validation["packet"], eval_sketch[score_shuffle]),
        ("qwen_derived_packet", qwen_derived_packet, qwen_eval_sketch),
        ("qwen_derived_score_sketch", validation["packet"], qwen_eval_sketch),
        ("random_same_byte_packet", random_eval_packet.astype(np.int64), random_eval_sketch.astype(np.float64)),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_packet, eval_score_sketch) in enumerate(control_specs):
        eval_features = _candidate_feature_tensor(
            source_packet=eval_packet.astype(np.int64),
            source_score_sketch=eval_score_sketch.astype(np.float64),
            qwen_scores=validation["qwen_scores"],
            qwen_target=validation["alternatives"]["qwen_target_score"],
            qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
            qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            feature_view=feature_view,
        )
        predictions = _decoder_predictions(
            train_features=feature_surfaces["train_features"],
            eval_features=eval_features,
            answers=calibration["answers"],
            fit_indices=fit_indices,
            ridge=ridge,
        )
        rows.append(
            _score_row(
                name=name,
                predictions=predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=15000 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={"feature_view": feature_view, "quantizer_bins": quantizer_bins, "ridge": ridge},
            )
        )
    permuted_predictions = _decoder_predictions(
        train_features=feature_surfaces["train_features"],
        eval_features=feature_surfaces["eval_features"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=ridge,
        label_permutation_seed=control_seed + 77,
    )
    rows.append(
        _score_row(
            name="label_permutation_decoder",
            predictions=permuted_predictions,
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=15099,
            bootstrap_samples=bootstrap_samples,
            extra={"feature_view": feature_view, "quantizer_bins": quantizer_bins, "ridge": ridge},
        )
    )
    packet_predictions = validation["packet"]
    rows.append(
        _score_row(
            name="packet_only",
            predictions=packet_predictions,
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=15100,
            bootstrap_samples=bootstrap_samples,
            extra={"feature_view": "baseline", "quantizer_bins": 0, "ridge": 0.0},
        )
    )
    return rows


def _summarize_quantizer(quantizer: dict[str, Any]) -> dict[str, Any]:
    return {
        "bins": int(quantizer["bins"]),
        "edges": [float(item) for item in np.asarray(quantizer["edges"], dtype=np.float64)],
        "centers": [float(item) for item in np.asarray(quantizer["centers"], dtype=np.float64)],
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Wyner-Ziv Residual-Logit Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- best prior receiver scout accuracy: `{h['best_prior_receiver_scout_accuracy']:.6f}`",
        f"- default delta vs best prior receiver scout: `{h['default_delta_vs_best_prior_receiver_scout']:.6f}`",
        f"- best scout accuracy: `{h['best_scout_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        f"- packet bits used by default: `{h['default_packet_bits_per_request']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = DEFAULT_TRAIN_PATH,
    tiny_train_cache_dir: pathlib.Path = DEFAULT_TINY_TRAIN_CACHE_DIR,
    qwen_train_cache_dir: pathlib.Path = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    tiny_eval_packet_jsonl: pathlib.Path = DEFAULT_TINY_EVAL_PACKET_JSONL,
    tiny_eval_artifact: pathlib.Path = DEFAULT_TINY_EVAL_ARTIFACT,
    tiny_eval_rows: pathlib.Path = DEFAULT_TINY_EVAL_ROWS,
    tiny_eval_score_cache: pathlib.Path = DEFAULT_TINY_EVAL_SCORE_CACHE,
    qwen_eval_packet_jsonl: pathlib.Path = DEFAULT_QWEN_EVAL_PACKET_JSONL,
    qwen_global_artifact: pathlib.Path = DEFAULT_QWEN_GLOBAL_ARTIFACT,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    decoder_ridges: tuple[float, ...] = DEFAULT_DECODER_RIDGES,
    quantizer_bins: tuple[int, ...] = DEFAULT_QUANTIZER_BINS,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = 500,
    control_seed: int = 15017,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    surfaces = _load_surfaces(
        train_path=train_path,
        tiny_train_cache_dir=tiny_train_cache_dir,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        ridges=ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
        tiny_eval_packet_jsonl=tiny_eval_packet_jsonl,
        qwen_eval_packet_jsonl=qwen_eval_packet_jsonl,
        qwen_global_artifact=qwen_global_artifact,
        tiny_eval_rows=tiny_eval_rows,
        tiny_eval_score_cache=tiny_eval_score_cache,
        tiny_aggregation_policy="mean_zscore",
    )
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    eval_answers = validation["answers"]
    packet_eval = validation["packet"]
    qwen_target_eval = validation["alternatives"]["qwen_target_score"]
    packet_only_accuracy = _accuracy(packet_eval, eval_answers)
    qwen_target_accuracy = _accuracy(qwen_target_eval, eval_answers)

    frontier_rows: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[int, str, float], np.ndarray] = {}
    quantizer_by_bins: dict[int, dict[str, Any]] = {}
    feature_views = ("full", "packet_only", "sketch_only", "target_side_only")
    for bins in quantizer_bins:
        bits = _packet_bits_per_request(candidate_count=CANDIDATE_COUNT, quantizer_bins=int(bins))
        if bits > RAW_PACKET_BYTES * 8:
            continue
        for feature_view in feature_views:
            feature_surfaces = _make_feature_surfaces(
                calibration=calibration,
                validation=validation,
                tiny_train_scores=surfaces["tiny_train_scores"],
                tiny_eval_scores=surfaces["tiny_eval_scores"],
                fit_indices=fit_indices,
                quantizer_bins=int(bins),
                feature_view=feature_view,
            )
            quantizer_by_bins[int(bins)] = feature_surfaces["quantizer"]
            for ridge in decoder_ridges:
                coef = _fit_candidate_decoder(
                    train_features=feature_surfaces["train_features"],
                    train_answers=calibration["answers"],
                    fit_indices=fit_indices,
                    ridge=float(ridge),
                )
                train_predictions = _predict_candidate_decoder(feature_surfaces["train_features"], coef)
                eval_predictions = _predict_candidate_decoder(feature_surfaces["eval_features"], coef)
                key = (int(bins), feature_view, float(ridge))
                predictions_by_key[key] = eval_predictions
                row = _score_row(
                    name="candidate_decoder",
                    predictions=eval_predictions,
                    answers=eval_answers,
                    packet_predictions=packet_eval,
                    qwen_target_predictions=qwen_target_eval,
                    seed=16000 + len(frontier_rows),
                    bootstrap_samples=bootstrap_samples,
                    extra={
                        "quantizer_bins": int(bins),
                        "packet_bits_per_request": bits,
                        "feature_view": feature_view,
                        "ridge": float(ridge),
                        "official_fit_accuracy": _accuracy(train_predictions[fit_indices], calibration["answers"][fit_indices]),
                        "official_dev_accuracy": _accuracy(train_predictions[dev_indices], calibration["answers"][dev_indices]),
                        "official_dev_delta_vs_packet": _accuracy(
                            train_predictions[dev_indices],
                            calibration["answers"][dev_indices],
                        )
                        - _accuracy(calibration["tiny_packet"][dev_indices], calibration["answers"][dev_indices]),
                    },
                )
                frontier_rows.append(row)

    if not frontier_rows:
        raise ValueError("no valid frontier rows fit inside packet byte budget")
    default_config = max(
        (row for row in frontier_rows if row["feature_view"] == "full"),
        key=lambda row: (
            row["official_dev_accuracy"],
            row["official_dev_delta_vs_packet"],
            row["quantizer_bins"],
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
    default_key = (
        int(default_config["quantizer_bins"]),
        str(default_config["feature_view"]),
        float(default_config["ridge"]),
    )
    default_predictions = predictions_by_key[default_key]
    default_feature_surfaces = _make_feature_surfaces(
        calibration=calibration,
        validation=validation,
        tiny_train_scores=surfaces["tiny_train_scores"],
        tiny_eval_scores=surfaces["tiny_eval_scores"],
        fit_indices=fit_indices,
        quantizer_bins=int(default_config["quantizer_bins"]),
        feature_view=str(default_config["feature_view"]),
    )
    default_coef = _fit_candidate_decoder(
        train_features=default_feature_surfaces["train_features"],
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(default_config["ridge"]),
    )
    default_blocks = _block_rows(
        selected=default_predictions,
        packet=packet_eval,
        answers=eval_answers,
    )
    control_rows = _control_rows(
        selected_config=default_config,
        surfaces=surfaces,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_max_delta = max(
        row["delta_vs_packet_only"] for row in control_rows if row["name"] != "packet_only"
    )
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_separation_gate = bool(
        default_config["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    prior_receiver_scout_gate = bool(
        default_config["accuracy"] - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY >= STRICT_PRIOR_SCOUT_DELTA
    )
    default_pass_gate = bool(
        default_config["delta_vs_packet_only"] >= STRICT_DELTA
        and default_config["ci95_low_vs_packet_only"] > 0.0
        and default_config["delta_vs_qwen_target"] >= STRICT_TARGET_DELTA
        and block_stability_gate
        and control_separation_gate
        and prior_receiver_scout_gate
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    pass_gate = bool(default_pass_gate)
    default_quantizer = quantizer_by_bins[int(default_config["quantizer_bins"])]
    headline = {
        "official_train_calibration_rows": int(len(calibration["answers"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "validation_rows": int(len(eval_answers)),
        "packet_only_accuracy": packet_only_accuracy,
        "qwen_target_accuracy": qwen_target_accuracy,
        "default_accuracy": default_config["accuracy"],
        "default_delta_vs_packet_only": default_config["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_config["ci95_low_vs_packet_only"],
        "default_delta_vs_qwen_target": default_config["delta_vs_qwen_target"],
        "best_prior_receiver_scout_accuracy": BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "default_delta_vs_best_prior_receiver_scout": default_config["accuracy"]
        - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "prior_receiver_scout_gate": prior_receiver_scout_gate,
        "default_feature_view": default_config["feature_view"],
        "default_quantizer_bins": default_config["quantizer_bins"],
        "default_ridge": default_config["ridge"],
        "default_packet_bits_per_request": default_config["packet_bits_per_request"],
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_scout_feature_view": best_scout["feature_view"],
        "best_scout_quantizer_bins": best_scout["quantizer_bins"],
        "best_scout_ridge": best_scout["ridge"],
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "default_pass_gate": default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "raw_payload_bytes": RAW_PACKET_BYTES,
        "framed_record_bytes": FRAMED_PACKET_BYTES,
        "strict_delta_required": STRICT_DELTA,
        "strict_prior_receiver_scout_delta_required": STRICT_PRIOR_SCOUT_DELTA,
    }
    payload = {
        "gate": "source_private_hellaswag_wyner_ziv_residual_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if the official-train-dev-selected full Wyner-Ziv residual-logit decoder beats "
            "TinyLlama packet-only by >=0.010 with positive paired CI95 low, beats Qwen target-only "
            "by >=0.02, beats the best prior official-train receiver scout by >=0.005, is positive "
            "on at least 4/5 contiguous blocks, and destructive controls remain within +0.002 of "
            "packet-only while separated from the real default by >=0.003."
        ),
        "packet_contract": {
            "packet_name": "wyner_ziv_residual_logit_packet",
            "raw_payload_bytes": RAW_PACKET_BYTES,
            "framed_record_bytes": FRAMED_PACKET_BYTES,
            "candidate_id_bits": int(math.ceil(math.log2(CANDIDATE_COUNT))),
            "score_residual_quantizer_bins": int(default_config["quantizer_bins"]),
            "score_residual_bits_per_candidate": int(math.ceil(math.log2(default_config["quantizer_bins"]))),
            "packet_bits_per_request": int(default_config["packet_bits_per_request"]),
            "fields": [
                "TinyLlama selected candidate id",
                "four quantized TinyLlama centered score residual bins",
            ],
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "quantized_score_residuals_transmitted": True,
            "decoder_uses_qwen_side_information": True,
        },
        "default_decoder_audit": {
            "feature_view": str(default_config["feature_view"]),
            "feature_names": _candidate_feature_names(str(default_config["feature_view"])),
            "feature_dim": int(default_coef.shape[0]),
            "ridge": float(default_config["ridge"]),
            "coefficients": [float(item) for item in default_coef],
        },
        "headline": headline,
        "frontier_rows": frontier_rows,
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "quantizer": _summarize_quantizer(default_quantizer),
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": FRAMED_PACKET_BYTES,
            "packet_bits_per_request": int(default_config["packet_bits_per_request"]),
            "logical_validation_raw_payload_bytes_total": int(len(eval_answers) * RAW_PACKET_BYTES),
            "logical_validation_framed_record_bytes_total": int(len(eval_answers) * FRAMED_PACKET_BYTES),
            "communication_object": "task_level_source_private_quantized_residual_logit_packet",
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
            "train_path": _display_path(train_path),
            "train_sha256": _sha256_file(train_path),
            "tiny_train_cache_dir": _display_path(tiny_train_cache_dir),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "tiny_eval_packet_jsonl": _display_path(tiny_eval_packet_jsonl),
            "tiny_eval_packet_jsonl_sha256": _sha256_file(tiny_eval_packet_jsonl),
            "tiny_eval_artifact": _display_path(tiny_eval_artifact),
            "tiny_eval_artifact_sha256": _sha256_file(tiny_eval_artifact),
            "tiny_eval_rows": _display_path(tiny_eval_rows),
            "tiny_eval_rows_sha256": _sha256_file(tiny_eval_rows),
            "tiny_eval_score_cache": _display_path(tiny_eval_score_cache),
            "tiny_eval_score_cache_sha256": _sha256_file(tiny_eval_score_cache),
            "qwen_eval_packet_jsonl": _display_path(qwen_eval_packet_jsonl),
            "qwen_eval_packet_jsonl_sha256": _sha256_file(qwen_eval_packet_jsonl),
            "qwen_global_artifact": _display_path(qwen_global_artifact),
            "qwen_global_artifact_sha256": _sha256_file(qwen_global_artifact),
        },
        "audit": surfaces["audit"],
        "interpretation": (
            "This gate tests the strongest Mac-feasible conditional-coding branch after receiver selectors "
            "saturated. Instead of sending only the TinyLlama candidate id, the packet also carries a tiny "
            "quantized TinyLlama score-residual sketch that the decoder combines with Qwen side information. "
            "A pass would promote a Wyner-Ziv-style conditional packet; a fail means score-residual source "
            "coding does not close the Tiny/Qwen receiver headroom on this calibration surface."
        ),
    }
    json_path = output_dir / "hellaswag_wyner_ziv_residual_packet_gate.json"
    md_path = output_dir / "hellaswag_wyner_ziv_residual_packet_gate.md"
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
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--tiny-train-cache-dir", type=pathlib.Path, default=DEFAULT_TINY_TRAIN_CACHE_DIR)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--tiny-eval-packet-jsonl", type=pathlib.Path, default=DEFAULT_TINY_EVAL_PACKET_JSONL)
    parser.add_argument("--tiny-eval-artifact", type=pathlib.Path, default=DEFAULT_TINY_EVAL_ARTIFACT)
    parser.add_argument("--tiny-eval-rows", type=pathlib.Path, default=DEFAULT_TINY_EVAL_ROWS)
    parser.add_argument("--tiny-eval-score-cache", type=pathlib.Path, default=DEFAULT_TINY_EVAL_SCORE_CACHE)
    parser.add_argument("--qwen-eval-packet-jsonl", type=pathlib.Path, default=DEFAULT_QWEN_EVAL_PACKET_JSONL)
    parser.add_argument("--qwen-global-artifact", type=pathlib.Path, default=DEFAULT_QWEN_GLOBAL_ARTIFACT)
    parser.add_argument("--sample-seeds", type=_parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--decoder-ridges", type=_parse_float_tuple, default=DEFAULT_DECODER_RIDGES)
    parser.add_argument("--quantizer-bins", type=_parse_int_tuple, default=DEFAULT_QUANTIZER_BINS)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=15017)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        tiny_train_cache_dir=args.tiny_train_cache_dir,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        tiny_eval_packet_jsonl=args.tiny_eval_packet_jsonl,
        tiny_eval_artifact=args.tiny_eval_artifact,
        tiny_eval_rows=args.tiny_eval_rows,
        tiny_eval_score_cache=args.tiny_eval_score_cache,
        qwen_eval_packet_jsonl=args.qwen_eval_packet_jsonl,
        qwen_global_artifact=args.qwen_global_artifact,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        decoder_ridges=args.decoder_ridges,
        quantizer_bins=args.quantizer_bins,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
