from __future__ import annotations

"""Qwen-to-Phi conditional innovation codec gate.

This gate asks whether Qwen needs to send the whole candidate frontier, or only
the part Phi cannot predict from its own score simplex.  A receiver-side
"ghost" model first predicts Qwen's four-candidate score geometry from Phi's
local scores.  The source packet then transmits a tiny quantized residual
innovation code.  The held-out receiver must beat fixed Qwen hybrid and
ghost/destructive controls to pass.
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as nonqwen  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_conditional_innovation_codec_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_PHI_TRAIN_SCORE_CACHE = (
    receiver_gate.DEFAULT_OUTPUT / "phi_official_train_score_cache.json"
)
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_RATE_BYTES = (1, 2, 3, 4)
RIDGE_LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
GHOST_RIDGES = (0.001, 0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _answers(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return denoise._paired_ci(
        selected=selected,
        baseline=baseline,
        answers=answers,
        seed=seed,
        samples=samples,
    )


def _z_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.where(scale > 1e-8, scale, 1.0)


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _phi_basis(phi_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(phi_scores, dtype=np.float64)
    z = _z_scores(scores)
    probs = _softmax_rows(scores)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(scores.shape[1])
    margin = scores[np.arange(scores.shape[0]), order[:, 0]] - scores[np.arange(scores.shape[0]), order[:, 1]]
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    parts = [
        np.ones((scores.shape[0], 1), dtype=np.float64),
        z,
        probs,
        ranks.astype(np.float64) / 3.0,
        margin[:, None],
        entropy[:, None],
    ]
    return np.concatenate(parts, axis=1)


def _fit_linear_map(x: np.ndarray, y: np.ndarray, *, ridges: tuple[float, ...]) -> dict[str, Any]:
    best: tuple[float, dict[str, Any]] | None = None
    for ridge in ridges:
        penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        try:
            weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)
        pred = x @ weights
        mse = float(np.mean((pred - y) ** 2))
        item = {"ridge": float(ridge), "weights": weights.tolist(), "fit_mse": mse}
        if best is None or mse < best[0]:
            best = (mse, item)
    if best is None:
        raise ValueError("no ghost map fit")
    return best[1]


def _predict_ghost(phi_scores: np.ndarray, ghost: dict[str, Any]) -> np.ndarray:
    return _phi_basis(phi_scores) @ np.asarray(ghost["weights"], dtype=np.float64)


def _quantile_bins(values: np.ndarray, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return []
    return sorted(set(float(value) for value in np.quantile(flat, quantiles)))


def _digitize(value: float, bins: list[float]) -> int:
    return int(np.digitize([float(value)], np.asarray(bins, dtype=np.float64))[0])


def _source_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["qwen_source_scores"] for row in rows], dtype=np.float64)


def _phi_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["phi_target_scores"] for row in rows], dtype=np.float64)


def _top2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-values, axis=1)
    return order[:, 0].astype(np.int64), order[:, 1].astype(np.int64)


def _fit_codec_state(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    qwen_z = _z_scores(qwen_scores)
    ghost = _fit_linear_map(_phi_basis(phi_scores[fit_indices]), qwen_z[fit_indices], ridges=GHOST_RIDGES)
    residual = qwen_z - _predict_ghost(phi_scores, ghost)
    q_top1, q_top2 = _top2(qwen_scores)
    q_margin = qwen_scores[np.arange(qwen_scores.shape[0]), q_top1] - qwen_scores[np.arange(qwen_scores.shape[0]), q_top2]
    return {
        "ghost": ghost,
        "residual_abs_bins": _quantile_bins(np.abs(residual[fit_indices])),
        "q_margin_bins": _quantile_bins(q_margin[fit_indices]),
    }


def _packet_bytes(rate_bytes: int) -> tuple[int, int]:
    return int(rate_bytes), int(rate_bytes) + 3


def _encode_innovation_codes(
    *,
    rows: list[dict[str, Any]],
    codec: dict[str, Any],
    condition: str = "matched",
    seed: int = 0,
) -> dict[int, np.ndarray]:
    qwen_scores = _source_scores(rows)
    phi_scores = _phi_scores(rows)
    qwen_z = _z_scores(qwen_scores)
    residual = qwen_z - _predict_ghost(phi_scores, codec["ghost"])
    q_top1, q_top2 = _top2(qwen_scores)
    top_abs_order = np.argsort(-np.abs(residual), axis=1)
    n = len(rows)
    base: dict[int, np.ndarray] = {}
    for rate_bytes in DEFAULT_RATE_BYTES:
        codes: list[int] = []
        for row_index in range(n):
            top = int(top_abs_order[row_index, 0])
            second = int(top_abs_order[row_index, 1])
            sign = int(residual[row_index, top] >= 0.0)
            sign2 = int(residual[row_index, second] >= 0.0)
            mag = _digitize(abs(float(residual[row_index, top])), codec["residual_abs_bins"])
            mag2 = _digitize(abs(float(residual[row_index, second])), codec["residual_abs_bins"])
            q_margin = float(qwen_scores[row_index, q_top1[row_index]] - qwen_scores[row_index, q_top2[row_index]])
            margin_bin = _digitize(q_margin, codec["q_margin_bins"])
            hybrid = int(rows[row_index]["qwen_hybrid_prediction"])
            if rate_bytes == 1:
                code = top | (sign << 2) | ((mag & 3) << 3) | ((hybrid & 3) << 5)
            elif rate_bytes == 2:
                code = (
                    top
                    | (sign << 2)
                    | ((mag & 3) << 3)
                    | ((hybrid & 3) << 5)
                    | ((second & 3) << 7)
                    | (sign2 << 9)
                    | ((mag2 & 3) << 10)
                    | ((int(q_top1[row_index]) & 3) << 12)
                    | ((margin_bin & 3) << 14)
                )
            elif rate_bytes == 3:
                quant = [
                    (_digitize(abs(float(residual[row_index, candidate])), codec["residual_abs_bins"]) & 3)
                    | (int(residual[row_index, candidate] >= 0.0) << 2)
                    for candidate in range(4)
                ]
                code = (
                    quant[0]
                    | (quant[1] << 3)
                    | (quant[2] << 6)
                    | (quant[3] << 9)
                    | ((hybrid & 3) << 12)
                    | ((int(q_top1[row_index]) & 3) << 14)
                    | ((margin_bin & 3) << 16)
                )
            else:
                quant4 = [
                    min(7, _digitize(abs(float(residual[row_index, candidate])), codec["residual_abs_bins"]) * 2)
                    | (int(residual[row_index, candidate] >= 0.0) << 3)
                    for candidate in range(4)
                ]
                code = (
                    quant4[0]
                    | (quant4[1] << 4)
                    | (quant4[2] << 8)
                    | (quant4[3] << 12)
                    | ((hybrid & 3) << 16)
                    | ((int(q_top1[row_index]) & 3) << 18)
                    | ((int(q_top2[row_index]) & 3) << 20)
                    | ((margin_bin & 3) << 22)
                )
            codes.append(int(code))
        base[int(rate_bytes)] = np.asarray(codes, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if condition == "matched":
        return base
    if condition == "source_row_shuffle":
        order = rng.permutation(n)
        return {rate: codes[order] for rate, codes in base.items()}
    if condition == "random_same_byte":
        return {rate: rng.choice(codes, size=n, replace=True) for rate, codes in base.items()}
    if condition == "code_value_permutation":
        out = {}
        for rate, codes in base.items():
            unique = np.unique(codes)
            shuffled = unique.copy()
            rng.shuffle(shuffled)
            mapping = {int(src): int(dst) for src, dst in zip(unique, shuffled, strict=True)}
            out[rate] = np.asarray([mapping[int(code)] for code in codes], dtype=np.int64)
        return out
    if condition == "candidate_roll_code":
        rolled = [dict(row) for row in rows]
        for row in rolled:
            row["qwen_source_scores"] = list(np.roll(np.asarray(row["qwen_source_scores"], dtype=np.float64), 1))
            for field in ("qwen_hybrid_prediction", "selected_prediction", "hidden_mean_prediction", "score_mean_prediction"):
                if field in row:
                    row[field] = (int(row[field]) + 1) % 4
        return _encode_innovation_codes(rows=rolled, codec=codec, condition="matched", seed=seed)
    if condition == "target_derived":
        target_rows = [dict(row) for row in rows]
        for row in target_rows:
            row["qwen_source_scores"] = list(row["phi_target_scores"])
            row["qwen_hybrid_prediction"] = int(row["phi_target_prediction"])
        return _encode_innovation_codes(rows=target_rows, codec=codec, condition="matched", seed=seed)
    raise ValueError(f"unknown condition: {condition}")


def _decode_candidate_packet_features(code: int, rate_bytes: int, candidate: int) -> list[float]:
    code = int(code)
    candidate = int(candidate)
    if rate_bytes == 1:
        top = code & 3
        sign = (code >> 2) & 1
        mag = (code >> 3) & 3
        hybrid = (code >> 5) & 3
        return [
            float(candidate == top),
            float(candidate == hybrid),
            float(sign * 2 - 1) * float(mag + 1) * float(candidate == top),
        ]
    if rate_bytes == 2:
        top = code & 3
        sign = (code >> 2) & 1
        mag = (code >> 3) & 3
        hybrid = (code >> 5) & 3
        second = (code >> 7) & 3
        sign2 = (code >> 9) & 1
        mag2 = (code >> 10) & 3
        qtop = (code >> 12) & 3
        margin = (code >> 14) & 3
        return [
            float(candidate == top),
            float(candidate == second),
            float(candidate == hybrid),
            float(candidate == qtop),
            float(sign * 2 - 1) * float(mag + 1) * float(candidate == top),
            float(sign2 * 2 - 1) * float(mag2 + 1) * float(candidate == second),
            float(margin),
        ]
    if rate_bytes == 3:
        quant = [(code >> (3 * candidate_id)) & 7 for candidate_id in range(4)]
        hybrid = (code >> 12) & 3
        qtop = (code >> 14) & 3
        margin = (code >> 16) & 3
        value = quant[candidate]
        sign = (value >> 2) & 1
        mag = value & 3
        return [
            float(candidate == hybrid),
            float(candidate == qtop),
            float(sign * 2 - 1) * float(mag + 1),
            float(margin),
        ]
    quant = [(code >> (4 * candidate_id)) & 15 for candidate_id in range(4)]
    hybrid = (code >> 16) & 3
    qtop = (code >> 18) & 3
    qtop2 = (code >> 20) & 3
    margin = (code >> 22) & 3
    value = quant[candidate]
    sign = (value >> 3) & 1
    mag = value & 7
    return [
        float(candidate == hybrid),
        float(candidate == qtop),
        float(candidate == qtop2),
        float(sign * 2 - 1) * float(mag + 1),
        float(margin),
    ]


def _candidate_features(
    row: dict[str, Any],
    *,
    candidate: int,
    code: int,
    rate_bytes: int,
    codec: dict[str, Any],
) -> np.ndarray:
    phi_scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    z = _z_scores(phi_scores.reshape(1, -1))[0]
    probs = _softmax_rows(phi_scores.reshape(1, -1))[0]
    order = np.argsort(-phi_scores)
    ranks = np.empty(4, dtype=np.int64)
    ranks[order] = np.arange(4)
    ghost = _predict_ghost(phi_scores.reshape(1, -1), codec["ghost"])[0]
    packet = _decode_candidate_packet_features(code, rate_bytes, candidate)
    features: list[float] = [
        1.0,
        z[candidate],
        probs[candidate],
        phi_scores[candidate],
        float(ranks[candidate]),
        float(candidate == int(row["phi_target_prediction"])),
        float(candidate == int(row["qwen_hybrid_prediction"])),
        float(candidate == int(row["selected_prediction"])),
        float(phi_scores[order[0]] - phi_scores[order[1]]),
        float(probs[order[0]]),
        float(-np.sum(probs * np.log(probs + 1e-12))),
        ghost[candidate],
    ]
    features.extend(packet)
    if rate_bytes > 0:
        hashed = (int(code) * 17 + int(candidate) * 31) % 64
        features.extend(float(index == hashed) for index in range(64))
    return np.asarray(features, dtype=np.float64)


def _row_features(
    rows: list[dict[str, Any]],
    *,
    rate_bytes: int,
    codec: dict[str, Any],
    codes: np.ndarray,
) -> np.ndarray:
    features = []
    for row_index, row in enumerate(rows):
        for candidate in range(4):
            features.append(
                _candidate_features(
                    row,
                    candidate=candidate,
                    code=int(codes[row_index]),
                    rate_bytes=rate_bytes,
                    codec=codec,
                )
            )
    width = max(item.shape[0] for item in features)
    return np.vstack([np.pad(item, (0, width - item.shape[0])) for item in features])


def _fit_decoder(
    rows: list[dict[str, Any]],
    *,
    rate_bytes: int,
    codec: dict[str, Any],
    codes: np.ndarray,
    l2: float,
    label_shift: int = 0,
) -> dict[str, Any]:
    x = _row_features(rows, rate_bytes=rate_bytes, codec=codec, codes=codes)
    answers = (np.repeat(_answers(rows), 4) + int(label_shift)) % 4
    candidate_ids = np.tile(np.arange(4, dtype=np.int64), len(rows))
    y = (candidate_ids == answers).astype(np.float64)
    penalty = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    try:
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)
    return {
        "rate_bytes": int(rate_bytes),
        "l2": float(l2),
        "weights": weights.tolist(),
        "feature_dim": int(x.shape[1]),
        "label_shift": int(label_shift),
    }


def _predict_decoder(
    rows: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    codec: dict[str, Any],
    codes: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    rate_bytes = int(model["rate_bytes"])
    out: list[int] = []
    for row_index, row in enumerate(rows):
        scores = []
        for candidate in range(4):
            features = _candidate_features(
                row,
                candidate=candidate,
                code=int(codes[row_index]),
                rate_bytes=rate_bytes,
                codec=codec,
            )
            if features.shape[0] < weights.shape[0]:
                features = np.pad(features, (0, weights.shape[0] - features.shape[0]))
            scores.append(float(features[: weights.shape[0]] @ weights))
        out.append(int(np.argmax(scores)))
    return np.asarray(out, dtype=np.int64)


def _select_decoder(
    *,
    fit_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    codec: dict[str, Any],
    fit_codes_by_rate: dict[int, np.ndarray],
    dev_codes_by_rate: dict[int, np.ndarray],
    rate_bytes: tuple[int, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answers = _answers(dev_rows)
    fixed_hybrid = _field_array(dev_rows, "qwen_hybrid_prediction")
    config_rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, str], dict[str, Any]] | None = None
    for rate in rate_bytes:
        for l2 in RIDGE_LAMBDAS:
            model = _fit_decoder(
                fit_rows,
                rate_bytes=int(rate),
                codec=codec,
                codes=fit_codes_by_rate[int(rate)],
                l2=float(l2),
            )
            predictions = _predict_decoder(dev_rows, model=model, codec=codec, codes=dev_codes_by_rate[int(rate)])
            paired = _paired_ci(
                selected=predictions,
                baseline=fixed_hybrid,
                answers=answers,
                seed=82760504 + int(rate) * 1000 + int(float(l2) * 1000),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            row = {
                "rate_bytes": int(rate),
                "l2": float(l2),
                "official_dev_accuracy": _accuracy(predictions, answers),
                "official_dev_delta_vs_hybrid": paired["delta"],
                "official_dev_ci95_low_vs_hybrid": paired["ci95_low"],
                "official_dev_helps_vs_hybrid": paired["helps"],
                "official_dev_harms_vs_hybrid": paired["harms"],
                "official_dev_override_count": int(np.sum(predictions != fixed_hybrid)),
            }
            config_rows.append(row)
            key = (
                float(row["official_dev_accuracy"]),
                float(row["official_dev_delta_vs_hybrid"]),
                float(row["official_dev_ci95_low_vs_hybrid"]),
                float(-int(rate)),
                json.dumps(row, sort_keys=True),
            )
            if best is None or key > best[0]:
                best = (key, model)
    if best is None:
        raise ValueError("no conditional innovation decoder selected")
    return best[1], sorted(config_rows, key=lambda item: item["official_dev_accuracy"], reverse=True)


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    ghost_only: np.ndarray,
    bootstrap_samples: int,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = _answers(rows)
    vs_hybrid = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=92760504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=92760604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=92760704 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_ghost = _paired_ci(
        selected=predictions,
        baseline=ghost_only,
        answers=answers,
        seed=92760804 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "target_only_accuracy": _accuracy(target_only, answers),
        "ghost_only_accuracy": _accuracy(ghost_only, answers),
        "delta_vs_fixed_hybrid": vs_hybrid["delta"],
        "ci95_low_vs_fixed_hybrid": vs_hybrid["ci95_low"],
        "ci95_high_vs_fixed_hybrid": vs_hybrid["ci95_high"],
        "helps_vs_fixed_hybrid": vs_hybrid["helps"],
        "harms_vs_fixed_hybrid": vs_hybrid["harms"],
        "delta_vs_candidate_only": vs_candidate["delta"],
        "ci95_low_vs_candidate_only": vs_candidate["ci95_low"],
        "delta_vs_target_only": vs_target["delta"],
        "ci95_low_vs_target_only": vs_target["ci95_low"],
        "delta_vs_ghost_only": vs_ghost["delta"],
        "ci95_low_vs_ghost_only": vs_ghost["ci95_low"],
        "override_count_vs_fixed_hybrid": int(np.sum(predictions != fixed_hybrid)),
        "override_rate_vs_fixed_hybrid": float(np.mean(predictions != fixed_hybrid)),
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_private": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "source_score_or_logit_vector_exposed": False,
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _slice_rows(
    *,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    answers = _answers(rows)
    starts = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    out: list[dict[str, Any]] = []
    for start in sorted(set(starts.tolist())):
        mask = starts == start
        paired = _paired_ci(
            selected=predictions[mask],
            baseline=fixed_hybrid[mask],
            answers=answers[mask],
            seed=93760504 + int(start),
            samples=max(200, min(bootstrap_samples, 1000)),
        )
        out.append(
            {
                "slice_start": int(start),
                "eval_rows": int(np.sum(mask)),
                "method_accuracy": _accuracy(predictions[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]),
                "delta_vs_fixed_hybrid": paired["delta"],
                "ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                "helps_vs_fixed_hybrid": paired["helps"],
                "harms_vs_fixed_hybrid": paired["harms"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Conditional Innovation Codec Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- selected rate bytes: `{h['selected_rate_bytes']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- conditional innovation accuracy: `{h['conditional_innovation_accuracy']:.6f}`",
        f"- delta vs fixed hybrid: `{h['conditional_innovation_delta_vs_fixed_hybrid']:.6f}`",
        f"- CI95 low vs fixed hybrid: `{h['conditional_innovation_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- ghost-only accuracy: `{h['ghost_only_accuracy']:.6f}`",
        f"- best destructive control: `{h['best_destructive_control_name']}` at `{h['best_destructive_control_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
    ]
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    train_path: pathlib.Path | str = DEFAULT_TRAIN_PATH,
    qwen_train_cache_dir: pathlib.Path | str = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    phi_train_score_cache: pathlib.Path | str = DEFAULT_PHI_TRAIN_SCORE_CACHE,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    component_ridges: tuple[float, ...] = DEFAULT_COMPONENT_RIDGES,
    fit_rows_per_slice: int = denoise.FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = denoise.SELECT_ROWS_PER_SLICE,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    max_calibration_rows: int | None = None,
    target_model: pathlib.Path | str = DEFAULT_TARGET_MODEL,
    target_device: str = "mps",
    target_dtype: str = "float16",
    target_max_length: int = 256,
    target_normalization: str = "mean",
    target_prompt_mode: str = "continuation",
    local_files_only: bool = True,
    rate_bytes: tuple[int, ...] = DEFAULT_RATE_BYTES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = source_gate._build_qwen_oob_calibration(
        train_path=train_path,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        component_ridges=component_ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
    )
    calibration = receiver_gate._limit_calibration(calibration, max_calibration_rows)
    fit_indices, dev_indices = official._official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )
    calibration_arc_rows, calibration_row_meta = receiver_gate._arc_rows_for_calibration(
        train_path=train_path,
        calibration_rows=calibration["rows"],
    )
    phi_cache_path = _resolve(phi_train_score_cache)
    phi_cache_existed = phi_cache_path.exists()
    phi_scores, phi_predictions, phi_state, phi_sha = receiver_gate._load_or_build_phi_scores(
        rows=calibration_arc_rows,
        score_cache=phi_cache_path,
        target_model=target_model,
        target_device=target_device,
        target_dtype=target_dtype,
        target_max_length=target_max_length,
        target_normalization=target_normalization,
        target_prompt_mode=target_prompt_mode,
        local_files_only=local_files_only,
    )
    codec = _fit_codec_state(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        fit_indices=fit_indices,
    )
    calibration_rows = [
        {
            "row_id": row["row_id"],
            "answer_index": int(answer),
            "qwen_hybrid_prediction": int(hybrid),
            "selected_prediction": int(mean),
            "phi_target_prediction": int(phi_pred),
            "phi_target_scores": [float(value) for value in phi_row],
            "qwen_source_scores": [float(value) for value in qwen_row],
        }
        for row, answer, hybrid, mean, phi_pred, phi_row, qwen_row in zip(
            calibration["rows"],
            calibration["answers"],
            calibration["hybrid"],
            calibration["mean"],
            phi_predictions,
            phi_scores,
            calibration["scores"],
            strict=True,
        )
    ]
    fit_rows = [calibration_rows[index] for index in fit_indices]
    dev_rows = [calibration_rows[index] for index in dev_indices]
    train_codes = _encode_innovation_codes(rows=calibration_rows, codec=codec, condition="matched", seed=20260504)
    fit_codes = {rate: codes[fit_indices] for rate, codes in train_codes.items()}
    dev_codes = {rate: codes[dev_indices] for rate, codes in train_codes.items()}
    model, config_rows = _select_decoder(
        fit_rows=fit_rows,
        dev_rows=dev_rows,
        codec=codec,
        fit_codes_by_rate=fit_codes,
        dev_codes_by_rate=dev_codes,
        rate_bytes=rate_bytes,
        bootstrap_samples=bootstrap_samples,
    )
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    eval_codes = _encode_innovation_codes(rows=eval_rows, codec=codec, condition="matched", seed=20260504)
    rate = int(model["rate_bytes"])
    raw_bytes, framed_bytes = _packet_bytes(rate)
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    selected_predictions = _predict_decoder(eval_rows, model=model, codec=codec, codes=eval_codes[rate])
    ghost_codes = _encode_innovation_codes(rows=eval_rows, codec=codec, condition="target_derived", seed=20260504)
    ghost_only = _predict_decoder(eval_rows, model=model, codec=codec, codes=ghost_codes[rate])
    label_model = _fit_decoder(
        fit_rows,
        rate_bytes=rate,
        codec=codec,
        codes=fit_codes[rate],
        l2=float(model["l2"]),
        label_shift=1,
    )
    label_permutation = _predict_decoder(eval_rows, model=label_model, codec=codec, codes=eval_codes[rate])
    method_rows = [
        _method_row(
            name="conditional_innovation_packet",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            ghost_only=ghost_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            details={"model": {key: value for key, value in model.items() if key != "weights"}},
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=eval_rows,
            predictions=fixed_hybrid,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            ghost_only=ghost_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="qwen_candidate_only",
            rows=eval_rows,
            predictions=candidate_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            ghost_only=ghost_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="phi_target_only",
            rows=eval_rows,
            predictions=target_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            ghost_only=ghost_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
        ),
        _method_row(
            name="ghost_only_receiver_control",
            rows=eval_rows,
            predictions=ghost_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            ghost_only=ghost_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"condition": "target_derived_no_source_innovation"},
        ),
        _method_row(
            name="label_permutation_decoder_control",
            rows=eval_rows,
            predictions=label_permutation,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            ghost_only=ghost_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            details={"condition": "official_train_label_shift"},
        ),
    ]
    for condition in (
        "source_row_shuffle",
        "random_same_byte",
        "code_value_permutation",
        "candidate_roll_code",
        "target_derived",
    ):
        codes = _encode_innovation_codes(
            rows=eval_rows,
            codec=codec,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )[rate]
        predictions = _predict_decoder(eval_rows, model=model, codec=codec, codes=codes)
        method_rows.append(
            _method_row(
                name=f"{condition}_innovation_control",
                rows=eval_rows,
                predictions=predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                ghost_only=ghost_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=raw_bytes if condition != "target_derived" else 0,
                framed_record_bytes=framed_bytes if condition != "target_derived" else 0,
                details={"condition": condition},
            )
        )
    method_rows = sorted(method_rows, key=lambda item: item["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "conditional_innovation_packet")
    ghost_row = next(row for row in method_rows if row["method"] == "ghost_only_receiver_control")
    target_derived_row = next(row for row in method_rows if row["method"] == "target_derived_innovation_control")
    destructive_rows = [
        row
        for row in method_rows
        if row["method"].endswith("_control")
        and row["method"] not in {"ghost_only_receiver_control", "target_derived_innovation_control"}
    ]
    best_destructive = max(destructive_rows, key=lambda item: item["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    pass_gate = bool(
        method_row["delta_vs_fixed_hybrid"] >= 0.005
        and method_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and method_row["ci95_low_vs_candidate_only"] > 0.0
        and method_row["ci95_low_vs_ghost_only"] > 0.0
        and method_row["accuracy"] > best_destructive["accuracy"]
        and method_row["accuracy"] > target_derived_row["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and method_row["helps_vs_fixed_hybrid"] > method_row["harms_vs_fixed_hybrid"]
    )
    train_content_ids = {row.content_id for row in calibration_arc_rows}
    eval_content_ids = {str(row.get("content_id", row["row_id"])) for row in eval_rows}
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_phi_target_accuracy": _accuracy(phi_predictions, calibration["answers"]),
        "official_train_qwen_hybrid_accuracy": _accuracy(calibration["hybrid"], calibration["answers"]),
        "ghost_fit_mse": float(codec["ghost"]["fit_mse"]),
        "ghost_ridge": float(codec["ghost"]["ridge"]),
        "selected_rate_bytes": int(rate),
        "selected_l2": float(model["l2"]),
        "eval_rows": len(eval_rows),
        "fixed_hybrid_accuracy": next(
            row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement"
        )["accuracy"],
        "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")[
            "accuracy"
        ],
        "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")[
            "accuracy"
        ],
        "ghost_only_accuracy": ghost_row["accuracy"],
        "target_derived_control_accuracy": target_derived_row["accuracy"],
        "conditional_innovation_accuracy": method_row["accuracy"],
        "conditional_innovation_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "conditional_innovation_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "conditional_innovation_delta_vs_candidate_only": method_row["delta_vs_candidate_only"],
        "conditional_innovation_ci95_low_vs_candidate_only": method_row["ci95_low_vs_candidate_only"],
        "conditional_innovation_delta_vs_ghost_only": method_row["delta_vs_ghost_only"],
        "conditional_innovation_ci95_low_vs_ghost_only": method_row["ci95_low_vs_ghost_only"],
        "conditional_innovation_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "conditional_innovation_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "conditional_innovation_overrides": method_row["override_count_vs_fixed_hybrid"],
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "raw_payload_bytes": raw_bytes,
        "framed_record_bytes": framed_bytes,
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_conditional_innovation_codec_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the official-train selected conditional innovation packet beats fixed Qwen hybrid by "
            "at least 0.005 with positive paired CI, beats candidate-only and ghost-only with positive paired CI, "
            "beats target-derived and destructive innovation controls, is nonnegative on both eval slices, and "
            "helps more than it harms."
        ),
        "headline": headline,
        "calibration_row_metadata": calibration_row_meta,
        "packet_contract": {
            "receiver_visible_payload": "byte-scale quantized residual between Qwen source frontier and Phi-predicted ghost frontier",
            "raw_payload_bytes": raw_bytes,
            "framed_record_bytes": framed_bytes,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "phi_scores_used_as_receiver_side_information": True,
            "qwen_scores_used_only_for_source_side_residual_quantization": True,
        },
        "source_score_metadata": source_score_metadata,
        "slice_metadata": metadata,
        "sample_cache_rows": calibration["sample_cache_rows"],
        "component_rows": calibration["component_rows"],
        "config_rows": config_rows,
        "method_rows": method_rows,
        "slice_rows": slice_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": raw_bytes,
            "framed_record_bytes_per_request": framed_bytes,
            "fit_and_eval_wall_time_s": float(time.perf_counter() - started),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
        },
        "inputs": {
            "train_path": official._display_path(train_path),
            "train_sha256": official._sha256_file(train_path),
            "qwen_train_cache_dir": official._display_path(qwen_train_cache_dir),
            "phi_train_score_cache": headroom._display_path(phi_cache_path),
            "phi_train_score_cache_sha256": phi_sha,
            "phi_train_score_model": phi_state,
            "source_score_cache": denoise._display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate tests the conditional innovation hypothesis: if Phi can already predict some of Qwen's "
            "candidate frontier from its local scores, the source should spend packet bits only on the residual. "
            "A pass would promote residual coding as the next cross-family method branch. A failure means this "
            "linear ghost plus discrete residual codec is not sufficient, even though richer residual channels may "
            "still be alive."
        ),
        "lay_explanation": (
            "First we ask Phi to guess what Qwen probably thinks about the four answers. Then Qwen sends only a "
            "tiny correction to that guess. The controls replace that correction with fake or target-only versions "
            "to check whether the real source correction is doing useful work."
        ),
    }
    _write_json(output_dir / "hellaswag_qwen_to_phi_conditional_innovation_codec_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_conditional_innovation_codec_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_conditional_innovation_codec_gate.json",
                "hellaswag_qwen_to_phi_conditional_innovation_codec_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "slice_rows.csv",
            ],
            "headline": headline,
            "inputs": payload["inputs"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--phi-train-score-cache", type=pathlib.Path, default=DEFAULT_PHI_TRAIN_SCORE_CACHE)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--sample-seeds", type=official._parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=official._parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--component-ridges", type=official._parse_float_tuple, default=DEFAULT_COMPONENT_RIDGES)
    parser.add_argument("--fit-rows-per-slice", type=int, default=denoise.FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=denoise.SELECT_ROWS_PER_SLICE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--max-calibration-rows", type=int, default=None)
    parser.add_argument("--target-model", type=pathlib.Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-device", type=str, default="mps")
    parser.add_argument("--target-dtype", type=str, default="float16")
    parser.add_argument("--target-max-length", type=int, default=256)
    parser.add_argument("--target-normalization", type=str, default="mean")
    parser.add_argument("--target-prompt-mode", type=str, default="continuation")
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--rate-bytes", type=_parse_int_tuple, default=DEFAULT_RATE_BYTES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        phi_train_score_cache=args.phi_train_score_cache,
        source_score_cache=args.source_score_cache,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        component_ridges=args.component_ridges,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        max_calibration_rows=args.max_calibration_rows,
        target_model=args.target_model,
        target_device=args.target_device,
        target_dtype=args.target_dtype,
        target_max_length=args.target_max_length,
        target_normalization=args.target_normalization,
        target_prompt_mode=args.target_prompt_mode,
        local_files_only=not args.no_local_files_only,
        rate_bytes=args.rate_bytes,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "selected_rate_bytes": h["selected_rate_bytes"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "conditional_innovation_accuracy": h["conditional_innovation_accuracy"],
                "conditional_innovation_delta_vs_fixed_hybrid": h[
                    "conditional_innovation_delta_vs_fixed_hybrid"
                ],
                "ghost_only_accuracy": h["ghost_only_accuracy"],
                "best_destructive_control_accuracy": h["best_destructive_control_accuracy"],
                "phi_train_score_cache_hit": h["phi_train_score_cache_hit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
