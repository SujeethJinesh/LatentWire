from __future__ import annotations

"""Qwen-to-Phi oracle-gap switch decomposition gate.

This gate asks a narrower question than the ridge denoising receiver: can a
train-only row-level controller learn when Phi's receiver-local answer should
override the source-private Qwen hybrid packet?
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate_20260504_validation1024_2048"
)
DEFAULT_SOURCE_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/"
    "source_eval_score_cache.json"
)
SOURCE_CODE_MODES = (
    "code8_fourier_hybrid_sign",
    "code16_fourier_hybrid_sign_mag",
    "code8_hybrid_selected_margin",
    "code8_hybrid_agreement_margin",
    "code16_policy_margin",
)
RIDGE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
FIT_ROWS_PER_SLICE = denoise.FIT_ROWS_PER_SLICE
SELECT_ROWS_PER_SLICE = denoise.SELECT_ROWS_PER_SLICE
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


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _field_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _answers(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    deltas = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(deltas), size=len(deltas))
        draws.append(float(np.mean(deltas[indices])))
    return {
        "delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "helps": int(np.sum(deltas > 0)),
        "harms": int(np.sum(deltas < 0)),
    }


def _packet_bytes(mode: str) -> tuple[int, int]:
    if mode.startswith("code8"):
        return 1, 4
    if mode.startswith("code16"):
        return 2, 5
    if mode in {"zero_byte_target_only", "target_derived_code"}:
        return 0, 0
    raise ValueError(f"unknown mode: {mode}")


def _load_source_scores(rows: list[dict[str, Any]], source_score_cache: pathlib.Path | str) -> dict[str, Any]:
    cache = denoise._read_json(source_score_cache)
    by_row_id = {
        str(row_id): (int(prediction), [float(value) for value in scores])
        for row_id, prediction, scores in zip(
            cache["row_ids"],
            cache["source_predictions"],
            cache["source_scores"],
            strict=True,
        )
    }
    missing = [str(row["row_id"]) for row in rows if str(row["row_id"]) not in by_row_id]
    if missing:
        raise ValueError(f"missing Qwen source scores for {len(missing)} rows")
    for row in rows:
        prediction, scores = by_row_id[str(row["row_id"])]
        row["qwen_source_score_prediction"] = int(prediction)
        row["qwen_source_scores"] = [float(value) for value in scores]
    return {
        "source_score_cache": denoise._display_path(source_score_cache),
        "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        "source_score_cache_rows": int(cache["row_count"]),
    }


def _contrast_basis() -> np.ndarray:
    return np.asarray(
        [
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
            [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
            [0.0, -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(12.0)],
            [0.0, 0.0, -3.0 / np.sqrt(12.0)],
        ],
        dtype=np.float64,
    )


def _source_score_z(row: dict[str, Any], *, source_transform: str = "matched") -> np.ndarray:
    scores = np.asarray(row["qwen_source_scores"], dtype=np.float64)
    centered = scores - np.mean(scores)
    scale = float(np.std(centered))
    z = centered / (scale if scale > 1e-8 else 1.0)
    if source_transform == "candidate_roll_code":
        z = np.roll(z, 1)
    return z


def _source_fourier_coefficients(row: dict[str, Any], *, source_transform: str = "matched") -> np.ndarray:
    coeff = _source_score_z(row, source_transform=source_transform) @ _contrast_basis()
    if source_transform == "fourier_sign_flip_source":
        coeff = -coeff
    elif source_transform == "fourier_basis_permutation_source":
        coeff = coeff[[1, 2, 0]]
    return coeff.astype(np.float64)


def _bin_value(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if float(value) < float(threshold):
            return int(index)
    return int(len(thresholds))


def _encode_packet_code(
    row: dict[str, Any],
    mode: str,
    *,
    source_transform: str = "matched",
) -> int:
    if mode in {"code8_hybrid_selected_margin", "code8_hybrid_agreement_margin", "code16_policy_margin"}:
        if source_transform == "candidate_roll_code":
            return denoise._encode_source_code(denoise._roll_source_row(row), mode)
        return denoise._encode_source_code(row, mode)
    if mode == "zero_byte_target_only":
        return 0
    if mode == "target_derived_code":
        return denoise._encode_source_code(row, mode)
    hybrid = int(row["qwen_hybrid_prediction"])
    score_pred = int(row["qwen_source_score_prediction"])
    source_z = _source_score_z(row, source_transform=source_transform)
    coeff = _source_fourier_coefficients(row, source_transform=source_transform)
    order = np.argsort(-source_z)
    top_rival = int(order[1] if int(order[0]) == hybrid else order[0])
    sign_bits = sum(int(coeff[index] >= 0.0) << index for index in range(3))
    energy_bin = _bin_value(float(np.linalg.norm(coeff)), (0.75, 1.25, 1.75))
    margin_bin = _bin_value(float(source_z[order[0]] - source_z[order[1]]), (0.5, 1.0, 1.5))
    if mode == "code8_fourier_hybrid_sign":
        return (
            hybrid
            | (sign_bits << 2)
            | (int(score_pred != hybrid) << 5)
            | (int(top_rival & 1) << 6)
            | (int(energy_bin > 1) << 7)
        )
    if mode == "code16_fourier_hybrid_sign_mag":
        return (
            hybrid
            | (score_pred << 2)
            | (top_rival << 4)
            | (sign_bits << 6)
            | (energy_bin << 9)
            | (margin_bin << 11)
            | (int(score_pred == hybrid) << 13)
        )
    raise ValueError(f"unknown mode: {mode}")


def _decode_packet_code(code: int, mode: str) -> dict[str, Any]:
    if mode in {"code8_hybrid_selected_margin", "code8_hybrid_agreement_margin", "code16_policy_margin"}:
        return denoise._decode_code(code, mode)
    if mode == "zero_byte_target_only":
        return {"ids": {}, "margin_bin": None, "flags": []}
    if mode == "target_derived_code":
        return denoise._decode_code(code, mode)
    hybrid = int(code) & 3
    if mode == "code8_fourier_hybrid_sign":
        sign_bits = (int(code) >> 2) & 7
        return {
            "ids": {"hybrid": hybrid},
            "margin_bin": None,
            "flags": [
                (sign_bits >> index) & 1 for index in range(3)
            ]
            + [
                (int(code) >> 5) & 1,
                (int(code) >> 6) & 1,
                (int(code) >> 7) & 1,
            ],
        }
    if mode == "code16_fourier_hybrid_sign_mag":
        sign_bits = (int(code) >> 6) & 7
        return {
            "ids": {
                "hybrid": hybrid,
                "score_pred": (int(code) >> 2) & 3,
                "top_rival": (int(code) >> 4) & 3,
            },
            "margin_bin": (int(code) >> 11) & 3,
            "flags": [(sign_bits >> index) & 1 for index in range(3)]
            + [
                (int(code) >> 9) & 3,
                (int(code) >> 13) & 1,
            ],
        }
    raise ValueError(f"unknown mode: {mode}")


def _decoded_packet_hybrid(code: int, mode: str, row: dict[str, Any]) -> int:
    if mode in {"zero_byte_target_only", "target_derived_code"}:
        return int(row["phi_target_prediction"])
    return int(_decode_packet_code(code, mode)["ids"]["hybrid"])


def _features_for_row(row: dict[str, Any], *, mode: str, code: int) -> np.ndarray:
    scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    order = np.argsort(-scores)
    ranks = np.empty(4, dtype=np.int64)
    ranks[order] = np.arange(4)
    probs = denoise._softmax(scores)
    target = int(row["phi_target_prediction"])
    packet_hybrid = _decoded_packet_hybrid(code, mode, row)
    decoded = _decode_packet_code(code, mode)
    margin_bin = decoded["margin_bin"]
    features: list[float] = [
        1.0,
        float(scores[order[0]] - scores[order[1]]),
        float(probs[order[0]]),
        float(-np.sum(probs * np.log(probs + 1e-12))),
        float(scores[target] - scores[packet_hybrid]),
        float(probs[target] - probs[packet_hybrid]),
        float(ranks[target]),
        float(ranks[packet_hybrid]),
        float(target == packet_hybrid),
    ]
    features.extend(float(target == item) for item in range(4))
    features.extend(float(packet_hybrid == item) for item in range(4))
    for name in ("selected", "hidden_mean", "score_mean", "vote", "score_pred", "top_rival"):
        value = decoded["ids"].get(name)
        if value is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
            features.append(0.0)
        else:
            features.extend(float(int(value) == item) for item in range(4))
            features.append(float(int(value) == target))
    if margin_bin is None:
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
        features.extend(float(int(margin_bin) == item) for item in range(4))
    features.extend(float(item) for item in decoded["flags"])
    if mode not in {"zero_byte_target_only", "target_derived_code"}:
        hashed = (int(code) * 17) % 64
        features.extend(float(item == hashed) for item in range(64))
    return np.asarray(features, dtype=np.float64)


def _row_features(rows: list[dict[str, Any]], *, mode: str, codes: np.ndarray | None = None) -> np.ndarray:
    features = []
    for index, row in enumerate(rows):
        code = int(_encode_packet_code(row, mode) if codes is None else codes[index])
        features.append(_features_for_row(row, mode=mode, code=code))
    return np.vstack(features)


def _packet_hybrid_array(rows: list[dict[str, Any]], *, mode: str, codes: np.ndarray | None = None) -> np.ndarray:
    out = []
    for index, row in enumerate(rows):
        code = int(_encode_packet_code(row, mode) if codes is None else codes[index])
        out.append(_decoded_packet_hybrid(code, mode, row))
    return np.asarray(out, dtype=np.int64)


def _fit_switcher(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    l2: float,
    label_shift: int = 0,
) -> dict[str, Any]:
    x = _row_features(rows, mode=mode)
    answers = (_answers(rows) + int(label_shift)) % 4
    packet_hybrid = _packet_hybrid_array(rows, mode=mode)
    target = _field_array(rows, "phi_target_prediction")
    y = (target == answers).astype(np.float64) - (packet_hybrid == answers).astype(np.float64)
    penalty = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    return {
        "mode": mode,
        "l2": float(l2),
        "threshold": None,
        "weights": weights.tolist(),
        "label_shift": int(label_shift),
    }


def _predict_with_model(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    *,
    codes: np.ndarray | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(model["mode"])
    weights = np.asarray(model["weights"], dtype=np.float64)
    scores = _row_features(rows, mode=mode, codes=codes) @ weights
    packet_hybrid = _packet_hybrid_array(rows, mode=mode, codes=codes)
    target = _field_array(rows, "phi_target_prediction")
    cutoff = float(model["threshold"] if threshold is None else threshold)
    predictions = packet_hybrid.copy()
    switch_mask = (target != packet_hybrid) & (scores > cutoff)
    predictions[switch_mask] = target[switch_mask]
    return predictions.astype(np.int64), scores


def _threshold_candidates(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[float]:
    mode = str(model["mode"])
    _, scores = _predict_with_model(rows, model, threshold=float("inf"))
    packet_hybrid = _packet_hybrid_array(rows, mode=mode)
    target = _field_array(rows, "phi_target_prediction")
    values = sorted(set(float(item) for item in scores[packet_hybrid != target]))
    values.append(float("inf"))
    if len(values) > 50:
        finite = values[:-1]
        reduced = sorted({finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 41)})
        values = reduced + [float("inf")]
    return values


def _select_switcher(
    *,
    fit_rows: list[dict[str, Any]],
    select_rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    l2_values: tuple[float, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    answers = _answers(select_rows)
    fixed_hybrid = _field_array(select_rows, "qwen_hybrid_prediction")
    config_rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, str], dict[str, Any]] | None = None
    best_forced: tuple[tuple[float, float, float, float, str], dict[str, Any]] | None = None
    for mode in modes:
        raw_bytes, framed_bytes = _packet_bytes(mode)
        for l2 in l2_values:
            base_model = _fit_switcher(fit_rows, mode=mode, l2=float(l2))
            for threshold in _threshold_candidates(select_rows, base_model):
                model = dict(base_model, threshold=float(threshold))
                predictions, scores = _predict_with_model(select_rows, model)
                paired = _paired_ci(
                    selected=predictions,
                    baseline=fixed_hybrid,
                    answers=answers,
                    seed=20260504 + int(float(l2) * 1000) + sum(ord(ch) for ch in mode),
                    samples=max(200, min(bootstrap_samples, 1000)),
                )
                overrides = int(np.sum(predictions != fixed_hybrid))
                row = {
                    "mode": mode,
                    "l2": float(l2),
                    "threshold": float(threshold),
                    "threshold_is_noop": bool(np.isinf(threshold)),
                    "raw_payload_bytes": raw_bytes,
                    "framed_record_bytes": framed_bytes,
                    "select_accuracy": _accuracy(predictions, answers),
                    "select_delta_vs_fixed_hybrid": paired["delta"],
                    "select_ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                    "select_helps_vs_fixed_hybrid": paired["helps"],
                    "select_harms_vs_fixed_hybrid": paired["harms"],
                    "select_override_count": overrides,
                    "select_switch_score_mean": float(np.mean(scores)),
                }
                config_rows.append(row)
                key = (
                    float(row["select_accuracy"]),
                    float(row["select_delta_vs_fixed_hybrid"]),
                    float(row["select_ci95_low_vs_fixed_hybrid"]),
                    float(-overrides),
                    f"{mode}:{l2}:{threshold}",
                )
                candidate = (key, model)
                if best is None or key > best[0]:
                    best = candidate
                if overrides > 0 and (best_forced is None or key > best_forced[0]):
                    best_forced = candidate
    if best is None:
        raise ValueError("no switcher configs")
    return best[1], (best_forced[1] if best_forced else None), sorted(
        config_rows,
        key=lambda item: (
            item["select_accuracy"],
            item["select_delta_vs_fixed_hybrid"],
            item["select_ci95_low_vs_fixed_hybrid"],
        ),
        reverse=True,
    )


def _best_eval_diagnostic(
    *,
    fit_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    l2_values: tuple[float, ...],
) -> dict[str, Any]:
    answers = _answers(eval_rows)
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    best: tuple[tuple[float, float, float, str], dict[str, Any], np.ndarray] | None = None
    for mode in modes:
        for l2 in l2_values:
            base_model = _fit_switcher(fit_rows, mode=mode, l2=float(l2))
            for threshold in _threshold_candidates(eval_rows, base_model):
                if np.isinf(threshold):
                    continue
                model = dict(base_model, threshold=float(threshold))
                predictions, _ = _predict_with_model(eval_rows, model)
                overrides = int(np.sum(predictions != fixed_hybrid))
                if overrides == 0:
                    continue
                key = (
                    _accuracy(predictions, answers),
                    float(np.mean((predictions == answers).astype(float) - (fixed_hybrid == answers).astype(float))),
                    float(-overrides),
                    f"{mode}:{l2}:{threshold}",
                )
                if best is None or key > best[0]:
                    best = (key, model, predictions)
    if best is None:
        raise ValueError("no eval diagnostic configs")
    return {"model": best[1], "predictions": best[2]}


def _codes_for_condition(rows: list[dict[str, Any]], *, mode: str, condition: str, seed: int) -> np.ndarray:
    if condition == "label_permutation_decoder":
        raise ValueError("label_permutation_decoder is a training condition, not a code condition")
    base = np.asarray([_encode_packet_code(row, mode) for row in rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if condition == "matched":
        return base
    if condition == "source_row_shuffle":
        return base[rng.permutation(len(base))]
    if condition == "random_same_byte":
        return rng.choice(base, size=len(base), replace=True)
    if condition == "code_value_permutation":
        unique = np.unique(base)
        permuted = unique.copy()
        rng.shuffle(permuted)
        mapping = {int(src): int(dst) for src, dst in zip(unique, permuted, strict=True)}
        return np.asarray([mapping[int(item)] for item in base], dtype=np.int64)
    if condition in {"candidate_roll_code", "fourier_sign_flip_source", "fourier_basis_permutation_source"}:
        return np.asarray(
            [_encode_packet_code(row, mode, source_transform=condition) for row in rows],
            dtype=np.int64,
        )
    if condition == "source_score_row_shuffle_before_encoding":
        score_rows = [dict(row) for row in rows]
        order = rng.permutation(len(rows))
        shuffled_scores = [rows[index]["qwen_source_scores"] for index in order]
        shuffled_preds = [rows[index]["qwen_source_score_prediction"] for index in order]
        for row, scores, prediction in zip(score_rows, shuffled_scores, shuffled_preds, strict=True):
            row["qwen_source_scores"] = scores
            row["qwen_source_score_prediction"] = prediction
        return np.asarray([_encode_packet_code(row, mode) for row in score_rows], dtype=np.int64)
    raise ValueError(f"unknown condition: {condition}")


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
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
        seed=30360504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=30360604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=30360704 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "target_only_accuracy": _accuracy(target_only, answers),
        "delta_vs_fixed_hybrid": vs_hybrid["delta"],
        "ci95_low_vs_fixed_hybrid": vs_hybrid["ci95_low"],
        "ci95_high_vs_fixed_hybrid": vs_hybrid["ci95_high"],
        "helps_vs_fixed_hybrid": vs_hybrid["helps"],
        "harms_vs_fixed_hybrid": vs_hybrid["harms"],
        "delta_vs_candidate_only": vs_candidate["delta"],
        "ci95_low_vs_candidate_only": vs_candidate["ci95_low"],
        "delta_vs_target_only": vs_target["delta"],
        "ci95_low_vs_target_only": vs_target["ci95_low"],
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


def _oracle_bucket_rows(rows: list[dict[str, Any]], split_name: str) -> dict[str, int | str]:
    answers = _answers(rows)
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    target = _field_array(rows, "phi_target_prediction")
    return {
        "split": split_name,
        "rows": len(rows),
        "both_correct": int(np.sum((hybrid == answers) & (target == answers))),
        "hybrid_only_correct": int(np.sum((hybrid == answers) & (target != answers))),
        "target_only_correct": int(np.sum((hybrid != answers) & (target == answers))),
        "both_wrong": int(np.sum((hybrid != answers) & (target != answers))),
        "hybrid_target_disagree": int(np.sum(hybrid != target)),
    }


def _qwen_score_topk_oracle(rows: list[dict[str, Any]], *, k: int) -> np.ndarray:
    answers = _answers(rows)
    out = []
    for row_index, row in enumerate(rows):
        scores = np.asarray(row["qwen_source_scores"], dtype=np.float64)
        topk = np.argsort(-scores)[: int(k)]
        out.append(int(answers[row_index]) if int(answers[row_index]) in topk else int(topk[0]))
    return np.asarray(out, dtype=np.int64)


def _target_hybrid_qwen_top2_oracle(rows: list[dict[str, Any]]) -> np.ndarray:
    answers = _answers(rows)
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    target = _field_array(rows, "phi_target_prediction")
    qwen_top2 = _qwen_score_topk_oracle(rows, k=2)
    out = []
    for index, answer in enumerate(answers):
        if int(hybrid[index]) == int(answer):
            out.append(int(hybrid[index]))
        elif int(target[index]) == int(answer):
            out.append(int(target[index]))
        elif int(qwen_top2[index]) == int(answer):
            out.append(int(answer))
        else:
            out.append(int(qwen_top2[index]))
    return np.asarray(out, dtype=np.int64)


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
            seed=40360504 + int(start),
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
        "# HellaSwag Qwen-To-Phi Oracle Switch Decomposition Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- target-or-hybrid oracle accuracy: `{h['target_or_hybrid_oracle_accuracy']:.6f}`",
        f"- Qwen score top-2 oracle accuracy: `{h['qwen_score_top2_oracle_accuracy']:.6f}`",
        f"- target+hybrid+Qwen-top2 oracle accuracy: `{h['target_hybrid_qwen_top2_oracle_accuracy']:.6f}`",
        f"- selected switcher accuracy: `{h['selected_switcher_accuracy']:.6f}`",
        f"- selected switcher delta: `{h['selected_switcher_delta_vs_fixed_hybrid']:.6f}`",
        f"- selected switcher CI95 low: `{h['selected_switcher_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- forced switcher accuracy: `{h['forced_switcher_accuracy']:.6f}`",
        f"- eval-label diagnostic best accuracy: `{h['eval_label_diagnostic_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
    ]
    _resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    fit_rows_per_slice: int = FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = SELECT_ROWS_PER_SLICE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = _load_source_scores(rows, source_score_cache)
    fit_rows = [row for row in rows if row["_split"] == "fit"]
    select_rows = [row for row in rows if row["_split"] == "select"]
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    selected_model, forced_model, config_rows = _select_switcher(
        fit_rows=fit_rows,
        select_rows=select_rows,
        modes=SOURCE_CODE_MODES,
        l2_values=RIDGE_LAMBDAS,
        bootstrap_samples=bootstrap_samples,
    )
    if forced_model is None:
        forced_model = selected_model
    eval_diagnostic = _best_eval_diagnostic(
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        modes=SOURCE_CODE_MODES,
        l2_values=RIDGE_LAMBDAS,
    )
    label_control_model = _fit_switcher(
        fit_rows,
        mode=str(forced_model["mode"]),
        l2=float(forced_model["l2"]),
        label_shift=1,
    )
    label_control_model["threshold"] = forced_model["threshold"]

    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    oracle = np.where(fixed_hybrid == answers, fixed_hybrid, target_only).astype(np.int64)
    qwen_top2_oracle = _qwen_score_topk_oracle(eval_rows, k=2)
    target_hybrid_qwen_top2_oracle = _target_hybrid_qwen_top2_oracle(eval_rows)
    selected_predictions, _ = _predict_with_model(eval_rows, selected_model)
    forced_predictions, _ = _predict_with_model(eval_rows, forced_model)
    label_control_predictions, _ = _predict_with_model(eval_rows, label_control_model)
    selected_raw, selected_framed = _packet_bytes(str(selected_model["mode"]))
    forced_raw, forced_framed = _packet_bytes(str(forced_model["mode"]))
    method_rows = [
        _method_row(
            name="selected_train_select_switcher",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=selected_raw,
            framed_record_bytes=selected_framed,
            details={
                "model": {key: value for key, value in selected_model.items() if key != "weights"},
                "eval_label_selected": False,
                "allows_noop": True,
            },
        ),
        _method_row(
            name="forced_nonzero_train_select_switcher",
            rows=eval_rows,
            predictions=forced_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=forced_raw,
            framed_record_bytes=forced_framed,
            details={
                "model": {key: value for key, value in forced_model.items() if key != "weights"},
                "eval_label_selected": False,
                "forced_nonzero": True,
            },
        ),
        _method_row(
            name="eval_label_best_switcher_diagnostic",
            rows=eval_rows,
            predictions=eval_diagnostic["predictions"],
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=_packet_bytes(str(eval_diagnostic["model"]["mode"]))[0],
            framed_record_bytes=_packet_bytes(str(eval_diagnostic["model"]["mode"]))[1],
            details={
                "model": {key: value for key, value in eval_diagnostic["model"].items() if key != "weights"},
                "eval_label_selected": True,
                "not_promotable": True,
            },
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=eval_rows,
            predictions=fixed_hybrid,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
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
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
        ),
        _method_row(
            name="target_or_hybrid_oracle",
            rows=eval_rows,
            predictions=oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True},
        ),
        _method_row(
            name="qwen_score_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=qwen_top2_oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True, "source_score_topk": 2},
        ),
        _method_row(
            name="target_hybrid_qwen_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=target_hybrid_qwen_top2_oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True, "source_score_topk": 2},
        ),
        _method_row(
            name="label_permutation_switcher_control",
            rows=eval_rows,
            predictions=label_control_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=forced_raw,
            framed_record_bytes=forced_framed,
            details={"label_shift": 1},
        ),
    ]
    for condition in (
        "source_row_shuffle",
        "source_score_row_shuffle_before_encoding",
        "code_value_permutation",
        "candidate_roll_code",
        "fourier_sign_flip_source",
        "fourier_basis_permutation_source",
        "random_same_byte",
    ):
        codes = _codes_for_condition(
            eval_rows,
            mode=str(forced_model["mode"]),
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        predictions, _ = _predict_with_model(eval_rows, forced_model, codes=codes)
        method_rows.append(
            _method_row(
                name=f"{condition}_switcher_control",
                rows=eval_rows,
                predictions=predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=forced_raw,
                framed_record_bytes=forced_framed,
                details={"condition": condition, "control_model": "forced_nonzero_train_select_switcher"},
            )
        )
    method_rows = sorted(method_rows, key=lambda item: item["accuracy"], reverse=True)
    selected_row = next(row for row in method_rows if row["method"] == "selected_train_select_switcher")
    forced_row = next(row for row in method_rows if row["method"] == "forced_nonzero_train_select_switcher")
    eval_diag_row = next(row for row in method_rows if row["method"] == "eval_label_best_switcher_diagnostic")
    destructive_rows = [
        row for row in method_rows if row["method"].endswith("_control") and row["method"] != "label_permutation_switcher_control"
    ]
    best_destructive = max(destructive_rows, key=lambda item: item["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    oracle_buckets = [
        _oracle_bucket_rows(fit_rows, "fit"),
        _oracle_bucket_rows(select_rows, "select"),
        _oracle_bucket_rows(eval_rows, "eval"),
    ]
    pass_gate = (
        selected_row["delta_vs_fixed_hybrid"] >= 0.005
        and selected_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and selected_row["ci95_low_vs_candidate_only"] > 0.0
        and selected_row["accuracy"] > best_destructive["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and selected_row["override_count_vs_fixed_hybrid"] > 0
    )
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if the fit/select-selected switcher beats fixed Qwen hybrid by at least 0.005 "
            "with positive paired CI, beats candidate-only with positive paired CI, is nonnegative on both "
            "cached Phi slices, beats destructive controls, and actually performs held-out overrides."
        ),
        "headline": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
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
            "target_or_hybrid_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "target_or_hybrid_oracle"
            )["accuracy"],
            "qwen_score_top2_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "qwen_score_top2_oracle_diagnostic"
            )["accuracy"],
            "target_hybrid_qwen_top2_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "target_hybrid_qwen_top2_oracle_diagnostic"
            )["accuracy"],
            "selected_switcher_accuracy": selected_row["accuracy"],
            "selected_switcher_delta_vs_fixed_hybrid": selected_row["delta_vs_fixed_hybrid"],
            "selected_switcher_ci95_low_vs_fixed_hybrid": selected_row["ci95_low_vs_fixed_hybrid"],
            "selected_switcher_overrides": selected_row["override_count_vs_fixed_hybrid"],
            "forced_switcher_accuracy": forced_row["accuracy"],
            "forced_switcher_delta_vs_fixed_hybrid": forced_row["delta_vs_fixed_hybrid"],
            "forced_switcher_overrides": forced_row["override_count_vs_fixed_hybrid"],
            "eval_label_diagnostic_accuracy": eval_diag_row["accuracy"],
            "eval_label_diagnostic_delta_vs_fixed_hybrid": eval_diag_row["delta_vs_fixed_hybrid"],
            "best_destructive_control_name": best_destructive["method"],
            "best_destructive_control_accuracy": best_destructive["accuracy"],
            "native_systems_claim_allowed": False,
        },
        "packet_contract": {
            "receiver_visible_payload": (
                "one byte-scale source packet decoded as Qwen hybrid plus optional Fourier score-contrast flags; "
                "Phi score simplex remains receiver-local side information"
            ),
            "raw_payload_bytes": selected_raw,
            "framed_record_bytes": selected_framed,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "target_scores_are_receiver_side_information": True,
        },
        "source_score_metadata": source_score_metadata,
        "slice_metadata": metadata,
        "method_rows": method_rows,
        "config_rows": config_rows,
        "slice_rows": slice_rows,
        "oracle_bucket_rows": oracle_buckets,
        "interpretation": (
            "This gate decomposes the target-or-hybrid oracle gap. It tests whether the byte-scale Qwen "
            "packet plus receiver-local Phi scores can learn when to switch from Qwen hybrid to Phi. If "
            "the eval-label diagnostic is also near fixed hybrid, the current feature/packet surface does "
            "not expose the oracle headroom and a richer source-code dictionary or interface is needed."
        ),
        "lay_explanation": (
            "Qwen's tiny hint is usually better, but Phi is sometimes right when Qwen is wrong. This test "
            "tries to learn those moments without looking at the held-out answers. It also reports a "
            "cheating diagnostic to ask whether this feature set could recover the oracle gap even if we "
            "were allowed to tune on the test answers."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_qwen_to_phi_oracle_switch_decomposition_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_csv(output_dir / "oracle_bucket_rows.csv", oracle_buckets)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_oracle_switch_decomposition_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_oracle_switch_decomposition_gate.json",
                "hellaswag_qwen_to_phi_oracle_switch_decomposition_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "slice_rows.csv",
                "oracle_bucket_rows.csv",
            ],
            "slice_metadata": metadata,
            "source_score_metadata": source_score_metadata,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--fit-rows-per-slice", type=int, default=FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=SELECT_ROWS_PER_SLICE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        source_score_cache=args.source_score_cache,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "target_or_hybrid_oracle_accuracy": h["target_or_hybrid_oracle_accuracy"],
                "selected_switcher_accuracy": h["selected_switcher_accuracy"],
                "selected_switcher_delta_vs_fixed_hybrid": h["selected_switcher_delta_vs_fixed_hybrid"],
                "forced_switcher_accuracy": h["forced_switcher_accuracy"],
                "eval_label_diagnostic_accuracy": h["eval_label_diagnostic_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
