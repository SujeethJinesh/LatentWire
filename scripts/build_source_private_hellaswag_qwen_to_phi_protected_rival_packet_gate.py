from __future__ import annotations

"""Qwen-to-Phi protected top-rival packet gate.

This gate tests the most direct follow-up to the Qwen score top-2 oracle:
instead of sending only a switch hint, the source packet names a protected
candidate pair, usually the fixed Qwen hybrid and its highest-scoring Qwen
source rival. The receiver can only choose within that low-rate pair using
Phi's local scores as side information.
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
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402

DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate_20260504_validation1024_2048"
)
DEFAULT_SOURCE_SCORE_CACHE = oracle.DEFAULT_SOURCE_SCORE_CACHE
SOURCE_PAIR_MODES = (
    "code8_hybrid_rival",
    "code8_source_top2",
    "code16_hybrid_rival_policy",
    "code16_source_top2_policy",
)
RIDGE_LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
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


def _packet_bytes(mode: str) -> tuple[int, int]:
    if mode.startswith("code8"):
        return 1, 4
    if mode.startswith("code16"):
        return 2, 5
    if mode in {"phi_target_top2_pair_control"}:
        return 0, 0
    raise ValueError(f"unknown mode: {mode}")


def _softmax(scores: np.ndarray) -> np.ndarray:
    return denoise._softmax(scores)


def _source_order(row: dict[str, Any], *, source_transform: str = "matched") -> np.ndarray:
    scores = np.asarray(row["qwen_source_scores"], dtype=np.float64)
    if source_transform == "candidate_roll_code":
        scores = np.roll(scores, 1)
    return np.argsort(-scores)


def _source_margin_bin(row: dict[str, Any], *, source_transform: str = "matched") -> int:
    scores = np.asarray(row["qwen_source_scores"], dtype=np.float64)
    if source_transform == "candidate_roll_code":
        scores = np.roll(scores, 1)
    order = np.argsort(-scores)
    margin = float(scores[order[0]] - scores[order[1]])
    return int(np.digitize(margin, (0.25, 0.5, 1.0, 1.5), right=False))


def _source_top_pair(row: dict[str, Any], *, source_transform: str = "matched") -> tuple[int, int]:
    order = _source_order(row, source_transform=source_transform)
    return int(order[0]), int(order[1])


def _hybrid_rival_pair(row: dict[str, Any], *, source_transform: str = "matched") -> tuple[int, int]:
    hybrid = int(row["qwen_hybrid_prediction"])
    top1, top2 = _source_top_pair(row, source_transform=source_transform)
    rival = top2 if top1 == hybrid else top1
    return int(hybrid), int(rival)


def _encode_packet_code(row: dict[str, Any], mode: str, *, source_transform: str = "matched") -> int:
    hybrid = int(row["qwen_hybrid_prediction"])
    selected = int(row["selected_prediction"])
    hidden_mean = int(row["hidden_mean_prediction"])
    score_mean = int(row["score_mean_prediction"])
    top1, top2 = _source_top_pair(row, source_transform=source_transform)
    pair_hybrid, rival = _hybrid_rival_pair(row, source_transform=source_transform)
    margin_bin = _source_margin_bin(row, source_transform=source_transform)
    if mode == "code8_hybrid_rival":
        return (
            pair_hybrid
            | (rival << 2)
            | (int(top1 == pair_hybrid) << 4)
            | ((margin_bin & 3) << 5)
            | (int(selected == pair_hybrid) << 7)
        )
    if mode == "code8_source_top2":
        return (
            top1
            | (top2 << 2)
            | ((margin_bin & 3) << 4)
            | (int(hybrid == top1) << 6)
            | (int(hybrid == top2) << 7)
        )
    if mode == "code16_hybrid_rival_policy":
        return (
            pair_hybrid
            | (rival << 2)
            | (top1 << 4)
            | (top2 << 6)
            | (selected << 8)
            | (hidden_mean << 10)
            | (score_mean << 12)
            | ((margin_bin & 3) << 14)
        )
    if mode == "code16_source_top2_policy":
        return (
            top1
            | (top2 << 2)
            | (hybrid << 4)
            | (selected << 6)
            | (hidden_mean << 8)
            | (score_mean << 10)
            | ((margin_bin & 7) << 12)
            | (int(hybrid in {top1, top2}) << 15)
        )
    raise ValueError(f"unknown mode: {mode}")


def _decode_packet_code(code: int, mode: str) -> dict[str, Any]:
    code = int(code)
    if mode == "code8_hybrid_rival":
        return {
            "pair": (code & 3, (code >> 2) & 3),
            "ids": {"hybrid": code & 3, "rival": (code >> 2) & 3},
            "margin_bin": (code >> 5) & 3,
            "flags": [(code >> 4) & 1, (code >> 7) & 1],
        }
    if mode == "code8_source_top2":
        return {
            "pair": (code & 3, (code >> 2) & 3),
            "ids": {"top1": code & 3, "top2": (code >> 2) & 3},
            "margin_bin": (code >> 4) & 3,
            "flags": [(code >> 6) & 1, (code >> 7) & 1],
        }
    if mode == "code16_hybrid_rival_policy":
        return {
            "pair": (code & 3, (code >> 2) & 3),
            "ids": {
                "hybrid": code & 3,
                "rival": (code >> 2) & 3,
                "top1": (code >> 4) & 3,
                "top2": (code >> 6) & 3,
                "selected": (code >> 8) & 3,
                "hidden_mean": (code >> 10) & 3,
                "score_mean": (code >> 12) & 3,
            },
            "margin_bin": (code >> 14) & 3,
            "flags": [],
        }
    if mode == "code16_source_top2_policy":
        return {
            "pair": (code & 3, (code >> 2) & 3),
            "ids": {
                "top1": code & 3,
                "top2": (code >> 2) & 3,
                "hybrid": (code >> 4) & 3,
                "selected": (code >> 6) & 3,
                "hidden_mean": (code >> 8) & 3,
                "score_mean": (code >> 10) & 3,
            },
            "margin_bin": (code >> 12) & 7,
            "flags": [(code >> 15) & 1],
        }
    if mode == "phi_target_top2_pair_control":
        return {"pair": (code & 3, (code >> 2) & 3), "ids": {}, "margin_bin": (code >> 4) & 3, "flags": []}
    raise ValueError(f"unknown mode: {mode}")


def _decoded_pair(code: int, mode: str) -> tuple[int, int]:
    first, second = _decode_packet_code(code, mode)["pair"]
    return int(first), int(second)


def _pair_array(rows: list[dict[str, Any]], *, mode: str, codes: np.ndarray | None = None) -> np.ndarray:
    out = []
    for index, row in enumerate(rows):
        code = int(_encode_packet_code(row, mode) if codes is None else codes[index])
        out.append(_decoded_pair(code, mode))
    return np.asarray(out, dtype=np.int64)


def _phi_target_top2_codes(rows: list[dict[str, Any]]) -> np.ndarray:
    codes = []
    for row in rows:
        scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
        order = np.argsort(-scores)
        margin_bin = int(np.digitize(float(scores[order[0]] - scores[order[1]]), (0.05, 0.1, 0.2, 0.4), right=False))
        codes.append(int(order[0]) | (int(order[1]) << 2) | ((margin_bin & 3) << 4))
    return np.asarray(codes, dtype=np.int64)


def _candidate_features(row: dict[str, Any], candidate: int, *, code: int, mode: str) -> np.ndarray:
    scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    order = np.argsort(-scores)
    ranks = np.empty(4, dtype=np.int64)
    ranks[order] = np.arange(4)
    probs = _softmax(scores)
    centered = scores - np.mean(scores)
    scale = float(np.std(centered))
    z_scores = centered / (scale if scale > 1e-8 else 1.0)
    first, second = _decoded_pair(code, mode)
    other = second if int(candidate) == first else first
    decoded = _decode_packet_code(code, mode)
    features: list[float] = [
        1.0,
        float(candidate == first),
        float(candidate == second),
        float(z_scores[candidate]),
        float(probs[candidate]),
        float(scores[candidate]),
        float(ranks[candidate]),
        float(scores[candidate] - scores[other]),
        float(probs[candidate] - probs[other]),
        float(scores[order[0]] - scores[order[1]]),
        float(probs[order[0]]),
        float(-np.sum(probs * np.log(probs + 1e-12))),
        float(candidate == int(row["phi_target_prediction"])),
    ]
    for value in decoded["ids"].values():
        features.append(float(candidate == int(value)))
    margin_bin = decoded["margin_bin"]
    if margin_bin is not None:
        features.extend(float(int(margin_bin) == item) for item in range(8))
    features.extend(float(item) for item in decoded["flags"])
    if mode != "phi_target_top2_pair_control":
        hashed = (int(code) * 17 + int(candidate) * 31) % 64
        features.extend(float(item == hashed) for item in range(64))
    return np.asarray(features, dtype=np.float64)


def _row_features(rows: list[dict[str, Any]], *, mode: str, codes: np.ndarray | None = None) -> np.ndarray:
    features: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        code = int(_encode_packet_code(row, mode) if codes is None else codes[row_index])
        first, second = _decoded_pair(code, mode)
        features.append(_candidate_features(row, first, code=code, mode=mode))
        features.append(_candidate_features(row, second, code=code, mode=mode))
    max_width = max(item.shape[0] for item in features)
    return np.vstack([np.pad(item, (0, max_width - item.shape[0])) for item in features])


def _fit_pair_decoder(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    l2: float,
    label_shift: int = 0,
) -> dict[str, Any]:
    x = _row_features(rows, mode=mode)
    pairs = _pair_array(rows, mode=mode).reshape(-1)
    answers = np.repeat((_answers(rows) + int(label_shift)) % 4, 2)
    y = (pairs == answers).astype(np.float64)
    penalty = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    return {"mode": mode, "l2": float(l2), "weights": weights.tolist(), "label_shift": int(label_shift)}


def _predict_with_model(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    *,
    codes: np.ndarray | None = None,
) -> np.ndarray:
    mode = str(model["mode"])
    weights = np.asarray(model["weights"], dtype=np.float64)
    predictions: list[int] = []
    for row_index, row in enumerate(rows):
        code = int(_encode_packet_code(row, mode) if codes is None else codes[row_index])
        first, second = _decoded_pair(code, mode)
        x_first = _candidate_features(row, first, code=code, mode=mode)
        x_second = _candidate_features(row, second, code=code, mode=mode)
        width = weights.shape[0]
        if x_first.shape[0] < width:
            x_first = np.pad(x_first, (0, width - x_first.shape[0]))
            x_second = np.pad(x_second, (0, width - x_second.shape[0]))
        scores = [float(x_first[:width] @ weights), float(x_second[:width] @ weights)]
        predictions.append(int(first if scores[0] >= scores[1] else second))
    return np.asarray(predictions, dtype=np.int64)


def _deterministic_pair_prediction(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    chooser: str,
    codes: np.ndarray | None = None,
) -> np.ndarray:
    pairs = _pair_array(rows, mode=mode, codes=codes)
    out = []
    for row, (first, second) in zip(rows, pairs, strict=True):
        if chooser == "phi":
            scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
        elif chooser == "source":
            scores = np.asarray(row["qwen_source_scores"], dtype=np.float64)
        else:
            raise ValueError(f"unknown chooser: {chooser}")
        out.append(int(first if scores[first] >= scores[second] else second))
    return np.asarray(out, dtype=np.int64)


def _pair_oracle(rows: list[dict[str, Any]], *, mode: str, codes: np.ndarray | None = None) -> np.ndarray:
    pairs = _pair_array(rows, mode=mode, codes=codes)
    answers = _answers(rows)
    out = []
    for answer, (first, second) in zip(answers, pairs, strict=True):
        if int(answer) in {int(first), int(second)}:
            out.append(int(answer))
        else:
            out.append(int(first))
    return np.asarray(out, dtype=np.int64)


def _select_model(
    *,
    fit_rows: list[dict[str, Any]],
    select_rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    l2_values: tuple[float, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answers = _answers(select_rows)
    fixed_hybrid = _field_array(select_rows, "qwen_hybrid_prediction")
    config_rows: list[dict[str, Any]] = []
    best_model: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, str] | None = None
    for mode in modes:
        raw_bytes, framed_bytes = _packet_bytes(mode)
        for l2 in l2_values:
            model = _fit_pair_decoder(fit_rows, mode=mode, l2=float(l2))
            predictions = _predict_with_model(select_rows, model)
            paired = _paired_ci(
                selected=predictions,
                baseline=fixed_hybrid,
                answers=answers,
                seed=20260504 + sum(ord(ch) for ch in mode) + int(float(l2) * 1000),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            row = {
                "mode": mode,
                "l2": float(l2),
                "raw_payload_bytes": raw_bytes,
                "framed_record_bytes": framed_bytes,
                "select_accuracy": _accuracy(predictions, answers),
                "select_delta_vs_fixed_hybrid": paired["delta"],
                "select_ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                "select_helps_vs_fixed_hybrid": paired["helps"],
                "select_harms_vs_fixed_hybrid": paired["harms"],
                "select_override_count": int(np.sum(predictions != fixed_hybrid)),
            }
            config_rows.append(row)
            key = (
                float(row["select_accuracy"]),
                float(row["select_delta_vs_fixed_hybrid"]),
                float(row["select_ci95_low_vs_fixed_hybrid"]),
                float(-row["raw_payload_bytes"]),
                f"{mode}:{l2}",
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = model
    if best_model is None:
        raise ValueError("no protected-rival model selected")
    return best_model, sorted(config_rows, key=lambda item: item["select_accuracy"], reverse=True)


def _codes_for_condition(rows: list[dict[str, Any]], *, mode: str, condition: str, seed: int) -> np.ndarray:
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
    if condition == "candidate_roll_code":
        return np.asarray(
            [_encode_packet_code(row, mode, source_transform="candidate_roll_code") for row in rows],
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
    if condition == "phi_target_top2_pair_control":
        return _phi_target_top2_codes(rows)
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
        "# HellaSwag Qwen-To-Phi Protected Rival Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- selected source mode: `{h['selected_source_mode']}`",
        f"- selected l2: `{h['selected_source_l2']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- protected pair decoder accuracy: `{h['protected_pair_decoder_accuracy']:.6f}`",
        f"- protected pair decoder delta: `{h['protected_pair_decoder_delta_vs_fixed_hybrid']:.6f}`",
        f"- protected pair decoder CI95 low: `{h['protected_pair_decoder_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- hybrid-rival oracle accuracy: `{h['hybrid_rival_oracle_accuracy']:.6f}`",
        f"- source top-2 oracle accuracy: `{h['source_top2_oracle_accuracy']:.6f}`",
        f"- best destructive control: `{h['best_destructive_control_name']}` at `{h['best_destructive_control_accuracy']:.6f}`",
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
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    fit_rows = [row for row in rows if row["_split"] == "fit"]
    select_rows = [row for row in rows if row["_split"] == "select"]
    eval_rows = [row for row in rows if row["_split"] == "eval"]

    selected_model, config_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        modes=SOURCE_PAIR_MODES,
        l2_values=RIDGE_LAMBDAS,
        bootstrap_samples=bootstrap_samples,
    )
    selected_mode = str(selected_model["mode"])
    raw_bytes, framed_bytes = _packet_bytes(selected_mode)
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")

    selected_predictions = _predict_with_model(eval_rows, selected_model)
    fit_select_model = _fit_pair_decoder(fit_rows + select_rows, mode=selected_mode, l2=float(selected_model["l2"]))
    fit_select_predictions = _predict_with_model(eval_rows, fit_select_model)
    label_permutation_model = _fit_pair_decoder(
        fit_rows,
        mode=selected_mode,
        l2=float(selected_model["l2"]),
        label_shift=1,
    )
    label_permutation_predictions = _predict_with_model(eval_rows, label_permutation_model)
    hybrid_rival_codes = np.asarray(
        [_encode_packet_code(row, "code8_hybrid_rival") for row in eval_rows], dtype=np.int64
    )
    source_top2_codes = np.asarray([_encode_packet_code(row, "code8_source_top2") for row in eval_rows], dtype=np.int64)
    phi_top2_codes = _phi_target_top2_codes(eval_rows)

    method_rows = [
        _method_row(
            name="selected_protected_pair_decoder",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            details={"model": {key: value for key, value in selected_model.items() if key != "weights"}},
        ),
        _method_row(
            name="fit_select_pair_decoder_diagnostic",
            rows=eval_rows,
            predictions=fit_select_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            details={"not_promotable": True, "uses_select_labels_for_training": True},
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
            name="hybrid_rival_phi_argmax_baseline",
            rows=eval_rows,
            predictions=_deterministic_pair_prediction(
                eval_rows, mode="code8_hybrid_rival", chooser="phi", codes=hybrid_rival_codes
            ),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="source_top2_phi_argmax_baseline",
            rows=eval_rows,
            predictions=_deterministic_pair_prediction(
                eval_rows, mode="code8_source_top2", chooser="phi", codes=source_top2_codes
            ),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="qwen_source_score_top1_baseline",
            rows=eval_rows,
            predictions=_deterministic_pair_prediction(
                eval_rows, mode="code8_source_top2", chooser="source", codes=source_top2_codes
            ),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="hybrid_rival_oracle_diagnostic",
            rows=eval_rows,
            predictions=_pair_oracle(eval_rows, mode="code8_hybrid_rival", codes=hybrid_rival_codes),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
        _method_row(
            name="source_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=_pair_oracle(eval_rows, mode="code8_source_top2", codes=source_top2_codes),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
        _method_row(
            name="phi_target_top2_pair_control",
            rows=eval_rows,
            predictions=_pair_oracle(eval_rows, mode="phi_target_top2_pair_control", codes=phi_top2_codes),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"target_derived": True, "not_source_packet": True, "oracle_within_phi_top2": True},
        ),
        _method_row(
            name="label_permutation_pair_decoder_control",
            rows=eval_rows,
            predictions=label_permutation_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            details={"label_shift": 1},
        ),
    ]
    for condition in (
        "source_row_shuffle",
        "source_score_row_shuffle_before_encoding",
        "code_value_permutation",
        "candidate_roll_code",
        "random_same_byte",
    ):
        codes = _codes_for_condition(
            eval_rows,
            mode=selected_mode,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        predictions = _predict_with_model(eval_rows, selected_model, codes=codes)
        method_rows.append(
            _method_row(
                name=f"{condition}_pair_decoder_control",
                rows=eval_rows,
                predictions=predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=raw_bytes,
                framed_record_bytes=framed_bytes,
                details={"condition": condition},
            )
        )

    method_rows = sorted(method_rows, key=lambda item: item["accuracy"], reverse=True)
    selected_row = next(row for row in method_rows if row["method"] == "selected_protected_pair_decoder")
    fit_select_row = next(row for row in method_rows if row["method"] == "fit_select_pair_decoder_diagnostic")
    destructive_rows = [
        row
        for row in method_rows
        if row["method"].endswith("_control") and row["method"] != "phi_target_top2_pair_control"
    ]
    best_destructive = max(destructive_rows, key=lambda item: item["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    pass_gate = (
        selected_row["delta_vs_fixed_hybrid"] >= 0.005
        and selected_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and selected_row["ci95_low_vs_candidate_only"] > 0.0
        and selected_row["accuracy"] > best_destructive["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and selected_row["override_count_vs_fixed_hybrid"] > 0
        and selected_row["helps_vs_fixed_hybrid"] > selected_row["harms_vs_fixed_hybrid"]
    )
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if the fit/select-selected pair decoder beats fixed Qwen hybrid by at least 0.005 "
            "with positive paired CI, beats candidate-only with positive paired CI, beats destructive controls, "
            "is nonnegative on both cached Phi slices, performs held-out overrides, and helps more than it harms."
        ),
        "headline": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
            "eval_rows": len(eval_rows),
            "selected_source_mode": selected_mode,
            "selected_source_l2": float(selected_model["l2"]),
            "raw_payload_bytes": raw_bytes,
            "framed_record_bytes": framed_bytes,
            "fixed_hybrid_accuracy": next(
                row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement"
            )["accuracy"],
            "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")[
                "accuracy"
            ],
            "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")[
                "accuracy"
            ],
            "protected_pair_decoder_accuracy": selected_row["accuracy"],
            "protected_pair_decoder_delta_vs_fixed_hybrid": selected_row["delta_vs_fixed_hybrid"],
            "protected_pair_decoder_ci95_low_vs_fixed_hybrid": selected_row["ci95_low_vs_fixed_hybrid"],
            "protected_pair_decoder_overrides": selected_row["override_count_vs_fixed_hybrid"],
            "protected_pair_decoder_helps": selected_row["helps_vs_fixed_hybrid"],
            "protected_pair_decoder_harms": selected_row["harms_vs_fixed_hybrid"],
            "fit_select_pair_decoder_accuracy": fit_select_row["accuracy"],
            "fit_select_pair_decoder_delta_vs_fixed_hybrid": fit_select_row["delta_vs_fixed_hybrid"],
            "hybrid_rival_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "hybrid_rival_oracle_diagnostic"
            )["accuracy"],
            "source_top2_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "source_top2_oracle_diagnostic"
            )["accuracy"],
            "best_destructive_control_name": best_destructive["method"],
            "best_destructive_control_accuracy": best_destructive["accuracy"],
            "native_systems_claim_allowed": False,
        },
        "packet_contract": {
            "receiver_visible_payload": (
                "one byte-scale source packet naming a protected candidate pair: fixed Qwen hybrid plus "
                "source-score rival, or Qwen source-score top-2; Phi score simplex remains receiver-local side information"
            ),
            "raw_payload_bytes": raw_bytes,
            "framed_record_bytes": framed_bytes,
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
        "interpretation": (
            "This gate directly tests the source-score top-2 headroom exposed by the previous oracle switch "
            "decomposition. The pair oracles are high, but the receiver must learn a reliable pairwise choice "
            "from only fit/select labels and Phi side information. A failure means the shared candidate-ID basis "
            "contains headroom but the current tiny calibration surface and low-rate pair features do not expose "
            "a safe decision rule."
        ),
        "lay_explanation": (
            "Qwen often has the right answer somewhere in its top two guesses. This test sends Phi the two "
            "Qwen-side choices that matter most and asks Phi to pick between them. If Phi still picks the wrong "
            "one too often, the problem is not that the rival was hidden; it is that we do not yet know how to "
            "teach Phi when that rival is better."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_qwen_to_phi_protected_rival_packet_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_protected_rival_packet_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_protected_rival_packet_gate.json",
                "hellaswag_qwen_to_phi_protected_rival_packet_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "slice_rows.csv",
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
                "protected_pair_decoder_accuracy": h["protected_pair_decoder_accuracy"],
                "protected_pair_decoder_delta_vs_fixed_hybrid": h["protected_pair_decoder_delta_vs_fixed_hybrid"],
                "hybrid_rival_oracle_accuracy": h["hybrid_rival_oracle_accuracy"],
                "source_top2_oracle_accuracy": h["source_top2_oracle_accuracy"],
                "best_destructive_control_accuracy": h["best_destructive_control_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
