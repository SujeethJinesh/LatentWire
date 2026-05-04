from __future__ import annotations

"""Train-only Qwen-to-Phi denoising syndrome packet gate.

The source sends a byte-scale discrete code derived from Qwen packet-policy
predictions. The receiver uses Phi's local four-choice score simplex as side
information and fits a tiny ridge denoiser to choose an answer candidate.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate_20260504_validation1024_2048"
)
DEFAULT_SLICES = (
    {
        "slice_start": 1024,
        "slice_end_exclusive": 1536,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/"
            "qwen_strict_packet_predictions_1024_1536.jsonl"
        ),
        "phi_target_score_cache": (
            "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536/"
            "target_score_cache.json"
        ),
    },
    {
        "slice_start": 1536,
        "slice_end_exclusive": 2048,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/"
            "qwen_strict_packet_predictions_1536_2048.jsonl"
        ),
        "phi_target_score_cache": (
            "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048/"
            "target_score_cache.json"
        ),
    },
)
SOURCE_CODE_MODES = (
    "code8_hybrid_selected_margin",
    "code8_hybrid_agreement_margin",
    "code16_policy_margin",
)
CONTROL_CODE_MODES = ("zero_byte_target_only", "target_derived_code")
RIDGE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
BOOTSTRAP_SAMPLES = 5000
FIT_ROWS_PER_SLICE = 64
SELECT_ROWS_PER_SLICE = 64


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hybrid_prediction(row: dict[str, Any]) -> int:
    if int(row["hidden_mean_prediction"]) == int(row["score_mean_prediction"]):
        return int(row["vote_prediction"])
    return int(row["hidden_mean_prediction"])


def _load_rows(
    *,
    slices: tuple[dict[str, Any], ...],
    fit_rows_per_slice: int,
    select_rows_per_slice: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for spec in slices:
        qwen_path = pathlib.Path(spec["qwen_predictions"])
        phi_path = pathlib.Path(spec["phi_target_score_cache"])
        qwen_rows = _read_jsonl(qwen_path)
        phi_cache = _read_json(phi_path)
        if len(qwen_rows) != int(phi_cache["row_count"]):
            raise ValueError(f"row-count mismatch for slice {spec['slice_start']}")
        for index, (row, row_id, target_pred, target_scores) in enumerate(
            zip(
                qwen_rows,
                phi_cache["row_ids"],
                phi_cache["source_predictions"],
                phi_cache["source_scores"],
                strict=True,
            )
        ):
            if str(row["row_id"]) != str(row_id):
                raise ValueError(f"row-id mismatch at slice {spec['slice_start']} index {index}")
            copied = dict(row)
            copied["phi_target_prediction"] = int(target_pred)
            copied["phi_target_scores"] = [float(value) for value in target_scores]
            copied["qwen_hybrid_prediction"] = _hybrid_prediction(row)
            copied["_slice_start"] = int(spec["slice_start"])
            copied["_slice_end_exclusive"] = int(spec["slice_end_exclusive"])
            copied["_within_slice_index"] = int(index)
            if index < int(fit_rows_per_slice):
                copied["_split"] = "fit"
            elif index < int(fit_rows_per_slice) + int(select_rows_per_slice):
                copied["_split"] = "select"
            else:
                copied["_split"] = "eval"
            rows.append(copied)
        metadata.append(
            {
                "slice_start": int(spec["slice_start"]),
                "slice_end_exclusive": int(spec["slice_end_exclusive"]),
                "rows": len(qwen_rows),
                "fit_rows": int(fit_rows_per_slice),
                "select_rows": int(select_rows_per_slice),
                "eval_rows": max(0, len(qwen_rows) - int(fit_rows_per_slice) - int(select_rows_per_slice)),
                "qwen_predictions": _display_path(qwen_path),
                "qwen_predictions_sha256": _sha256_file(qwen_path),
                "phi_target_score_cache": _display_path(phi_path),
                "phi_target_score_cache_sha256": _sha256_file(phi_path),
            }
        )
    return rows, metadata


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


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _margin_bin(value: float, thresholds: tuple[float, float, float]) -> int:
    if value < thresholds[0]:
        return 0
    if value < thresholds[1]:
        return 1
    if value < thresholds[2]:
        return 2
    return 3


def _encode_source_code(row: dict[str, Any], mode: str) -> int:
    hybrid = int(row["qwen_hybrid_prediction"])
    selected = int(row["selected_prediction"])
    hidden_mean = int(row["hidden_mean_prediction"])
    score_mean = int(row["score_mean_prediction"])
    vote = int(row["vote_prediction"])
    margin_bin = _margin_bin(float(row.get("selected_margin", 0.0)), (0.5, 1.0, 2.0))
    if mode == "zero_byte_target_only":
        return 0
    if mode == "target_derived_code":
        scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
        order = np.argsort(-scores)
        target_margin_bin = _margin_bin(float(scores[order[0]] - scores[order[1]]), (0.1, 0.3, 0.6))
        return int(row["phi_target_prediction"]) | (target_margin_bin << 2)
    if mode == "code8_hybrid_selected_margin":
        return (
            hybrid
            | (selected << 2)
            | (margin_bin << 4)
            | (int(hidden_mean == score_mean) << 6)
            | (int(hybrid == selected) << 7)
        )
    if mode == "code8_hybrid_agreement_margin":
        return (
            hybrid
            | (margin_bin << 2)
            | (int(hidden_mean == score_mean) << 4)
            | (int(hybrid == selected) << 5)
            | (int(vote == hybrid) << 6)
            | (int(score_mean == hybrid) << 7)
        )
    if mode == "code16_policy_margin":
        return (
            hybrid
            | (selected << 2)
            | (hidden_mean << 4)
            | (score_mean << 6)
            | (vote << 8)
            | (margin_bin << 10)
            | (int(hidden_mean == score_mean) << 12)
            | (int(hybrid == selected) << 13)
        )
    raise ValueError(f"unknown code mode: {mode}")


def _packet_bytes(mode: str) -> tuple[int, int]:
    if mode in {"zero_byte_target_only", "target_derived_code"}:
        return 0, 0
    if mode.startswith("code8"):
        return 1, 4
    if mode.startswith("code16"):
        return 2, 5
    raise ValueError(f"unknown code mode: {mode}")


def _decode_code(code: int, mode: str) -> dict[str, Any]:
    if mode == "zero_byte_target_only":
        return {"ids": {}, "margin_bin": None, "flags": []}
    if mode == "target_derived_code":
        return {"ids": {}, "margin_bin": (int(code) >> 2) & 3, "flags": []}
    hybrid = int(code) & 3
    if mode == "code8_hybrid_selected_margin":
        return {
            "ids": {"hybrid": hybrid, "selected": (int(code) >> 2) & 3},
            "margin_bin": (int(code) >> 4) & 3,
            "flags": [(int(code) >> 6) & 1, (int(code) >> 7) & 1],
        }
    if mode == "code8_hybrid_agreement_margin":
        return {
            "ids": {"hybrid": hybrid},
            "margin_bin": (int(code) >> 2) & 3,
            "flags": [
                (int(code) >> 4) & 1,
                (int(code) >> 5) & 1,
                (int(code) >> 6) & 1,
                (int(code) >> 7) & 1,
            ],
        }
    if mode == "code16_policy_margin":
        return {
            "ids": {
                "hybrid": hybrid,
                "selected": (int(code) >> 2) & 3,
                "hidden_mean": (int(code) >> 4) & 3,
                "score_mean": (int(code) >> 6) & 3,
                "vote": (int(code) >> 8) & 3,
            },
            "margin_bin": (int(code) >> 10) & 3,
            "flags": [(int(code) >> 12) & 1, (int(code) >> 13) & 1],
        }
    raise ValueError(f"unknown code mode: {mode}")


def _candidate_features(row: dict[str, Any], candidate: int, *, code: int, mode: str) -> np.ndarray:
    scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    order = np.argsort(-scores)
    ranks = np.empty(4, dtype=np.int64)
    ranks[order] = np.arange(4)
    probs = _softmax(scores)
    centered = scores - np.mean(scores)
    scale = np.std(centered)
    z_scores = centered / (scale if scale > 1e-8 else 1.0)
    decoded = _decode_code(code, mode)
    features: list[float] = [
        1.0,
        float(z_scores[candidate]),
        float(probs[candidate]),
        float(scores[candidate]),
        float(ranks[candidate] == 0),
        float(ranks[candidate]),
        float(scores[order[0]] - scores[order[1]]),
        float(probs[order[0]]),
        float(-np.sum(probs * np.log(probs + 1e-12))),
    ]
    for value in decoded["ids"].values():
        features.append(float(candidate == int(value)))
    margin_bin = decoded["margin_bin"]
    if margin_bin is not None:
        features.extend(float(int(margin_bin) == item) for item in range(4))
    features.extend(float(item) for item in decoded["flags"])
    if mode not in {"zero_byte_target_only", "target_derived_code"}:
        hashed = (int(code) * 17 + int(candidate) * 31) % 64
        features.extend(float(item == hashed) for item in range(64))
    return np.asarray(features, dtype=np.float64)


def _row_features(rows: list[dict[str, Any]], *, mode: str, codes: np.ndarray | None = None) -> np.ndarray:
    features: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        code = int(_encode_source_code(row, mode) if codes is None else codes[row_index])
        for candidate in range(4):
            features.append(_candidate_features(row, candidate, code=code, mode=mode))
    return np.vstack(features)


def _fit_ridge(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    l2: float,
    label_shift: int = 0,
) -> dict[str, Any]:
    x = _row_features(rows, mode=mode)
    answers = (np.repeat(_answers(rows), 4) + int(label_shift)) % 4
    candidate_ids = np.tile(np.arange(4, dtype=np.int64), len(rows))
    y = (candidate_ids == answers).astype(np.float64)
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
        code = int(_encode_source_code(row, mode) if codes is None else codes[row_index])
        scores = [
            float(_candidate_features(row, candidate, code=code, mode=mode) @ weights)
            for candidate in range(4)
        ]
        predictions.append(int(np.argmax(scores)))
    return np.asarray(predictions, dtype=np.int64)


def _roll_source_row(row: dict[str, Any], *, amount: int = 1) -> dict[str, Any]:
    rolled = dict(row)
    for field in (
        "qwen_hybrid_prediction",
        "selected_prediction",
        "hidden_mean_prediction",
        "score_mean_prediction",
        "vote_prediction",
    ):
        if field in rolled:
            rolled[field] = (int(rolled[field]) + int(amount)) % 4
    return rolled


def _codes_for_condition(rows: list[dict[str, Any]], *, mode: str, condition: str, seed: int) -> np.ndarray:
    base = np.asarray([_encode_source_code(row, mode) for row in rows], dtype=np.int64)
    if condition == "matched":
        return base
    rng = np.random.default_rng(seed)
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
            [_encode_source_code(_roll_source_row(row), mode) for row in rows],
            dtype=np.int64,
        )
    raise ValueError(f"unknown condition: {condition}")


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
            model = _fit_ridge(fit_rows, mode=mode, l2=float(l2))
            predictions = _predict_with_model(select_rows, model)
            paired = _paired_ci(
                selected=predictions,
                baseline=fixed_hybrid,
                answers=answers,
                seed=20260504 + sum(ord(ch) for ch in mode) + int(float(l2) * 1000),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            accuracy = _accuracy(predictions, answers)
            row = {
                "mode": mode,
                "l2": float(l2),
                "raw_payload_bytes": raw_bytes,
                "framed_record_bytes": framed_bytes,
                "select_accuracy": accuracy,
                "select_delta_vs_fixed_hybrid": paired["delta"],
                "select_ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                "select_helps_vs_fixed_hybrid": paired["helps"],
                "select_harms_vs_fixed_hybrid": paired["harms"],
                "select_override_count": int(np.sum(predictions != fixed_hybrid)),
            }
            config_rows.append(row)
            key = (
                float(accuracy),
                float(paired["delta"]),
                float(paired["ci95_low"]),
                float(-row["raw_payload_bytes"]),
                f"{mode}:{l2}",
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = model
    if best_model is None:
        raise ValueError("no selectable denoising model")
    return best_model, sorted(config_rows, key=lambda item: item["select_accuracy"], reverse=True)


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
        "# HellaSwag Qwen-To-Phi Denoising Syndrome Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- selected source mode: `{h['selected_source_mode']}`",
        f"- selected source l2: `{h['selected_source_l2']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- denoising syndrome accuracy: `{h['denoising_syndrome_accuracy']:.6f}`",
        f"- delta vs fixed hybrid: `{h['denoising_delta_vs_fixed_hybrid']:.6f}`",
        f"- CI95 low vs fixed hybrid: `{h['denoising_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- target-or-hybrid oracle accuracy: `{h['target_or_hybrid_oracle_accuracy']:.6f}`",
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
    _resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    slices: tuple[dict[str, Any], ...] = DEFAULT_SLICES,
    fit_rows_per_slice: int = FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = SELECT_ROWS_PER_SLICE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, metadata = _load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    fit_rows = [row for row in rows if row["_split"] == "fit"]
    select_rows = [row for row in rows if row["_split"] == "select"]
    eval_rows = [row for row in rows if row["_split"] == "eval"]

    source_model, source_config_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        modes=SOURCE_CODE_MODES,
        l2_values=RIDGE_LAMBDAS,
        bootstrap_samples=bootstrap_samples,
    )
    zero_model, zero_config_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        modes=("zero_byte_target_only",),
        l2_values=RIDGE_LAMBDAS,
        bootstrap_samples=bootstrap_samples,
    )
    target_code_model, target_code_config_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        modes=("target_derived_code",),
        l2_values=RIDGE_LAMBDAS,
        bootstrap_samples=bootstrap_samples,
    )

    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    raw_bytes, framed_bytes = _packet_bytes(str(source_model["mode"]))

    denoising = _predict_with_model(eval_rows, source_model)
    zero_byte = _predict_with_model(eval_rows, zero_model)
    target_code = _predict_with_model(eval_rows, target_code_model)
    label_permutation_model = _fit_ridge(
        fit_rows,
        mode=str(source_model["mode"]),
        l2=float(source_model["l2"]),
        label_shift=1,
    )
    label_permutation = _predict_with_model(eval_rows, label_permutation_model)
    oracle = np.where(fixed_hybrid == answers, fixed_hybrid, target_only).astype(np.int64)

    method_rows = [
        _method_row(
            name="denoising_syndrome_packet",
            rows=eval_rows,
            predictions=denoising,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            details={"model": {k: v for k, v in source_model.items() if k != "weights"}},
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
            name="zero_byte_target_ridge_control",
            rows=eval_rows,
            predictions=zero_byte,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"model": {k: v for k, v in zero_model.items() if k != "weights"}},
        ),
        _method_row(
            name="target_derived_code_control",
            rows=eval_rows,
            predictions=target_code,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"model": {k: v for k, v in target_code_model.items() if k != "weights"}},
        ),
        _method_row(
            name="label_permutation_decoder_control",
            rows=eval_rows,
            predictions=label_permutation,
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
        "code_value_permutation",
        "candidate_roll_code",
        "random_same_byte",
    ):
        codes = _codes_for_condition(
            eval_rows,
            mode=str(source_model["mode"]),
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        predictions = _predict_with_model(eval_rows, source_model, codes=codes)
        method_rows.append(
            _method_row(
                name=f"{condition}_control",
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
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    denoising_row = next(row for row in method_rows if row["method"] == "denoising_syndrome_packet")
    zero_row = next(row for row in method_rows if row["method"] == "zero_byte_target_ridge_control")
    target_code_row = next(row for row in method_rows if row["method"] == "target_derived_code_control")
    destructive_rows = [
        row
        for row in method_rows
        if row["method"].endswith("_control") and row["method"] not in {
            "zero_byte_target_ridge_control",
            "target_derived_code_control",
        }
    ]
    best_destructive = max(destructive_rows, key=lambda row: row["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=denoising,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    pass_gate = (
        denoising_row["delta_vs_fixed_hybrid"] >= 0.005
        and denoising_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and denoising_row["ci95_low_vs_candidate_only"] > 0.0
        and denoising_row["accuracy"] > zero_row["accuracy"]
        and denoising_row["accuracy"] > target_code_row["accuracy"]
        and denoising_row["accuracy"] > best_destructive["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
    )
    config_rows = [
        {**row, "selection_family": "source_syndrome"} for row in source_config_rows
    ] + [
        {**row, "selection_family": "zero_byte_target"} for row in zero_config_rows
    ] + [
        {**row, "selection_family": "target_derived_code"} for row in target_code_config_rows
    ]
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if the train-fit/select denoising syndrome packet beats fixed Qwen hybrid by "
            "at least 0.005 with positive paired CI, beats Qwen candidate-only with positive paired CI, "
            "beats zero-byte and target-derived-code controls, beats destructive code controls, and is "
            "nonnegative on both cached Phi slices."
        ),
        "headline": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
            "eval_rows": len(eval_rows),
            "selected_source_mode": source_model["mode"],
            "selected_source_l2": source_model["l2"],
            "raw_payload_bytes": raw_bytes,
            "framed_record_bytes": framed_bytes,
            "fixed_hybrid_accuracy": next(
                row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement"
            )["accuracy"],
            "candidate_only_accuracy": next(
                row for row in method_rows if row["method"] == "qwen_candidate_only"
            )["accuracy"],
            "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")[
                "accuracy"
            ],
            "denoising_syndrome_accuracy": denoising_row["accuracy"],
            "denoising_delta_vs_fixed_hybrid": denoising_row["delta_vs_fixed_hybrid"],
            "denoising_ci95_low_vs_fixed_hybrid": denoising_row["ci95_low_vs_fixed_hybrid"],
            "denoising_delta_vs_candidate_only": denoising_row["delta_vs_candidate_only"],
            "denoising_ci95_low_vs_candidate_only": denoising_row["ci95_low_vs_candidate_only"],
            "denoising_helps_vs_fixed_hybrid": denoising_row["helps_vs_fixed_hybrid"],
            "denoising_harms_vs_fixed_hybrid": denoising_row["harms_vs_fixed_hybrid"],
            "zero_byte_target_ridge_accuracy": zero_row["accuracy"],
            "target_derived_code_accuracy": target_code_row["accuracy"],
            "best_destructive_control_name": best_destructive["method"],
            "best_destructive_control_accuracy": best_destructive["accuracy"],
            "target_or_hybrid_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "target_or_hybrid_oracle"
            )["accuracy"],
            "native_systems_claim_allowed": False,
        },
        "packet_contract": {
            "receiver_visible_payload": (
                "one byte-scale source syndrome packet plus receiver-local Phi score side information"
            ),
            "raw_payload_bytes": raw_bytes,
            "framed_record_bytes": framed_bytes,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "target_scores_are_receiver_side_information": True,
        },
        "slice_metadata": metadata,
        "method_rows": method_rows,
        "config_rows": config_rows,
        "slice_rows": slice_rows,
        "interpretation": (
            "This gate tests the highest-priority Mac-local rate-distortion branch: a tiny source "
            "syndrome is decoded with Phi's local score simplex as side information. A pass would be a "
            "real receiver-side positive method over fixed hybrid. A failure weakens this ridge-denoising "
            "version of the syndrome idea while preserving oracle headroom for stronger methods."
        ),
        "lay_explanation": (
            "The source sends a tiny correction clue. Phi then uses its own four answer scores to decide "
            "whether that clue should repair its answer. The controls check whether the clue only helps "
            "when it comes from the right source example."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.json",
                "hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "slice_rows.csv",
            ],
            "slice_metadata": metadata,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-rows-per-slice", type=int, default=FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=SELECT_ROWS_PER_SLICE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
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
                "selected_source_mode": h["selected_source_mode"],
                "selected_source_l2": h["selected_source_l2"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "denoising_syndrome_accuracy": h["denoising_syndrome_accuracy"],
                "denoising_delta_vs_fixed_hybrid": h["denoising_delta_vs_fixed_hybrid"],
                "denoising_ci95_low_vs_fixed_hybrid": h["denoising_ci95_low_vs_fixed_hybrid"],
                "target_or_hybrid_oracle_accuracy": h["target_or_hybrid_oracle_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
