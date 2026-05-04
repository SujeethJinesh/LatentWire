from __future__ import annotations

"""Official-train receiver-calibrated Qwen-to-Phi packet gate.

The source-only official-train dictionary failed because Qwen's own score
geometry did not tell us when its protected rival should replace the hybrid on
Qwen-to-Phi. This gate adds the missing receiver-side calibration: Phi scores
the same official-train rows locally, then the frozen receiver learns when to
choose Qwen hybrid, Qwen rival, or Phi's own top candidate.
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

from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as nonqwen  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402

DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_RIDGES = source_gate.DICTIONARY_RIDGES
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


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    return order[:, 0].astype(np.int64), order[:, 1].astype(np.int64)


def _pair_from_scores(scores: np.ndarray, hybrid: np.ndarray) -> np.ndarray:
    return source_gate._pair_from_scores(scores, hybrid)


def _arc_rows_for_calibration(
    *,
    train_path: pathlib.Path | str,
    calibration_rows: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    all_train_rows = official.arc_gate._load_rows(official._resolve(train_path))
    by_row_id = {str(row.row_id): row for row in all_train_rows}
    missing = [str(row["row_id"]) for row in calibration_rows if str(row["row_id"]) not in by_row_id]
    if missing:
        raise ValueError(f"missing official train rows for {len(missing)} calibration ids")
    selected = [by_row_id[str(row["row_id"])] for row in calibration_rows]
    return selected, {
        "official_train_rows": len(all_train_rows),
        "calibration_rows": len(selected),
        "calibration_content_digest": headroom._content_digest(selected),
    }


def _limit_calibration(calibration: dict[str, Any], max_rows: int | None) -> dict[str, Any]:
    if max_rows is None:
        return calibration
    if max_rows <= 0:
        raise ValueError("max_calibration_rows must be positive")
    limit = min(int(max_rows), len(calibration["rows"]))
    limited = dict(calibration)
    limited["rows"] = calibration["rows"][:limit]
    for key in ("answers", "scores", "hybrid", "mean", "margin"):
        limited[key] = calibration[key][:limit]
    limited["max_calibration_rows"] = int(max_rows)
    return limited


def _load_or_build_phi_scores(
    *,
    rows: list[Any],
    score_cache: pathlib.Path,
    target_model: pathlib.Path | str,
    target_device: str,
    target_dtype: str,
    target_max_length: int,
    target_normalization: str,
    target_prompt_mode: str,
    local_files_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    scores, predictions, state, sha = headroom._source_scores(
        rows=rows,
        score_cache=score_cache,
        source_lm_model=str(target_model),
        source_lm_device=target_device,
        source_lm_dtype=target_dtype,
        source_lm_max_length=target_max_length,
        source_lm_normalization=target_normalization,
        source_lm_prompt_mode=target_prompt_mode,
        local_files_only=local_files_only,
    )
    if sha is None:
        sha = headroom._sha256_file(score_cache)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(predictions, dtype=np.int64),
        state,
        str(sha),
    )


def _candidate_features(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    rival: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_margin: np.ndarray,
    candidate: np.ndarray,
    role: int,
) -> np.ndarray:
    qz = _z_scores(qwen_scores)
    pz = _z_scores(phi_scores)
    q_top1, q_top2 = _top2(qwen_scores)
    p_top1, p_top2 = _top2(phi_scores)
    row_ids = np.arange(qwen_scores.shape[0])
    q_margin = qwen_scores[row_ids, q_top1] - qwen_scores[row_ids, q_top2]
    p_margin = phi_scores[row_ids, p_top1] - phi_scores[row_ids, p_top2]
    parts: list[np.ndarray] = [
        np.ones(qwen_scores.shape[0], dtype=np.float64),
        np.full(qwen_scores.shape[0], float(role), dtype=np.float64),
        qz[row_ids, candidate],
        pz[row_ids, candidate],
        qz[row_ids, candidate] - qz[row_ids, hybrid],
        pz[row_ids, candidate] - pz[row_ids, hybrid],
        qz[row_ids, hybrid],
        qz[row_ids, rival],
        pz[row_ids, hybrid],
        pz[row_ids, rival],
        q_margin,
        p_margin,
        qwen_margin,
        (candidate == hybrid).astype(np.float64),
        (candidate == rival).astype(np.float64),
        (candidate == q_top1).astype(np.float64),
        (candidate == q_top2).astype(np.float64),
        (candidate == p_top1).astype(np.float64),
        (candidate == p_top2).astype(np.float64),
        (qwen_mean == hybrid).astype(np.float64),
        (qwen_mean == rival).astype(np.float64),
        (p_top1 == hybrid).astype(np.float64),
        (p_top1 == rival).astype(np.float64),
    ]
    for ids in (candidate, hybrid, rival, q_top1, p_top1):
        for option in range(4):
            parts.append((ids == option).astype(np.float64))
    for values in (q_margin, p_margin, qwen_margin, pz[row_ids, candidate] - pz[row_ids, hybrid]):
        for threshold in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            parts.append((values < threshold).astype(np.float64))
    return np.vstack(parts).T.astype(np.float64)


def _stack_action_features(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_margin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = _pair_from_scores(qwen_scores, hybrid)
    rival = pair[:, 1]
    phi_top1, _ = _top2(phi_scores)
    rival_features = _candidate_features(
        qwen_scores=qwen_scores,
        phi_scores=phi_scores,
        hybrid=hybrid,
        rival=rival,
        qwen_mean=qwen_mean,
        qwen_margin=qwen_margin,
        candidate=rival,
        role=1,
    )
    phi_features = _candidate_features(
        qwen_scores=qwen_scores,
        phi_scores=phi_scores,
        hybrid=hybrid,
        rival=rival,
        qwen_mean=qwen_mean,
        qwen_margin=qwen_margin,
        candidate=phi_top1,
        role=2,
    )
    return np.stack([rival_features, phi_features], axis=1), np.stack([rival, phi_top1], axis=1), pair


def _fit_receiver(
    *,
    action_features: np.ndarray,
    action_candidates: np.ndarray,
    hybrid: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    ridges: tuple[float, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat_x = action_features[fit_indices].reshape(-1, action_features.shape[-1])
    flat_candidates = action_candidates[fit_indices].reshape(-1)
    repeated_answers = np.repeat(answers[fit_indices], action_candidates.shape[1])
    repeated_hybrid = np.repeat(hybrid[fit_indices], action_candidates.shape[1])
    target = (flat_candidates == repeated_answers).astype(np.float64) - (
        repeated_hybrid == repeated_answers
    ).astype(np.float64)
    base = hybrid[dev_indices]
    config_rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, str], dict[str, Any]] | None = None
    for l2 in ridges:
        penalty = float(l2) * np.eye(flat_x.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        lhs = flat_x.T @ flat_x + penalty
        rhs = flat_x.T @ target
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs
        dev_scores = action_features[dev_indices] @ weights
        max_scores = np.max(dev_scores, axis=1)
        thresholds = sorted(set(float(item) for item in max_scores))
        noop_threshold = float(np.max(max_scores) + max(1e-9, abs(float(np.max(max_scores))) * 1e-6))
        thresholds.append(noop_threshold)
        if len(thresholds) > 80:
            finite = thresholds[:-1]
            thresholds = sorted({finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 61)})
            thresholds.append(noop_threshold)
        for threshold in thresholds:
            predictions = _predict_receiver(
                action_features[dev_indices],
                action_candidates[dev_indices],
                base,
                {"weights": weights.tolist(), "threshold": float(threshold), "l2": float(l2)},
            )
            paired = _paired_ci(
                selected=predictions,
                baseline=base,
                answers=answers[dev_indices],
                seed=50560504 + int(float(l2) * 1000),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            row = {
                "l2": float(l2),
                "threshold": float(threshold),
                "threshold_is_noop": bool(threshold == noop_threshold),
                "official_dev_accuracy": _accuracy(predictions, answers[dev_indices]),
                "official_dev_delta_vs_hybrid": paired["delta"],
                "official_dev_ci95_low_vs_hybrid": paired["ci95_low"],
                "official_dev_helps_vs_hybrid": paired["helps"],
                "official_dev_harms_vs_hybrid": paired["harms"],
                "official_dev_override_count": int(np.sum(predictions != base)),
            }
            config_rows.append(row)
            key = (
                float(row["official_dev_accuracy"]),
                float(row["official_dev_delta_vs_hybrid"]),
                float(row["official_dev_ci95_low_vs_hybrid"]),
                float(-row["official_dev_override_count"]),
                f"{l2}:{threshold}",
            )
            model = {"l2": float(l2), "threshold": float(threshold), "weights": weights.tolist()}
            if best is None or key > best[0]:
                best = (key, model)
    if best is None:
        raise ValueError("no receiver configs")
    return best[1], sorted(config_rows, key=lambda item: item["official_dev_accuracy"], reverse=True)


def _predict_receiver(
    action_features: np.ndarray,
    action_candidates: np.ndarray,
    hybrid: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    scores = action_features @ weights
    best_action = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_action]
    predictions = hybrid.copy()
    mask = best_score > float(model["threshold"])
    predictions[mask] = action_candidates[np.arange(scores.shape[0]), best_action][mask]
    return predictions.astype(np.int64)


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
        seed=60660504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=60660604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=60660704 + sum(ord(ch) for ch in name),
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


def _pair_oracle(pair: np.ndarray, phi_top1: np.ndarray, answers: np.ndarray) -> np.ndarray:
    actions = np.stack([pair[:, 0], pair[:, 1], phi_top1], axis=1)
    predictions = pair[:, 0].copy()
    for index in range(len(predictions)):
        for action in actions[index]:
            if int(action) == int(answers[index]):
                predictions[index] = int(action)
                break
    return predictions.astype(np.int64)


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
            seed=70660504 + int(start),
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


def _permuted_source_inputs(
    *,
    qwen_scores: np.ndarray,
    hybrid: np.ndarray,
    mean: np.ndarray,
    margin: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if condition == "matched":
        return qwen_scores, hybrid, mean, margin
    if condition == "source_row_shuffle":
        order = rng.permutation(len(hybrid))
        return qwen_scores[order], hybrid[order], mean[order], margin[order]
    if condition == "candidate_roll_source":
        return np.roll(qwen_scores, shift=1, axis=1), (hybrid + 1) % 4, (mean + 1) % 4, margin
    if condition == "code_value_permutation":
        perm = rng.permutation(4)
        inverse = np.empty(4, dtype=np.int64)
        inverse[perm] = np.arange(4)
        return qwen_scores[:, perm], inverse[hybrid], inverse[mean], margin
    raise ValueError(f"unknown source condition {condition!r}")


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Official-Train Receiver-Calibrated Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- Phi train score cache hit: `{h['phi_train_score_cache_hit']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- receiver-calibrated accuracy: `{h['receiver_calibrated_accuracy']:.6f}`",
        f"- receiver-calibrated delta: `{h['receiver_calibrated_delta_vs_fixed_hybrid']:.6f}`",
        f"- receiver-calibrated CI95 low: `{h['receiver_calibrated_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- hybrid/rival/Phi oracle accuracy: `{h['hybrid_rival_phi_oracle_accuracy']:.6f}`",
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
    train_path: pathlib.Path | str = DEFAULT_TRAIN_PATH,
    qwen_train_cache_dir: pathlib.Path | str = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    phi_train_score_cache: pathlib.Path | str | None = None,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    component_ridges: tuple[float, ...] = DEFAULT_COMPONENT_RIDGES,
    receiver_ridges: tuple[float, ...] = DEFAULT_RIDGES,
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
    calibration = _limit_calibration(calibration, max_calibration_rows)
    fit_indices, dev_indices = official._official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )
    calibration_arc_rows, calibration_row_meta = _arc_rows_for_calibration(
        train_path=train_path,
        calibration_rows=calibration["rows"],
    )
    train_cache_path = _resolve(phi_train_score_cache) if phi_train_score_cache is not None else output_dir / "phi_official_train_score_cache.json"
    phi_cache_existed = train_cache_path.exists()
    phi_scores, phi_predictions, phi_state, phi_sha = _load_or_build_phi_scores(
        rows=calibration_arc_rows,
        score_cache=train_cache_path,
        target_model=target_model,
        target_device=target_device,
        target_dtype=target_dtype,
        target_max_length=target_max_length,
        target_normalization=target_normalization,
        target_prompt_mode=target_prompt_mode,
        local_files_only=local_files_only,
    )
    action_features, action_candidates, train_pair = _stack_action_features(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
    )
    model, config_rows = _fit_receiver(
        action_features=action_features,
        action_candidates=action_candidates,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=receiver_ridges,
        bootstrap_samples=bootstrap_samples,
    )
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    eval_scores = np.asarray([row["qwen_source_scores"] for row in eval_rows], dtype=np.float64)
    eval_hybrid = np.asarray([int(row["qwen_hybrid_prediction"]) for row in eval_rows], dtype=np.int64)
    eval_mean = np.asarray([int(row["selected_prediction"]) for row in eval_rows], dtype=np.int64)
    eval_margin = np.asarray([float(row.get("selected_margin", 0.0)) for row in eval_rows], dtype=np.float64)
    eval_phi_scores = np.asarray([row["phi_target_scores"] for row in eval_rows], dtype=np.float64)
    eval_features, eval_candidates, eval_pair = _stack_action_features(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
    )
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    selected_predictions = _predict_receiver(eval_features, eval_candidates, fixed_hybrid, model)
    selected_row = next(row for row in config_rows if row["l2"] == model["l2"] and row["threshold"] == model["threshold"])
    label_rng = np.random.default_rng(20260504)
    permuted_answers = calibration["answers"][label_rng.permutation(len(calibration["answers"]))]
    label_model, _ = _fit_receiver(
        action_features=action_features,
        action_candidates=action_candidates,
        hybrid=calibration["hybrid"],
        answers=permuted_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=receiver_ridges,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    label_predictions = _predict_receiver(eval_features, eval_candidates, fixed_hybrid, label_model)
    method_rows = [
        _method_row(
            name="official_train_receiver_calibrated_packet",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            details={"model": {key: value for key, value in model.items() if key != "weights"}},
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
            name="hybrid_rival_phi_oracle_diagnostic",
            rows=eval_rows,
            predictions=_pair_oracle(eval_pair[:, :2], np.argmax(eval_phi_scores, axis=1), answers),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
        _method_row(
            name="official_train_label_permutation_receiver_control",
            rows=eval_rows,
            predictions=label_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            details={"condition": "official_train_label_permutation"},
        ),
    ]
    for condition in ("source_row_shuffle", "candidate_roll_source", "code_value_permutation"):
        c_scores, c_hybrid, c_mean, c_margin = _permuted_source_inputs(
            qwen_scores=eval_scores,
            hybrid=eval_hybrid,
            mean=eval_mean,
            margin=eval_margin,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        c_features, c_candidates, _ = _stack_action_features(
            qwen_scores=c_scores,
            phi_scores=eval_phi_scores,
            hybrid=c_hybrid,
            qwen_mean=c_mean,
            qwen_margin=c_margin,
        )
        corrupted = _predict_receiver(c_features, c_candidates, c_hybrid, model)
        method_rows.append(
            _method_row(
                name=f"{condition}_receiver_control",
                rows=eval_rows,
                predictions=corrupted,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=2,
                framed_record_bytes=5,
                details={"condition": condition},
            )
        )
    eval_diag_best: tuple[tuple[float, float, float, str], np.ndarray] | None = None
    weights = np.asarray(model["weights"], dtype=np.float64)
    eval_action_scores = eval_features @ weights
    eval_max_scores = np.max(eval_action_scores, axis=1)
    for threshold in sorted(set(float(item) for item in eval_max_scores)):
        diag_model = dict(model, threshold=float(threshold))
        diag_predictions = _predict_receiver(eval_features, eval_candidates, fixed_hybrid, diag_model)
        overrides = int(np.sum(diag_predictions != fixed_hybrid))
        if overrides == 0:
            continue
        key = (
            _accuracy(diag_predictions, answers),
            float(np.mean((diag_predictions == answers).astype(float) - (fixed_hybrid == answers).astype(float))),
            float(-overrides),
            str(threshold),
        )
        if eval_diag_best is None or key > eval_diag_best[0]:
            eval_diag_best = (key, diag_predictions)
    if eval_diag_best is not None:
        method_rows.append(
            _method_row(
                name="eval_label_best_threshold_receiver_diagnostic",
                rows=eval_rows,
                predictions=eval_diag_best[1],
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=2,
                framed_record_bytes=5,
                details={"not_promotable": True, "eval_label_selected": True},
            )
        )
    method_rows = sorted(method_rows, key=lambda item: item["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "official_train_receiver_calibrated_packet")
    destructive_rows = [row for row in method_rows if row["method"].endswith("_control")]
    best_destructive = max(destructive_rows, key=lambda item: item["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    train_content_ids = {row.content_id for row in calibration_arc_rows}
    eval_content_ids = {str(row.get("content_id", row["row_id"])) for row in eval_rows}
    pass_gate = bool(
        method_row["delta_vs_fixed_hybrid"] >= 0.005
        and method_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and method_row["ci95_low_vs_candidate_only"] > 0.0
        and method_row["accuracy"] > best_destructive["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and method_row["helps_vs_fixed_hybrid"] > method_row["harms_vs_fixed_hybrid"]
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "official_train_qwen_hybrid_accuracy": _accuracy(calibration["hybrid"], calibration["answers"]),
        "official_train_phi_target_accuracy": _accuracy(phi_predictions, calibration["answers"]),
        "official_train_hybrid_rival_phi_oracle_accuracy": _accuracy(
            _pair_oracle(train_pair[:, :2], phi_predictions, calibration["answers"]),
            calibration["answers"],
        ),
        "official_dev_selected_accuracy": selected_row["official_dev_accuracy"],
        "official_dev_selected_delta_vs_hybrid": selected_row["official_dev_delta_vs_hybrid"],
        "official_dev_selected_ci95_low_vs_hybrid": selected_row["official_dev_ci95_low_vs_hybrid"],
        "selected_l2": float(model["l2"]),
        "selected_threshold": float(model["threshold"]),
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
        "receiver_calibrated_accuracy": method_row["accuracy"],
        "receiver_calibrated_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "receiver_calibrated_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "receiver_calibrated_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "receiver_calibrated_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "receiver_calibrated_override_count": method_row["override_count_vs_fixed_hybrid"],
        "hybrid_rival_phi_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "hybrid_rival_phi_oracle_diagnostic"
        )["accuracy"],
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the official-train receiver-calibrated packet beats fixed Qwen hybrid by at least "
            "0.005 with positive paired CI, beats candidate-only with positive paired CI, is nonnegative on "
            "both eval slices, beats destructive controls, and helps more than it harms."
        ),
        "headline": headline,
        "calibration_row_metadata": calibration_row_meta,
        "packet_contract": {
            "receiver_visible_payload": (
                "byte-scale Qwen hybrid/rival candidate packet plus receiver-local Phi score side information"
            ),
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "phi_official_train_scores_used_for_training": True,
        },
        "source_score_metadata": source_score_metadata,
        "slice_metadata": metadata,
        "sample_cache_rows": calibration["sample_cache_rows"],
        "component_rows": calibration["component_rows"],
        "method_rows": method_rows,
        "config_rows": config_rows,
        "slice_rows": slice_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": 2,
            "framed_record_bytes_per_request": 5,
            "fit_and_eval_wall_time_s": float(time.perf_counter() - started),
            "phi_train_score_cache_hit": bool(phi_cache_existed),
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
            "phi_train_score_cache": headroom._display_path(train_cache_path),
            "phi_train_score_cache_sha256": phi_sha,
            "phi_train_score_model": phi_state,
            "source_score_cache": denoise._display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate tests the next live branch after the source-only dictionary failed: learn the receiver "
            "decision on official-train rows where both Qwen packet features and Phi local score features are "
            "available. A pass would show that receiver-side calibration, not more source-only fitting, unlocks "
            "the Qwen-to-Phi candidate frontier."
        ),
        "lay_explanation": (
            "We let Phi practice on training questions too. For each question, the rule sees Qwen's safe and "
            "backup answers plus Phi's own four answer scores, then learns whether to keep Qwen's safe answer, "
            "take Qwen's backup, or trust Phi. The frozen rule is then tested on held-out questions."
        ),
    }
    _write_json(output_dir / "hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.json",
                "hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.md",
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
    parser.add_argument("--phi-train-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--sample-seeds", type=official._parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=official._parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--component-ridges", type=official._parse_float_tuple, default=DEFAULT_COMPONENT_RIDGES)
    parser.add_argument("--receiver-ridges", type=official._parse_float_tuple, default=DEFAULT_RIDGES)
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
        receiver_ridges=args.receiver_ridges,
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
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "official_train_calibration_rows": h["official_train_calibration_rows"],
                "official_dev_selected_delta_vs_hybrid": h["official_dev_selected_delta_vs_hybrid"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "receiver_calibrated_accuracy": h["receiver_calibrated_accuracy"],
                "receiver_calibrated_delta_vs_fixed_hybrid": h[
                    "receiver_calibrated_delta_vs_fixed_hybrid"
                ],
                "hybrid_rival_phi_oracle_accuracy": h["hybrid_rival_phi_oracle_accuracy"],
                "phi_train_score_cache_hit": h["phi_train_score_cache_hit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
