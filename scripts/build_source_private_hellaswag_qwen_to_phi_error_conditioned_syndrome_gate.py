from __future__ import annotations

"""Official-train target-error-conditioned syndrome gate for Qwen-to-Phi.

The target-error repair audit found substantial held-out oracle headroom in
rows where the fixed Qwen hybrid packet is wrong but Qwen's source top-2 still
contains the gold candidate. This gate asks the next stricter question: can a
frozen receiver trained on official-train rows learn when to spend a tiny
source-private syndrome packet to repair those errors without leaking raw
source scores, source text, source hidden states, or evaluation labels?
"""

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as nonqwen  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate as router_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate as top2_gate  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_error_conditioned_syndrome_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_PHI_TRAIN_SCORE_CACHE = receiver_gate.DEFAULT_OUTPUT / "phi_official_train_score_cache.json"
DEFAULT_RECEIVER_RIDGES = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
)
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES

SOURCE_CONDITIONS = top2_gate.SOURCE_CONDITIONS
ERROR_SCHEMES: dict[str, tuple[str, ...]] = {
    "error_role_source_receiver": (
        "role",
        "candidate_eq_phi_top1",
        "candidate_eq_source_top1",
        "candidate_eq_source_top2",
        "source_top1_eq_phi_top1",
        "source_top2_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "q_margin_bin",
        "q_entropy_bin",
        "q_hybrid_gap_bin",
        "phi_action_adv_bin",
        "phi_margin_bin",
        "phi_hybrid_gap_bin",
    ),
    "source_unique_detector": (
        "role",
        "candidate_eq_source_top1",
        "candidate_eq_source_top2",
        "candidate_eq_phi_top1",
        "candidate_eq_phi_top2",
        "source_top1_eq_phi_top1",
        "source_top2_eq_phi_top1",
        "q_margin_bin",
        "phi_action_adv_bin",
    ),
    "compact_error_syndrome": (
        "role",
        "candidate_eq_hybrid",
        "candidate_eq_source_top1",
        "candidate_eq_source_top2",
        "candidate_eq_phi_top1",
        "q_selected_margin_bin",
        "q_hybrid_gap_bin",
        "phi_hybrid_gap_bin",
    ),
}
FOCUS_NAMES = (
    "all_rows",
    "fixed_hybrid_wrong",
    "fixed_hybrid_wrong_source_top2_contains_gold",
    "phi_target_wrong",
    "fixed_and_phi_wrong",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _answers(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == np.asarray(answers, dtype=np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return denoise._paired_ci(
        selected=np.asarray(selected, dtype=np.int64),
        baseline=np.asarray(baseline, dtype=np.int64),
        answers=np.asarray(answers, dtype=np.int64),
        seed=seed,
        samples=samples,
    )


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return router_gate._top2(scores)


def _fit_feature_spec(fields: dict[str, np.ndarray], scheme_keys: tuple[str, ...]) -> dict[str, Any]:
    spec_rows: list[dict[str, int | str]] = []
    for key in scheme_keys:
        values = np.asarray(fields[key], dtype=np.int64)
        spec_rows.append({"key": key, "min_value": int(np.min(values)), "max_value": int(np.max(values))})
    return {"scheme_keys": list(scheme_keys), "fields": spec_rows}


def _encode_features(fields: dict[str, np.ndarray], spec: dict[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    reference = np.asarray(fields[spec["scheme_keys"][0]], dtype=np.int64)
    n, action_count = reference.shape
    role = np.asarray(fields.get("role", np.zeros((n, action_count), dtype=np.int64)), dtype=np.int64)
    parts.append(np.ones((n, action_count, 1), dtype=np.float64))
    for item in spec["fields"]:
        key = str(item["key"])
        values = np.asarray(fields[key], dtype=np.int64)
        max_value = int(item["max_value"])
        scale = float(max(1, max_value))
        parts.append((values.astype(np.float64) / scale)[..., None])
        for value in range(max_value + 1):
            one_hot = (values == value).astype(np.float64)
            parts.append(one_hot[..., None])
            if key != "role":
                for role_id in range(action_count):
                    parts.append(((role == role_id) & (values == value)).astype(np.float64)[..., None])
    return np.concatenate(parts, axis=2)


def _focus_indices(
    *,
    focus: str,
    indices: np.ndarray,
    answers: np.ndarray,
    hybrid: np.ndarray,
    phi_top1: np.ndarray,
    source_top1: np.ndarray,
    source_top2: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if focus == "all_rows":
        mask = np.ones_like(indices, dtype=bool)
    elif focus == "fixed_hybrid_wrong":
        mask = hybrid[indices] != answers[indices]
    elif focus == "fixed_hybrid_wrong_source_top2_contains_gold":
        mask = (hybrid[indices] != answers[indices]) & (
            (source_top1[indices] == answers[indices]) | (source_top2[indices] == answers[indices])
        )
    elif focus == "phi_target_wrong":
        mask = phi_top1[indices] != answers[indices]
    elif focus == "fixed_and_phi_wrong":
        mask = (hybrid[indices] != answers[indices]) & (phi_top1[indices] != answers[indices])
    else:
        raise ValueError(f"unknown focus {focus!r}")
    selected = indices[mask]
    return selected if len(selected) >= 16 else np.asarray([], dtype=np.int64)


def _evaluate_predictions(
    *,
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    paired = _paired_ci(selected=predictions, baseline=baseline, answers=answers, seed=seed, samples=samples)
    helps = int(paired["helps"])
    harms = int(paired["harms"])
    return {
        "accuracy": _accuracy(predictions, answers),
        "delta": float(paired["delta"]),
        "ci95_low": float(paired["ci95_low"]),
        "ci95_high": float(paired["ci95_high"]),
        "helps": helps,
        "harms": harms,
        "net_help": int(helps - harms),
        "override_count": int(np.sum(predictions != baseline)),
        "accepted_harm_rate": float(harms / max(1, helps + harms)),
    }


def _quick_evaluate_predictions(
    *,
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
) -> dict[str, float | int]:
    predictions = np.asarray(predictions, dtype=np.int64)
    baseline = np.asarray(baseline, dtype=np.int64)
    answers = np.asarray(answers, dtype=np.int64)
    diff = (predictions == answers).astype(np.int64) - (baseline == answers).astype(np.int64)
    helps = int(np.sum(diff > 0))
    harms = int(np.sum(diff < 0))
    delta = float(np.mean(diff))
    return {
        "accuracy": _accuracy(predictions, answers),
        "delta": delta,
        "ci95_low": delta,
        "ci95_high": delta,
        "helps": helps,
        "harms": harms,
        "net_help": int(helps - harms),
        "override_count": int(np.sum(predictions != baseline)),
        "accepted_harm_rate": float(harms / max(1, helps + harms)),
    }


def _predict_receiver(
    *,
    actions: np.ndarray,
    fields: dict[str, np.ndarray],
    hybrid: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    predictions = np.asarray(hybrid, dtype=np.int64).copy()
    if model["scheme"] == "no_op":
        return predictions
    features = _encode_features(fields, model["feature_spec"])
    weights = np.asarray(model["weights"], dtype=np.float64)
    scores = features @ weights
    best_action = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_action]
    mask = best_score > float(model["threshold"])
    predictions[mask] = actions[np.arange(actions.shape[0]), best_action][mask]
    return predictions.astype(np.int64)


def _fit_receiver(
    *,
    actions: np.ndarray,
    fields: dict[str, np.ndarray],
    hybrid: np.ndarray,
    answers: np.ndarray,
    phi_top1: np.ndarray,
    source_top1: np.ndarray,
    source_top2: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    ridges: tuple[float, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_rows: list[dict[str, Any]] = []
    baseline_eval = _quick_evaluate_predictions(
        predictions=hybrid[dev_indices],
        baseline=hybrid[dev_indices],
        answers=answers[dev_indices],
    )
    noop_model = {
        "scheme": "no_op",
        "scheme_keys": [],
        "feature_spec": {"scheme_keys": [], "fields": []},
        "selection": {"reason": "no error-conditioned syndrome receiver beat no-op on official dev"},
    }
    best: tuple[tuple[float, float, float, float, float, str], dict[str, Any]] = (
        (
            float(baseline_eval["accuracy"]),
            float(baseline_eval["delta"]),
            float(baseline_eval["ci95_low"]),
            float(baseline_eval["net_help"]),
            -float(baseline_eval["harms"]),
            "no_op",
        ),
        noop_model,
    )
    for scheme_name, scheme_keys in ERROR_SCHEMES.items():
        spec = _fit_feature_spec(fields, scheme_keys)
        features = _encode_features(fields, spec)
        for focus in FOCUS_NAMES:
            focus_indices = _focus_indices(
                focus=focus,
                indices=fit_indices,
                answers=answers,
                hybrid=hybrid,
                phi_top1=phi_top1,
                source_top1=source_top1,
                source_top2=source_top2,
            )
            if len(focus_indices) == 0:
                continue
            flat_x = features[focus_indices].reshape(-1, features.shape[-1])
            flat_actions = actions[focus_indices].reshape(-1)
            repeated_answers = np.repeat(answers[focus_indices], actions.shape[1])
            repeated_hybrid = np.repeat(hybrid[focus_indices], actions.shape[1])
            target = (flat_actions == repeated_answers).astype(np.float64) - (
                repeated_hybrid == repeated_answers
            ).astype(np.float64)
            for l2 in ridges:
                penalty = float(l2) * np.eye(flat_x.shape[1], dtype=np.float64)
                penalty[0, 0] = 0.0
                lhs = flat_x.T @ flat_x + penalty
                rhs = flat_x.T @ target
                try:
                    weights = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError:
                    weights = np.linalg.pinv(lhs) @ rhs
                dev_scores = features[dev_indices] @ weights
                max_scores = np.max(dev_scores, axis=1)
                thresholds = sorted(set(float(item) for item in max_scores))
                noop_threshold = float(np.max(max_scores) + max(1e-9, abs(float(np.max(max_scores))) * 1e-6))
                thresholds.append(noop_threshold)
                if len(thresholds) > 80:
                    finite = thresholds[:-1]
                    thresholds = sorted(
                        {finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 61)}
                    )
                    thresholds.append(noop_threshold)
                for threshold in thresholds:
                    model = {
                        "scheme": scheme_name,
                        "scheme_keys": list(scheme_keys),
                        "feature_spec": spec,
                        "focus": focus,
                        "focus_fit_rows": int(len(focus_indices)),
                        "l2": float(l2),
                        "threshold": float(threshold),
                        "threshold_is_noop": bool(threshold == noop_threshold),
                        "weights": weights.tolist(),
                    }
                    predictions = _predict_receiver(
                        actions=actions[dev_indices],
                        fields={key: value[dev_indices] for key, value in fields.items()},
                        hybrid=hybrid[dev_indices],
                        model=model,
                    )
                    metrics = _quick_evaluate_predictions(
                        predictions=predictions,
                        baseline=hybrid[dev_indices],
                        answers=answers[dev_indices],
                    )
                    if int(metrics["override_count"]) == 0:
                        continue
                    row = {
                        "scheme": scheme_name,
                        "scheme_keys": ",".join(scheme_keys),
                        "focus": focus,
                        "focus_fit_rows": int(len(focus_indices)),
                        "l2": float(l2),
                        "threshold": float(threshold),
                        "threshold_is_noop": bool(threshold == noop_threshold),
                        "official_dev_accuracy": metrics["accuracy"],
                        "official_dev_delta_vs_hybrid": metrics["delta"],
                        "official_dev_ci95_low_vs_hybrid": metrics["ci95_low"],
                        "official_dev_ci95_high_vs_hybrid": metrics["ci95_high"],
                        "official_dev_helps_vs_hybrid": metrics["helps"],
                        "official_dev_harms_vs_hybrid": metrics["harms"],
                        "official_dev_net_help": metrics["net_help"],
                        "official_dev_override_count": metrics["override_count"],
                        "official_dev_accepted_harm_rate": metrics["accepted_harm_rate"],
                    }
                    config_rows.append(row)
                    key = (
                        float(metrics["accuracy"]),
                        float(metrics["delta"]),
                        float(metrics["ci95_low"]),
                        float(metrics["net_help"]),
                        -float(metrics["harms"]),
                        json.dumps(row, sort_keys=True),
                    )
                    if key > best[0]:
                        best = (key, model)
    return best[1], sorted(config_rows, key=lambda row: row["official_dev_accuracy"], reverse=True)


def _method_row(
    *,
    name: str,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    bootstrap_samples: int,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    source_score_or_logit_vector_exposed: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return top2_gate._method_row(
        name=name,
        rows=rows,
        predictions=predictions,
        fixed_hybrid=fixed_hybrid,
        candidate_only=candidate_only,
        target_only=target_only,
        bootstrap_samples=bootstrap_samples,
        raw_payload_bytes=raw_payload_bytes,
        framed_record_bytes=framed_record_bytes,
        source_score_or_logit_vector_exposed=source_score_or_logit_vector_exposed,
        details=details,
    )


def _oracle_from_candidates(fallback: np.ndarray, candidate_sets: np.ndarray, answers: np.ndarray) -> np.ndarray:
    predictions = np.asarray(fallback, dtype=np.int64).copy()
    for index, answer in enumerate(np.asarray(answers, dtype=np.int64)):
        if int(answer) in {int(item) for item in candidate_sets[index]}:
            predictions[index] = int(answer)
    return predictions.astype(np.int64)


def _slice_rows(
    *,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    return top2_gate._slice_rows(
        rows=rows,
        predictions=predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Error-Conditioned Syndrome Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- error-conditioned accuracy: `{h['error_conditioned_syndrome_accuracy']:.6f}`",
        f"- error-conditioned delta: `{h['error_conditioned_syndrome_delta_vs_fixed_hybrid']:.6f}`",
        f"- error-conditioned CI95 low: `{h['error_conditioned_syndrome_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- overrides / helps / harms: `{h['error_conditioned_syndrome_override_count']} / {h['error_conditioned_syndrome_helps_vs_fixed_hybrid']} / {h['error_conditioned_syndrome_harms_vs_fixed_hybrid']}`",
        f"- fixed-or-source-top2 oracle accuracy: `{h['fixed_hybrid_or_qwen_top2_oracle_accuracy']:.6f}`",
        f"- best destructive: `{h['best_destructive_control_name']}` (`{h['best_destructive_control_accuracy']:.6f}`)",
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
    phi_train_score_cache: pathlib.Path | str = DEFAULT_PHI_TRAIN_SCORE_CACHE,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    component_ridges: tuple[float, ...] = DEFAULT_COMPONENT_RIDGES,
    receiver_ridges: tuple[float, ...] = DEFAULT_RECEIVER_RIDGES,
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
    calibration = receiver_gate._limit_calibration(calibration, max_calibration_rows)
    fit_indices, dev_indices = official._official_split_indices(
        len(calibration["rows"]), dev_fraction=dev_fraction, seed=4242
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
    bins = top2_gate._fit_bins(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_margin=calibration["margin"],
        fit_indices=fit_indices,
    )
    train_actions, train_fields, train_diag = top2_gate._action_fields(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
        bins=bins,
    )
    model, config_rows = _fit_receiver(
        actions=train_actions,
        fields=train_fields,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        phi_top1=np.asarray(phi_predictions, dtype=np.int64),
        source_top1=train_diag["source_top1"],
        source_top2=train_diag["source_top2"],
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
    eval_phi_scores = np.asarray([row["phi_target_scores"] for row in eval_rows], dtype=np.float64)
    eval_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    eval_mean = _field_array(eval_rows, "selected_prediction")
    eval_margin = np.asarray([float(row.get("selected_margin", 0.0)) for row in eval_rows], dtype=np.float64)
    fixed_hybrid = eval_hybrid.copy()
    candidate_only = eval_mean.copy()
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    eval_actions, eval_fields, eval_diag = top2_gate._action_fields(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
        bins=bins,
    )
    selected_predictions = _predict_receiver(actions=eval_actions, fields=eval_fields, hybrid=eval_hybrid, model=model)
    source_top1 = eval_diag["source_top1"]
    source_top2 = eval_diag["source_top2"]
    phi_top1, phi_top2 = _top2(eval_phi_scores)
    qwen_top2 = np.stack([source_top1, source_top2], axis=1)
    phi_top2_set = np.stack([phi_top1, phi_top2], axis=1)
    source_pair_oracle = _oracle_from_candidates(source_top1, qwen_top2, answers)
    fixed_or_qwen_top2_oracle = _oracle_from_candidates(fixed_hybrid, qwen_top2, answers)
    fixed_or_phi_top2_oracle = _oracle_from_candidates(fixed_hybrid, phi_top2_set, answers)
    fixed_or_union_top2_oracle = _oracle_from_candidates(
        fixed_hybrid, np.concatenate([qwen_top2, phi_top2_set], axis=1), answers
    )
    oracle_help_count = int(
        np.sum((fixed_or_qwen_top2_oracle == answers).astype(int) - (fixed_hybrid == answers).astype(int) > 0)
    )
    no_syndrome_actions, no_syndrome_fields, _ = top2_gate._action_fields(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
        bins=bins,
        zero_source_bins=True,
    )
    no_syndrome_predictions = _predict_receiver(
        actions=no_syndrome_actions,
        fields=no_syndrome_fields,
        hybrid=eval_hybrid,
        model=model,
    )
    label_rng = np.random.default_rng(20260504)
    permuted_answers = calibration["answers"][label_rng.permutation(len(calibration["answers"]))]
    label_model, _ = _fit_receiver(
        actions=train_actions,
        fields=train_fields,
        hybrid=calibration["hybrid"],
        answers=permuted_answers,
        phi_top1=np.asarray(phi_predictions, dtype=np.int64),
        source_top1=train_diag["source_top1"],
        source_top2=train_diag["source_top2"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=receiver_ridges,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    label_predictions = _predict_receiver(actions=eval_actions, fields=eval_fields, hybrid=eval_hybrid, model=label_model)
    method_rows = [
        _method_row(
            name="error_conditioned_syndrome_packet",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            details={"model": {key: value for key, value in model.items() if key not in {"weights", "feature_spec"}}},
        ),
        _method_row(
            name="source_pair_no_syndrome_receiver_control",
            rows=eval_rows,
            predictions=no_syndrome_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            details={"source_top1_top2_visible": True, "source_syndrome_bins_zeroed": True},
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
            name="source_top1_label_control",
            rows=eval_rows,
            predictions=source_top1,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            details={"source_rank": 1},
        ),
        _method_row(
            name="source_top2_label_control",
            rows=eval_rows,
            predictions=source_top2,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            details={"source_rank": 2},
        ),
        _method_row(
            name="source_top1_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=source_pair_oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
        _method_row(
            name="fixed_hybrid_or_qwen_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=fixed_or_qwen_top2_oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
        _method_row(
            name="fixed_hybrid_or_phi_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=fixed_or_phi_top2_oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
        _method_row(
            name="fixed_hybrid_or_union_top2_oracle_diagnostic",
            rows=eval_rows,
            predictions=fixed_or_union_top2_oracle,
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
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    if weights.size:
        features = _encode_features(eval_fields, model["feature_spec"])
        scores = features @ weights
        max_scores = np.max(scores, axis=1)
        eval_diag_best: tuple[tuple[float, float, int, str], np.ndarray] | None = None
        for threshold in sorted(set(float(item) for item in max_scores)):
            diag_model = dict(model, threshold=float(threshold))
            diag_predictions = _predict_receiver(actions=eval_actions, fields=eval_fields, hybrid=eval_hybrid, model=diag_model)
            overrides = int(np.sum(diag_predictions != fixed_hybrid))
            if overrides == 0:
                continue
            delta = float(
                np.mean((diag_predictions == answers).astype(float) - (fixed_hybrid == answers).astype(float))
            )
            key = (_accuracy(diag_predictions, answers), delta, -overrides, str(threshold))
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
    for condition in SOURCE_CONDITIONS:
        c_scores, c_hybrid, c_mean, c_margin = top2_gate._source_control_inputs(
            qwen_scores=eval_scores,
            phi_scores=eval_phi_scores,
            hybrid=eval_hybrid,
            mean=eval_mean,
            margin=eval_margin,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        c_actions, c_fields, _ = top2_gate._action_fields(
            qwen_scores=c_scores,
            phi_scores=eval_phi_scores,
            hybrid=c_hybrid,
            qwen_mean=c_mean,
            qwen_margin=c_margin,
            bins=bins,
        )
        control_predictions = _predict_receiver(actions=c_actions, fields=c_fields, hybrid=c_hybrid, model=model)
        method_rows.append(
            _method_row(
                name=f"{condition}_receiver_control",
                rows=eval_rows,
                predictions=control_predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=2,
                framed_record_bytes=5,
                details={"condition": condition},
            )
        )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "error_conditioned_syndrome_packet")
    no_syndrome_row = next(row for row in method_rows if row["method"] == "source_pair_no_syndrome_receiver_control")
    source_top1_row = next(row for row in method_rows if row["method"] == "source_top1_label_control")
    destructive_rows = [
        row
        for row in method_rows
        if row["method"].endswith("_control")
        and row["method"]
        not in {
            "source_pair_no_syndrome_receiver_control",
            "source_top1_label_control",
            "source_top2_label_control",
        }
    ]
    best_destructive = max(destructive_rows, key=lambda row: row["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    source_unique_rows = int(
        np.sum(
            (fixed_hybrid != answers)
            & ((source_top1 == answers) | (source_top2 == answers))
            & ~((phi_top1 == answers) | (phi_top2 == answers))
        )
    )
    train_content_ids = {row.content_id for row in calibration_arc_rows}
    eval_content_ids = {str(row.get("content_id", row["row_id"])) for row in eval_rows}
    source_index_margin = float(method_row["accuracy"] - max(source_top1_row["accuracy"], no_syndrome_row["accuracy"]))
    pass_gate = bool(
        method_row["delta_vs_fixed_hybrid"] >= 0.005
        and method_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and method_row["ci95_low_vs_candidate_only"] > 0.0
        and method_row["helps_vs_fixed_hybrid"] > method_row["harms_vs_fixed_hybrid"]
        and method_row["harms_vs_fixed_hybrid"] <= max(1, int(math.floor(0.25 * method_row["helps_vs_fixed_hybrid"])))
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and method_row["accuracy"] > best_destructive["accuracy"]
        and source_index_margin >= 0.005
        and len(train_content_ids & eval_content_ids) == 0
    )
    selected_dev = next(
        (
            row
            for row in config_rows
            if row["scheme"] == model.get("scheme")
            and row["focus"] == model.get("focus")
            and row["l2"] == model.get("l2")
            and row["threshold"] == model.get("threshold")
        ),
        {},
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "selected_scheme": model["scheme"],
        "selected_focus": model.get("focus", "no_op"),
        "selected_l2": model.get("l2"),
        "selected_threshold": model.get("threshold"),
        "official_dev_selected_accuracy": selected_dev.get(
            "official_dev_accuracy", _accuracy(calibration["hybrid"][dev_indices], calibration["answers"][dev_indices])
        ),
        "official_dev_selected_delta_vs_hybrid": selected_dev.get("official_dev_delta_vs_hybrid", 0.0),
        "official_dev_selected_ci95_low_vs_hybrid": selected_dev.get("official_dev_ci95_low_vs_hybrid", 0.0),
        "eval_rows": len(eval_rows),
        "fixed_hybrid_accuracy": next(row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement")[
            "accuracy"
        ],
        "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")[
            "accuracy"
        ],
        "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")["accuracy"],
        "error_conditioned_syndrome_accuracy": method_row["accuracy"],
        "error_conditioned_syndrome_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "error_conditioned_syndrome_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "error_conditioned_syndrome_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "error_conditioned_syndrome_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "error_conditioned_syndrome_override_count": method_row["override_count_vs_fixed_hybrid"],
        "source_pair_no_syndrome_accuracy": no_syndrome_row["accuracy"],
        "source_top1_accuracy": source_top1_row["accuracy"],
        "source_index_margin": source_index_margin,
        "source_unique_top2_repair_rows": source_unique_rows,
        "fixed_hybrid_or_qwen_top2_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "fixed_hybrid_or_qwen_top2_oracle_diagnostic"
        )["accuracy"],
        "fixed_hybrid_or_phi_top2_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "fixed_hybrid_or_phi_top2_oracle_diagnostic"
        )["accuracy"],
        "fixed_hybrid_or_union_top2_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "fixed_hybrid_or_union_top2_oracle_diagnostic"
        )["accuracy"],
        "oracle_headroom_capture": float(method_row["helps_vs_fixed_hybrid"] / max(1, oracle_help_count)),
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_error_conditioned_syndrome_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the official-train selected error-conditioned syndrome packet beats fixed hybrid by "
            "at least 0.005 with positive paired CI, beats candidate-only with positive paired CI, helps more "
            "than harms with harm <= 25% of helps, is nonnegative on all eval slices, beats destructive "
            "controls, beats source-index/no-syndrome controls by at least 0.005, and has zero train/eval "
            "content overlap."
        ),
        "headline": headline,
        "packet_contract": {
            "receiver_visible_payload": (
                "Qwen source top-1/top-2 candidate IDs plus quantized source-side syndrome bins; "
                "Phi-local score bins are decoder side information."
            ),
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "source_packet_fields": [
                "source_top1_candidate_id",
                "source_top2_candidate_id",
                "qwen_margin_bin",
                "qwen_entropy_bin",
                "qwen_hybrid_gap_bin",
                "qwen_selected_margin_bin",
            ],
            "receiver_local_fields": [
                "phi_top1_candidate_id",
                "phi_top2_candidate_id",
                "phi_margin_bin",
                "phi_entropy_bin",
                "phi_hybrid_gap_bin",
                "phi_action_advantage_bin",
            ],
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "raw_qwen_scores_used_only_for_source_side_quantized_packet": True,
        },
        "calibration_row_metadata": calibration_row_meta,
        "quantization_bins": bins,
        "source_score_metadata": source_score_metadata,
        "slice_metadata": metadata,
        "sample_cache_rows": calibration["sample_cache_rows"],
        "component_rows": calibration["component_rows"],
        "method_rows": method_rows,
        "config_row_count": int(len(config_rows)),
        "top_config_rows": config_rows[:100],
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
            "train_path": _display_path(train_path),
            "train_sha256": official._sha256_file(train_path),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "phi_train_score_cache": _display_path(phi_cache_path),
            "phi_train_score_cache_sha256": phi_sha,
            "phi_train_score_model": phi_state,
            "source_score_cache": denoise._display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate directly tests the branch promoted by the target-error repair audit: train on official "
            "rows where the fixed packet tends to fail and ask whether quantized source top-2 syndrome fields "
            "can safely trigger repairs on held-out Qwen-to-Phi HellaSwag. A negative result weakens "
            "score/top2 syndrome repair and promotes learned target-resonance soft-prefix encoders as the next "
            "highest-value branch."
        ),
        "lay_explanation": (
            "Qwen sends Phi only a tiny hint: its two favorite answer choices and a few coarse confidence bins. "
            "The receiver practices on training questions to decide when that hint means Phi should change the "
            "existing packet answer. The final test checks whether this rule improves held-out questions or just "
            "creates new mistakes."
        ),
    }
    _write_json(output_dir / "hellaswag_qwen_to_phi_error_conditioned_syndrome_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_error_conditioned_syndrome_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_error_conditioned_syndrome_gate.json",
                "hellaswag_qwen_to_phi_error_conditioned_syndrome_gate.md",
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
    parser.add_argument("--receiver-ridges", type=official._parse_float_tuple, default=DEFAULT_RECEIVER_RIDGES)
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
                "selected_scheme": h["selected_scheme"],
                "selected_focus": h["selected_focus"],
                "official_dev_selected_delta_vs_hybrid": h["official_dev_selected_delta_vs_hybrid"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "error_conditioned_syndrome_accuracy": h["error_conditioned_syndrome_accuracy"],
                "error_conditioned_syndrome_delta_vs_fixed_hybrid": h[
                    "error_conditioned_syndrome_delta_vs_fixed_hybrid"
                ],
                "best_destructive_control_accuracy": h["best_destructive_control_accuracy"],
                "fixed_hybrid_or_qwen_top2_oracle_accuracy": h["fixed_hybrid_or_qwen_top2_oracle_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
