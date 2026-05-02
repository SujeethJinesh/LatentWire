from __future__ import annotations

"""HellaSwag switch-observability gate for the packet/Qwen surface.

This artifact is diagnostic, not a promoted method. It asks whether the current
source packet/score and Qwen score surface contains a learnable signal for
when switching from the source packet to a Qwen alternative helps rather than
harms. The goal is to decide whether more Mac-local selector/source-code tuning
is scientifically justified.
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_conditional_selector_syndrome_gate as linear  # noqa: E402
from scripts import build_source_private_hellaswag_nonlinear_selector_syndrome_gate as nonlinear  # noqa: E402
from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_switch_observability_gate_20260502")
DEFAULT_RIDGE = 10.0
DEFAULT_RFF_COMPONENTS = 64
DEFAULT_RFF_GAMMA = 1.0
DEFAULT_RFF_SEED = 19
STRICT_DELTA = 0.010
STRICT_AUC = 0.600
STRICT_HEADROOM_CAPTURE = 0.100
SCOUT_DELTA = 0.005
CANDIDATE_COUNT = 4


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


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return linear._accuracy(predictions, answers)


def _safe_div(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values / np.where(np.abs(scale) < 1e-8, 1.0, scale)


def _row_zscores(scores: np.ndarray) -> np.ndarray:
    centered = scores.astype(np.float64) - np.mean(scores.astype(np.float64), axis=1, keepdims=True)
    return _safe_div(centered, np.std(centered, axis=1, keepdims=True))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores.astype(np.float64) - np.max(scores.astype(np.float64), axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores.astype(np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _candidate_rank(zscores: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    order = np.argsort(-zscores, axis=1)
    ranks = np.zeros(len(candidate), dtype=np.float64)
    for row_index, row_order in enumerate(order):
        ranks[row_index] = float(np.where(row_order == int(candidate[row_index]))[0][0])
    return ranks / float(CANDIDATE_COUNT - 1)


def _one_hot(values: np.ndarray, width: int = CANDIDATE_COUNT) -> np.ndarray:
    values = values.astype(np.int64)
    eye = np.eye(int(width), dtype=np.float64)
    return eye[np.mod(values, int(width))]


def _score_features(scores: np.ndarray, packet: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    z = _row_zscores(scores)
    probs = _softmax(scores)
    packet = packet.astype(np.int64)
    row_index = np.arange(len(packet), dtype=np.int64)
    top = np.argmax(z, axis=1).astype(np.int64)
    features = [
        z,
        probs,
        z[row_index, packet][:, None],
        probs[row_index, packet][:, None],
        _top2_margin(z)[:, None],
        _candidate_rank(z, packet)[:, None],
        _entropy(probs)[:, None],
        _one_hot(top),
    ]
    names = (
        [f"{prefix}_z_{idx}" for idx in range(CANDIDATE_COUNT)]
        + [f"{prefix}_prob_{idx}" for idx in range(CANDIDATE_COUNT)]
        + [
            f"{prefix}_packet_z",
            f"{prefix}_packet_prob",
            f"{prefix}_top2_margin",
            f"{prefix}_packet_rank",
            f"{prefix}_entropy",
        ]
        + [f"{prefix}_top_onehot_{idx}" for idx in range(CANDIDATE_COUNT)]
    )
    return np.concatenate(features, axis=1).astype(np.float64), names


def _feature_matrix(
    *,
    view: str,
    source_scores: np.ndarray,
    qwen_scores: np.ndarray,
    packet: np.ndarray,
    alternative: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    packet = packet.astype(np.int64)
    alternative = alternative.astype(np.int64)
    packet_alt = [
        _one_hot(packet),
        _one_hot(alternative),
        (packet == alternative)[:, None].astype(np.float64),
        ((alternative - packet) % CANDIDATE_COUNT)[:, None].astype(np.float64) / float(
            CANDIDATE_COUNT - 1
        ),
    ]
    packet_alt_names = (
        [f"packet_onehot_{idx}" for idx in range(CANDIDATE_COUNT)]
        + [f"alternative_onehot_{idx}" for idx in range(CANDIDATE_COUNT)]
        + ["packet_equals_alternative", "alternative_minus_packet_mod"]
    )
    source_features, source_names = _score_features(source_scores, packet, "source")
    qwen_features, qwen_names = _score_features(qwen_scores, packet, "qwen")
    parts: list[np.ndarray] = packet_alt.copy()
    names = packet_alt_names.copy()
    if view == "source_score_only":
        parts.append(source_features)
        names.extend(source_names)
    elif view == "qwen_score_only":
        parts.append(qwen_features)
        names.extend(qwen_names)
    elif view == "source_plus_qwen":
        parts.extend([source_features, qwen_features, source_features * qwen_features])
        names.extend(source_names)
        names.extend(qwen_names)
        names.extend([f"source_x_qwen_{idx}" for idx in range(source_features.shape[1])])
    elif view == "packet_id_only":
        pass
    else:
        raise ValueError(f"unsupported feature view: {view}")
    return np.concatenate(parts, axis=1).astype(np.float64), names


def _benefit_targets(*, packet: np.ndarray, alternative: np.ndarray, answers: np.ndarray) -> np.ndarray:
    helps = (alternative.astype(np.int64) == answers.astype(np.int64)) & (
        packet.astype(np.int64) != answers.astype(np.int64)
    )
    harms = (alternative.astype(np.int64) != answers.astype(np.int64)) & (
        packet.astype(np.int64) == answers.astype(np.int64)
    )
    return helps.astype(np.float64) - harms.astype(np.float64)


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(np.sum(positives))
    neg_count = int(np.sum(negatives))
    if pos_count == 0 or neg_count == 0:
        return None
    ranks = _average_tie_ranks(scores.astype(np.float64))
    auc = (float(np.sum(ranks[positives])) - pos_count * (pos_count + 1) / 2.0) / (
        pos_count * neg_count
    )
    return float(auc)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    pos_count = int(np.sum(labels == 1))
    if pos_count == 0:
        return None
    order = np.argsort(-scores.astype(np.float64), kind="mergesort")
    sorted_labels = labels[order]
    true_positive_cumsum = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    precision = true_positive_cumsum / ranks
    return float(np.sum(precision[sorted_labels == 1]) / pos_count)


def _decisive_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    decisive = targets != 0.0
    if not np.any(decisive):
        return {
            "decisive_rows": 0,
            "help_rows": 0,
            "harm_rows": 0,
            "auc_help_vs_harm": None,
            "average_precision_help": None,
            "mean_score_help": None,
            "mean_score_harm": None,
        }
    labels = (targets[decisive] > 0.0).astype(np.int64)
    decisive_scores = scores[decisive].astype(np.float64)
    helps = labels == 1
    harms = labels == 0
    return {
        "decisive_rows": int(np.sum(decisive)),
        "help_rows": int(np.sum(helps)),
        "harm_rows": int(np.sum(harms)),
        "auc_help_vs_harm": _roc_auc(decisive_scores, labels),
        "average_precision_help": _average_precision(decisive_scores, labels),
        "mean_score_help": float(np.mean(decisive_scores[helps])) if np.any(helps) else None,
        "mean_score_harm": float(np.mean(decisive_scores[harms])) if np.any(harms) else None,
    }


def _fit_linear_score(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    targets: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_std, eval_std, standardizer = nonlinear._standardize_fit_eval(
        train_features,
        eval_features,
        fit_indices,
    )
    train_with_bias = np.concatenate(
        [np.ones((train_std.shape[0], 1), dtype=np.float64), train_std],
        axis=1,
    )
    eval_with_bias = np.concatenate(
        [np.ones((eval_std.shape[0], 1), dtype=np.float64), eval_std],
        axis=1,
    )
    coef = linear._fit_ridge(train_with_bias, targets, fit_indices, float(ridge))
    return (
        linear._predict_score(train_with_bias, coef),
        linear._predict_score(eval_with_bias, coef),
        {
            "standardizer": standardizer,
            "coef": coef,
            "feature_dim": int(train_with_bias.shape[1]),
            "model_public_resident_bytes": int(
                standardizer["center"].nbytes + standardizer["scale"].nbytes + coef.nbytes
            ),
        },
    )


def _fit_rff_score(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    targets: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
    rff_components: int,
    rff_gamma: float,
    rff_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_std, eval_std, standardizer = nonlinear._standardize_fit_eval(
        train_features,
        eval_features,
        fit_indices,
    )
    params = nonlinear._rff_parameters(
        train_std.shape[1],
        components=int(rff_components),
        gamma=float(rff_gamma),
        seed=int(rff_seed),
    )
    train_nl = nonlinear._nonlinear_feature_matrix(train_std, params)
    eval_nl = nonlinear._nonlinear_feature_matrix(eval_std, params)
    coef = linear._fit_ridge(train_nl, targets, fit_indices, float(ridge))
    model_bytes = nonlinear._model_resident_bytes(
        standardizer=standardizer,
        params=params,
        coef=coef,
    )
    return (
        linear._predict_score(train_nl, coef),
        linear._predict_score(eval_nl, coef),
        {
            "standardizer": standardizer,
            "rff_params": params,
            "coef": coef,
            "feature_dim": int(train_nl.shape[1]),
            "model_public_resident_bytes": int(model_bytes),
        },
    )


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.asarray([0.0], dtype=np.float64)
    quantiles = np.quantile(scores.astype(np.float64), np.linspace(0.0, 1.0, 101))
    return np.unique(np.concatenate([quantiles, np.asarray([-0.3, 0.0, 0.3], dtype=np.float64)]))


def _select_threshold(
    *,
    scores: np.ndarray,
    packet: np.ndarray,
    alternative: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in _threshold_candidates(scores[indices]):
        predictions = linear._threshold_predictions(
            packet=packet,
            alternative=alternative,
            benefit_scores=scores,
            threshold=float(threshold),
        )
        selected_acc = _accuracy(predictions[indices], answers[indices])
        packet_acc = _accuracy(packet[indices], answers[indices])
        override_rate = float(np.mean(predictions[indices] != packet[indices]))
        helps = int(np.sum((predictions[indices] == answers[indices]) & (packet[indices] != answers[indices])))
        harms = int(np.sum((predictions[indices] != answers[indices]) & (packet[indices] == answers[indices])))
        row = {
            "threshold": float(threshold),
            "accuracy": selected_acc,
            "delta_vs_packet": selected_acc - packet_acc,
            "override_rate_vs_packet": override_rate,
            "help_count": helps,
            "harm_count": harms,
            "net_help": helps - harms,
        }
        if best is None or (
            row["delta_vs_packet"],
            row["net_help"],
            -row["override_rate_vs_packet"],
        ) > (
            best["delta_vs_packet"],
            best["net_help"],
            -best["override_rate_vs_packet"],
        ):
            best = row
    if best is None:
        raise RuntimeError("no threshold candidates")
    return best


def _evaluate_view(
    *,
    view: str,
    model: str,
    alternative_name: str,
    surfaces: dict[str, Any],
    train_alternative: np.ndarray,
    eval_alternative: np.ndarray,
    ridge: float,
    rff_components: int,
    rff_gamma: float,
    rff_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    train_features, feature_names = _feature_matrix(
        view=view,
        source_scores=surfaces["tiny_train_scores"],
        qwen_scores=calibration["qwen_scores"],
        packet=calibration["tiny_packet"],
        alternative=train_alternative,
    )
    eval_features, _ = _feature_matrix(
        view=view,
        source_scores=surfaces["tiny_eval_scores"],
        qwen_scores=validation["qwen_scores"],
        packet=validation["packet"],
        alternative=eval_alternative,
    )
    train_targets = _benefit_targets(
        packet=calibration["tiny_packet"],
        alternative=train_alternative,
        answers=calibration["answers"],
    )
    eval_targets = _benefit_targets(
        packet=validation["packet"],
        alternative=eval_alternative,
        answers=validation["answers"],
    )
    if model == "linear":
        train_scores, eval_scores, state = _fit_linear_score(
            train_features=train_features,
            eval_features=eval_features,
            targets=train_targets,
            fit_indices=fit_indices,
            ridge=float(ridge),
        )
    elif model == "rff":
        train_scores, eval_scores, state = _fit_rff_score(
            train_features=train_features,
            eval_features=eval_features,
            targets=train_targets,
            fit_indices=fit_indices,
            ridge=float(ridge),
            rff_components=int(rff_components),
            rff_gamma=float(rff_gamma),
            rff_seed=int(rff_seed),
        )
    else:
        raise ValueError(f"unsupported model: {model}")
    selected_threshold = _select_threshold(
        scores=train_scores,
        packet=calibration["tiny_packet"],
        alternative=train_alternative,
        answers=calibration["answers"],
        indices=dev_indices,
    )
    oracle_eval_threshold = _select_threshold(
        scores=eval_scores,
        packet=validation["packet"],
        alternative=eval_alternative,
        answers=validation["answers"],
        indices=np.arange(len(validation["answers"]), dtype=np.int64),
    )
    predictions = linear._threshold_predictions(
        packet=validation["packet"],
        alternative=eval_alternative,
        benefit_scores=eval_scores,
        threshold=float(selected_threshold["threshold"]),
    )
    scored = linear._score_row(
        name=f"{view}_{model}",
        predictions=predictions,
        answers=validation["answers"],
        packet=validation["packet"],
        target=validation["alternatives"]["qwen_target_score"],
        seed=33001 + len(view) * 17 + len(model),
        bootstrap_samples=bootstrap_samples,
        extra={
            "view": view,
            "model": model,
            "alternative": alternative_name,
            "ridge": float(ridge),
            "rff_components": int(rff_components) if model == "rff" else 0,
            "rff_gamma": float(rff_gamma) if model == "rff" else 0.0,
            "rff_seed": int(rff_seed) if model == "rff" else 0,
            "feature_dim": int(state["feature_dim"]),
            "source_feature_count": int(train_features.shape[1]),
            "model_public_resident_bytes": int(state["model_public_resident_bytes"]),
            "official_dev_threshold": float(selected_threshold["threshold"]),
            "official_dev_delta_vs_packet": float(selected_threshold["delta_vs_packet"]),
            "official_dev_help_count": int(selected_threshold["help_count"]),
            "official_dev_harm_count": int(selected_threshold["harm_count"]),
            "official_dev_override_rate_vs_packet": float(selected_threshold["override_rate_vs_packet"]),
            "validation_oracle_threshold": float(oracle_eval_threshold["threshold"]),
            "validation_oracle_threshold_delta_vs_packet": float(
                oracle_eval_threshold["delta_vs_packet"]
            ),
            "validation_oracle_threshold_help_count": int(oracle_eval_threshold["help_count"]),
            "validation_oracle_threshold_harm_count": int(oracle_eval_threshold["harm_count"]),
            "feature_names": feature_names,
            "fit_observability": _decisive_metrics(train_scores[fit_indices], train_targets[fit_indices]),
            "dev_observability": _decisive_metrics(train_scores[dev_indices], train_targets[dev_indices]),
            "validation_observability": _decisive_metrics(eval_scores, eval_targets),
        },
    )
    return scored


def _baseline_rows(
    *,
    surfaces: dict[str, Any],
    alternatives: dict[str, tuple[np.ndarray, np.ndarray]],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    return linear._baseline_rows(
        surfaces=surfaces,
        alternatives=alternatives,
        bootstrap_samples=bootstrap_samples,
    )


def _top_rows(rows: list[dict[str, Any]], *, key: str, limit: int) -> list[dict[str, Any]]:
    return [linear._strip_large(row) for row in sorted(rows, key=lambda row: row[key], reverse=True)[:limit]]


def _top_rows_by_auc(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        linear._strip_large(row)
        for row in sorted(
            rows,
            key=lambda row: (
                row["validation_observability"]["auc_help_vs_harm"]
                if row["validation_observability"]["auc_help_vs_harm"] is not None
                else -1.0,
                row["delta_vs_packet_only"],
            ),
            reverse=True,
        )[:limit]
    ]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Switch Observability Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- default row: `{h['default_row_name']}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- default validation AUC help-vs-harm: `{h['default_validation_auc_help_vs_harm']}`",
        f"- best validation-oracle threshold delta: `{h['best_validation_oracle_threshold_delta_vs_packet']:.6f}`",
        f"- best diagnostic AUC help-vs-harm: `{h['best_validation_auc_help_vs_harm']}`",
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
    ridge: float = DEFAULT_RIDGE,
    rff_components: int = DEFAULT_RFF_COMPONENTS,
    rff_gamma: float = DEFAULT_RFF_GAMMA,
    rff_seed: int = DEFAULT_RFF_SEED,
    bootstrap_samples: int = 500,
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
    rows: list[dict[str, Any]] = []
    for alternative_name, (train_alt, eval_alt) in alternatives.items():
        for view in ("packet_id_only", "source_score_only", "qwen_score_only", "source_plus_qwen"):
            for model in ("linear", "rff"):
                rows.append(
                    _evaluate_view(
                        view=view,
                        model=model,
                        alternative_name=alternative_name,
                        surfaces=surfaces,
                        train_alternative=train_alt,
                        eval_alternative=eval_alt,
                        ridge=float(ridge),
                        rff_components=int(rff_components),
                        rff_gamma=float(rff_gamma),
                        rff_seed=int(rff_seed),
                        bootstrap_samples=bootstrap_samples,
                    )
                )
    default_row = next(
        row
        for row in rows
        if row["view"] == "source_plus_qwen"
        and row["model"] == "rff"
        and row["alternative"] == "qwen_hybrid"
    )
    best_auc_row = max(
        rows,
        key=lambda row: (
            row["validation_observability"]["auc_help_vs_harm"]
            if row["validation_observability"]["auc_help_vs_harm"] is not None
            else -1.0,
            row["delta_vs_packet_only"],
        ),
    )
    best_oracle_threshold_row = max(
        rows,
        key=lambda row: (
            row["validation_oracle_threshold_delta_vs_packet"],
            row["validation_observability"]["auc_help_vs_harm"]
            if row["validation_observability"]["auc_help_vs_harm"] is not None
            else -1.0,
        ),
    )
    baselines = _baseline_rows(
        surfaces=surfaces,
        alternatives=alternatives,
        bootstrap_samples=bootstrap_samples,
    )
    oracle_delta = next(
        row["delta_vs_packet_only"] for row in baselines if row["name"] == "packet_or_any_qwen_oracle"
    )
    default_auc = default_row["validation_observability"]["auc_help_vs_harm"]
    best_auc = best_auc_row["validation_observability"]["auc_help_vs_harm"]
    default_headroom_capture = (
        float(default_row["delta_vs_packet_only"] / oracle_delta) if oracle_delta > 0.0 else 0.0
    )
    best_oracle_threshold_capture = (
        float(best_oracle_threshold_row["validation_oracle_threshold_delta_vs_packet"] / oracle_delta)
        if oracle_delta > 0.0
        else 0.0
    )
    pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and default_auc is not None
        and default_auc >= STRICT_AUC
        and default_headroom_capture >= STRICT_HEADROOM_CAPTURE
    )
    branch_kill_gate = bool(
        (best_auc is None or best_auc < STRICT_AUC)
        and best_oracle_threshold_row["validation_oracle_threshold_delta_vs_packet"] < SCOUT_DELTA
        and best_oracle_threshold_capture < STRICT_HEADROOM_CAPTURE
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["answers"])),
        "official_train_fit_rows": int(len(surfaces["fit_indices"])),
        "official_train_dev_rows": int(len(surfaces["dev_indices"])),
        "validation_rows": int(len(validation["answers"])),
        "packet_only_accuracy": _accuracy(validation["packet"], validation["answers"]),
        "qwen_hybrid_accuracy": _accuracy(
            validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            validation["answers"],
        ),
        "oracle_delta_vs_packet": oracle_delta,
        "default_row_name": str(default_row["name"]),
        "default_alternative": str(default_row["alternative"]),
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_validation_auc_help_vs_harm": default_auc,
        "default_validation_average_precision_help": default_row["validation_observability"][
            "average_precision_help"
        ],
        "default_oracle_headroom_capture": default_headroom_capture,
        "best_validation_auc_row_name": str(best_auc_row["name"]),
        "best_validation_auc_alternative": str(best_auc_row["alternative"]),
        "best_validation_auc_help_vs_harm": best_auc,
        "best_validation_auc_delta_vs_packet_only": best_auc_row["delta_vs_packet_only"],
        "best_validation_oracle_threshold_row_name": str(best_oracle_threshold_row["name"]),
        "best_validation_oracle_threshold_alternative": str(best_oracle_threshold_row["alternative"]),
        "best_validation_oracle_threshold_delta_vs_packet": best_oracle_threshold_row[
            "validation_oracle_threshold_delta_vs_packet"
        ],
        "best_validation_oracle_threshold_headroom_capture": best_oracle_threshold_capture,
        "pass_gate": pass_gate,
        "branch_kill_gate": branch_kill_gate,
        "strict_delta_required": STRICT_DELTA,
        "strict_auc_required": STRICT_AUC,
    }
    lay_explanation = (
        "This experiment does not try to make a new model answer better. It checks whether the "
        "available signals contain enough information to tell when Qwen should overrule the "
        "TinyLlama packet. If that signal is not visible even to simple diagnostic learners, then "
        "more selector tuning is unlikely to produce an ICLR-strength method."
    )
    interpretation = (
        "This gate is a decision surface for the HellaSwag branch. A pass would mean the current "
        "packet/Qwen score surface still has learnable switch information worth turning into a "
        "source-private code. A fail, especially with a weak validation-oracle threshold row, means "
        "the branch should stop tuning selectors and move to a different source representation or a "
        "true joint connector."
    )
    payload = {
        "gate": "source_private_hellaswag_switch_observability_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "branch_kill_gate": branch_kill_gate,
        "pass_rule": (
            "Pass if the predeclared source_plus_qwen RFF qwen_hybrid diagnostic beats packet-only "
            "by >=0.010 with positive CI95 low, validation AUC >=0.600 on help-vs-harm decisive "
            "rows, and >=10% oracle-headroom capture."
        ),
        "branch_kill_rule": (
            "Kill Mac-local selector/source-score tuning if the best validation AUC remains below "
            "the strict observability bar and the best validation-oracle threshold diagnostic "
            "captures <10% of oracle headroom with delta <0.005. Validation-oracle rows are "
            "diagnostic only and not promotable."
        ),
        "headline": headline,
        "baselines": baselines,
        "observability_rows": [linear._strip_large(row) for row in rows],
        "rows_top_by_delta": _top_rows(rows, key="delta_vs_packet_only", limit=30),
        "rows_top_by_validation_auc": _top_rows_by_auc(rows, limit=30),
        "default_row": linear._strip_large(default_row),
        "best_auc_row": linear._strip_large(best_auc_row),
        "best_oracle_threshold_row": linear._strip_large(best_oracle_threshold_row),
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
        "systems_boundary": {
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "diagnostic_only": True,
            "total_wall_time_s": float(time.perf_counter() - started),
        },
    }
    json_path = output_dir / "hellaswag_switch_observability_gate.json"
    md_path = output_dir / "hellaswag_switch_observability_gate.md"
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
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--rff-components", type=int, default=DEFAULT_RFF_COMPONENTS)
    parser.add_argument("--rff-gamma", type=float, default=DEFAULT_RFF_GAMMA)
    parser.add_argument("--rff-seed", type=int, default=DEFAULT_RFF_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        ridge=float(args.ridge),
        rff_components=int(args.rff_components),
        rff_gamma=float(args.rff_gamma),
        rff_seed=int(args.rff_seed),
        bootstrap_samples=int(args.bootstrap_samples),
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")
    print(f"branch_kill_gate={payload['branch_kill_gate']}")


if __name__ == "__main__":
    main()
