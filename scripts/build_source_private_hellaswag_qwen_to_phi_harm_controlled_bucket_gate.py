from __future__ import annotations

"""Harm-controlled bucket receiver for Qwen-to-Phi HellaSwag packets.

The previous receiver-calibrated gate nearly matched the fixed Qwen packet, but
it used raw source score features at the receiver despite the packet contract
disallowing raw score transmission. This gate tightens the interface: source
scores may only be used by the source-side encoder to emit quantized packet
fields. The receiver then combines those byte-scale fields with Phi-local score
features and accepts an override only in official-train buckets with calibrated
positive net benefit.
"""

import argparse
import csv
import datetime as dt
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

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as nonqwen  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_harm_controlled_bucket_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_PHI_TRAIN_SCORE_CACHE = (
    receiver_gate.DEFAULT_OUTPUT / "phi_official_train_score_cache.json"
)
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES

SCHEMES: dict[str, tuple[str, ...]] = {
    "role_agreement": (
        "role",
        "candidate_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "rival_eq_phi_top1",
        "q_top_relation",
    ),
    "compact_advantage": (
        "role",
        "candidate_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "q_rival_adv_bin",
        "phi_action_adv_bin",
    ),
    "compact_margin": (
        "role",
        "candidate_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "q_margin_bin",
        "phi_margin_bin",
    ),
    "source_receiver_bucket": (
        "role",
        "candidate_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "q_top_relation",
        "q_rival_adv_bin",
        "phi_action_adv_bin",
    ),
    "packet_candidate_bucket": (
        "role",
        "candidate",
        "hybrid",
        "candidate_eq_phi_top1",
        "q_rival_adv_bin",
        "phi_action_adv_bin",
    ),
    "mean_relation_bucket": (
        "role",
        "mean_relation",
        "candidate_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "q_margin_bin",
        "phi_action_adv_bin",
    ),
}


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


def _quantile_bins(values: np.ndarray, quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    bins = np.quantile(values, quantiles)
    return np.asarray(sorted(set(float(item) for item in bins)), dtype=np.float64)


def _digitize(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=np.float64), bins).astype(np.int64)


def _z_scores(scores: np.ndarray) -> np.ndarray:
    return receiver_gate._z_scores(scores)


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return receiver_gate._top2(scores)


def _pair_from_scores(scores: np.ndarray, hybrid: np.ndarray) -> np.ndarray:
    return source_gate._pair_from_scores(scores, hybrid)


def _base_signals(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_margin: np.ndarray,
) -> dict[str, np.ndarray]:
    qwen_scores = np.asarray(qwen_scores, dtype=np.float64)
    phi_scores = np.asarray(phi_scores, dtype=np.float64)
    hybrid = np.asarray(hybrid, dtype=np.int64)
    qwen_mean = np.asarray(qwen_mean, dtype=np.int64)
    qwen_margin = np.asarray(qwen_margin, dtype=np.float64)
    row_ids = np.arange(qwen_scores.shape[0])
    qz = _z_scores(qwen_scores)
    pz = _z_scores(phi_scores)
    q_top1, q_top2 = _top2(qwen_scores)
    p_top1, p_top2 = _top2(phi_scores)
    pair = _pair_from_scores(qwen_scores, hybrid)
    rival = pair[:, 1]
    q_margin = qwen_scores[row_ids, q_top1] - qwen_scores[row_ids, q_top2]
    p_margin = phi_scores[row_ids, p_top1] - phi_scores[row_ids, p_top2]
    q_rival_adv = qz[row_ids, rival] - qz[row_ids, hybrid]
    actions = np.stack([rival, p_top1], axis=1).astype(np.int64)
    p_action_adv = np.stack(
        [
            pz[row_ids, rival] - pz[row_ids, hybrid],
            pz[row_ids, p_top1] - pz[row_ids, hybrid],
        ],
        axis=1,
    )
    return {
        "hybrid": hybrid,
        "rival": rival.astype(np.int64),
        "phi_top1": p_top1.astype(np.int64),
        "qwen_mean": qwen_mean,
        "qwen_selected_margin": qwen_margin,
        "q_top1": q_top1.astype(np.int64),
        "q_top2": q_top2.astype(np.int64),
        "p_top1": p_top1.astype(np.int64),
        "p_top2": p_top2.astype(np.int64),
        "q_margin": q_margin,
        "p_margin": p_margin,
        "q_rival_adv": q_rival_adv,
        "p_action_adv": p_action_adv,
        "actions": actions,
    }


def _fit_bins(signals: dict[str, np.ndarray], fit_indices: np.ndarray) -> dict[str, list[float]]:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    return {
        "q_margin": _quantile_bins(signals["q_margin"][fit_indices]).tolist(),
        "q_rival_adv": _quantile_bins(signals["q_rival_adv"][fit_indices]).tolist(),
        "qwen_selected_margin": _quantile_bins(signals["qwen_selected_margin"][fit_indices]).tolist(),
        "p_margin": _quantile_bins(signals["p_margin"][fit_indices]).tolist(),
        "p_action_adv": _quantile_bins(signals["p_action_adv"][fit_indices].reshape(-1)).tolist(),
    }


def _action_fields(signals: dict[str, np.ndarray], bins: dict[str, list[float]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(signals["hybrid"])
    actions = np.asarray(signals["actions"], dtype=np.int64)
    role = np.repeat(np.asarray([[0, 1]], dtype=np.int64), n, axis=0)
    hybrid = np.repeat(signals["hybrid"][:, None], 2, axis=1).astype(np.int64)
    rival = np.repeat(signals["rival"][:, None], 2, axis=1).astype(np.int64)
    phi_top1 = np.repeat(signals["phi_top1"][:, None], 2, axis=1).astype(np.int64)
    q_top1 = np.repeat(signals["q_top1"][:, None], 2, axis=1).astype(np.int64)
    q_top2 = np.repeat(signals["q_top2"][:, None], 2, axis=1).astype(np.int64)
    q_margin_bin = np.repeat(
        _digitize(signals["q_margin"], np.asarray(bins["q_margin"], dtype=np.float64))[:, None],
        2,
        axis=1,
    )
    q_rival_adv_bin = np.repeat(
        _digitize(signals["q_rival_adv"], np.asarray(bins["q_rival_adv"], dtype=np.float64))[:, None],
        2,
        axis=1,
    )
    selected_margin_bin = np.repeat(
        _digitize(
            signals["qwen_selected_margin"],
            np.asarray(bins["qwen_selected_margin"], dtype=np.float64),
        )[:, None],
        2,
        axis=1,
    )
    phi_margin_bin = np.repeat(
        _digitize(signals["p_margin"], np.asarray(bins["p_margin"], dtype=np.float64))[:, None],
        2,
        axis=1,
    )
    phi_action_adv_bin = _digitize(
        signals["p_action_adv"],
        np.asarray(bins["p_action_adv"], dtype=np.float64),
    )
    q_top_relation_row = np.where(
        signals["q_top1"] == signals["hybrid"],
        0,
        np.where(signals["q_top1"] == signals["rival"], 1, 2),
    ).astype(np.int64)
    mean_relation_row = np.where(
        signals["qwen_mean"] == signals["hybrid"],
        0,
        np.where(signals["qwen_mean"] == signals["rival"], 1, 2),
    ).astype(np.int64)
    fields = {
        "role": role,
        "candidate": actions,
        "hybrid": hybrid,
        "rival": rival,
        "phi_top1": phi_top1,
        "q_margin_bin": q_margin_bin,
        "q_rival_adv_bin": q_rival_adv_bin,
        "selected_margin_bin": selected_margin_bin,
        "phi_margin_bin": phi_margin_bin,
        "phi_action_adv_bin": phi_action_adv_bin,
        "q_top_relation": np.repeat(q_top_relation_row[:, None], 2, axis=1),
        "mean_relation": np.repeat(mean_relation_row[:, None], 2, axis=1),
        "candidate_eq_phi_top1": (actions == phi_top1).astype(np.int64),
        "hybrid_eq_phi_top1": (hybrid == phi_top1).astype(np.int64),
        "rival_eq_phi_top1": (rival == phi_top1).astype(np.int64),
        "candidate_eq_qtop1": (actions == q_top1).astype(np.int64),
        "candidate_eq_qtop2": (actions == q_top2).astype(np.int64),
    }
    return actions, fields


def _bucket_key(fields: dict[str, np.ndarray], row_index: int, action_index: int, keys: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(fields[key][row_index, action_index]) for key in keys)


def _bucket_stats(
    *,
    fields: dict[str, np.ndarray],
    actions: np.ndarray,
    hybrid: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
    scheme_keys: tuple[str, ...],
) -> dict[tuple[int, ...], dict[str, float | int | tuple[int, ...]]]:
    stats: dict[tuple[int, ...], dict[str, float | int | tuple[int, ...]]] = {}
    for row_index in np.asarray(indices, dtype=np.int64):
        for action_index in range(actions.shape[1]):
            key = _bucket_key(fields, int(row_index), action_index, scheme_keys)
            delta = int(actions[row_index, action_index] == answers[row_index]) - int(
                hybrid[row_index] == answers[row_index]
            )
            entry = stats.setdefault(
                key,
                {"bucket_key": key, "support": 0, "helps": 0, "harms": 0, "delta_sum": 0.0, "delta_sq_sum": 0.0},
            )
            entry["support"] = int(entry["support"]) + 1
            entry["helps"] = int(entry["helps"]) + int(delta > 0)
            entry["harms"] = int(entry["harms"]) + int(delta < 0)
            entry["delta_sum"] = float(entry["delta_sum"]) + float(delta)
            entry["delta_sq_sum"] = float(entry["delta_sq_sum"]) + float(delta * delta)
    return stats


def _eligible_buckets(
    *,
    stats: dict[tuple[int, ...], dict[str, float | int | tuple[int, ...]]],
    min_support: int,
    z_value: float,
    min_mean_delta: float,
    max_harm_rate: float,
    min_net_help: int,
) -> dict[tuple[int, ...], dict[str, float | int | tuple[int, ...]]]:
    eligible: dict[tuple[int, ...], dict[str, float | int | tuple[int, ...]]] = {}
    for key, entry in stats.items():
        support = int(entry["support"])
        helps = int(entry["helps"])
        harms = int(entry["harms"])
        delta_sum = float(entry["delta_sum"])
        delta_sq_sum = float(entry["delta_sq_sum"])
        if support < int(min_support):
            continue
        mean = delta_sum / float(support)
        variance = max(0.0, delta_sq_sum / float(support) - mean * mean)
        lower = mean - float(z_value) * math.sqrt(variance / float(max(1, support)))
        harm_rate = harms / float(max(1, helps + harms))
        if (
            mean >= float(min_mean_delta)
            and lower > 0.0
            and harm_rate <= float(max_harm_rate)
            and helps - harms >= int(min_net_help)
        ):
            item = dict(entry)
            item.update(
                {
                    "mean_delta": float(mean),
                    "lower_delta": float(lower),
                    "harm_rate": float(harm_rate),
                    "net_help": int(helps - harms),
                    "score": float(lower + 0.01 * mean),
                }
            )
            eligible[key] = item
    return eligible


def _predict_bucket_model(
    *,
    fields: dict[str, np.ndarray],
    actions: np.ndarray,
    hybrid: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    predictions = np.asarray(hybrid, dtype=np.int64).copy()
    scheme_keys = tuple(model["scheme_keys"])
    buckets = {
        tuple(int(part) for part in key.split("|")): value
        for key, value in model["eligible_buckets"].items()
    }
    for row_index in range(len(predictions)):
        best: tuple[float, int] | None = None
        for action_index in range(actions.shape[1]):
            key = _bucket_key(fields, row_index, action_index, scheme_keys)
            entry = buckets.get(key)
            if entry is None:
                continue
            score = float(entry["score"])
            if best is None or score > best[0]:
                best = (score, action_index)
        if best is not None:
            predictions[row_index] = int(actions[row_index, best[1]])
    return predictions.astype(np.int64)


def _evaluate_predictions(
    *,
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    paired = _paired_ci(
        selected=predictions,
        baseline=baseline,
        answers=answers,
        seed=seed,
        samples=samples,
    )
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
        "override_rate": float(np.mean(predictions != baseline)),
        "accepted_harm_rate": float(harms / max(1, helps + harms)),
    }


def _select_bucket_model(
    *,
    actions: np.ndarray,
    fields: dict[str, np.ndarray],
    hybrid: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config_rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, float, str], dict[str, Any]] | None = None
    grid = {
        "min_support": (8, 12, 16, 24, 32, 48, 64),
        "z_value": (0.0, 0.5, 1.0, 1.64),
        "min_mean_delta": (0.0, 0.02, 0.05, 0.08, 0.10),
        "max_harm_rate": (0.0, 0.10, 0.20, 0.33),
        "min_net_help": (1, 2, 3, 5),
    }
    baseline_predictions = hybrid[dev_indices]
    baseline_eval = _evaluate_predictions(
        predictions=baseline_predictions,
        baseline=baseline_predictions,
        answers=answers[dev_indices],
        seed=72760504,
        samples=max(200, min(bootstrap_samples, 1000)),
    )
    noop_model = {
        "scheme": "no_op",
        "scheme_keys": (),
        "eligible_buckets": {},
        "selection": {"reason": "no eligible official-train bucket selected"},
    }
    best = (
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
    for scheme_name, scheme_keys in SCHEMES.items():
        stats = _bucket_stats(
            fields=fields,
            actions=actions,
            hybrid=hybrid,
            answers=answers,
            indices=fit_indices,
            scheme_keys=scheme_keys,
        )
        for min_support in grid["min_support"]:
            for z_value in grid["z_value"]:
                for min_mean_delta in grid["min_mean_delta"]:
                    for max_harm_rate in grid["max_harm_rate"]:
                        for min_net_help in grid["min_net_help"]:
                            eligible = _eligible_buckets(
                                stats=stats,
                                min_support=min_support,
                                z_value=z_value,
                                min_mean_delta=min_mean_delta,
                                max_harm_rate=max_harm_rate,
                                min_net_help=min_net_help,
                            )
                            if not eligible:
                                continue
                            model = {
                                "scheme": scheme_name,
                                "scheme_keys": list(scheme_keys),
                                "eligible_buckets": {
                                    "|".join(str(part) for part in key): value for key, value in eligible.items()
                                },
                                "selection": {
                                    "min_support": int(min_support),
                                    "z_value": float(z_value),
                                    "min_mean_delta": float(min_mean_delta),
                                    "max_harm_rate": float(max_harm_rate),
                                    "min_net_help": int(min_net_help),
                                },
                            }
                            predictions = _predict_bucket_model(
                                fields={key: value[dev_indices] for key, value in fields.items()},
                                actions=actions[dev_indices],
                                hybrid=hybrid[dev_indices],
                                model=model,
                            )
                            metrics = _evaluate_predictions(
                                predictions=predictions,
                                baseline=hybrid[dev_indices],
                                answers=answers[dev_indices],
                                seed=72860504 + len(config_rows),
                                samples=max(200, min(bootstrap_samples, 1000)),
                            )
                            if int(metrics["override_count"]) == 0:
                                continue
                            row = {
                                "scheme": scheme_name,
                                "scheme_keys": ",".join(scheme_keys),
                                "min_support": int(min_support),
                                "z_value": float(z_value),
                                "min_mean_delta": float(min_mean_delta),
                                "max_harm_rate": float(max_harm_rate),
                                "min_net_help": int(min_net_help),
                                "eligible_bucket_count": int(len(eligible)),
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
                            if best is None or key > best[0]:
                                best = (key, model)
    if best is None:
        raise ValueError("failed to select bucket model")
    selected_model = best[1]
    selected_bucket_rows = _selected_bucket_rows(selected_model)
    return selected_model, sorted(config_rows, key=lambda row: row["official_dev_accuracy"], reverse=True), selected_bucket_rows


def _selected_bucket_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket_id, entry in model["eligible_buckets"].items():
        rows.append(
            {
                "scheme": model["scheme"],
                "scheme_keys": ",".join(model["scheme_keys"]),
                "bucket_id": bucket_id,
                "support": int(entry["support"]),
                "fit_helps": int(entry["helps"]),
                "fit_harms": int(entry["harms"]),
                "fit_net_help": int(entry["net_help"]),
                "fit_mean_delta": float(entry["mean_delta"]),
                "fit_lower_delta": float(entry["lower_delta"]),
                "fit_harm_rate": float(entry["harm_rate"]),
                "score": float(entry["score"]),
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    oracle_help_count: int,
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
        seed=93960504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=93960604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=93960704 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    helps = int(vs_hybrid["helps"])
    harms = int(vs_hybrid["harms"])
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
        "helps_vs_fixed_hybrid": helps,
        "harms_vs_fixed_hybrid": harms,
        "net_help_vs_fixed_hybrid": int(helps - harms),
        "accepted_harm_rate": float(harms / max(1, helps + harms)),
        "oracle_headroom_capture": float(helps / max(1, oracle_help_count)),
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
    return receiver_gate._pair_oracle(pair[:, :2], phi_top1, answers)


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
            seed=94960504 + int(start),
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


def _corrupt_source_inputs(
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
    if condition == "source_score_row_shuffle_before_encoding":
        order = rng.permutation(len(hybrid))
        return qwen_scores[order], hybrid, mean[order], margin[order]
    if condition == "candidate_roll_source":
        return np.roll(qwen_scores, shift=1, axis=1), (hybrid + 1) % 4, (mean + 1) % 4, margin
    if condition == "code_value_permutation":
        perm = rng.permutation(4)
        inverse = np.empty(4, dtype=np.int64)
        inverse[perm] = np.arange(4)
        return qwen_scores[:, perm], inverse[hybrid], inverse[mean], margin
    if condition == "random_same_byte_source":
        order = rng.integers(0, len(hybrid), size=len(hybrid))
        return qwen_scores[order], hybrid[order], mean[order], margin[order]
    raise ValueError(f"unknown source condition {condition!r}")


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Harm-Controlled Bucket Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- harm-controlled accuracy: `{h['harm_controlled_accuracy']:.6f}`",
        f"- harm-controlled delta: `{h['harm_controlled_delta_vs_fixed_hybrid']:.6f}`",
        f"- harm-controlled CI95 low: `{h['harm_controlled_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- overrides / helps / harms: `{h['harm_controlled_override_count']} / {h['harm_controlled_helps_vs_fixed_hybrid']} / {h['harm_controlled_harms_vs_fixed_hybrid']}`",
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
    train_signals = _base_signals(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
    )
    bins = _fit_bins(train_signals, fit_indices)
    train_actions, train_fields = _action_fields(train_signals, bins)
    model, config_rows, selected_bucket_rows = _select_bucket_model(
        actions=train_actions,
        fields=train_fields,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
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
    eval_signals = _base_signals(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
    )
    eval_actions, eval_fields = _action_fields(eval_signals, bins)
    selected_predictions = _predict_bucket_model(
        fields=eval_fields,
        actions=eval_actions,
        hybrid=eval_hybrid,
        model=model,
    )
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    oracle_predictions = _pair_oracle(
        _pair_from_scores(eval_scores, eval_hybrid),
        np.argmax(eval_phi_scores, axis=1),
        answers,
    )
    oracle_help_count = int(
        np.sum((oracle_predictions == answers).astype(int) - (fixed_hybrid == answers).astype(int) > 0)
    )
    method_rows = [
        _method_row(
            name="harm_controlled_bucket_accept_defer_packet",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            oracle_help_count=oracle_help_count,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=3,
            framed_record_bytes=6,
            details={"model": {key: value for key, value in model.items() if key != "eligible_buckets"}},
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=eval_rows,
            predictions=fixed_hybrid,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            oracle_help_count=oracle_help_count,
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
            oracle_help_count=oracle_help_count,
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
            oracle_help_count=oracle_help_count,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
        ),
        _method_row(
            name="hybrid_rival_phi_oracle_diagnostic",
            rows=eval_rows,
            predictions=oracle_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            oracle_help_count=oracle_help_count,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
    ]
    label_rng = np.random.default_rng(20260504)
    permuted_answers = calibration["answers"][label_rng.permutation(len(calibration["answers"]))]
    label_model, _, _ = _select_bucket_model(
        actions=train_actions,
        fields=train_fields,
        hybrid=calibration["hybrid"],
        answers=permuted_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    label_predictions = _predict_bucket_model(
        fields=eval_fields,
        actions=eval_actions,
        hybrid=eval_hybrid,
        model=label_model,
    )
    method_rows.append(
        _method_row(
            name="official_train_label_permutation_bucket_control",
            rows=eval_rows,
            predictions=label_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            oracle_help_count=oracle_help_count,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=3,
            framed_record_bytes=6,
            details={"condition": "official_train_label_permutation"},
        )
    )
    for condition in (
        "source_row_shuffle",
        "source_score_row_shuffle_before_encoding",
        "candidate_roll_source",
        "code_value_permutation",
        "random_same_byte_source",
    ):
        c_scores, c_hybrid, c_mean, c_margin = _corrupt_source_inputs(
            qwen_scores=eval_scores,
            hybrid=eval_hybrid,
            mean=eval_mean,
            margin=eval_margin,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        c_signals = _base_signals(
            qwen_scores=c_scores,
            phi_scores=eval_phi_scores,
            hybrid=c_hybrid,
            qwen_mean=c_mean,
            qwen_margin=c_margin,
        )
        c_actions, c_fields = _action_fields(c_signals, bins)
        corrupted = _predict_bucket_model(
            fields=c_fields,
            actions=c_actions,
            hybrid=c_hybrid,
            model=model,
        )
        method_rows.append(
            _method_row(
                name=f"{condition}_bucket_control",
                rows=eval_rows,
                predictions=corrupted,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                oracle_help_count=oracle_help_count,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=3,
                framed_record_bytes=6,
                details={"condition": condition},
            )
        )
    eval_diag_best: tuple[tuple[float, float, float, str], np.ndarray] | None = None
    selected_keys = tuple(model["scheme_keys"])
    selected_stats = _bucket_stats(
        fields=eval_fields,
        actions=eval_actions,
        hybrid=fixed_hybrid,
        answers=answers,
        indices=np.arange(len(eval_rows), dtype=np.int64),
        scheme_keys=selected_keys,
    )
    for min_support in (1, 2, 4, 8, 12, 16):
        eligible = _eligible_buckets(
            stats=selected_stats,
            min_support=min_support,
            z_value=0.0,
            min_mean_delta=0.0,
            max_harm_rate=0.0,
            min_net_help=1,
        )
        if not eligible:
            continue
        diag_model = {
            "scheme": model["scheme"],
            "scheme_keys": list(selected_keys),
            "eligible_buckets": {"|".join(str(part) for part in key): value for key, value in eligible.items()},
            "selection": {"eval_label_selected": True, "min_support": min_support},
        }
        diag_predictions = _predict_bucket_model(
            fields=eval_fields,
            actions=eval_actions,
            hybrid=fixed_hybrid,
            model=diag_model,
        )
        overrides = int(np.sum(diag_predictions != fixed_hybrid))
        if overrides == 0:
            continue
        key = (
            _accuracy(diag_predictions, answers),
            float(np.mean((diag_predictions == answers).astype(float) - (fixed_hybrid == answers).astype(float))),
            float(-overrides),
            str(min_support),
        )
        if eval_diag_best is None or key > eval_diag_best[0]:
            eval_diag_best = (key, diag_predictions)
    if eval_diag_best is not None:
        method_rows.append(
            _method_row(
                name="eval_label_best_bucket_diagnostic",
                rows=eval_rows,
                predictions=eval_diag_best[1],
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                oracle_help_count=oracle_help_count,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=3,
                framed_record_bytes=6,
                details={"not_promotable": True, "eval_label_selected": True},
            )
        )
    method_rows = sorted(method_rows, key=lambda item: item["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "harm_controlled_bucket_accept_defer_packet")
    destructive_rows = [row for row in method_rows if row["method"].endswith("_control")]
    best_destructive = max(destructive_rows, key=lambda item: item["accuracy"])
    selected_bucket_csv_rows = selected_bucket_rows or [
        {
            "scheme": model["scheme"],
            "scheme_keys": ",".join(model["scheme_keys"]),
            "bucket_id": "none",
            "support": 0,
            "fit_helps": 0,
            "fit_harms": 0,
            "fit_net_help": 0,
            "fit_mean_delta": 0.0,
            "fit_lower_delta": 0.0,
            "fit_harm_rate": 0.0,
            "score": 0.0,
        }
    ]
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
        and method_row["helps_vs_fixed_hybrid"] > method_row["harms_vs_fixed_hybrid"]
        and method_row["harms_vs_fixed_hybrid"] <= max(1, int(math.floor(0.25 * method_row["helps_vs_fixed_hybrid"])))
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and method_row["accuracy"] > best_destructive["accuracy"]
        and len(train_content_ids & eval_content_ids) == 0
    )
    selected_dev = next(
        (
            row
            for row in config_rows
            if row["scheme"] == model["scheme"]
            and row["min_support"] == model.get("selection", {}).get("min_support")
            and row["z_value"] == model.get("selection", {}).get("z_value")
            and row["min_mean_delta"] == model.get("selection", {}).get("min_mean_delta")
            and row["max_harm_rate"] == model.get("selection", {}).get("max_harm_rate")
            and row["min_net_help"] == model.get("selection", {}).get("min_net_help")
        ),
        {},
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "selected_scheme": model["scheme"],
        "selected_scheme_keys": ",".join(model["scheme_keys"]),
        "selected_eligible_bucket_count": int(len(model["eligible_buckets"])),
        "official_dev_selected_accuracy": selected_dev.get("official_dev_accuracy", _accuracy(calibration["hybrid"][dev_indices], calibration["answers"][dev_indices])),
        "official_dev_selected_delta_vs_hybrid": selected_dev.get("official_dev_delta_vs_hybrid", 0.0),
        "official_dev_selected_ci95_low_vs_hybrid": selected_dev.get("official_dev_ci95_low_vs_hybrid", 0.0),
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
        "harm_controlled_accuracy": method_row["accuracy"],
        "harm_controlled_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "harm_controlled_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "harm_controlled_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "harm_controlled_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "harm_controlled_net_help_vs_fixed_hybrid": method_row["net_help_vs_fixed_hybrid"],
        "harm_controlled_override_count": method_row["override_count_vs_fixed_hybrid"],
        "harm_controlled_accepted_harm_rate": method_row["accepted_harm_rate"],
        "oracle_headroom_capture": method_row["oracle_headroom_capture"],
        "hybrid_rival_phi_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "hybrid_rival_phi_oracle_diagnostic"
        )["accuracy"],
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "raw_payload_bytes": 3,
        "framed_record_bytes": 6,
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_harm_controlled_bucket_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the harm-controlled bucket packet beats fixed Qwen hybrid by at least 0.005 "
            "with positive paired CI, beats candidate-only with positive paired CI, helps more than harms, "
            "keeps harms <= max(1, floor(0.25 * helps)), is nonnegative on both eval slices, beats destructive "
            "controls, and has zero official-train/eval content overlap."
        ),
        "headline": headline,
        "packet_contract": {
            "receiver_visible_payload": (
                "Qwen hybrid/rival candidate IDs plus quantized source-side packet bins; receiver-local Phi "
                "scores are used as decoder side information."
            ),
            "raw_payload_bytes": 3,
            "framed_record_bytes": 6,
            "source_packet_fields": [
                "hybrid_candidate_id",
                "rival_candidate_id",
                "qwen_margin_bin",
                "qwen_rival_advantage_bin",
                "qwen_selected_margin_bin",
                "qwen_top1_relation",
                "qwen_mean_relation",
            ],
            "receiver_local_fields": [
                "phi_top1_candidate_id",
                "phi_margin_bin",
                "phi_action_advantage_bin",
                "candidate_matches_phi_top1",
                "hybrid_matches_phi_top1",
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
        "config_rows": config_rows,
        "selected_bucket_rows": selected_bucket_rows,
        "slice_rows": slice_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": 3,
            "framed_record_bytes_per_request": 6,
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
            "phi_train_score_cache": headroom._display_path(phi_cache_path),
            "phi_train_score_cache_sha256": phi_sha,
            "phi_train_score_model": phi_state,
            "source_score_cache": denoise._display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate tests the highest-priority receiver branch after shallow linear switchers failed. It "
            "uses official-train labels to select only low-harm complementarity buckets, then freezes that rule "
            "on Qwen-to-Phi validation. The receiver sees quantized source packet fields and Phi-local side "
            "information, not raw Qwen score vectors."
        ),
        "lay_explanation": (
            "Qwen sends Phi a tiny message naming its safe answer, its backup answer, and a few coarse confidence "
            "levels. Phi then uses a rule learned on training questions: only switch away from the safe answer "
            "in buckets where training showed switches usually helped and rarely hurt."
        ),
    }
    _write_json(output_dir / "hellaswag_qwen_to_phi_harm_controlled_bucket_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "selected_bucket_rows.csv", selected_bucket_csv_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_harm_controlled_bucket_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_harm_controlled_bucket_gate.json",
                "hellaswag_qwen_to_phi_harm_controlled_bucket_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "selected_bucket_rows.csv",
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
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "selected_scheme": h["selected_scheme"],
                "selected_eligible_bucket_count": h["selected_eligible_bucket_count"],
                "official_dev_selected_delta_vs_hybrid": h["official_dev_selected_delta_vs_hybrid"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "harm_controlled_accuracy": h["harm_controlled_accuracy"],
                "harm_controlled_delta_vs_fixed_hybrid": h["harm_controlled_delta_vs_fixed_hybrid"],
                "harm_controlled_helps_vs_fixed_hybrid": h["harm_controlled_helps_vs_fixed_hybrid"],
                "harm_controlled_harms_vs_fixed_hybrid": h["harm_controlled_harms_vs_fixed_hybrid"],
                "hybrid_rival_phi_oracle_accuracy": h["hybrid_rival_phi_oracle_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
