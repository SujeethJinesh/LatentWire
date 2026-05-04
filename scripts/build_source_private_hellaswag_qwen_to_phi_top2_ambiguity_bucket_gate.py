from __future__ import annotations

"""Official-train top1/top2 ambiguity-bucket gate for Qwen-to-Phi HellaSwag.

This is a stricter follow-up to the held-out uncertainty router and
harm-controlled bucket gates. The source-visible packet carries only the
source top-1/top-2 candidates plus quantized decision-syndrome fields. The
receiver can combine those packet fields with Phi-local score bins, but it
cannot see raw Qwen scores, logits, hidden states, text, or KV cache.
"""

import argparse
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
from scripts import build_source_private_hellaswag_qwen_to_phi_harm_controlled_bucket_gate as bucket_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate as router_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_PHI_TRAIN_SCORE_CACHE = receiver_gate.DEFAULT_OUTPUT / "phi_official_train_score_cache.json"
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES

ACTION_NAMES = ("source_top1", "source_top2", "phi_top1", "qwen_mean")
SOURCE_CONDITIONS = (
    "source_row_shuffle",
    "source_score_row_shuffle_before_encoding",
    "candidate_roll_source",
    "code_value_permutation",
    "target_derived_source_packet",
    "zero_source_packet",
    "random_same_byte_source",
)
AMBIGUITY_SCHEMES: dict[str, tuple[str, ...]] = {
    "source_pair_receiver_agreement": (
        "role",
        "candidate_eq_phi_top1",
        "candidate_eq_source_top1",
        "candidate_eq_source_top2",
        "source_top1_eq_phi_top1",
        "source_top2_eq_phi_top1",
        "q_margin_bin",
        "phi_action_adv_bin",
    ),
    "source_pair_syndrome": (
        "role",
        "candidate",
        "source_top1",
        "source_top2",
        "source_pair_relation_to_hybrid",
        "q_margin_bin",
        "q_entropy_bin",
        "q_hybrid_gap_bin",
        "phi_action_adv_bin",
    ),
    "source_phi_conflict": (
        "role",
        "candidate_eq_phi_top1",
        "candidate_eq_phi_top2",
        "source_top1_eq_phi_top1",
        "source_top2_eq_phi_top1",
        "hybrid_eq_phi_top1",
        "q_selected_margin_bin",
        "phi_margin_bin",
        "phi_hybrid_gap_bin",
    ),
    "compact_top2_packet": (
        "role",
        "candidate_eq_hybrid",
        "candidate_eq_source_top1",
        "candidate_eq_source_top2",
        "candidate_eq_qwen_mean",
        "q_margin_bin",
        "phi_action_adv_bin",
    ),
}


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def _fit_bins(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    qwen_margin: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, list[float]]:
    return router_gate._fit_bins(
        qwen_scores=qwen_scores,
        phi_scores=phi_scores,
        hybrid=hybrid,
        qwen_margin=qwen_margin,
        fit_indices=fit_indices,
    )


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return router_gate._top2(scores)


def _z_scores(scores: np.ndarray) -> np.ndarray:
    return router_gate._z_scores(scores)


def _digitize(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    return router_gate._digitize(values, bins)


def _action_fields(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_margin: np.ndarray,
    bins: dict[str, list[float]],
    zero_source_bins: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    qwen_scores = np.asarray(qwen_scores, dtype=np.float64)
    phi_scores = np.asarray(phi_scores, dtype=np.float64)
    hybrid = np.asarray(hybrid, dtype=np.int64)
    qwen_mean = np.asarray(qwen_mean, dtype=np.int64)
    qwen_margin = np.asarray(qwen_margin, dtype=np.float64)
    row_ids = np.arange(qwen_scores.shape[0])
    q_top1, q_top2 = _top2(qwen_scores)
    p_top1, p_top2 = _top2(phi_scores)
    qz = _z_scores(qwen_scores)
    pz = _z_scores(phi_scores)
    q_margin_value = qwen_scores[row_ids, q_top1] - qwen_scores[row_ids, q_top2]
    q_entropy_value = router_gate._entropy(qwen_scores)
    q_hybrid_gap_value = qz[row_ids, q_top1] - qz[row_ids, hybrid]
    p_margin_value = phi_scores[row_ids, p_top1] - phi_scores[row_ids, p_top2]
    p_entropy_value = router_gate._entropy(phi_scores)
    p_hybrid_gap_value = pz[row_ids, p_top1] - pz[row_ids, hybrid]
    source_bins = {
        "q_margin_bin": _digitize(q_margin_value, bins["q_margin"]),
        "q_entropy_bin": _digitize(q_entropy_value, bins["q_entropy"]),
        "q_hybrid_gap_bin": _digitize(q_hybrid_gap_value, bins["q_hybrid_gap"]),
        "q_selected_margin_bin": _digitize(qwen_margin, bins["qwen_selected_margin"]),
    }
    if zero_source_bins:
        source_bins = {key: np.zeros_like(value) for key, value in source_bins.items()}
    receiver_bins = {
        "phi_margin_bin": _digitize(p_margin_value, bins["p_margin"]),
        "phi_entropy_bin": _digitize(p_entropy_value, bins["p_entropy"]),
        "phi_hybrid_gap_bin": _digitize(p_hybrid_gap_value, bins["p_hybrid_gap"]),
    }
    actions = np.stack([q_top1, q_top2, p_top1, qwen_mean], axis=1).astype(np.int64)
    role = np.repeat(np.arange(actions.shape[1], dtype=np.int64)[None, :], actions.shape[0], axis=0)
    repeated = {
        "source_top1": np.repeat(q_top1[:, None], actions.shape[1], axis=1),
        "source_top2": np.repeat(q_top2[:, None], actions.shape[1], axis=1),
        "hybrid": np.repeat(hybrid[:, None], actions.shape[1], axis=1),
        "qwen_mean": np.repeat(qwen_mean[:, None], actions.shape[1], axis=1),
        "phi_top1": np.repeat(p_top1[:, None], actions.shape[1], axis=1),
        "phi_top2": np.repeat(p_top2[:, None], actions.shape[1], axis=1),
    }
    p_action_adv = pz[row_ids[:, None], actions] - pz[row_ids, hybrid][:, None]
    phi_action_adv_bin = _digitize(p_action_adv, bins["p_hybrid_gap"])
    fields: dict[str, np.ndarray] = {
        "role": role,
        "candidate": actions,
        "phi_action_adv_bin": phi_action_adv_bin,
        "source_pair_relation_to_hybrid": np.repeat(
            np.where(q_top1 == hybrid, 0, np.where(q_top2 == hybrid, 1, 2))[:, None],
            actions.shape[1],
            axis=1,
        ),
    }
    fields.update(repeated)
    for key, value in source_bins.items():
        fields[key] = np.repeat(value[:, None], actions.shape[1], axis=1)
    for key, value in receiver_bins.items():
        fields[key] = np.repeat(value[:, None], actions.shape[1], axis=1)
    fields.update(
        {
            "candidate_eq_hybrid": (actions == repeated["hybrid"]).astype(np.int64),
            "candidate_eq_source_top1": (actions == repeated["source_top1"]).astype(np.int64),
            "candidate_eq_source_top2": (actions == repeated["source_top2"]).astype(np.int64),
            "candidate_eq_phi_top1": (actions == repeated["phi_top1"]).astype(np.int64),
            "candidate_eq_phi_top2": (actions == repeated["phi_top2"]).astype(np.int64),
            "candidate_eq_qwen_mean": (actions == repeated["qwen_mean"]).astype(np.int64),
            "source_top1_eq_phi_top1": (repeated["source_top1"] == repeated["phi_top1"]).astype(np.int64),
            "source_top2_eq_phi_top1": (repeated["source_top2"] == repeated["phi_top1"]).astype(np.int64),
            "hybrid_eq_phi_top1": (repeated["hybrid"] == repeated["phi_top1"]).astype(np.int64),
        }
    )
    diagnostics = {
        "source_top1": q_top1,
        "source_top2": q_top2,
        "phi_top1": p_top1,
        "phi_top2": p_top2,
        **source_bins,
        **receiver_bins,
    }
    return actions, fields, diagnostics


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
        "accepted_harm_rate": float(harms / max(1, helps + harms)),
    }


def _select_model(
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
    baseline_eval = _evaluate_predictions(
        predictions=hybrid[dev_indices],
        baseline=hybrid[dev_indices],
        answers=answers[dev_indices],
        seed=81660504,
        samples=max(200, min(bootstrap_samples, 1000)),
    )
    noop_model = {
        "scheme": "no_op",
        "scheme_keys": (),
        "eligible_buckets": {},
        "selection": {"reason": "no eligible official-train ambiguity bucket selected"},
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
    grid = {
        "min_support": (8, 12, 16, 24, 32, 48, 64),
        "z_value": (0.0, 0.5, 1.0, 1.64),
        "min_mean_delta": (0.0, 0.02, 0.05, 0.08, 0.10),
        "max_harm_rate": (0.0, 0.10, 0.20, 0.33),
        "min_net_help": (1, 2, 3, 5),
    }
    for scheme_name, scheme_keys in AMBIGUITY_SCHEMES.items():
        stats = bucket_gate._bucket_stats(
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
                            eligible = bucket_gate._eligible_buckets(
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
                            predictions = bucket_gate._predict_bucket_model(
                                fields={key: value[dev_indices] for key, value in fields.items()},
                                actions=actions[dev_indices],
                                hybrid=hybrid[dev_indices],
                                model=model,
                            )
                            metrics = _evaluate_predictions(
                                predictions=predictions,
                                baseline=hybrid[dev_indices],
                                answers=answers[dev_indices],
                                seed=81760504 + len(config_rows),
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
                            if key > best[0]:
                                best = (key, model)
    return best[1], sorted(config_rows, key=lambda row: row["official_dev_accuracy"], reverse=True), _selected_bucket_rows(best[1])


def _selected_bucket_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    if not model.get("eligible_buckets"):
        return []
    return bucket_gate._selected_bucket_rows(model)


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
    return router_gate._method_row(
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


def _source_pair_oracle(source_top1: np.ndarray, source_top2: np.ndarray, answers: np.ndarray) -> np.ndarray:
    return np.where(source_top1 == answers, source_top1, np.where(source_top2 == answers, source_top2, source_top1)).astype(np.int64)


def _source_control_inputs(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    mean: np.ndarray,
    margin: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if condition == "zero_source_packet":
        return (
            np.zeros_like(qwen_scores, dtype=np.float64),
            np.zeros_like(hybrid, dtype=np.int64),
            np.zeros_like(mean, dtype=np.int64),
            np.zeros_like(margin, dtype=np.float64),
        )
    return router_gate._corrupt_source_inputs(
        qwen_scores=qwen_scores,
        phi_scores=phi_scores,
        hybrid=hybrid,
        mean=mean,
        margin=margin,
        condition=condition,
        seed=seed,
    )


def _slice_rows(
    *,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    return router_gate._slice_rows(
        rows=rows,
        predictions=predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Top1/Top2 Ambiguity Bucket Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- ambiguity-bucket accuracy: `{h['top2_ambiguity_bucket_accuracy']:.6f}`",
        f"- ambiguity-bucket delta: `{h['top2_ambiguity_bucket_delta_vs_fixed_hybrid']:.6f}`",
        f"- ambiguity-bucket CI95 low: `{h['top2_ambiguity_bucket_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- overrides / helps / harms: `{h['top2_ambiguity_bucket_override_count']} / {h['top2_ambiguity_bucket_helps_vs_fixed_hybrid']} / {h['top2_ambiguity_bucket_harms_vs_fixed_hybrid']}`",
        f"- source top1/top2 oracle accuracy: `{h['source_top1_or_top2_oracle_accuracy']:.6f}`",
        f"- no-syndrome top-pair accuracy: `{h['source_pair_no_syndrome_accuracy']:.6f}`",
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
    phi_scores, _, phi_state, phi_sha = receiver_gate._load_or_build_phi_scores(
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
    bins = _fit_bins(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_margin=calibration["margin"],
        fit_indices=fit_indices,
    )
    train_actions, train_fields, _ = _action_fields(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
        bins=bins,
    )
    model, config_rows, selected_bucket_rows = _select_model(
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
    eval_phi_scores = np.asarray([row["phi_target_scores"] for row in eval_rows], dtype=np.float64)
    eval_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    eval_mean = _field_array(eval_rows, "selected_prediction")
    eval_margin = np.asarray([float(row.get("selected_margin", 0.0)) for row in eval_rows], dtype=np.float64)
    fixed_hybrid = eval_hybrid.copy()
    candidate_only = eval_mean.copy()
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    eval_actions, eval_fields, eval_diag = _action_fields(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
        bins=bins,
    )
    selected_predictions = bucket_gate._predict_bucket_model(
        fields=eval_fields,
        actions=eval_actions,
        hybrid=eval_hybrid,
        model=model,
    )
    source_top1 = eval_diag["source_top1"]
    source_top2 = eval_diag["source_top2"]
    source_pair_oracle = _source_pair_oracle(source_top1, source_top2, answers)
    oracle_help_count = int(
        np.sum((source_pair_oracle == answers).astype(int) - (fixed_hybrid == answers).astype(int) > 0)
    )
    no_syndrome_actions, no_syndrome_fields, _ = _action_fields(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
        bins=bins,
        zero_source_bins=True,
    )
    no_syndrome_predictions = bucket_gate._predict_bucket_model(
        fields=no_syndrome_fields,
        actions=no_syndrome_actions,
        hybrid=eval_hybrid,
        model=model,
    )
    fusion_model = router_gate._fit_logit_fusion(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        dev_indices=dev_indices,
    )
    fusion_predictions = router_gate._predict_logit_fusion(eval_scores, eval_phi_scores, fusion_model)
    method_rows = [
        _method_row(
            name="top2_ambiguity_bucket_packet",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=4,
            details={"model": {key: value for key, value in model.items() if key != "eligible_buckets"}},
        ),
        _method_row(
            name="source_pair_no_syndrome_bucket_control",
            rows=eval_rows,
            predictions=no_syndrome_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            details={"source_top1_top2_only": True},
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
            name="raw_source_score_logit_fusion_control",
            rows=eval_rows,
            predictions=fusion_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=32,
            framed_record_bytes=32,
            source_score_or_logit_vector_exposed=True,
            details={"alpha": fusion_model["alpha"]},
        ),
    ]
    if eval_rows and "source_rank_only_bagged_prediction" in eval_rows[0]:
        method_rows.append(
            _method_row(
                name="source_rank_only_bagged_control",
                rows=eval_rows,
                predictions=_field_array(eval_rows, "source_rank_only_bagged_prediction"),
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                details={"existing_audit_control": True},
            )
        )
    label_rng = np.random.default_rng(20260504)
    permuted_answers = calibration["answers"][label_rng.permutation(len(calibration["answers"]))]
    label_model, _, _ = _select_model(
        actions=train_actions,
        fields=train_fields,
        hybrid=calibration["hybrid"],
        answers=permuted_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    label_predictions = bucket_gate._predict_bucket_model(
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
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=4,
            details={"condition": "official_train_label_permutation"},
        )
    )
    for condition in SOURCE_CONDITIONS:
        c_scores, c_hybrid, c_mean, c_margin = _source_control_inputs(
            qwen_scores=eval_scores,
            phi_scores=eval_phi_scores,
            hybrid=eval_hybrid,
            mean=eval_mean,
            margin=eval_margin,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        c_actions, c_fields, _ = _action_fields(
            qwen_scores=c_scores,
            phi_scores=eval_phi_scores,
            hybrid=c_hybrid,
            qwen_mean=c_mean,
            qwen_margin=c_margin,
            bins=bins,
        )
        control_predictions = bucket_gate._predict_bucket_model(
            fields=c_fields,
            actions=c_actions,
            hybrid=c_hybrid,
            model=model,
        )
        method_rows.append(
            _method_row(
                name=f"{condition}_bucket_control",
                rows=eval_rows,
                predictions=control_predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=2,
                framed_record_bytes=4,
                details={"condition": condition},
            )
        )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "top2_ambiguity_bucket_packet")
    no_syndrome_row = next(row for row in method_rows if row["method"] == "source_pair_no_syndrome_bucket_control")
    source_top1_row = next(row for row in method_rows if row["method"] == "source_top1_label_control")
    destructive_rows = [
        row
        for row in method_rows
        if row["method"].endswith("_control")
        and row["method"] not in {"source_pair_no_syndrome_bucket_control", "source_top1_label_control", "source_top2_label_control", "source_rank_only_bagged_control"}
    ]
    best_destructive = max(destructive_rows, key=lambda row: row["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
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
            if row["scheme"] == model["scheme"]
            and row["min_support"] == model.get("selection", {}).get("min_support")
            and row["z_value"] == model.get("selection", {}).get("z_value")
            and row["min_mean_delta"] == model.get("selection", {}).get("min_mean_delta")
            and row["max_harm_rate"] == model.get("selection", {}).get("max_harm_rate")
            and row["min_net_help"] == model.get("selection", {}).get("min_net_help")
        ),
        {},
    )
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
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "selected_scheme": model["scheme"],
        "selected_scheme_keys": ",".join(model["scheme_keys"]),
        "selected_eligible_bucket_count": int(len(model["eligible_buckets"])),
        "official_dev_selected_accuracy": selected_dev.get("official_dev_accuracy", _accuracy(calibration["hybrid"][dev_indices], calibration["answers"][dev_indices])),
        "official_dev_selected_delta_vs_hybrid": selected_dev.get("official_dev_delta_vs_hybrid", 0.0),
        "official_dev_selected_ci95_low_vs_hybrid": selected_dev.get("official_dev_ci95_low_vs_hybrid", 0.0),
        "eval_rows": len(eval_rows),
        "fixed_hybrid_accuracy": next(row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement")["accuracy"],
        "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")["accuracy"],
        "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")["accuracy"],
        "top2_ambiguity_bucket_accuracy": method_row["accuracy"],
        "top2_ambiguity_bucket_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "top2_ambiguity_bucket_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "top2_ambiguity_bucket_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "top2_ambiguity_bucket_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "top2_ambiguity_bucket_override_count": method_row["override_count_vs_fixed_hybrid"],
        "source_pair_no_syndrome_accuracy": no_syndrome_row["accuracy"],
        "source_top1_accuracy": source_top1_row["accuracy"],
        "source_index_margin": source_index_margin,
        "source_top1_or_top2_oracle_accuracy": next(row for row in method_rows if row["method"] == "source_top1_top2_oracle_diagnostic")["accuracy"],
        "oracle_headroom_capture": float(method_row["helps_vs_fixed_hybrid"] / max(1, oracle_help_count)),
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "source_pair_disagreement_rate": float(np.mean(source_top1 != source_top2)),
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "raw_payload_bytes": 2,
        "framed_record_bytes": 4,
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the official-train selected top1/top2 ambiguity bucket beats fixed hybrid by at "
            "least 0.005 with positive paired CI, beats candidate-only with positive paired CI, helps more "
            "than harms, is nonnegative on all eval slices, beats destructive controls, beats source-index/"
            "no-syndrome packet controls by at least 0.005, and has zero official-train/eval content overlap."
        ),
        "headline": headline,
        "packet_contract": {
            "receiver_visible_payload": (
                "Qwen source top-1/top-2 candidate IDs plus quantized source-side decision-syndrome bins; "
                "Phi-local score bins are decoder side information."
            ),
            "raw_payload_bytes": 2,
            "framed_record_bytes": 4,
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
        "config_rows": config_rows,
        "selected_bucket_rows": selected_bucket_rows,
        "slice_rows": slice_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": 2,
            "framed_record_bytes_per_request": 4,
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
            "This gate tests whether the remaining Qwen top1/top2 oracle headroom can be recovered by a "
            "source-private ambiguity packet selected on official train and frozen on validation. It is not "
            "promotable unless it separates from no-syndrome source-index and destructive packet controls."
        ),
        "lay_explanation": (
            "Qwen sends Phi its two best guesses and a few tiny confidence clues. Phi only switches away from "
            "the fixed safe answer in clue-patterns that helped on training questions. If the same behavior "
            "appears when the clues are shuffled, zeroed, or made from Phi itself, the packet is not useful "
            "cross-model communication."
        ),
    }
    bucket_gate._write_json(output_dir / "hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.json", payload)
    bucket_gate._write_csv(output_dir / "method_rows.csv", method_rows)
    bucket_gate._write_csv(output_dir / "config_rows.csv", config_rows)
    bucket_gate._write_csv(output_dir / "selected_bucket_rows.csv", selected_bucket_csv_rows)
    bucket_gate._write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.md", payload)
    bucket_gate._write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.json",
                "hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.md",
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
                "top2_ambiguity_bucket_accuracy": h["top2_ambiguity_bucket_accuracy"],
                "top2_ambiguity_bucket_delta_vs_fixed_hybrid": h["top2_ambiguity_bucket_delta_vs_fixed_hybrid"],
                "top2_ambiguity_bucket_helps_vs_fixed_hybrid": h["top2_ambiguity_bucket_helps_vs_fixed_hybrid"],
                "top2_ambiguity_bucket_harms_vs_fixed_hybrid": h["top2_ambiguity_bucket_harms_vs_fixed_hybrid"],
                "source_pair_no_syndrome_accuracy": h["source_pair_no_syndrome_accuracy"],
                "source_top1_or_top2_oracle_accuracy": h["source_top1_or_top2_oracle_accuracy"],
                "best_destructive_control_name": h["best_destructive_control_name"],
                "best_destructive_control_accuracy": h["best_destructive_control_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
