"""Schema and decision contracts for second-stage hybrid-quant gates.

These contracts intentionally do not run models. They validate reduced row
packets for the Mac-local follow-up gates that become relevant only after the
first real SSQ-LR, HORN, or HBSM gate promotes. The point is to make the later
claim boundary executable before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

BASE_REQUIRED_FILES = ("config.json", "raw_rows.jsonl", "summary.json", "decision.md")
BASE_CONFIG_FIELDS = (
    "gate_name",
    "project",
    "source_gate_packet_sha256",
    "preregistration_sha256",
    "seed_list",
    "command",
)
BASE_SUMMARY_FIELDS = ("decision", "gate_name", "gate_status", "gate_pass", "row_count", "claim_boundary")

FOLLOWUP_ROW_FIELDS: dict[str, tuple[str, ...]] = {
    "ssq_lr_s2": (
        "model_id",
        "prompt_id",
        "recipe_id",
        "precision",
        "scale_granularity",
        "block_size",
        "effective_bits",
        "bf16_state_bytes",
        "quantized_state_bytes",
        "scale_bytes",
        "metadata_bytes",
        "bf16_accuracy",
        "quantized_accuracy",
        "accuracy_delta_abs",
        "bf16_nll",
        "quantized_nll",
        "nll_delta",
        "paired_ci_low",
        "paired_ci_high",
        "bf16_noop_delta",
        "control_type",
    ),
    "ssq_lr_s3": (
        "model_id",
        "prompt_id",
        "recipe_id",
        "frozen_recipe_sha256",
        "source_s2_packet_sha256",
        "retuned",
        "model_role",
        "bf16_accuracy",
        "quantized_accuracy",
        "accuracy_delta_abs",
        "paired_ci_high",
        "bf16_nll",
        "quantized_nll",
        "nll_delta",
        "control_type",
    ),
    "horn_h2": (
        "model_id",
        "prompt_id",
        "prompt_cluster_id",
        "selected_direction_from_h1",
        "boundary_direction",
        "noise_side",
        "noise_std_basis",
        "noise_scale",
        "seed",
        "clean_nll",
        "noisy_nll",
        "delta_nll",
        "hook_off_delta",
        "control_type",
    ),
    "horn_h3": (
        "model_id",
        "model_role",
        "architecture_family",
        "prompt_id",
        "selected_direction_from_h1",
        "boundary_direction",
        "directional_drift_ratio",
        "directional_ratio_ci_low",
        "pure_control_expected_null",
        "control_type",
    ),
    "hbsm_b2": (
        "model_id",
        "predictor_name",
        "predictor_registry_sha256",
        "hyperparams_sha256",
        "registry_status",
        "train_test_split",
        "selection_split",
        "spearman",
        "spearman_ci_low",
        "baseline_name",
        "baseline_spearman",
        "baseline_ci_low",
        "margin_vs_best_baseline",
        "selected_predictor",
        "control_type",
    ),
    "hbsm_b3": (
        "model_id",
        "prompt_id",
        "layer",
        "boundary_direction",
        "noise_scale",
        "horn_alignment_sign",
        "attention_output_drift",
        "ssm_output_drift",
        "paired_delta",
        "paired_ci_low",
        "paired_ci_high",
        "control_type",
    ),
}

FOLLOWUP_REQUIRED_CONTROLS: dict[str, set[str]] = {
    "ssq_lr_s2": {
        "candidate_recipe",
        "bf16_noop",
        "same_byte_uniform",
        "int8_state",
        "fp8_state",
        "mxfp4_state",
        "random_same_l2",
        "shuffled_scales",
    },
    "ssq_lr_s3": {"transfer_eval", "retune_probe"},
    "horn_h2": {"directional_noise", "hook_off", "flipped_direction_label"},
    "horn_h3": {"hybrid_validation", "pure_attention_control", "pure_mamba_control"},
    "hbsm_b2": {"candidate_predictor", "baseline_predictor"},
    "hbsm_b3": {"matched_noise", "noise_off", "direction_flip"},
}

FOLLOWUP_SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "ssq_lr_s2": (
        "recipe_count",
        "selected_recipe_id",
        "selected_memory_reduction",
        "selected_accuracy_delta_abs",
        "selected_accuracy_ci_high",
        "selected_nll_delta_abs",
        "selected_nll_delta_ci_high",
        "bf16_noop_max_delta",
    ),
    "ssq_lr_s3": (
        "transfer_model_count",
        "passing_model_count",
        "max_accuracy_delta_abs",
        "max_ci_high",
        "retuned_row_count",
        "frozen_recipe_sha256",
    ),
    "horn_h2": (
        "selected_direction_from_h1",
        "selected_h2_direction",
        "directional_drift_ratio",
        "directional_ratio_ci_low",
        "selected_direction_support_fraction",
        "hook_off_max_delta",
        "noise_std_basis",
        "seed_count",
    ),
    "horn_h3": (
        "hybrid_model_count",
        "passing_hybrid_model_count",
        "pure_control_count",
        "pure_control_max_ratio",
        "selected_direction_consistent",
    ),
    "hbsm_b2": (
        "selected_predictor_name",
        "test_spearman",
        "test_spearman_ci_low",
        "best_baseline_spearman",
        "margin_vs_best_baseline",
        "predictor_registry_sha256",
        "selected_on_train_only",
    ),
    "hbsm_b3": (
        "horn_alignment_sign",
        "mechanism_effect",
        "mechanism_ci_low",
        "noise_off_max_drift",
        "direction_flip_effect",
    ),
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"raw_rows.jsonl line {line_number} is not an object")
            rows.append(row)
    return rows


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0
    return float(numerator / denominator)


def _bootstrap_ratio_low(paired: dict[str, dict[str, float]]) -> float:
    ratios = [
        _safe_ratio(max(values.values()), min(values.values()))
        for values in paired.values()
        if len(values) == 2 and min(values.values()) > 0.0
    ]
    if not ratios:
        return 0.0
    ratios.sort()
    index = max(0, int(0.025 * (len(ratios) - 1)))
    return float(ratios[index])


def _signed_direction_ratio_low(
    paired: dict[str, dict[str, float]],
    *,
    selected_direction: str,
) -> tuple[float, float]:
    ratios: list[float] = []
    wins = 0
    for values in paired.values():
        if selected_direction not in values or len(values) != 2:
            continue
        other_values = [value for direction, value in values.items() if direction != selected_direction]
        if len(other_values) != 1 or other_values[0] <= 0.0:
            continue
        ratio = _safe_ratio(values[selected_direction], other_values[0])
        ratios.append(ratio)
        if ratio > 1.0:
            wins += 1
    if not ratios:
        return 0.0, 0.0
    ratios.sort()
    index = max(0, int(0.025 * (len(ratios) - 1)))
    return float(ratios[index]), float(wins / len(ratios))


def _hash_values(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {str(row.get(field, "")).strip().lower() for row in rows if row.get(field)}


def evaluate_ssq_lr_s2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recipe_rows = [row for row in rows if str(row.get("control_type")) == "candidate_recipe"]
    recipes = sorted({str(row["recipe_id"]) for row in recipe_rows})
    noop_rows = [row for row in rows if str(row.get("control_type")) == "bf16_noop"]
    noop_max = max((abs(float(row["bf16_noop_delta"])) for row in noop_rows), default=float("inf"))
    candidates: list[dict[str, Any]] = []
    for recipe_id in recipes:
        current = [row for row in recipe_rows if str(row["recipe_id"]) == recipe_id]
        total_bytes = [
            float(row["quantized_state_bytes"]) + float(row["scale_bytes"]) + float(row["metadata_bytes"])
            for row in current
        ]
        memory_reduction = min(
            _safe_ratio(float(row["bf16_state_bytes"]), total)
            for row, total in zip(current, total_bytes)
        )
        accuracy_delta = max(abs(float(row["accuracy_delta_abs"])) for row in current)
        ci_high = max(float(row["paired_ci_high"]) for row in current)
        nll_delta_abs = max(abs(float(row["nll_delta"])) for row in current)
        nll_ci_high = max(
            float(row.get("nll_delta_abs_ci_high", abs(float(row["nll_delta"]))))
            for row in current
        )
        candidate = {
            "recipe_id": recipe_id,
            "memory_reduction": memory_reduction,
            "accuracy_delta": accuracy_delta,
            "ci_high": ci_high,
            "nll_delta_abs": nll_delta_abs,
            "nll_ci_high": nll_ci_high,
        }
        candidate["quality_pass"] = ci_high <= 0.01 or nll_ci_high <= 0.01
        candidate["gate_pass"] = (
            memory_reduction >= 4.0 and candidate["quality_pass"] and noop_max <= 1e-5
        )
        candidates.append(candidate)
    passing_candidates = [candidate for candidate in candidates if candidate["gate_pass"]]
    quality_only_candidates = [candidate for candidate in candidates if candidate["quality_pass"]]
    if passing_candidates:
        best = min(
            passing_candidates,
            key=lambda item: (
                item["nll_ci_high"],
                item["ci_high"],
                -item["memory_reduction"],
                item["recipe_id"],
            ),
        )
    elif quality_only_candidates:
        best = min(
            quality_only_candidates,
            key=lambda item: (
                -item["memory_reduction"],
                item["nll_ci_high"],
                item["ci_high"],
                item["recipe_id"],
            ),
        )
    elif candidates:
        best = min(
            candidates,
            key=lambda item: (
                item["accuracy_delta"],
                item["nll_ci_high"],
                -item["memory_reduction"],
                item["recipe_id"],
            ),
        )
    else:
        best = {
            "recipe_id": "",
            "memory_reduction": 1.0,
            "accuracy_delta": float("inf"),
            "ci_high": float("inf"),
            "nll_delta_abs": float("inf"),
            "nll_ci_high": float("inf"),
            "quality_pass": False,
            "gate_pass": False,
        }
    gate_pass = bool(best["gate_pass"])
    return {
        "gate_name": "ssq_lr_s2_state_quantization_sensitivity",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY"
        if gate_pass
        else "FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY",
        "recipe_count": len(recipes),
        "selected_recipe_id": best["recipe_id"],
        "selected_memory_reduction": best["memory_reduction"],
        "selected_accuracy_delta_abs": best["accuracy_delta"],
        "selected_accuracy_ci_high": best["ci_high"],
        "selected_nll_delta_abs": best["nll_delta_abs"],
        "selected_nll_delta_ci_high": best["nll_ci_high"],
        "bf16_noop_max_delta": noop_max,
    }


def evaluate_ssq_lr_s3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transfer_rows = [row for row in rows if str(row.get("control_type")) == "transfer_eval"]
    model_ids = sorted({str(row["model_id"]) for row in transfer_rows})
    passing_models = []
    max_accuracy_delta = max((abs(float(row["accuracy_delta_abs"])) for row in transfer_rows), default=float("inf"))
    max_ci_high = max((float(row["paired_ci_high"]) for row in transfer_rows), default=float("inf"))
    retuned_count = sum(1 for row in rows if bool(row.get("retuned")))
    for model_id in model_ids:
        model_rows = [row for row in transfer_rows if str(row["model_id"]) == model_id]
        if (
            model_rows
            and max(abs(float(row["accuracy_delta_abs"])) for row in model_rows) <= 0.02
            and max(float(row["paired_ci_high"]) for row in model_rows) <= 0.02
            and all(not bool(row.get("retuned")) for row in model_rows)
        ):
            passing_models.append(model_id)
    frozen_hashes = _hash_values(transfer_rows, "frozen_recipe_sha256")
    gate_pass = (
        len(model_ids) >= 2
        and len(passing_models) >= 2
        and retuned_count == 0
        and len(frozen_hashes) == 1
        and all(SHA256_PATTERN.fullmatch(value) for value in frozen_hashes)
    )
    return {
        "gate_name": "ssq_lr_s3_cross_model_recipe_transfer",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER"
        if gate_pass
        else "FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER",
        "transfer_model_count": len(model_ids),
        "passing_model_count": len(passing_models),
        "max_accuracy_delta_abs": max_accuracy_delta,
        "max_ci_high": max_ci_high,
        "retuned_row_count": retuned_count,
        "frozen_recipe_sha256": next(iter(frozen_hashes), ""),
    }


def evaluate_horn_h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    noise_rows = [row for row in rows if str(row.get("control_type")) == "directional_noise"]
    directions = sorted({str(row["boundary_direction"]) for row in noise_rows})
    mean_by_direction = {
        direction: _mean([float(row["delta_nll"]) for row in noise_rows if str(row["boundary_direction"]) == direction])
        for direction in directions
    }
    if len(mean_by_direction) == 2:
        selected_h2 = max(mean_by_direction, key=mean_by_direction.get)
        ratio = _safe_ratio(max(mean_by_direction.values()), min(mean_by_direction.values()))
    else:
        selected_h2 = ""
        ratio = 1.0
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in noise_rows:
        key = "|".join(
            [
                str(row.get("prompt_cluster_id") or row.get("prompt_id")),
                str(row.get("seed")),
                str(row.get("noise_side")),
            ]
        )
        paired[key][str(row["boundary_direction"])] = float(row["delta_nll"])
    paired_unit_count = sum(1 for values in paired.values() if len(values) == 2)
    hook_off_max = max(
        (abs(float(row["hook_off_delta"])) for row in rows if str(row.get("control_type")) == "hook_off"),
        default=float("inf"),
    )
    selected_from_h1 = str(noise_rows[0].get("selected_direction_from_h1", "")) if noise_rows else ""
    signed_ratio_low, selected_support_fraction = _signed_direction_ratio_low(
        paired,
        selected_direction=selected_from_h1,
    )
    noise_bases = {str(row.get("noise_std_basis", "")) for row in noise_rows}
    seed_count = len({int(row["seed"]) for row in noise_rows if isinstance(row.get("seed"), int)})
    gate_pass = (
        selected_h2 == selected_from_h1
        and ratio >= 1.5
        and signed_ratio_low > 1.0
        and selected_support_fraction >= 0.75
        and hook_off_max <= 1e-5
        and len(noise_bases) == 1
        and seed_count >= 3
        and paired_unit_count == len(paired)
        and paired_unit_count > 0
    )
    return {
        "gate_name": "horn_h2_directional_noise_propagation",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION"
        if gate_pass
        else "FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION",
        "selected_direction_from_h1": selected_from_h1,
        "selected_h2_direction": selected_h2,
        "directional_drift_ratio": ratio,
        "directional_ratio_ci_low": signed_ratio_low,
        "selected_direction_support_fraction": selected_support_fraction,
        "hook_off_max_delta": hook_off_max,
        "noise_std_basis": next(iter(noise_bases), ""),
        "seed_count": seed_count,
        "paired_unit_count": paired_unit_count,
    }


def evaluate_horn_h3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hybrid_rows = [row for row in rows if str(row.get("control_type")) == "hybrid_validation"]
    pure_rows = [
        row
        for row in rows
        if str(row.get("control_type")) in {"pure_attention_control", "pure_mamba_control"}
    ]
    hybrid_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hybrid_rows:
        hybrid_by_model[str(row["model_id"])].append(row)
    selected_directions = {str(row.get("selected_direction_from_h1", "")) for row in hybrid_rows}
    passing_models = [
        model_id
        for model_id, model_rows in hybrid_by_model.items()
        if min(float(row["directional_ratio_ci_low"]) for row in model_rows) > 1.0
        and min(float(row["directional_drift_ratio"]) for row in model_rows) >= 1.5
    ]
    pure_max = max((float(row["directional_drift_ratio"]) for row in pure_rows), default=1.0)
    pure_controls_fold = all(
        bool(row.get("pure_control_expected_null"))
        and (
            float(row["directional_drift_ratio"]) < 1.2
            or float(row["directional_ratio_ci_low"]) <= 1.0
        )
        for row in pure_rows
    )
    consistent = len(selected_directions - {""}) == 1
    gate_pass = (
        len(hybrid_by_model) >= 2
        and len(passing_models) >= 2
        and len(pure_rows) >= 2
        and pure_controls_fold
        and consistent
    )
    return {
        "gate_name": "horn_h3_cross_model_and_architecture_controls",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_HORN_H3_CROSS_MODEL_CONTROLS"
        if gate_pass
        else "FAIL_REAL_HORN_H3_CROSS_MODEL_CONTROLS",
        "hybrid_model_count": len(hybrid_by_model),
        "passing_hybrid_model_count": len(passing_models),
        "pure_control_count": len(pure_rows),
        "pure_control_max_ratio": pure_max,
        "selected_direction_consistent": consistent,
    }


def evaluate_hbsm_b2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if bool(row.get("selected_predictor"))]
    selected_names = {str(row.get("predictor_name", "")) for row in selected}
    test_rows = [row for row in selected if str(row.get("train_test_split")) == "test"]
    train_selected = [row for row in selected if str(row.get("selection_split")) == "train"]
    test_spearman = min((float(row["spearman"]) for row in test_rows), default=0.0)
    test_low = min((float(row["spearman_ci_low"]) for row in test_rows), default=0.0)
    best_baseline = max((float(row["baseline_spearman"]) for row in rows), default=0.0)
    margin = min((float(row["margin_vs_best_baseline"]) for row in test_rows), default=0.0)
    registries = _hash_values(rows, "predictor_registry_sha256")
    preregistered = all(str(row.get("registry_status")) == "preregistered" for row in rows)
    selected_on_train_only = bool(train_selected) and all(
        str(row.get("selection_split")) == "train" for row in selected
    )
    gate_pass = (
        len(selected_names) == 1
        and test_spearman >= 0.6
        and test_low >= 0.5
        and margin >= 0.05
        and len(registries) == 1
        and all(SHA256_PATTERN.fullmatch(value) for value in registries)
        and preregistered
        and selected_on_train_only
    )
    return {
        "gate_name": "hbsm_b2_no_forward_predictor",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_HBSM_B2_NO_FORWARD_PREDICTOR"
        if gate_pass
        else "FAIL_REAL_HBSM_B2_NO_FORWARD_PREDICTOR",
        "selected_predictor_name": next(iter(selected_names), ""),
        "test_spearman": test_spearman,
        "test_spearman_ci_low": test_low,
        "best_baseline_spearman": best_baseline,
        "margin_vs_best_baseline": margin,
        "predictor_registry_sha256": next(iter(registries), ""),
        "selected_on_train_only": selected_on_train_only,
    }


def evaluate_hbsm_b3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if str(row.get("control_type")) == "matched_noise"]
    noise_off = [row for row in rows if str(row.get("control_type")) == "noise_off"]
    direction_flip = [row for row in rows if str(row.get("control_type")) == "direction_flip"]
    signs = {str(row.get("horn_alignment_sign", "")) for row in matched}
    effect = _mean([float(row["paired_delta"]) for row in matched])
    ci_low = min((float(row["paired_ci_low"]) for row in matched), default=0.0)
    noise_off_max = max(
        (
            max(abs(float(row["attention_output_drift"])), abs(float(row["ssm_output_drift"])))
            for row in noise_off
        ),
        default=float("inf"),
    )
    flip_effect = _mean([float(row["paired_delta"]) for row in direction_flip])
    gate_pass = (
        len(signs - {""}) == 1
        and effect > 0.0
        and ci_low > 0.0
        and noise_off_max <= 1e-5
        and flip_effect <= 0.0
    )
    return {
        "gate_name": "hbsm_b3_softmax_amplification_mechanism",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_HBSM_B3_SOFTMAX_MECHANISM"
        if gate_pass
        else "FAIL_REAL_HBSM_B3_SOFTMAX_MECHANISM",
        "horn_alignment_sign": next(iter(signs), ""),
        "mechanism_effect": effect,
        "mechanism_ci_low": ci_low,
        "noise_off_max_drift": noise_off_max,
        "direction_flip_effect": flip_effect,
    }


EVALUATORS: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {
    "ssq_lr_s2": evaluate_ssq_lr_s2,
    "ssq_lr_s3": evaluate_ssq_lr_s3,
    "horn_h2": evaluate_horn_h2,
    "horn_h3": evaluate_horn_h3,
    "hbsm_b2": evaluate_hbsm_b2,
    "hbsm_b3": evaluate_hbsm_b3,
}


def _validate_common(
    *,
    gate: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    for field in BASE_CONFIG_FIELDS:
        if field not in config:
            errors.append(f"config.json missing {field}")
    for field in ("source_gate_packet_sha256", "preregistration_sha256"):
        value = str(config.get(field, "")).strip().lower()
        if not SHA256_PATTERN.fullmatch(value):
            errors.append(f"config.json {field} must be sha256:<64 lowercase hex chars>")
    if config.get("gate_name") != gate:
        errors.append(f"config.json gate_name must be {gate!r}")
    if not isinstance(config.get("seed_list"), list) or not config.get("seed_list"):
        errors.append("config.json seed_list must be a non-empty list")
    for field in BASE_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"summary.json missing {field}")
    if summary.get("row_count") != len(rows):
        errors.append("summary.json row_count must match raw_rows.jsonl")
    if not isinstance(summary.get("claim_boundary"), list):
        errors.append("summary.json claim_boundary must be a list")
    required = FOLLOWUP_ROW_FIELDS[gate]
    for index, row in enumerate(rows):
        missing = [field for field in required if field not in row]
        if missing:
            errors.append(f"row {index} missing fields: {', '.join(missing)}")
    observed_controls = {str(row.get("control_type")) for row in rows}
    missing_controls = FOLLOWUP_REQUIRED_CONTROLS[gate] - observed_controls
    if missing_controls:
        errors.append(f"missing required controls: {', '.join(sorted(missing_controls))}")
    unknown_controls = observed_controls - FOLLOWUP_REQUIRED_CONTROLS[gate]
    if unknown_controls:
        errors.append(f"unknown controls: {', '.join(sorted(unknown_controls))}")


def _validate_numeric_rows(gate: str, rows: list[dict[str, Any]], errors: list[str]) -> None:
    numeric_by_gate = {
        "ssq_lr_s2": (
            "block_size",
            "effective_bits",
            "bf16_state_bytes",
            "quantized_state_bytes",
            "scale_bytes",
            "metadata_bytes",
            "bf16_accuracy",
            "quantized_accuracy",
            "accuracy_delta_abs",
            "bf16_nll",
            "quantized_nll",
            "nll_delta",
            "paired_ci_low",
            "paired_ci_high",
            "bf16_noop_delta",
        ),
        "ssq_lr_s3": (
            "bf16_accuracy",
            "quantized_accuracy",
            "accuracy_delta_abs",
            "paired_ci_high",
            "bf16_nll",
            "quantized_nll",
            "nll_delta",
        ),
        "horn_h2": ("noise_scale", "seed", "clean_nll", "noisy_nll", "delta_nll", "hook_off_delta"),
        "horn_h3": ("directional_drift_ratio", "directional_ratio_ci_low"),
        "hbsm_b2": (
            "spearman",
            "spearman_ci_low",
            "baseline_spearman",
            "baseline_ci_low",
            "margin_vs_best_baseline",
        ),
        "hbsm_b3": (
            "layer",
            "noise_scale",
            "attention_output_drift",
            "ssm_output_drift",
            "paired_delta",
            "paired_ci_low",
            "paired_ci_high",
        ),
    }
    for index, row in enumerate(rows):
        for field in numeric_by_gate[gate]:
            if not _finite_number(row.get(field)):
                errors.append(f"row {index} {field} must be finite numeric")
        if gate == "ssq_lr_s3" and not isinstance(row.get("retuned"), bool):
            errors.append(f"row {index} retuned must be boolean")
        if gate == "horn_h3" and not isinstance(row.get("pure_control_expected_null"), bool):
            errors.append(f"row {index} pure_control_expected_null must be boolean")
        if gate == "hbsm_b2" and not isinstance(row.get("selected_predictor"), bool):
            errors.append(f"row {index} selected_predictor must be boolean")


def _validate_gate_specific_invariants(gate: str, rows: list[dict[str, Any]], errors: list[str]) -> None:
    if gate == "ssq_lr_s2":
        noop_rows = [row for row in rows if str(row.get("control_type")) == "bf16_noop"]
        for index, row in enumerate(noop_rows):
            if abs(float(row.get("bf16_noop_delta", float("inf")))) > 1e-5:
                errors.append(f"ssq_lr_s2 bf16_noop row {index} must have near-zero drift")
    if gate == "ssq_lr_s3":
        for index, row in enumerate(rows):
            if bool(row.get("retuned")):
                errors.append(f"ssq_lr_s3 row {index} retuned must be false")
        frozen_hashes = _hash_values(rows, "frozen_recipe_sha256")
        source_hashes = _hash_values(rows, "source_s2_packet_sha256")
        if len(frozen_hashes) != 1 or not all(SHA256_PATTERN.fullmatch(value) for value in frozen_hashes):
            errors.append("ssq_lr_s3 rows must cite one frozen_recipe_sha256")
        if len(source_hashes) != 1 or not all(SHA256_PATTERN.fullmatch(value) for value in source_hashes):
            errors.append("ssq_lr_s3 rows must cite one source_s2_packet_sha256")
    if gate == "horn_h2":
        selected = {str(row.get("selected_direction_from_h1", "")) for row in rows}
        noise_basis = {str(row.get("noise_std_basis", "")) for row in rows}
        seed_values = {row.get("seed") for row in rows if str(row.get("control_type")) == "directional_noise"}
        if len(selected - {""}) != 1:
            errors.append("horn_h2 rows must share one selected_direction_from_h1")
        if len(noise_basis - {""}) != 1:
            errors.append("horn_h2 rows must share one noise_std_basis")
        if len(seed_values) < 3:
            errors.append("horn_h2 directional_noise rows must include at least 3 seeds")
        directional: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in rows:
            if str(row.get("control_type")) != "directional_noise":
                continue
            key = (
                str(row.get("prompt_cluster_id") or row.get("prompt_id")),
                str(row.get("seed")),
                str(row.get("noise_side")),
            )
            directional[key].add(str(row.get("boundary_direction")))
        unpaired = [key for key, directions in directional.items() if len(directions) != 2]
        if unpaired:
            errors.append("horn_h2 directional_noise rows must pair both boundary directions per cluster/seed/noise_side")
    if gate == "horn_h3":
        for index, row in enumerate(rows):
            control_type = str(row.get("control_type"))
            if control_type.startswith("pure_") and row.get("pure_control_expected_null") is not True:
                errors.append(f"horn_h3 pure control row {index} must set pure_control_expected_null=true")
            if control_type.startswith("pure_") and float(row.get("directional_drift_ratio", float("inf"))) >= 1.2 and float(
                row.get("directional_ratio_ci_low", float("inf"))
            ) > 1.0:
                errors.append(f"horn_h3 pure control row {index} must fold below 1.2 or have CI overlap 1.0")
    if gate == "hbsm_b2":
        registries = _hash_values(rows, "predictor_registry_sha256")
        if len(registries) != 1 or not all(SHA256_PATTERN.fullmatch(value) for value in registries):
            errors.append("hbsm_b2 rows must cite one predictor_registry_sha256")
        for index, row in enumerate(rows):
            if str(row.get("registry_status")) != "preregistered":
                errors.append(f"hbsm_b2 row {index} registry_status must be preregistered")
            if bool(row.get("selected_predictor")) and str(row.get("selection_split")) != "train":
                errors.append(f"hbsm_b2 selected row {index} must be selected from train split only")
    if gate == "hbsm_b3":
        signs = {str(row.get("horn_alignment_sign", "")) for row in rows}
        if len(signs - {""}) != 1:
            errors.append("hbsm_b3 rows must share one horn_alignment_sign")
        for index, row in enumerate(rows):
            if str(row.get("control_type")) == "noise_off":
                attention = abs(float(row.get("attention_output_drift", float("inf"))))
                ssm = abs(float(row.get("ssm_output_drift", float("inf"))))
                if max(attention, ssm) > 1e-5:
                    errors.append(f"hbsm_b3 noise_off row {index} must have near-zero drift")


def validate_followup_gate_packet(packet_dir: Path, *, gate: str) -> dict[str, Any]:
    if gate not in EVALUATORS:
        return {
            "packet_dir": str(packet_dir),
            "ok": False,
            "errors": [f"unknown follow-up gate {gate!r}"],
            "gate": gate,
        }
    errors: list[str] = []
    for filename in BASE_REQUIRED_FILES:
        if not (packet_dir / filename).exists():
            errors.append(f"missing {filename}")
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if (packet_dir / "config.json").exists():
        try:
            config = _load_json(packet_dir / "config.json")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"config.json is invalid: {exc}")
    if (packet_dir / "summary.json").exists():
        try:
            summary = _load_json(packet_dir / "summary.json")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"summary.json is invalid: {exc}")
    if (packet_dir / "raw_rows.jsonl").exists():
        try:
            rows = _load_rows(packet_dir / "raw_rows.jsonl")
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            errors.append(f"raw_rows.jsonl is invalid: {exc}")
    if not rows:
        errors.append("raw_rows.jsonl contains no rows")
    if rows:
        _validate_common(gate=gate, config=config, summary=summary, rows=rows, errors=errors)
        _validate_numeric_rows(gate, rows, errors)
        _validate_gate_specific_invariants(gate, rows, errors)
        evaluated = EVALUATORS[gate](rows)
        for field in ("gate_name", "gate_status", "gate_pass", *FOLLOWUP_SUMMARY_FIELDS[gate]):
            if field not in summary:
                errors.append(f"summary.json missing {field}")
                continue
            expected = evaluated.get(field)
            observed = summary.get(field)
            if _finite_number(expected) and _finite_number(observed):
                if abs(float(expected) - float(observed)) > 1e-9:
                    errors.append(f"summary.json {field} must match evaluator output")
            elif observed != expected:
                errors.append(f"summary.json {field} must match evaluator output")
        if summary.get("decision") != evaluated.get("gate_status"):
            errors.append("summary.json decision must equal recomputed gate_status")
    decision_text = (packet_dir / "decision.md").read_text(encoding="utf-8") if (packet_dir / "decision.md").exists() else ""
    if summary and str(summary.get("decision")) not in decision_text:
        errors.append("decision.md does not contain summary decision")
    return {
        "packet_dir": str(packet_dir),
        "ok": not errors,
        "errors": errors,
        "gate": gate,
        "row_count": len(rows),
        "decision": summary.get("decision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--gate", choices=sorted(EVALUATORS), required=True)
    args = parser.parse_args()
    report = validate_followup_gate_packet(args.packet_dir, gate=args.gate)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
