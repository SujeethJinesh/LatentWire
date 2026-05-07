from __future__ import annotations

from experimental.shared.hybrid_gate_evaluators import (
    evaluate_hbsm_b1,
    evaluate_horn_h1,
    evaluate_ssq_lr_s1,
)


SSQ_BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")


def test_ssq_lr_s1_evaluator_requires_enough_passing_layers() -> None:
    rows = []
    for prompt_index in range(12):
        for layer in range(4):
            for bucket in SSQ_BUCKETS:
                boosted = bucket == "final_minus_128" and layer < 3
                rows.append(
                    {
                        "prompt_id": f"p{prompt_index}",
                        "layer": layer,
                        "position_bucket": bucket,
                        "max_abs": 3.0 if boosted else 1.0,
                        "std": 3.0 if boosted else 1.0,
                        "kurtosis": 1.0,
                    }
                )

    result = evaluate_ssq_lr_s1(rows)

    assert result["gate_pass"] is True
    assert result["passing_layer_count"] == 3
    assert result["required_passing_layer_count"] == 3


def test_ssq_lr_s1_evaluator_fails_without_layer_support() -> None:
    rows = []
    for prompt_index in range(12):
        for layer in range(4):
            for bucket in SSQ_BUCKETS:
                boosted = bucket == "final_minus_128" and layer == 0
                rows.append(
                    {
                        "prompt_id": f"p{prompt_index}",
                        "layer": layer,
                        "position_bucket": bucket,
                        "max_abs": 3.0 if boosted else 1.0,
                        "std": 3.0 if boosted else 1.0,
                        "kurtosis": 1.0,
                    }
                )

    result = evaluate_ssq_lr_s1(rows)

    assert result["gate_pass"] is False
    assert result["passing_layer_count"] == 1


def test_ssq_lr_s1_evaluator_rejects_tiny_distribution_only_shift() -> None:
    rows = []
    for prompt_index in range(12):
        for layer in range(4):
            for bucket in SSQ_BUCKETS:
                shifted = bucket == "final_minus_128"
                rows.append(
                    {
                        "prompt_id": f"p{prompt_index}",
                        "layer": layer,
                        "position_bucket": bucket,
                        "max_abs": 1.01 if shifted else 1.0,
                        "std": 1.01 if shifted else 1.0,
                        "kurtosis": 1.01 if shifted else 1.0,
                    }
                )

    result = evaluate_ssq_lr_s1(rows)

    assert result["gate_pass"] is False
    assert result["distribution_effect_floor_pass"] is False
    assert result["selected_s1_ratio"] < 1.25


def test_horn_h1_evaluator_uses_directional_support_and_controls() -> None:
    rows = []
    for prompt_index in range(12):
        for direction, max_abs in [("attention->ssm", 10.0), ("ssm->attention", 1.0)]:
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": max_abs,
                    "kurtosis": 1.0,
                    "control_type": "boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": 1.0,
                    "kurtosis": 1.0,
                    "control_type": "non_boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": "ssm->attention" if direction == "attention->ssm" else "attention->ssm",
                    "max_abs": 1.0,
                    "kurtosis": 1.0,
                    "control_type": "permuted_direction",
                }
            )

    result = evaluate_horn_h1(rows)

    assert result["gate_pass"] is True
    assert result["selected_h1_metric"] == "max_abs"
    assert result["selected_h1_ratio"] == 10.0
    assert result["non_boundary_control_ratio"] == 1.0
    assert result["permuted_direction_ratio"] == 1.0


def test_horn_h1_evaluator_allows_faithful_label_flip_null() -> None:
    rows = []
    for prompt_index in range(12):
        boundary_values = [("attention->ssm", 10.0), ("ssm->attention", 1.0)]
        for direction, max_abs in boundary_values:
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": max_abs,
                    "kurtosis": 1.0,
                    "control_type": "boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": 1.0,
                    "kurtosis": 1.0,
                    "control_type": "non_boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": "ssm->attention" if direction == "attention->ssm" else "attention->ssm",
                    "max_abs": max_abs,
                    "kurtosis": 1.0,
                    "control_type": "permuted_direction",
                }
            )

    result = evaluate_horn_h1(rows)

    assert result["gate_pass"] is True
    assert result["selected_h1_direction"] == "attention->ssm"
    assert result["permuted_direction_ratio"] == 0.1


def test_horn_h1_evaluator_fails_when_non_boundary_control_has_same_ratio() -> None:
    rows = []
    for prompt_index in range(12):
        for direction, max_abs in [("attention->ssm", 10.0), ("ssm->attention", 1.0)]:
            for control_type in ("boundary", "non_boundary", "permuted_direction"):
                rows.append(
                    {
                        "prompt_id": f"p{prompt_index}",
                        "direction": direction,
                        "max_abs": max_abs,
                        "kurtosis": 1.0,
                        "control_type": control_type,
                    }
                )

    result = evaluate_horn_h1(rows)

    assert result["gate_pass"] is False
    assert result["non_boundary_control_ratio"] == result["selected_h1_ratio"]


def test_horn_h1_evaluator_rejects_near_boundary_non_boundary_control() -> None:
    rows = []
    for prompt_index in range(12):
        for direction, boundary_abs, non_boundary_abs in [
            ("attention->ssm", 10.0, 9.9),
            ("ssm->attention", 1.0, 1.0),
        ]:
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": boundary_abs,
                    "kurtosis": 1.0,
                    "control_type": "boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": non_boundary_abs,
                    "kurtosis": 1.0,
                    "control_type": "non_boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": "ssm->attention" if direction == "attention->ssm" else "attention->ssm",
                    "max_abs": boundary_abs,
                    "kurtosis": 1.0,
                    "control_type": "permuted_direction",
                }
            )

    result = evaluate_horn_h1(rows)

    assert result["gate_pass"] is False
    assert result["non_boundary_control_ratio"] > result["selected_h1_threshold"]


def test_horn_h1_evaluator_fails_when_permuted_control_preserves_signal() -> None:
    rows = []
    for prompt_index in range(12):
        for direction, max_abs in [("attention->ssm", 10.0), ("ssm->attention", 1.0)]:
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": max_abs,
                    "kurtosis": 1.0,
                    "control_type": "boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": 1.0,
                    "kurtosis": 1.0,
                    "control_type": "non_boundary",
                }
            )
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "direction": direction,
                    "max_abs": max_abs,
                    "kurtosis": 1.0,
                    "control_type": "permuted_direction",
                }
            )

    result = evaluate_horn_h1(rows)

    assert result["gate_pass"] is False
    assert result["permuted_direction_ratio"] == result["selected_h1_ratio"]


def test_hbsm_b1_evaluator_requires_boundary_enrichment_over_random() -> None:
    rows = []
    for index in range(60):
        boundary = index < 30
        rows.append(
            {
                "model_id": "hybrid",
                "prompt_id": f"p{index % 30}",
                "layer": index,
                "boundary_flag": boundary,
                "top_decile_flag": boundary and index < 6,
                "random_top_decile": index in {0, 1, 2, 30, 31, 32},
                "train_test_split": "train" if index % 2 == 0 else "test",
                "control_type": "boundary_only",
                "cheap_predictor": float(100 - index),
                "kl_or_nll_drift": float(100 - index),
                "parameter_count": float(index + 100),
                "weight_norm": float(index + 1),
            }
        )
    for control in [
        "perturbation_off",
        "random_flags",
        "layer_index",
        "parameter_count_norm",
        "kl_lens_rank",
        "activation_outlier",
    ]:
        rows.append(
            {
                "prompt_id": "control",
                "boundary_flag": False,
                "top_decile_flag": False,
                "random_top_decile": False,
                "train_test_split": "train",
                "control_type": control,
                "cheap_predictor": 0.0,
                "kl_or_nll_drift": 0.0,
            }
        )

    result = evaluate_hbsm_b1(rows)

    assert result["gate_pass"] is True
    assert result["scoring_layer_count"] == 60
    assert result["expected_top_decile_count"] == 6
    assert result["boundary_top_decile_enrichment"] > 1.0
    assert result["random_boundary_top_decile_enrichment"] == 1.0
    assert result["fisher_p_boundary_top_decile"] < 0.05
    assert result["cheap_predictor_spearman"] > 0.9
    assert set(result["baseline_spearman"]) == {
        "layer_index",
        "parameter_count_norm",
        "weight_norm",
        "boundary_flag",
        "kl_lens_rank",
        "activation_outlier",
    }


def test_hbsm_b1_evaluator_aggregates_prompt_rows_by_layer() -> None:
    rows = []
    for prompt_index in range(12):
        for layer in range(20):
            boundary = layer < 10
            rows.append(
                {
                    "model_id": "hybrid",
                    "prompt_id": f"p{prompt_index}",
                    "layer": layer,
                    "boundary_flag": boundary,
                    "top_decile_flag": boundary and layer < 2,
                    "random_top_decile": layer in {0, 10},
                    "train_test_split": "train" if layer % 2 == 0 else "test",
                    "control_type": "boundary_only",
                    "cheap_predictor": float(100 - layer),
                    "kl_or_nll_drift": float(100 - layer),
                    "parameter_count": float(100 + layer),
                    "weight_norm": float(layer),
                }
            )

    result = evaluate_hbsm_b1(rows)

    assert result["primary_row_count"] == 240
    assert result["scoring_layer_count"] == 20
    assert result["expected_top_decile_count"] == 2
    assert result["top_decile_count"] == 2
    assert result["prompt_count"] == 12
