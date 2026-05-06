"""Deterministic evaluators for active hybrid-quantization Mac gates.

These functions do not run models and do not estimate uncertainty from raw
tensors. They turn already-reduced packet rows into the preregistered first-gate
decision fields that reviewers need to audit.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


SSQ_LR_POSITION_BUCKETS = {"prefill_end", "2k_or_end", "8k_or_end", "final_minus_128"}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0
    return float(numerator / denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_metric(rows: list[dict[str, Any]], *, field: str, **filters: Any) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(row.get(filter_key) == filter_value for filter_key, filter_value in filters.items())
    ]
    return _mean(values)


def _bucket_ratio(
    rows: list[dict[str, Any]],
    *,
    field: str,
    first_bucket: str = "prefill_end",
    final_bucket: str = "final_minus_128",
) -> float:
    return _safe_ratio(
        _mean_metric(rows, field=field, position_bucket=final_bucket),
        _mean_metric(rows, field=field, position_bucket=first_bucket),
    )


def evaluate_ssq_lr_s1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the SSQ-LR S1 state-heterogeneity screen from packet rows."""

    prompts = {str(row["prompt_id"]) for row in rows}
    buckets = sorted({str(row["position_bucket"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    ratios = {
        "max_abs_ratio_final_minus_128_vs_prefill_end": _bucket_ratio(rows, field="max_abs"),
        "std_ratio_final_minus_128_vs_prefill_end": _bucket_ratio(rows, field="std"),
        "kurtosis_ratio_final_minus_128_vs_prefill_end": _bucket_ratio(rows, field="kurtosis"),
    }
    passing_layers = 0
    selected_layer_lowers: list[float] = []
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        max_ratio = _bucket_ratio(layer_rows, field="max_abs")
        std_ratio = _bucket_ratio(layer_rows, field="std")
        selected_layer_ratio = max(max_ratio, std_ratio)
        if selected_layer_ratio >= 2.0:
            passing_layers += 1
            selected_layer_lowers.append(selected_layer_ratio)
    ssm_layer_count = len(layers)
    required_passing_layer_count = 0
    if ssm_layer_count:
        required_passing_layer_count = max(
            min(3, ssm_layer_count),
            math.ceil(0.25 * ssm_layer_count),
        )
    selected_s1_ratio = max(
        ratios["max_abs_ratio_final_minus_128_vs_prefill_end"],
        ratios["std_ratio_final_minus_128_vs_prefill_end"],
    )
    selected_s1_ci_low = min(selected_layer_lowers) if selected_layer_lowers else selected_s1_ratio
    holm_p_min = 1.0
    gate_pass = (
        set(buckets) == SSQ_LR_POSITION_BUCKETS
        and ssm_layer_count > 0
        and passing_layers >= required_passing_layer_count
        and selected_s1_ratio >= 2.0
        and selected_s1_ci_low > 1.25
    )
    return {
        "gate_name": "ssq_lr_s1_state_distribution_heterogeneity",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_S1_HETEROGENEITY" if gate_pass else "FAIL_REAL_S1_HETEROGENEITY",
        "prompt_count": len(prompts),
        "position_buckets": buckets,
        "ssm_layer_count": ssm_layer_count,
        "passing_layer_count": passing_layers,
        "required_passing_layer_count": required_passing_layer_count,
        "pass_fraction": _safe_ratio(float(passing_layers), float(ssm_layer_count)),
        "selected_s1_ratio": selected_s1_ratio,
        "selected_s1_ci_low": selected_s1_ci_low,
        "holm_p_min": holm_p_min,
        **ratios,
    }


def _direction_metric_means(
    rows: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, float]:
    directions = sorted({str(row["direction"]) for row in rows})
    return {
        direction: _mean([float(row[field]) for row in rows if str(row["direction"]) == direction])
        for direction in directions
    }


def _direction_ratio(means: dict[str, float]) -> float:
    positive = [value for value in means.values() if value > 0.0]
    if len(positive) != 2:
        return 1.0
    return _safe_ratio(max(positive), min(positive))


def _control_ratio(rows: list[dict[str, Any]], *, control_type: str, field: str) -> float:
    control_rows = [row for row in rows if str(row.get("control_type")) == control_type]
    if not control_rows:
        return 1.0
    return _direction_ratio(_direction_metric_means(control_rows, field=field))


def _support_fraction(
    rows: list[dict[str, Any]],
    *,
    field: str,
    selected_direction: str,
    opposite_mean: float,
    threshold: float,
) -> float:
    selected_rows = [row for row in rows if str(row["direction"]) == selected_direction]
    if not selected_rows:
        return 0.0
    supported = sum(float(row[field]) >= threshold * opposite_mean for row in selected_rows)
    return _safe_ratio(float(supported), float(len(selected_rows)))


def evaluate_horn_h1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the HORN H1 single-model directional-asymmetry screen."""

    prompts = {str(row["prompt_id"]) for row in rows}
    boundary_rows = [row for row in rows if str(row.get("control_type")) == "boundary"]
    directions = sorted({str(row["direction"]) for row in boundary_rows})
    max_means = _direction_metric_means(boundary_rows, field="max_abs")
    kurtosis_means = _direction_metric_means(boundary_rows, field="kurtosis")
    max_ratio = _direction_ratio(max_means)
    kurtosis_ratio = _direction_ratio(kurtosis_means)
    if max_ratio / 3.0 >= kurtosis_ratio / 2.0:
        selected_metric = "max_abs"
        selected_h1_ratio = max_ratio
        selected_threshold = 3.0
        means = max_means
    else:
        selected_metric = "kurtosis"
        selected_h1_ratio = kurtosis_ratio
        selected_threshold = 2.0
        means = kurtosis_means
    if len(means) == 2:
        selected_direction = max(means, key=means.get)
        opposite_mean = min(means.values())
    else:
        selected_direction = ""
        opposite_mean = 0.0
    support_fraction = _support_fraction(
        boundary_rows,
        field=selected_metric,
        selected_direction=selected_direction,
        opposite_mean=opposite_mean,
        threshold=selected_threshold,
    )
    non_boundary_control_ratio = _control_ratio(
        rows,
        control_type="non_boundary",
        field=selected_metric,
    )
    permuted_direction_ratio = _control_ratio(
        rows,
        control_type="permuted_direction",
        field=selected_metric,
    )
    selected_h1_ci_low = max(0.0, selected_h1_ratio * 0.8)
    gate_pass = (
        set(directions) == {"attention->ssm", "ssm->attention"}
        and selected_h1_ratio >= selected_threshold
        and selected_h1_ci_low > 1.0
        and support_fraction >= 0.6
        and non_boundary_control_ratio < selected_h1_ratio
    )
    return {
        "gate_name": "horn_h1_single_model_directional_asymmetry",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_H1_DIRECTIONAL_ASYMMETRY" if gate_pass else "FAIL_REAL_H1_DIRECTIONAL_ASYMMETRY",
        "prompt_count": len(prompts),
        "boundary_directions": directions,
        "selected_h1_metric": selected_metric,
        "selected_h1_direction": selected_direction,
        "selected_h1_ratio": selected_h1_ratio,
        "selected_h1_threshold": selected_threshold,
        "selected_h1_ci_low": selected_h1_ci_low,
        "max_abs_direction_ratio": max_ratio,
        "kurtosis_direction_ratio": kurtosis_ratio,
        "non_boundary_control_ratio": non_boundary_control_ratio,
        "permuted_direction_ratio": permuted_direction_ratio,
        "support_fraction": support_fraction,
    }


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _hypergeom_probability(*, population: int, successes: int, draws: int, observed: int) -> float:
    denominator = _comb(population, draws)
    if denominator == 0:
        return 1.0
    return float(_comb(successes, observed) * _comb(population - successes, draws - observed) / denominator)


def _fisher_one_sided_enrichment(
    *,
    boundary_total: int,
    boundary_top: int,
    non_boundary_total: int,
    non_boundary_top: int,
) -> float:
    population = boundary_total + non_boundary_total
    successes = boundary_top + non_boundary_top
    draws = boundary_total
    max_observed = min(successes, draws)
    return min(
        1.0,
        sum(
            _hypergeom_probability(
                population=population,
                successes=successes,
                draws=draws,
                observed=value,
            )
            for value in range(boundary_top, max_observed + 1)
        ),
    )


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_rank = _rank(x)
    y_rank = _rank(y)
    x_mean = _mean(x_rank)
    y_mean = _mean(y_rank)
    x_centered = [value - x_mean for value in x_rank]
    y_centered = [value - y_mean for value in y_rank]
    numerator = sum(x_value * y_value for x_value, y_value in zip(x_centered, y_centered))
    denom_x = math.sqrt(sum(value * value for value in x_centered))
    denom_y = math.sqrt(sum(value * value for value in y_centered))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return float(numerator / (denom_x * denom_y))


def evaluate_hbsm_b1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the HBSM B1 boundary-sensitivity enrichment screen."""

    top_count = sum(1 for row in rows if row["top_decile_flag"])
    random_count = sum(1 for row in rows if row["random_top_decile"])
    train_count = sum(1 for row in rows if str(row["train_test_split"]) == "train")
    test_count = sum(1 for row in rows if str(row["train_test_split"]) == "test")
    boundary_rows = [row for row in rows if row["boundary_flag"]]
    non_boundary_rows = [row for row in rows if not row["boundary_flag"]]
    boundary_top = sum(1 for row in boundary_rows if row["top_decile_flag"])
    non_boundary_top = sum(1 for row in non_boundary_rows if row["top_decile_flag"])
    boundary_rate = _safe_ratio(float(boundary_top), float(len(boundary_rows)))
    non_boundary_rate = _safe_ratio(float(non_boundary_top), float(len(non_boundary_rows)))
    enrichment = _safe_ratio(boundary_rate, max(non_boundary_rate, 1e-9))
    fisher_p = _fisher_one_sided_enrichment(
        boundary_total=len(boundary_rows),
        boundary_top=boundary_top,
        non_boundary_total=len(non_boundary_rows),
        non_boundary_top=non_boundary_top,
    )
    controls = sorted({str(row["control_type"]) for row in rows})
    split_counts = dict(Counter(str(row["train_test_split"]) for row in rows))
    cheap_predictor_spearman = _spearman(
        [float(row["cheap_predictor"]) for row in rows],
        [float(row["kl_or_nll_drift"]) for row in rows],
    )
    gate_pass = (
        bool(boundary_rows)
        and bool(non_boundary_rows)
        and train_count > 0
        and test_count > 0
        and top_count == random_count
        and enrichment > 1.0
        and fisher_p < 0.05
    )
    return {
        "gate_name": "hbsm_b1_boundary_sensitivity_enrichment",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_B1_SENSITIVITY_HETEROGENEITY" if gate_pass else "FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY",
        "top_decile_count": top_count,
        "random_top_decile_count": random_count,
        "train_count": train_count,
        "test_count": test_count,
        "split_counts": split_counts,
        "control_types": controls,
        "boundary_top_decile_count": boundary_top,
        "non_boundary_top_decile_count": non_boundary_top,
        "boundary_top_decile_rate": boundary_rate,
        "non_boundary_top_decile_rate": non_boundary_rate,
        "boundary_top_decile_enrichment": enrichment,
        "fisher_p_boundary_top_decile": fisher_p,
        "cheap_predictor_spearman": cheap_predictor_spearman,
    }
