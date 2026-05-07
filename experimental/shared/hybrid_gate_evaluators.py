"""Deterministic evaluators for active hybrid-quantization Mac gates.

These functions do not run models and do not estimate uncertainty from raw
tensors. They turn already-reduced packet rows into the preregistered first-gate
decision fields that reviewers need to audit.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any


SSQ_LR_POSITION_BUCKETS = {"prefill_end", "2k_or_end", "8k_or_end", "final_minus_128"}
SSQ_LR_DISTRIBUTION_MIN_RATIO = 1.25


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


def _prompt_bucket_ratios(
    rows: list[dict[str, Any]],
    *,
    field: str,
    first_bucket: str = "prefill_end",
    final_bucket: str = "final_minus_128",
) -> list[float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["prompt_id"])][str(row["position_bucket"])].append(float(row[field]))
    ratios: list[float] = []
    for prompt_buckets in grouped.values():
        first = _mean(prompt_buckets.get(first_bucket, []))
        final = _mean(prompt_buckets.get(final_bucket, []))
        if first > 0.0:
            ratios.append(final / first)
    return ratios


def _bootstrap_mean_low(values: list[float], *, draws: int = 1000, quantile: float = 0.025) -> float:
    if not values:
        return 0.0
    rng = random.Random(1729)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    return float(means[int(quantile * (draws - 1))])


def _ks_2samp_pvalue(first: list[float], second: list[float]) -> float:
    """Return an asymptotic two-sample KS p-value without requiring scipy."""

    if len(first) < 2 or len(second) < 2:
        return 1.0
    first_sorted = sorted(float(value) for value in first)
    second_sorted = sorted(float(value) for value in second)
    i = j = 0
    n_first = len(first_sorted)
    n_second = len(second_sorted)
    statistic = 0.0
    values = sorted(set(first_sorted + second_sorted))
    for value in values:
        while i < n_first and first_sorted[i] <= value:
            i += 1
        while j < n_second and second_sorted[j] <= value:
            j += 1
        statistic = max(statistic, abs(i / n_first - j / n_second))
    if statistic <= 0.0:
        return 1.0
    effective_n = n_first * n_second / (n_first + n_second)
    lam = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic
    terms = [
        ((-1) ** (term_index - 1)) * math.exp(-2.0 * (term_index**2) * (lam**2))
        for term_index in range(1, 101)
    ]
    return min(1.0, max(0.0, 2.0 * sum(terms)))


def _holm_adjusted_pvalues(tests: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """Apply Holm correction and return ``(layer, field, adjusted_p)`` rows."""

    if not tests:
        return []
    ordered = sorted(tests, key=lambda item: item[2])
    adjusted: list[tuple[int, str, float]] = []
    running = 0.0
    total = len(ordered)
    for rank, (layer, field, p_value) in enumerate(ordered):
        corrected = min(1.0, p_value * (total - rank))
        running = max(running, corrected)
        adjusted.append((layer, field, running))
    return adjusted


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
    distribution_tests: list[tuple[int, str, float]] = []
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        max_values = _prompt_bucket_ratios(layer_rows, field="max_abs")
        std_values = _prompt_bucket_ratios(layer_rows, field="std")
        max_ratio = _mean(max_values)
        std_ratio = _mean(std_values)
        if max_ratio >= std_ratio:
            selected_layer_ratio = max_ratio
            selected_layer_low = _bootstrap_mean_low(max_values)
        else:
            selected_layer_ratio = std_ratio
            selected_layer_low = _bootstrap_mean_low(std_values)
        if selected_layer_ratio >= 2.0 and selected_layer_low > 1.25:
            passing_layers += 1
            selected_layer_lowers.append(selected_layer_low)
        for field in ["max_abs", "std", "kurtosis"]:
            first_values = [
                float(row[field])
                for row in layer_rows
                if str(row["position_bucket"]) == "prefill_end"
            ]
            final_values = [
                float(row[field])
                for row in layer_rows
                if str(row["position_bucket"]) == "final_minus_128"
            ]
            distribution_tests.append((layer, field, _ks_2samp_pvalue(first_values, final_values)))
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
    adjusted_tests = _holm_adjusted_pvalues(distribution_tests)
    holm_p_min = min((p_value for _, _, p_value in adjusted_tests), default=1.0)
    distribution_passing_layers = len(
        {layer for layer, _, p_value in adjusted_tests if p_value < 0.01}
    )
    magnitude_gate_pass = (
        set(buckets) == SSQ_LR_POSITION_BUCKETS
        and ssm_layer_count > 0
        and passing_layers >= required_passing_layer_count
        and selected_s1_ratio >= 2.0
        and selected_s1_ci_low > 1.25
    )
    distribution_effect_floor_pass = selected_s1_ratio >= SSQ_LR_DISTRIBUTION_MIN_RATIO
    distribution_gate_pass = (
        set(buckets) == SSQ_LR_POSITION_BUCKETS
        and ssm_layer_count > 0
        and distribution_passing_layers >= required_passing_layer_count
        and holm_p_min < 0.01
        and distribution_effect_floor_pass
    )
    gate_pass = magnitude_gate_pass or distribution_gate_pass
    return {
        "gate_name": "ssq_lr_s1_state_distribution_heterogeneity",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_S1_HETEROGENEITY" if gate_pass else "FAIL_REAL_S1_HETEROGENEITY",
        "prompt_count": len(prompts),
        "position_buckets": buckets,
        "ssm_layer_count": ssm_layer_count,
        "passing_layer_count": passing_layers,
        "distribution_passing_layer_count": distribution_passing_layers,
        "required_passing_layer_count": required_passing_layer_count,
        "pass_fraction": _safe_ratio(float(passing_layers), float(ssm_layer_count)),
        "selected_s1_ratio": selected_s1_ratio,
        "selected_s1_ci_low": selected_s1_ci_low,
        "holm_p_min": holm_p_min,
        "magnitude_gate_pass": magnitude_gate_pass,
        "distribution_effect_floor_pass": distribution_effect_floor_pass,
        "distribution_gate_pass": distribution_gate_pass,
        **ratios,
    }


def _horn_direction_label(row: dict[str, Any]) -> str:
    control_type = str(row.get("control_type", ""))
    if control_type == "non_boundary":
        matched = str(row.get("matched_boundary_direction", "")).strip()
        if matched:
            return matched
    return str(row["direction"])


def _direction_metric_means(
    rows: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, float]:
    directions = sorted({_horn_direction_label(row) for row in rows})
    return {
        direction: _mean([float(row[field]) for row in rows if _horn_direction_label(row) == direction])
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


def _selected_direction_control_ratio(
    rows: list[dict[str, Any]],
    *,
    control_type: str,
    field: str,
    selected_direction: str,
) -> float:
    """Measure whether a control preserves the selected boundary direction.

    A faithful permutation flips direction labels while keeping values tied to
    the observed tuple. Its max/min asymmetry can therefore remain large even
    though the selected direction no longer carries the high-magnitude signal.
    The H1 null should reject only controls that preserve the same selected
    direction, not controls that merely preserve unsigned asymmetry magnitude.
    """

    control_rows = [row for row in rows if str(row.get("control_type")) == control_type]
    if not control_rows or not selected_direction:
        return 1.0
    means = _direction_metric_means(control_rows, field=field)
    if selected_direction not in means or len(means) != 2:
        return 1.0
    opposite_values = [value for direction, value in means.items() if direction != selected_direction]
    if not opposite_values:
        return 1.0
    return _safe_ratio(means[selected_direction], opposite_values[0])


def _direction_count(rows: list[dict[str, Any]], *, control_type: str) -> int:
    return len({_horn_direction_label(row) for row in rows if str(row.get("control_type")) == control_type})


def _prompt_direction_ratio_low(
    rows: list[dict[str, Any]],
    *,
    field: str,
) -> float:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    ratios = [
        _direction_ratio(_direction_metric_means(prompt_rows, field=field))
        for prompt_rows in grouped.values()
        if _direction_count(prompt_rows, control_type="boundary") >= 2
    ]
    if not ratios:
        return 0.0
    ratios = sorted(ratios)
    index = max(0, math.floor(0.05 * (len(ratios) - 1)))
    return float(ratios[index])


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
    non_boundary_control_ratio = _selected_direction_control_ratio(
        rows,
        control_type="non_boundary",
        field=selected_metric,
        selected_direction=selected_direction,
    )
    permuted_direction_ratio = _selected_direction_control_ratio(
        rows,
        control_type="permuted_direction",
        field=selected_metric,
        selected_direction=selected_direction,
    )
    selected_h1_ci_low = _prompt_direction_ratio_low(boundary_rows, field=selected_metric)
    non_boundary_direction_count = _direction_count(rows, control_type="non_boundary")
    permuted_direction_count = _direction_count(rows, control_type="permuted_direction")
    gate_pass = (
        set(directions) == {"attention->ssm", "ssm->attention"}
        and selected_h1_ratio >= selected_threshold
        and selected_h1_ci_low > 1.0
        and support_fraction >= 0.6
        and non_boundary_direction_count >= 2
        and permuted_direction_count >= 2
        and non_boundary_control_ratio < selected_threshold
        and permuted_direction_ratio <= 1.0
    )
    return {
        "gate_name": "horn_h1a_single_model_directional_asymmetry_screen",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN" if gate_pass else "FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN",
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
        "non_boundary_direction_count": non_boundary_direction_count,
        "permuted_direction_count": permuted_direction_count,
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


def _aggregate_hbsm_primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model_id", "")), int(row["layer"]))].append(row)
    aggregated: list[dict[str, Any]] = []
    for (model_id, layer), layer_rows in sorted(grouped.items(), key=lambda item: item[0]):
        aggregated.append(
            {
                "model_id": model_id,
                "layer": layer,
                "boundary_flag": any(bool(row["boundary_flag"]) for row in layer_rows),
                "top_decile_flag": any(bool(row["top_decile_flag"]) for row in layer_rows),
                "random_top_decile": any(bool(row["random_top_decile"]) for row in layer_rows),
                "train_test_split": (
                    "test"
                    if any(str(row["train_test_split"]) == "test" for row in layer_rows)
                    else str(layer_rows[0]["train_test_split"])
                ),
                "cheap_predictor": _mean([float(row["cheap_predictor"]) for row in layer_rows]),
                "kl_or_nll_drift": _mean([float(row["kl_or_nll_drift"]) for row in layer_rows]),
                "parameter_count": _mean([float(row.get("parameter_count", 0.0)) for row in layer_rows]),
                "weight_norm": _mean([float(row.get("weight_norm", 0.0)) for row in layer_rows]),
                "prompt_count": len({str(row.get("prompt_id")) for row in layer_rows}),
            }
        )
    return aggregated


def _derive_hbsm_top_decile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    expected_top_decile_count = math.ceil(0.10 * len(rows))
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["kl_or_nll_drift"]),
            str(row["model_id"]),
            int(row["layer"]),
        ),
    )
    measured_top_keys = {
        (str(row["model_id"]), int(row["layer"]))
        for row in ranked[:expected_top_decile_count]
    }
    derived: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["top_decile_flag"] = (
            str(row["model_id"]),
            int(row["layer"]),
        ) in measured_top_keys
        derived.append(copied)
    return derived


def _hbsm_control_spearman(
    rows: list[dict[str, Any]],
    control_type: str,
    scoring_rows: list[dict[str, Any]],
) -> float:
    control_by_key = {
        (str(row.get("model_id", "")), int(row["layer"])): row
        for row in rows
        if str(row.get("control_type")) == control_type and "layer" in row
    }
    scoring_keys = [(str(row["model_id"]), int(row["layer"])) for row in scoring_rows]
    if len(scoring_keys) < 2 or set(control_by_key) != set(scoring_keys):
        return 0.0
    control_rows = [control_by_key[key] for key in scoring_keys]
    return _spearman(
        [float(row.get("cheap_predictor", 0.0)) for row in control_rows],
        [float(row["kl_or_nll_drift"]) for row in scoring_rows],
    )


def evaluate_hbsm_b1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the HBSM B1 boundary-sensitivity enrichment screen."""

    primary_rows = [row for row in rows if str(row["control_type"]) == "boundary_only"]
    scoring_rows = _aggregate_hbsm_primary_rows(primary_rows) if primary_rows else rows
    scoring_rows = _derive_hbsm_top_decile(scoring_rows)
    expected_top_decile_count = math.ceil(0.10 * len(scoring_rows)) if scoring_rows else 0
    top_count = sum(1 for row in scoring_rows if row["top_decile_flag"])
    random_count = sum(1 for row in scoring_rows if row["random_top_decile"])
    train_count = sum(1 for row in scoring_rows if str(row["train_test_split"]) == "train")
    test_count = sum(1 for row in scoring_rows if str(row["train_test_split"]) == "test")
    boundary_rows = [row for row in scoring_rows if row["boundary_flag"]]
    non_boundary_rows = [row for row in scoring_rows if not row["boundary_flag"]]
    boundary_top = sum(1 for row in boundary_rows if row["top_decile_flag"])
    non_boundary_top = sum(1 for row in non_boundary_rows if row["top_decile_flag"])
    boundary_random_top = sum(1 for row in boundary_rows if row["random_top_decile"])
    non_boundary_random_top = sum(1 for row in non_boundary_rows if row["random_top_decile"])
    boundary_rate = _safe_ratio(float(boundary_top), float(len(boundary_rows)))
    non_boundary_rate = _safe_ratio(float(non_boundary_top), float(len(non_boundary_rows)))
    enrichment = _safe_ratio(boundary_rate, max(non_boundary_rate, 1e-9))
    random_boundary_rate = _safe_ratio(float(boundary_random_top), float(len(boundary_rows)))
    random_non_boundary_rate = _safe_ratio(float(non_boundary_random_top), float(len(non_boundary_rows)))
    random_enrichment = _safe_ratio(random_boundary_rate, max(random_non_boundary_rate, 1e-9))
    fisher_p = _fisher_one_sided_enrichment(
        boundary_total=len(boundary_rows),
        boundary_top=boundary_top,
        non_boundary_total=len(non_boundary_rows),
        non_boundary_top=non_boundary_top,
    )
    random_fisher_p = _fisher_one_sided_enrichment(
        boundary_total=len(boundary_rows),
        boundary_top=boundary_random_top,
        non_boundary_total=len(non_boundary_rows),
        non_boundary_top=non_boundary_random_top,
    )
    controls = sorted({str(row["control_type"]) for row in rows})
    split_counts = dict(Counter(str(row["train_test_split"]) for row in scoring_rows))
    cheap_predictor_spearman = _spearman(
        [float(row["cheap_predictor"]) for row in scoring_rows],
        [float(row["kl_or_nll_drift"]) for row in scoring_rows],
    )
    baseline_spearman = {
        "layer_index": _spearman(
            [float(row["layer"]) for row in scoring_rows],
            [float(row["kl_or_nll_drift"]) for row in scoring_rows],
        ),
        "parameter_count_norm": _spearman(
            [float(row.get("parameter_count", 0.0)) for row in scoring_rows],
            [float(row["kl_or_nll_drift"]) for row in scoring_rows],
        ),
        "weight_norm": _spearman(
            [float(row.get("weight_norm", 0.0)) for row in scoring_rows],
            [float(row["kl_or_nll_drift"]) for row in scoring_rows],
        ),
        "boundary_flag": _spearman(
            [1.0 if row["boundary_flag"] else 0.0 for row in scoring_rows],
            [float(row["kl_or_nll_drift"]) for row in scoring_rows],
        ),
        "kl_lens_rank": _hbsm_control_spearman(rows, "kl_lens_rank", scoring_rows),
        "activation_outlier": _hbsm_control_spearman(rows, "activation_outlier", scoring_rows),
    }
    prompt_count = len({str(row.get("prompt_id")) for row in primary_rows if "prompt_id" in row})
    scoring_layer_count = len(scoring_rows)
    gate_pass = (
        bool(boundary_rows)
        and bool(non_boundary_rows)
        and train_count > 0
        and test_count > 0
        and top_count == expected_top_decile_count
        and random_count == expected_top_decile_count
        and enrichment > 1.0
        and enrichment > random_enrichment
        and fisher_p < 0.05
        and random_fisher_p >= 0.05
    )
    return {
        "gate_name": "hbsm_b1_boundary_sensitivity_enrichment",
        "gate_pass": gate_pass,
        "gate_status": "PASS_REAL_B1_SENSITIVITY_HETEROGENEITY" if gate_pass else "FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY",
        "primary_row_count": len(primary_rows),
        "scoring_layer_count": scoring_layer_count,
        "prompt_count": prompt_count,
        "expected_top_decile_count": expected_top_decile_count,
        "top_decile_count": top_count,
        "random_top_decile_count": random_count,
        "train_count": train_count,
        "test_count": test_count,
        "split_counts": split_counts,
        "control_types": controls,
        "boundary_top_decile_count": boundary_top,
        "non_boundary_top_decile_count": non_boundary_top,
        "boundary_random_top_decile_count": boundary_random_top,
        "non_boundary_random_top_decile_count": non_boundary_random_top,
        "boundary_top_decile_rate": boundary_rate,
        "non_boundary_top_decile_rate": non_boundary_rate,
        "boundary_random_top_decile_rate": random_boundary_rate,
        "non_boundary_random_top_decile_rate": random_non_boundary_rate,
        "boundary_top_decile_enrichment": enrichment,
        "random_boundary_top_decile_enrichment": random_enrichment,
        "fisher_p_boundary_top_decile": fisher_p,
        "fisher_p_random_boundary_top_decile": random_fisher_p,
        "cheap_predictor_spearman": cheap_predictor_spearman,
        "baseline_spearman": baseline_spearman,
        "cheap_predictor_margin_vs_best_baseline": cheap_predictor_spearman - max(baseline_spearman.values(), default=0.0),
    }
