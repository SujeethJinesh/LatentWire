#!/usr/bin/env python3
"""Offline RiskGuard calibration for Granite M11b top-5 cached data."""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts/riskguard"
PACK_DIR = ROOT / "artifacts/external_review_pack/om_positive_method_pack_20260528_0536"
RUN_DIR = ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z"

BOOTSTRAP_SEED = 20260527
BOOTSTRAP_SAMPLES = 1000
SELECTED_POSITIONS = tuple(list(range(100, 1001, 100)) + [2000, 5000, 10000])
EARLY_POSITIONS = tuple(range(100, 1001, 100))
EPS = 1e-12


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap_median_ci(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(BOOTSTRAP_SEED)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(median(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def worst_quartile_cvar(values: list[float]) -> float | None:
    if not values:
        return None
    count = max(1, math.ceil(len(values) * 0.25))
    return float(mean(sorted(values)[:count]))


def jaccard_distance(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def load_recoveries() -> dict[int, dict[str, Any]]:
    rows = {}
    for row in load_json(RUN_DIR / "per_trace_metrics.json")["traces"]:
        idx = int(row["prompt_index"])
        rows[idx] = {
            "prompt_id": row["prompt_id"],
            "no_gap": bool(row["no_recoverable_static_gap"]),
            "static_gap": float(row["static_gap"]),
            "recoveries": row["recoveries"],
            "perplexities": row["perplexities"],
            "mean_nll": row["mean_nll"],
        }
    return rows


def load_global_top5_sets() -> dict[int, set[int]]:
    protected = load_json(RUN_DIR / "protected_sets.json")
    layers = protected["regimes"]["m11b_top5"]["layers"]
    return {int(layer): set(map(int, spec["protected_channels"])) for layer, spec in layers.items()}


def activation_features(global_top5: dict[int, set[int]]) -> dict[int, dict[str, float]]:
    row_stats: dict[int, dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    wanted = {f'"decode_position": {position},' for position in SELECTED_POSITIONS}
    activation_path = RUN_DIR / "activation_magnitudes.jsonl.gz"

    with gzip.open(activation_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not any(marker in line for marker in wanted):
                continue
            row = json.loads(line)
            position = int(row["decode_position"])
            if position not in SELECTED_POSITIONS:
                continue
            prompt_idx = int(row["prompt_index"])
            layer_idx = int(row["layer_index"])
            values = [float(value) for value in row["channel_magnitudes"]]
            channel_count = int(row["channel_count"])
            k = max(1, math.ceil(channel_count * 0.05))
            top_indices = heapq.nlargest(k + 1, range(channel_count), key=values.__getitem__)
            top_set = set(top_indices[:k])
            top_values = [values[channel] for channel in top_indices[:k]]
            kth = values[top_indices[k - 1]]
            kth_plus_one = values[top_indices[k]] if len(top_indices) > k else 0.0
            total = sum(values)
            mean_value = total / channel_count
            global_set = global_top5[layer_idx]
            row_stats[prompt_idx][layer_idx][position] = {
                "top_set": top_set,
                "mean_value": mean_value,
                "max_value": max(top_values) if top_values else 0.0,
                "top1_to_mean": (max(top_values) if top_values else 0.0) / (mean_value + EPS),
                "top5_mass_fraction": sum(top_values) / (total + EPS),
                "margin_norm": (kth - kth_plus_one) / (abs(kth) + EPS),
                "global_overlap": len(top_set & global_set) / len(top_set | global_set),
            }

    features: dict[int, dict[str, float]] = {}
    for prompt_idx, layers in sorted(row_stats.items()):
        early_churn: list[float] = []
        early_margin: list[float] = []
        early_concentration: list[float] = []
        early_top1_ratio: list[float] = []
        early_max_abs: list[float] = []
        early_global_overlap: list[float] = []
        pos1000_global_overlap: list[float] = []
        final_global_overlap: list[float] = []
        initial_to_final_drift: list[float] = []
        p1000_to_final_drift: list[float] = []
        scale_growth_1000_over_100: list[float] = []
        mean_growth_1000_over_100: list[float] = []

        for by_position in layers.values():
            early_positions = [pos for pos in EARLY_POSITIONS if pos in by_position]
            for left, right in zip(early_positions, early_positions[1:]):
                early_churn.append(jaccard_distance(by_position[left]["top_set"], by_position[right]["top_set"]))
            for pos in early_positions:
                stat = by_position[pos]
                early_margin.append(float(stat["margin_norm"]))
                early_concentration.append(float(stat["top5_mass_fraction"]))
                early_top1_ratio.append(float(stat["top1_to_mean"]))
                early_max_abs.append(float(stat["max_value"]))
                early_global_overlap.append(float(stat["global_overlap"]))
            if 1000 in by_position:
                pos1000_global_overlap.append(float(by_position[1000]["global_overlap"]))
            if 10000 in by_position:
                final_global_overlap.append(float(by_position[10000]["global_overlap"]))
            if 100 in by_position and 10000 in by_position:
                initial_to_final_drift.append(jaccard_distance(by_position[100]["top_set"], by_position[10000]["top_set"]))
            if 1000 in by_position and 10000 in by_position:
                p1000_to_final_drift.append(jaccard_distance(by_position[1000]["top_set"], by_position[10000]["top_set"]))
            if 100 in by_position and 1000 in by_position:
                scale_growth_1000_over_100.append(
                    float(by_position[1000]["top1_to_mean"]) / (float(by_position[100]["top1_to_mean"]) + EPS)
                )
                mean_growth_1000_over_100.append(
                    float(by_position[1000]["mean_value"]) / (float(by_position[100]["mean_value"]) + EPS)
                )

        def avg(values: list[float]) -> float:
            return float(mean(values)) if values else float("nan")

        def min_or_nan(values: list[float]) -> float:
            return float(min(values)) if values else float("nan")

        def max_or_nan(values: list[float]) -> float:
            return float(max(values)) if values else float("nan")

        features[prompt_idx] = {
            "early_churn_mean": avg(early_churn),
            "early_margin_mean": avg(early_margin),
            "early_margin_min": min_or_nan(early_margin),
            "early_top5_mass_fraction_mean": avg(early_concentration),
            "early_top1_to_mean_max": max_or_nan(early_top1_ratio),
            "early_max_abs_max": max_or_nan(early_max_abs),
            "early_global_overlap_mean": avg(early_global_overlap),
            "pos1000_global_overlap_mean": avg(pos1000_global_overlap),
            "final_global_overlap_mean": avg(final_global_overlap),
            "initial_to_final_drift_mean": avg(initial_to_final_drift),
            "p1000_to_final_drift_mean": avg(p1000_to_final_drift),
            "scale_growth_1000_over_100_mean": avg(scale_growth_1000_over_100),
            "mean_growth_1000_over_100_mean": avg(mean_growth_1000_over_100),
        }
    return features


def candidate_results(
    recoveries: dict[int, dict[str, Any]],
    features: dict[int, dict[str, float]],
    feature_names: list[str],
    included_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    included = included_indices or [idx for idx, row in recoveries.items() if not row["no_gap"]]
    results: list[dict[str, Any]] = []
    for feature in feature_names:
        values = [(idx, features[idx][feature]) for idx in included if not math.isnan(features[idx][feature])]
        unique = sorted({value for _, value in values})
        if len(unique) < 2:
            continue
        thresholds = [(unique[i] + unique[i + 1]) / 2.0 for i in range(len(unique) - 1)]
        thresholds = [unique[0] - 1e-12, *thresholds, unique[-1] + 1e-12]
        for direction in ("<=", ">="):
            for threshold in thresholds:
                flagged = {
                    idx
                    for idx, value in values
                    if (value <= threshold if direction == "<=" else value >= threshold)
                }
                if not flagged:
                    continue
                before = [float(recoveries[idx]["recoveries"]["m11b_top5"]) for idx in included]
                after = [
                    0.0 if idx in flagged else float(recoveries[idx]["recoveries"]["m11b_top5"])
                    for idx in included
                ]
                ci = bootstrap_median_ci(after)
                positives_flagged = sum(
                    1 for idx in flagged if float(recoveries[idx]["recoveries"]["m11b_top5"]) > 0.0
                )
                negatives_flagged = sum(
                    1 for idx in flagged if float(recoveries[idx]["recoveries"]["m11b_top5"]) < 0.0
                )
                results.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "flagged": sorted(flagged),
                        "flagged_count": len(flagged),
                        "negatives_flagged": negatives_flagged,
                        "positives_flagged": positives_flagged,
                        "before_median": float(median(before)),
                        "after_median": float(median(after)),
                        "after_ci_low": ci["ci95_low"],
                        "after_ci_high": ci["ci95_high"],
                        "before_cvar_worst_quartile": worst_quartile_cvar(before),
                        "after_cvar_worst_quartile": worst_quartile_cvar(after),
                    }
                )
    results.sort(
        key=lambda item: (
            float("-inf") if item["after_ci_low"] is None else float(item["after_ci_low"]),
            float(item["after_median"]),
            float(item["after_cvar_worst_quartile"]),
            -int(item["flagged_count"]),
        ),
        reverse=True,
    )
    return results


def leave_one_out_validation(
    recoveries: dict[int, dict[str, Any]],
    features: dict[int, dict[str, float]],
    feature_names: list[str],
) -> dict[str, Any]:
    included = [idx for idx, row in recoveries.items() if not row["no_gap"]]
    rows: list[dict[str, Any]] = []
    for heldout in included:
        train = [idx for idx in included if idx != heldout]
        candidates = candidate_results(recoveries, features, feature_names, included_indices=train)
        if not candidates:
            before = float(recoveries[heldout]["recoveries"]["m11b_top5"])
            rows.append(
                {
                    "trace_id": heldout,
                    "before": before,
                    "after": before,
                    "triggered": False,
                    "trained_feature": None,
                    "trained_direction": None,
                    "trained_threshold": None,
                    "heldout_feature_value": None,
                }
            )
            continue
        best = candidates[0]
        value = float(features[heldout][best["feature"]])
        triggered = value <= best["threshold"] if best["direction"] == "<=" else value >= best["threshold"]
        before = float(recoveries[heldout]["recoveries"]["m11b_top5"])
        rows.append(
            {
                "trace_id": heldout,
                "before": before,
                "after": 0.0 if triggered else before,
                "triggered": bool(triggered),
                "trained_feature": best["feature"],
                "trained_direction": best["direction"],
                "trained_threshold": best["threshold"],
                "heldout_feature_value": value,
            }
        )
    after = [float(row["after"]) for row in rows]
    return {
        "rows": rows,
        "median": float(median(after)) if after else None,
        "ci": bootstrap_median_ci(after),
        "cvar_worst_quartile": worst_quartile_cvar(after),
        "negative_trace_ids_after": [int(row["trace_id"]) for row in rows if float(row["after"]) < 0.0],
        "triggered_trace_ids": [int(row["trace_id"]) for row in rows if row["triggered"]],
    }


def oracle_summaries(recoveries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    included = [idx for idx, row in recoveries.items() if not row["no_gap"]]
    before = [float(recoveries[idx]["recoveries"]["m11b_top5"]) for idx in included]
    cap_negative = [max(0.0, value) for value in before]
    bf16_fallback = [1.0 if value < 0.0 else value for value in before]
    return {
        "before": {
            "values": before,
            "median": float(median(before)),
            "ci": bootstrap_median_ci(before),
            "cvar_worst_quartile": worst_quartile_cvar(before),
        },
        "oracle_negative_to_static": {
            "values": cap_negative,
            "median": float(median(cap_negative)),
            "ci": bootstrap_median_ci(cap_negative),
            "cvar_worst_quartile": worst_quartile_cvar(cap_negative),
        },
        "oracle_negative_to_bf16": {
            "values": bf16_fallback,
            "median": float(median(bf16_fallback)),
            "ci": bootstrap_median_ci(bf16_fallback),
            "cvar_worst_quartile": worst_quartile_cvar(bf16_fallback),
        },
    }


def render_report(
    recoveries: dict[int, dict[str, Any]],
    features: dict[int, dict[str, float]],
    early_results: list[dict[str, Any]],
    full_source_results: list[dict[str, Any]],
    loo: dict[str, Any],
    oracles: dict[str, Any],
) -> str:
    best_early = early_results[0] if early_results else None
    best_full = full_source_results[0] if full_source_results else None
    source_files = [
        str(PACK_DIR / "EXECUTIVE_SUMMARY.md"),
        str(PACK_DIR / "DECISIONS_SNAPSHOT.md"),
        str(PACK_DIR / "experiments/05_m11b_granite/decision.json"),
        str(PACK_DIR / "experiments/05_m11b_granite/per_trace.csv"),
        str(RUN_DIR / "per_trace_metrics.json"),
        str(RUN_DIR / "score_cache"),
        str(RUN_DIR / "activation_magnitudes.jsonl.gz"),
        str(RUN_DIR / "protected_sets.json"),
        str(RUN_DIR / "protected_trajectories.json"),
    ]
    lines = [
        "# RiskGuard Granite M11b Top-5 Calibration",
        "",
        "## Method",
        "",
        "Offline-only calibration on cached Granite M11b top-5 data. Candidate guards set predicted recovery to the static-1pct baseline value (0.0) when triggered, then recompute the preregistered bootstrap median CI with seed 20260527. This tests whether a cheap guard could cap negative tails without using recovery labels at deployment time.",
        "",
        "## Available Data",
        "",
        "- Final per-trace score caches are present for BF16, static_1pct, m11b_top1/top5/top10, and static_top10.",
        "- Per-step or streaming PPL/loss ratios are not present; final loss ratios are diagnostic only and were not used for deployable promotion.",
        "- Source activation magnitudes are present for all 12 prompts, 40 layers, every 100 decode positions from 100 to 10000.",
        "- Protected-set trajectories are present, but the M11b policy builds global EMA sets from mean activation magnitudes across traces; per-trace churn here is recomputed from cached source activations.",
        "- No target-side per-step activations, logits, top-k margins, or loss deltas are cached.",
        "",
        "## Baseline Tail",
        "",
        f"- Included recoverable traces: {sum(1 for row in recoveries.values() if not row['no_gap'])}/12.",
        f"- M11b top-5 median: {oracles['before']['median']:.6f}.",
        f"- M11b top-5 bootstrap CI: [{oracles['before']['ci']['ci95_low']:.6f}, {oracles['before']['ci']['ci95_high']:.6f}].",
        f"- Worst-quartile CVaR: {oracles['before']['cvar_worst_quartile']:.6f}.",
        "",
        "## Trigger Candidates",
        "",
    ]
    if best_early:
        lines.extend(
            [
                "Best early source-activation trigger:",
                f"- Rule: `{best_early['feature']} {best_early['direction']} {best_early['threshold']:.9g}`.",
                f"- Flagged recoverable traces: {best_early['flagged']} ({best_early['negatives_flagged']} negative, {best_early['positives_flagged']} positive).",
                f"- Predicted post-guard median: {best_early['after_median']:.6f}.",
                f"- Predicted post-guard CI: [{best_early['after_ci_low']:.6f}, {best_early['after_ci_high']:.6f}].",
                f"- Predicted worst-quartile CVaR: {best_early['after_cvar_worst_quartile']:.6f}.",
                "",
            ]
        )
    if best_full:
        lines.extend(
            [
                "Best full-source activation trigger, using position 10000 source activations but no target outcome:",
                f"- Rule: `{best_full['feature']} {best_full['direction']} {best_full['threshold']:.9g}`.",
                f"- Flagged recoverable traces: {best_full['flagged']} ({best_full['negatives_flagged']} negative, {best_full['positives_flagged']} positive).",
                f"- Predicted post-guard median: {best_full['after_median']:.6f}.",
                f"- Predicted post-guard CI: [{best_full['after_ci_low']:.6f}, {best_full['after_ci_high']:.6f}].",
                f"- Predicted worst-quartile CVaR: {best_full['after_cvar_worst_quartile']:.6f}.",
                "",
            ]
        )
    lines.extend(
        [
            "Robustness check:",
            f"- Leave-one-trace-out nested calibration gives median {loo['median']:.6f}, CI [{loo['ci']['ci95_low']:.6f}, {loo['ci']['ci95_high']:.6f}], CVaR {loo['cvar_worst_quartile']:.6f}.",
            f"- LOO-triggered traces: {loo['triggered_trace_ids']}; remaining negative traces after LOO guard: {loo['negative_trace_ids_after']}.",
            "- Interpretation: the positive in-sample CI depends on a threshold selected from the same 8 recoverable outcomes; when each trace is withheld, the guard misses one catastrophic trace and adds positive-trace fallbacks.",
            "",
            "Oracle diagnostics, not deployable:",
            f"- Replacing exactly negative M11b top-5 traces with static baseline gives median {oracles['oracle_negative_to_static']['median']:.6f}, CI [{oracles['oracle_negative_to_static']['ci']['ci95_low']:.6f}, {oracles['oracle_negative_to_static']['ci']['ci95_high']:.6f}], CVaR {oracles['oracle_negative_to_static']['cvar_worst_quartile']:.6f}.",
            f"- Replacing exactly negative traces with BF16/full-precision gives median {oracles['oracle_negative_to_bf16']['median']:.6f}, CI [{oracles['oracle_negative_to_bf16']['ci']['ci95_low']:.6f}, {oracles['oracle_negative_to_bf16']['ci']['ci95_high']:.6f}], CVaR {oracles['oracle_negative_to_bf16']['cvar_worst_quartile']:.6f}.",
            "",
            "## Decision",
            "",
            "Do not promote to GPU confirmation. A source-activation churn trigger has positive in-sample CI, but the cutoff is outcome-fit on 8 traces and fails the leave-one-trace-out robustness check. This is not defensible enough to spend GPU confirmation time.",
            "",
            "## Caveats",
            "",
            "- All threshold search is in-sample over 8 recoverable traces; it is suitable for falsification/triage, not a claim.",
            "- Source activation features may be deployable only in protocols that already run the source/BF16 pass before target scoring.",
            "- Final score-cache loss ratios are available but intentionally excluded from promotion because they require outcome scoring and leak the quantity the guard is supposed to predict.",
            "- No GPU was used.",
            "",
            "## Source Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in source_files)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recoveries = load_recoveries()
    global_top5 = load_global_top5_sets()
    features = activation_features(global_top5)

    early_feature_names = [
        "early_churn_mean",
        "early_margin_mean",
        "early_margin_min",
        "early_top5_mass_fraction_mean",
        "early_top1_to_mean_max",
        "early_max_abs_max",
        "early_global_overlap_mean",
        "pos1000_global_overlap_mean",
        "scale_growth_1000_over_100_mean",
        "mean_growth_1000_over_100_mean",
    ]
    full_source_feature_names = [
        *early_feature_names,
        "final_global_overlap_mean",
        "initial_to_final_drift_mean",
        "p1000_to_final_drift_mean",
    ]
    early_results = candidate_results(recoveries, features, early_feature_names)
    full_source_results = candidate_results(recoveries, features, full_source_feature_names)
    loo = leave_one_out_validation(recoveries, features, early_feature_names)
    oracles = oracle_summaries(recoveries)
    best = early_results[0] if early_results else None
    loo_by_trace = {int(row["trace_id"]): row for row in loo["rows"]}

    metrics_rows = []
    for idx in sorted(recoveries):
        row = recoveries[idx]
        before = row["recoveries"]["m11b_top5"]
        triggered = bool(best and idx in set(best["flagged"]))
        after = None if before is None else (0.0 if triggered else float(before))
        metric_row = {
            "trace_id": idx,
            "prompt_id": row["prompt_id"],
            "no_gap_flag": row["no_gap"],
            "m11b_top5_recovery_before": before,
            "in_sample_best_triggered": triggered,
            "in_sample_predicted_recovery_after_static_fallback": after,
            "loo_triggered": bool(loo_by_trace.get(idx, {}).get("triggered", False)),
            "loo_predicted_recovery_after_static_fallback": loo_by_trace.get(idx, {}).get("after"),
            "loo_trained_feature": loo_by_trace.get(idx, {}).get("trained_feature"),
            "loo_trained_direction": loo_by_trace.get(idx, {}).get("trained_direction"),
            "loo_trained_threshold": loo_by_trace.get(idx, {}).get("trained_threshold"),
            "static_gap": row["static_gap"],
        }
        metric_row.update(features.get(idx, {}))
        metrics_rows.append(metric_row)

    with (OUT_DIR / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(metrics_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    report = render_report(recoveries, features, early_results, full_source_results, loo, oracles)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    reason = (
        "Best cached-signal trigger clears ci_low only in-sample; leave-one-trace-out ci_low is not > 0 "
        "and one catastrophic trace remains negative, so the trigger is not defensible for GPU confirmation."
    )
    decision = {
        "status": "DEFER_IN_SAMPLE_RAZOR_TRIGGER",
        "promote_to_gpu_confirm": False,
        "predicted_ci_low": loo["ci"]["ci95_low"],
        "trigger": None
        if not best
        else {
            "name": "best_in_sample_early_source_activation_static_fallback",
            "feature": best["feature"],
            "direction": best["direction"],
            "threshold": best["threshold"],
            "flagged_recoverable_traces": best["flagged"],
            "predicted_median": best["after_median"],
            "predicted_ci_low_in_sample": best["after_ci_low"],
            "predicted_ci_high": best["after_ci_high"],
            "predicted_worst_quartile_cvar": best["after_cvar_worst_quartile"],
            "leave_one_out_ci_low": loo["ci"]["ci95_low"],
            "leave_one_out_median": loo["median"],
            "leave_one_out_worst_quartile_cvar": loo["cvar_worst_quartile"],
            "deployability": "requires cached/online source activation magnitudes through decode position <=1000; no target outcomes",
            "promoted": False,
        },
        "reason": reason,
        "source_files": [
            str(PACK_DIR / "EXECUTIVE_SUMMARY.md"),
            str(PACK_DIR / "DECISIONS_SNAPSHOT.md"),
            str(PACK_DIR / "experiments/05_m11b_granite/decision.json"),
            str(PACK_DIR / "experiments/05_m11b_granite/per_trace.csv"),
            str(RUN_DIR / "per_trace_metrics.json"),
            str(RUN_DIR / "score_cache"),
            str(RUN_DIR / "activation_magnitudes.jsonl.gz"),
            str(RUN_DIR / "protected_sets.json"),
            str(RUN_DIR / "protected_trajectories.json"),
        ],
        "diagnostics": {
            "baseline": oracles["before"],
            "oracle_negative_to_static": oracles["oracle_negative_to_static"],
            "oracle_negative_to_bf16": oracles["oracle_negative_to_bf16"],
            "leave_one_out_validation": loo,
            "best_full_source_activation_trigger": None if not full_source_results else full_source_results[0],
        },
    }
    write_json(OUT_DIR / "decision.json", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
