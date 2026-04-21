"""Confidence-gated stochastic route expansion with interpretable telemetry."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from typing import Any, Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.evaluate import (  # noqa: E402
    add_paired_prediction_summary,
    write_prediction_records,
    write_prediction_sidecar,
)
from scripts.aggregate_stochastic_routes import load_records  # noqa: E402
from scripts.rerank_stochastic_routes import (  # noqa: E402
    _annotate_candidates,
    _candidate_metadata,
    _choose,
    _reranked_record,
    _rows_by_index,
)


def target_confidence_proxy(target: dict[str, Any]) -> float:
    """Target-only quality proxy used before deciding whether to spend routes."""

    format_score = float(target.get("candidate_format_score", 0.0))
    numeric_score = float(target.get("candidate_numeric_consistency_score", 0.0))
    completion = float(target.get("candidate_completion_score", 0.0))
    generated_tokens = target.get("generated_tokens", 0)
    cap_penalty = 0.75 if isinstance(generated_tokens, (int, float)) and generated_tokens >= 64 else 0.0
    return float(format_score + 0.75 * numeric_score + 0.5 * completion - cap_penalty)


def choose_seed_budget(
    *,
    target_proxy: float,
    low_threshold: float,
    high_threshold: float,
    medium_budget: int,
    max_budget: int,
) -> int:
    if target_proxy >= high_threshold:
        return 0
    if target_proxy >= low_threshold:
        return min(max(int(medium_budget), 0), int(max_budget))
    return int(max_budget)


def _indices_for_record_sets(
    record_sets: list[list[dict[str, Any]]],
    *,
    method: str,
    baseline_method: str,
) -> tuple[list[int], dict[int, dict[str, Any]], list[dict[int, dict[str, Any]]]]:
    baseline_rows = _rows_by_index(record_sets[0], baseline_method)
    seed_method_rows = [_rows_by_index(records, method) for records in record_sets]
    indices = sorted(set(baseline_rows).intersection(*(set(rows) for rows in seed_method_rows)))
    if not indices:
        raise ValueError(f"No paired examples for method={method!r} and baseline={baseline_method!r}")
    return indices, baseline_rows, seed_method_rows


def _candidate_bundle(
    idx: int,
    *,
    baseline_rows: dict[int, dict[str, Any]],
    seed_method_rows: list[dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = dict(baseline_rows[idx])
    baseline["method"] = "target_alone"
    seed_rows = [rows[idx] for rows in seed_method_rows]
    return baseline, _annotate_candidates(baseline, seed_rows)


def _split_indices(indices: list[int], *, calibration_fraction: float) -> tuple[list[int], list[int]]:
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be in [0, 1)")
    if calibration_fraction <= 0.0 or len(indices) < 4:
        return [], list(indices)
    cutoff = max(1, min(len(indices) - 1, int(round(len(indices) * calibration_fraction))))
    return indices[:cutoff], indices[cutoff:]


def _candidate_thresholds(values: list[float]) -> list[float]:
    if not values:
        return [0.0]
    unique = sorted(set(float(value) for value in values))
    mids = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    return sorted(set(unique + mids))


def _choose_from_budget(
    candidates: list[dict[str, Any]],
    *,
    seed_budget: int,
    selection_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subset = [candidates[0], *candidates[1 : 1 + max(seed_budget, 0)]]
    if seed_budget <= 0:
        return candidates[0], subset
    return _choose(subset, selection_policy), subset


def _record_for_choice(
    *,
    method_name: str,
    policy_name: str,
    chosen: dict[str, Any],
    baseline: dict[str, Any],
    candidate_subset: list[dict[str, Any]],
    full_candidates: list[dict[str, Any]],
    seed_budget: int,
    target_proxy: float,
    low_threshold: float | None,
    high_threshold: float | None,
    split: str,
) -> dict[str, Any]:
    record = _reranked_record(
        method_name=method_name,
        policy=policy_name,
        chosen=chosen,
        baseline=baseline,
        candidates=candidate_subset,
    )
    full_meta = _candidate_metadata(full_candidates)
    record["selector_seed_budget"] = int(seed_budget)
    record["selector_target_confidence_proxy"] = float(target_proxy)
    record["selector_low_threshold"] = low_threshold
    record["selector_high_threshold"] = high_threshold
    record["selector_split"] = split
    record["selector_available_seed_count"] = max(len(full_candidates) - 1, 0)
    record["selector_subset_candidate_count"] = len(candidate_subset)
    record["selector_full_oracle_correct"] = bool(full_meta["candidate_oracle_correct"])
    record["selector_full_seed_correct_count"] = int(full_meta["seed_correct_count"])
    record["selector_full_unique_predictions"] = int(full_meta["candidate_unique_predictions"])
    record["selector_full_vote_entropy"] = float(full_meta["candidate_vote_entropy"])
    return record


def _accuracy(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    return sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)


def _avg(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return sum(values) / max(len(values), 1)


def _evaluate_policy_rows(
    indices: list[int],
    *,
    baseline_rows: dict[int, dict[str, Any]],
    seed_method_rows: list[dict[int, dict[str, Any]]],
    selection_policy: str,
    method_name: str,
    policy_name: str,
    seed_budget: int | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    medium_budget: int = 1,
    random_budget_probs: dict[int, float] | None = None,
    random_seed: int = 0,
    split: str = "eval",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_probs: list[tuple[int, float]] = []
    if random_budget_probs is not None:
        running = 0.0
        for budget, probability in sorted(random_budget_probs.items()):
            running += float(probability)
            cumulative_probs.append((int(budget), running))

    max_budget = len(seed_method_rows)
    for idx in indices:
        baseline, candidates = _candidate_bundle(idx, baseline_rows=baseline_rows, seed_method_rows=seed_method_rows)
        target_proxy = target_confidence_proxy(candidates[0])
        if seed_budget is not None:
            budget = min(max(int(seed_budget), 0), max_budget)
        elif random_budget_probs is not None:
            rng = random.Random(f"{random_seed}:{idx}")
            u = rng.random()
            budget = max_budget
            for candidate_budget, cutoff in cumulative_probs:
                if u <= cutoff:
                    budget = min(max(candidate_budget, 0), max_budget)
                    break
        else:
            assert low_threshold is not None and high_threshold is not None
            budget = choose_seed_budget(
                target_proxy=target_proxy,
                low_threshold=float(low_threshold),
                high_threshold=float(high_threshold),
                medium_budget=medium_budget,
                max_budget=max_budget,
            )
        chosen, subset = _choose_from_budget(candidates, seed_budget=budget, selection_policy=selection_policy)
        rows.append(
            _record_for_choice(
                method_name=method_name,
                policy_name=policy_name,
                chosen=chosen,
                baseline=baseline,
                candidate_subset=subset,
                full_candidates=candidates,
                seed_budget=budget,
                target_proxy=target_proxy,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                split=split,
            )
        )
    return rows


def _calibrate_thresholds(
    train_indices: list[int],
    *,
    baseline_rows: dict[int, dict[str, Any]],
    seed_method_rows: list[dict[int, dict[str, Any]]],
    selection_policy: str,
    medium_budget: int,
    target_avg_seed_budget: float,
    budget_penalty: float,
) -> dict[str, Any]:
    proxies = [
        target_confidence_proxy(
            _candidate_bundle(idx, baseline_rows=baseline_rows, seed_method_rows=seed_method_rows)[1][0]
        )
        for idx in train_indices
    ]
    thresholds = _candidate_thresholds(proxies)
    best: dict[str, Any] | None = None
    for low in thresholds:
        for high in thresholds:
            if low > high:
                continue
            rows = _evaluate_policy_rows(
                train_indices,
                baseline_rows=baseline_rows,
                seed_method_rows=seed_method_rows,
                selection_policy=selection_policy,
                method_name="confidence_gated_route_expansion",
                policy_name="confidence_gated_route_expansion_train",
                low_threshold=float(low),
                high_threshold=float(high),
                medium_budget=medium_budget,
                split="calibration",
            )
            acc = _accuracy(rows)
            avg_seed_budget = _avg(rows, "selector_seed_budget")
            score = acc - float(budget_penalty) * abs(avg_seed_budget - target_avg_seed_budget) / max(
                target_avg_seed_budget, 1e-8
            )
            candidate = {
                "low_threshold": float(low),
                "high_threshold": float(high),
                "train_accuracy": float(acc),
                "train_avg_seed_budget": float(avg_seed_budget),
                "train_score": float(score),
                "train_count": len(train_indices),
            }
            if best is None:
                best = candidate
                continue
            better = candidate["train_score"] > best["train_score"]
            tied = math.isclose(candidate["train_score"], best["train_score"], rel_tol=1e-9, abs_tol=1e-9)
            closer_budget = abs(candidate["train_avg_seed_budget"] - target_avg_seed_budget) < abs(
                best["train_avg_seed_budget"] - target_avg_seed_budget
            )
            higher_acc = candidate["train_accuracy"] > best["train_accuracy"]
            if better or (tied and (higher_acc or closer_budget)):
                best = candidate
    if best is None:
        return {
            "low_threshold": 0.0,
            "high_threshold": 0.0,
            "train_accuracy": 0.0,
            "train_avg_seed_budget": 0.0,
            "train_score": 0.0,
            "train_count": 0,
        }
    return best


def _budget_histogram(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(rows), 1)
    budgets = sorted({int(row.get("selector_seed_budget", 0)) for row in rows})
    return {str(budget): sum(int(row.get("selector_seed_budget", 0)) == budget for row in rows) / total for budget in budgets}


def _subgroup_rows(rows: list[dict[str, Any]], *, bins: int) -> list[dict[str, Any]]:
    if not rows or bins <= 0:
        return []
    values = sorted(float(row["selector_target_confidence_proxy"]) for row in rows)
    boundaries = [values[min(len(values) - 1, round(i * (len(values) - 1) / bins))] for i in range(bins + 1)]
    out: list[dict[str, Any]] = []
    for bin_idx in range(bins):
        left = boundaries[bin_idx]
        right = boundaries[bin_idx + 1]
        if bin_idx < bins - 1:
            subset = [
                row
                for row in rows
                if float(row["selector_target_confidence_proxy"]) >= left
                and float(row["selector_target_confidence_proxy"]) < right
            ]
        else:
            subset = [
                row
                for row in rows
                if float(row["selector_target_confidence_proxy"]) >= left
                and float(row["selector_target_confidence_proxy"]) <= right
            ]
        if not subset:
            continue
        out.append(
            {
                "group": f"target_proxy_{bin_idx}",
                "count": len(subset),
                "accuracy": _accuracy(subset),
                "avg_seed_budget": _avg(subset, "selector_seed_budget"),
                "full_oracle_accuracy": sum(bool(row["selector_full_oracle_correct"]) for row in subset)
                / max(len(subset), 1),
                "target_selection_rate": sum(row.get("selected_candidate_source") == "target" for row in subset)
                / max(len(subset), 1),
                "proxy_min": min(float(row["selector_target_confidence_proxy"]) for row in subset),
                "proxy_max": max(float(row["selector_target_confidence_proxy"]) for row in subset),
            }
        )
    return out


def summarize_records(records: list[dict[str, Any]], *, subgroup_bins: int = 3) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for method in sorted({str(record["method"]) for record in records}):
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = _accuracy(rows)
        if method.startswith("confidence_gated") or method.startswith("fixed_route_budget") or method.startswith(
            "random_route_budget"
        ):
            results[f"{method}_avg_seed_budget"] = _avg(rows, "selector_seed_budget")
            results[f"{method}_target_selection_rate"] = sum(
                row.get("selected_candidate_source") == "target" for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_full_oracle_accuracy"] = sum(
                bool(row.get("selector_full_oracle_correct")) for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_oracle_gap"] = float(
                results[f"{method}_full_oracle_accuracy"] - results[method]
            )
            results[f"{method}_budget_histogram"] = _budget_histogram(rows)
    add_paired_prediction_summary(results, records)
    gated_rows = [row for row in records if row.get("method") == "confidence_gated_route_expansion"]
    results["confidence_gated_route_expansion_subgroups"] = _subgroup_rows(gated_rows, bins=subgroup_bins)
    return results


def _random_budget_probs(train_rows: list[dict[str, Any]]) -> dict[int, float]:
    histogram = _budget_histogram(train_rows)
    return {int(budget): float(probability) for budget, probability in histogram.items()}


def confidence_gated_route_records(
    record_sets: list[list[dict[str, Any]]],
    *,
    method: str,
    baseline_method: str = "target_alone",
    selection_policy: str = "target_on_strict_format",
    calibration_fraction: float = 0.5,
    medium_budget: int = 1,
    target_avg_seed_budget: float = 1.5,
    budget_penalty: float = 0.15,
    random_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices, baseline_rows, seed_method_rows = _indices_for_record_sets(
        record_sets,
        method=method,
        baseline_method=baseline_method,
    )
    calibration_indices, eval_indices = _split_indices(indices, calibration_fraction=calibration_fraction)
    calibration = _calibrate_thresholds(
        calibration_indices,
        baseline_rows=baseline_rows,
        seed_method_rows=seed_method_rows,
        selection_policy=selection_policy,
        medium_budget=medium_budget,
        target_avg_seed_budget=target_avg_seed_budget,
        budget_penalty=budget_penalty,
    )
    gated_train_rows = _evaluate_policy_rows(
        calibration_indices,
        baseline_rows=baseline_rows,
        seed_method_rows=seed_method_rows,
        selection_policy=selection_policy,
        method_name="confidence_gated_route_expansion",
        policy_name="confidence_gated_route_expansion_train",
        low_threshold=calibration["low_threshold"],
        high_threshold=calibration["high_threshold"],
        medium_budget=medium_budget,
        split="calibration",
    )
    random_probs = _random_budget_probs(gated_train_rows) if gated_train_rows else {0: 1.0}

    output: list[dict[str, Any]] = []
    for idx in eval_indices:
        baseline = dict(baseline_rows[idx])
        baseline["method"] = baseline_method
        output.append(baseline)

    max_budget = len(seed_method_rows)
    for fixed_budget in [0, medium_budget, max_budget]:
        output.extend(
            _evaluate_policy_rows(
                eval_indices,
                baseline_rows=baseline_rows,
                seed_method_rows=seed_method_rows,
                selection_policy=selection_policy,
                method_name=f"fixed_route_budget_{fixed_budget}",
                policy_name=f"fixed_route_budget_{fixed_budget}_{selection_policy}",
                seed_budget=fixed_budget,
                split="eval",
            )
        )
    output.extend(
        _evaluate_policy_rows(
            eval_indices,
            baseline_rows=baseline_rows,
            seed_method_rows=seed_method_rows,
            selection_policy=selection_policy,
            method_name="random_route_budget_matched",
            policy_name=f"random_route_budget_matched_{selection_policy}",
            random_budget_probs=random_probs,
            random_seed=random_seed,
            split="eval",
        )
    )
    output.extend(
        _evaluate_policy_rows(
            eval_indices,
            baseline_rows=baseline_rows,
            seed_method_rows=seed_method_rows,
            selection_policy=selection_policy,
            method_name="confidence_gated_route_expansion",
            policy_name=f"confidence_gated_route_expansion_{selection_policy}",
            low_threshold=calibration["low_threshold"],
            high_threshold=calibration["high_threshold"],
            medium_budget=medium_budget,
            split="eval",
        )
    )
    metadata = {
        "indices": indices,
        "calibration_indices": calibration_indices,
        "eval_indices": eval_indices,
        "calibration": calibration,
        "random_budget_probs": random_probs,
        "selection_policy": selection_policy,
        "medium_budget": int(medium_budget),
        "max_seed_budget": int(max_budget),
        "target_avg_seed_budget": float(target_avg_seed_budget),
        "budget_penalty": float(budget_penalty),
    }
    return output, metadata


def write_markdown_summary(results: dict[str, Any], metadata: dict[str, Any], output_md: str | pathlib.Path) -> None:
    methods = [
        "target_alone",
        "fixed_route_budget_0",
        f"fixed_route_budget_{metadata['medium_budget']}",
        f"fixed_route_budget_{metadata['max_seed_budget']}",
        "random_route_budget_matched",
        "confidence_gated_route_expansion",
    ]
    seen: set[str] = set()
    target = float(results.get("target_alone", 0.0))
    lines = [
        "# Confidence-Gated Route Expansion Summary",
        "",
        f"- Selection policy: `{metadata['selection_policy']}`",
        f"- Calibration examples: `{len(metadata['calibration_indices'])}`",
        f"- Eval examples: `{len(metadata['eval_indices'])}`",
        f"- Low threshold: `{metadata['calibration']['low_threshold']:.4f}`",
        f"- High threshold: `{metadata['calibration']['high_threshold']:.4f}`",
        f"- Train accuracy: `{metadata['calibration']['train_accuracy']:.4f}`",
        f"- Train avg seed budget: `{metadata['calibration']['train_avg_seed_budget']:.4f}`",
        "",
        "| Method | Accuracy | Delta vs target | Avg seed budget | Oracle gap | Target selected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        if method in seen or method not in results:
            continue
        seen.add(method)
        avg_budget = results.get(f"{method}_avg_seed_budget")
        oracle_gap = results.get(f"{method}_oracle_gap")
        target_rate = results.get(f"{method}_target_selection_rate")
        lines.append(
            "| {method} | {accuracy:.4f} | {delta:+.4f} | {avg_budget} | {oracle_gap} | {target_rate} |".format(
                method=method,
                accuracy=float(results[method]),
                delta=float(results[method]) - target,
                avg_budget="-" if avg_budget is None else f"{float(avg_budget):.4f}",
                oracle_gap="-" if oracle_gap is None else f"{float(oracle_gap):.4f}",
                target_rate="-" if target_rate is None else f"{float(target_rate):.4f}",
            )
        )
    subgroups = results.get("confidence_gated_route_expansion_subgroups", [])
    if subgroups:
        lines.extend(
            [
                "",
                "## Confidence-Gated Subgroups",
                "",
                "| Group | Count | Accuracy | Avg seed budget | Full oracle accuracy | Target selected | Proxy range |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in subgroups:
            lines.append(
                "| {group} | {count} | {accuracy:.4f} | {avg_budget:.4f} | {oracle:.4f} | {target:.4f} | {lo:.2f}-{hi:.2f} |".format(
                    group=row["group"],
                    count=int(row["count"]),
                    accuracy=float(row["accuracy"]),
                    avg_budget=float(row["avg_seed_budget"]),
                    oracle=float(row["full_oracle_accuracy"]),
                    target=float(row["target_selection_rate"]),
                    lo=float(row["proxy_min"]),
                    hi=float(row["proxy_max"]),
                )
            )
    pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confidence-gated stochastic route expansion.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--method", default="rotalign_kv_gate_0.10")
    parser.add_argument("--baseline-method", default="target_alone")
    parser.add_argument("--selection-policy", default="target_on_strict_format")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--medium-budget", type=int, default=1)
    parser.add_argument("--target-avg-seed-budget", type=float, default=1.5)
    parser.add_argument("--budget-penalty", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--subgroup-bins", type=int, default=3)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record_sets = [load_records(path) for path in args.inputs]
    records, metadata = confidence_gated_route_records(
        record_sets,
        method=args.method,
        baseline_method=args.baseline_method,
        selection_policy=args.selection_policy,
        calibration_fraction=args.calibration_fraction,
        medium_budget=args.medium_budget,
        target_avg_seed_budget=args.target_avg_seed_budget,
        budget_penalty=args.budget_penalty,
        random_seed=args.random_seed,
    )
    results = summarize_records(records, subgroup_bins=args.subgroup_bins)
    write_prediction_records(args.output_jsonl, records)
    write_prediction_sidecar(
        args.output_jsonl,
        records,
        results,
        {
            "inputs": [str(path) for path in args.inputs],
            "method": args.method,
            "baseline_method": args.baseline_method,
            **metadata,
        },
    )
    if args.output_md:
        write_markdown_summary(results, metadata, args.output_md)
    for key, value in sorted(results.items()):
        if isinstance(value, (int, float)) and not key.startswith("paired_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
