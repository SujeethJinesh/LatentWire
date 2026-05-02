"""Calibrate an interpretable candidate selector on stochastic route pools."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
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
    _reranked_record,
    _rows_by_index,
)


FEATURE_NAMES = (
    "format_score",
    "numeric_consistency",
    "completion",
    "answer_agreement",
    "is_target",
    "is_seed",
    "seed_index",
)


def candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    is_target = 1.0 if candidate.get("candidate_source") == "target" else 0.0
    seed_index = float(candidate.get("source_input_index", -1))
    return {
        "format_score": float(candidate.get("candidate_format_score", 0.0)),
        "numeric_consistency": float(candidate.get("candidate_numeric_consistency_score", 0.0)),
        "completion": float(candidate.get("candidate_completion_score", 0.0)),
        "answer_agreement": float(candidate.get("candidate_answer_agreement", 0.0)),
        "is_target": is_target,
        "is_seed": 1.0 - is_target,
        "seed_index": seed_index,
    }


def candidate_score(candidate: dict[str, Any], weights: dict[str, float]) -> float:
    features = candidate_features(candidate)
    return float(sum(float(weights.get(name, 0.0)) * features[name] for name in FEATURE_NAMES))


def _candidate_weight_grid() -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for format_w, numeric_w, completion_w, agreement_w, target_w, seed_w in itertools.product(
        [0.0, 0.5, 1.0, 2.0],
        [0.0, 0.5, 1.0, 2.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-1.0, 0.0, 1.0],
    ):
        candidates.append(
            {
                "format_score": format_w,
                "numeric_consistency": numeric_w,
                "completion": completion_w,
                "answer_agreement": agreement_w,
                "is_target": target_w,
                "is_seed": seed_w,
                "seed_index": 0.0,
            }
        )
    candidates.extend(
        [
            {
                "format_score": 1.0,
                "numeric_consistency": 0.75,
                "completion": 0.5,
                "answer_agreement": 0.5,
                "is_target": 0.0,
                "is_seed": 0.0,
                "seed_index": 0.0,
            },
            {
                "format_score": 0.0,
                "numeric_consistency": 1.0,
                "completion": 1.0,
                "answer_agreement": 0.0,
                "is_target": 0.0,
                "is_seed": 0.0,
                "seed_index": 0.0,
            },
            {
                "format_score": 1.0,
                "numeric_consistency": 1.0,
                "completion": 1.0,
                "answer_agreement": 1.0,
                "is_target": -1.0,
                "is_seed": 0.5,
                "seed_index": 0.0,
            },
        ]
    )
    deduped: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
    for weights in candidates:
        key = tuple(sorted(weights.items()))
        deduped[key] = weights
    return list(deduped.values())


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


def _split_indices(indices: list[int], *, calibration_fraction: float) -> tuple[list[int], list[int]]:
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be in [0, 1)")
    if calibration_fraction <= 0.0 or len(indices) < 4:
        return [], list(indices)
    cutoff = max(1, min(len(indices) - 1, int(round(len(indices) * calibration_fraction))))
    return indices[:cutoff], indices[cutoff:]


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


def choose_candidate(candidates: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            candidate_score(row, weights),
            row.get("candidate_source") == "target",
            -float(row.get("source_input_index", -1)),
        ),
    )


def _accuracy(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    return sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)


def _select_rows(
    indices: list[int],
    *,
    baseline_rows: dict[int, dict[str, Any]],
    seed_method_rows: list[dict[int, dict[str, Any]]],
    weights: dict[str, float],
    method_name: str,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in indices:
        baseline, candidates = _candidate_bundle(idx, baseline_rows=baseline_rows, seed_method_rows=seed_method_rows)
        chosen = choose_candidate(candidates, weights)
        record = _reranked_record(
            method_name=method_name,
            policy="calibrated_feature_score",
            chosen=chosen,
            baseline=baseline,
            candidates=candidates,
        )
        meta = _candidate_metadata(candidates)
        record["selector_split"] = split
        record["selector_weights"] = weights
        record["selected_candidate_feature_score"] = candidate_score(chosen, weights)
        record["selected_candidate_features"] = candidate_features(chosen)
        record["selector_candidate_feature_scores"] = [
            {
                "source": row.get("candidate_source"),
                "normalized_prediction": row.get("normalized_prediction"),
                "feature_score": candidate_score(row, weights),
                "features": candidate_features(row),
                "correct": bool(row.get("correct")),
            }
            for row in candidates
        ]
        record["selector_full_oracle_correct"] = bool(meta["candidate_oracle_correct"])
        record["selector_full_seed_correct_count"] = int(meta["seed_correct_count"])
        record["selector_full_vote_entropy"] = float(meta["candidate_vote_entropy"])
        rows.append(record)
    return rows


def calibrate_weights(
    train_indices: list[int],
    *,
    baseline_rows: dict[int, dict[str, Any]],
    seed_method_rows: list[dict[int, dict[str, Any]]],
    target_selection_penalty: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for weights in _candidate_weight_grid():
        rows = _select_rows(
            train_indices,
            baseline_rows=baseline_rows,
            seed_method_rows=seed_method_rows,
            weights=weights,
            method_name="calibrated_feature_selector",
            split="calibration",
        )
        accuracy = _accuracy(rows)
        target_selection_rate = sum(row.get("selected_candidate_source") == "target" for row in rows) / max(
            len(rows), 1
        )
        score = accuracy - float(target_selection_penalty) * target_selection_rate
        candidate = {
            "weights": weights,
            "train_accuracy": accuracy,
            "train_target_selection_rate": target_selection_rate,
            "train_score": score,
            "train_count": len(train_indices),
        }
        if best is None:
            best = candidate
            continue
        better = candidate["train_score"] > best["train_score"]
        tied = abs(candidate["train_score"] - best["train_score"]) <= 1e-12
        higher_accuracy = candidate["train_accuracy"] > best["train_accuracy"]
        lower_target_rate = candidate["train_target_selection_rate"] < best["train_target_selection_rate"]
        if better or (tied and (higher_accuracy or lower_target_rate)):
            best = candidate
    if best is None:
        weights = {name: 0.0 for name in FEATURE_NAMES}
        weights["is_target"] = 1.0
        return {
            "weights": weights,
            "train_accuracy": 0.0,
            "train_target_selection_rate": 1.0,
            "train_score": 0.0,
            "train_count": 0,
        }
    return best


def calibrated_candidate_selector_records(
    record_sets: list[list[dict[str, Any]]],
    *,
    method: str,
    baseline_method: str = "target_alone",
    calibration_fraction: float = 0.5,
    target_selection_penalty: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices, baseline_rows, seed_method_rows = _indices_for_record_sets(
        record_sets,
        method=method,
        baseline_method=baseline_method,
    )
    calibration_indices, eval_indices = _split_indices(indices, calibration_fraction=calibration_fraction)
    calibration = calibrate_weights(
        calibration_indices,
        baseline_rows=baseline_rows,
        seed_method_rows=seed_method_rows,
        target_selection_penalty=target_selection_penalty,
    )

    output: list[dict[str, Any]] = []
    for idx in eval_indices:
        baseline = dict(baseline_rows[idx])
        baseline["method"] = baseline_method
        output.append(baseline)
    output.extend(
        _select_rows(
            eval_indices,
            baseline_rows=baseline_rows,
            seed_method_rows=seed_method_rows,
            weights=calibration["weights"],
            method_name="calibrated_feature_selector",
            split="eval",
        )
    )
    metadata = {
        "indices": indices,
        "calibration_indices": calibration_indices,
        "eval_indices": eval_indices,
        "calibration": calibration,
        "target_selection_penalty": float(target_selection_penalty),
        "feature_names": list(FEATURE_NAMES),
    }
    return output, metadata


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for method in sorted({str(record["method"]) for record in records}):
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = _accuracy(rows)
        if method == "calibrated_feature_selector":
            results["calibrated_feature_selector_target_selection_rate"] = sum(
                row.get("selected_candidate_source") == "target" for row in rows
            ) / max(len(rows), 1)
            results["calibrated_feature_selector_seed_selection_rate"] = sum(
                row.get("selected_candidate_source") != "target" for row in rows
            ) / max(len(rows), 1)
            results["calibrated_feature_selector_full_oracle_accuracy"] = sum(
                bool(row.get("selector_full_oracle_correct")) for row in rows
            ) / max(len(rows), 1)
            results["calibrated_feature_selector_oracle_gap"] = (
                float(results["calibrated_feature_selector_full_oracle_accuracy"]) - float(results[method])
            )
    add_paired_prediction_summary(results, records)
    return results


def write_markdown_summary(results: dict[str, Any], metadata: dict[str, Any], output_md: str | pathlib.Path) -> None:
    method = "calibrated_feature_selector"
    target = float(results.get("target_alone", 0.0))
    prefix = f"paired_{method}_vs_target_alone"
    lines = [
        "# Calibrated Feature Selector Summary",
        "",
        f"- Calibration examples: `{len(metadata['calibration_indices'])}`",
        f"- Eval examples: `{len(metadata['eval_indices'])}`",
        f"- Train accuracy: `{metadata['calibration']['train_accuracy']:.4f}`",
        f"- Train target-selection rate: `{metadata['calibration']['train_target_selection_rate']:.4f}`",
        f"- Target-selection penalty: `{metadata['target_selection_penalty']:.4f}`",
        f"- Weights: `{json.dumps(metadata['calibration']['weights'], sort_keys=True)}`",
        "",
        "| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Target selected | Oracle gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {method} | {acc:.4f} | {delta:+.4f} | {method_only:.0f} | {baseline_only:.0f} | {both_correct:.0f} | {both_wrong:.0f} | {target_selected:.4f} | {oracle_gap:.4f} |".format(
            method=method,
            acc=float(results.get(method, 0.0)),
            delta=float(results.get(f"{prefix}_delta_accuracy", float(results.get(method, 0.0)) - target)),
            method_only=float(results.get(f"{prefix}_method_only", 0.0)),
            baseline_only=float(results.get(f"{prefix}_baseline_only", 0.0)),
            both_correct=float(results.get(f"{prefix}_both_correct", 0.0)),
            both_wrong=float(results.get(f"{prefix}_both_wrong", 0.0)),
            target_selected=float(results.get("calibrated_feature_selector_target_selection_rate", 0.0)),
            oracle_gap=float(results.get("calibrated_feature_selector_oracle_gap", 0.0)),
        ),
        "",
        "Interpretation:",
        "",
        "This selector calibrates transparent candidate-feature weights on a held-out calibration prefix, then",
        "evaluates on the remaining examples. It is not an oracle: labels are used only to choose the weights",
        "on the calibration split, and every candidate score and feature vector is logged for audit.",
    ]
    pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate transparent candidate selector features.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--method", default="rotalign_kv_gate_0.10")
    parser.add_argument("--baseline-method", default="target_alone")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--target-selection-penalty", type=float, default=0.0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record_sets = [load_records(path) for path in args.inputs]
    records, metadata = calibrated_candidate_selector_records(
        record_sets,
        method=args.method,
        baseline_method=args.baseline_method,
        calibration_fraction=args.calibration_fraction,
        target_selection_penalty=args.target_selection_penalty,
    )
    results = summarize_results(records)
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
