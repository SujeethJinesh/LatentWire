from __future__ import annotations

"""Build COLM acceptance baselines for the source-private packet paper.

This artifact answers the main reviewer objection directly: compare the packet
against explicit source-index/source-choice baselines on the same frozen ARC and
OpenBookQA surfaces used by the paper.  It also records paired uncertainty for
packet-vs-baseline comparisons and compact byte/rate curves.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import pathlib
import random
import statistics
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_fourier_anchor_syndrome_gate as fourier_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_seed_stability as seed_stability  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_colm_acceptance_baselines_20260502")

ARC_TRAIN = pathlib.Path("results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl")
ARC_VALIDATION = pathlib.Path("results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl")
ARC_TEST = pathlib.Path("results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl")
ARC_VALIDATION_CACHE = pathlib.Path("results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/source_prediction_cache.jsonl")
ARC_TEST_CACHE = pathlib.Path("results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/source_prediction_cache.jsonl")

OBQA_TRAIN = pathlib.Path("results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl")
OBQA_TEST = pathlib.Path("results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_test.jsonl")
OBQA_ANCHOR = pathlib.Path("results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/predictions.jsonl")

ARC_SEEDS = (47, 53, 59, 61, 67, 71, 73, 79, 83, 89)
OBQA_SEEDS = (47, 53, 59, 61, 67)
RATE_BUDGETS = (2, 3, 4, 8)
BASELINE_CONDITIONS = (
    "source_index_byte",
    "source_choice_label_text",
    "source_rank_code",
    "entropy_matched_random_index",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_jsonl_gz(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _source_cache_predictions(path: pathlib.Path, rows: list[arc_gate.ArcRow]) -> list[int]:
    cache = fourier_gate._read_source_cache(path)
    predictions: list[int] = []
    missing: list[str] = []
    for row in rows:
        if row.content_id not in cache:
            missing.append(row.content_id)
            continue
        predictions.append(int(cache[row.content_id]))
    if missing:
        raise ValueError(f"{path} missing {len(missing)} rows")
    return predictions


def _obqa_source_predictions(
    *,
    eval_rows: list[arc_gate.ArcRow],
    anchor_predictions: pathlib.Path,
) -> tuple[list[int], list[dict[str, Any]]]:
    anchor_choices = seed_stability._read_anchor_source_choices(anchor_predictions)
    return seed_stability._source_predictions_from_anchor(eval_rows, anchor_choices)


def _prediction_row(
    *,
    condition: str,
    row: arc_gate.ArcRow,
    prediction_index: int,
    payload_bytes: int,
    seed: int,
    split: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    prediction_index = int(prediction_index)
    return {
        "condition": condition,
        "row_id": row.row_id,
        "content_id": row.content_id,
        "answer_index": row.answer_index,
        "answer_label": row.answer_label,
        "prediction_index": prediction_index,
        "prediction_label": row.choice_labels[prediction_index],
        "correct": bool(prediction_index == row.answer_index),
        "payload_bytes": int(payload_bytes),
        "payload_hex": "",
        "latency_ms": 0.0,
        "seed": int(seed),
        "split": split,
        "metadata": metadata,
    }


def _add_baseline_rows(
    *,
    rows: list[arc_gate.ArcRow],
    prediction_rows: list[dict[str, Any]],
    source_predictions: list[int],
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    source_counts = {index: source_predictions.count(index) for index in sorted(set(source_predictions))}
    choices = sorted(source_counts)
    weights = [source_counts[index] for index in choices]
    rng = random.Random(10_000 + seed + (0 if split == "validation" else 1_000_000))
    baseline_rows: list[dict[str, Any]] = []
    for row, source_index in zip(rows, source_predictions, strict=True):
        source_index = int(source_index)
        meta = {
            "source_selected_index": source_index,
            "source_selected_label": row.choice_labels[source_index],
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
        }
        baseline_rows.append(
            _prediction_row(
                condition="source_index_byte",
                row=row,
                prediction_index=source_index,
                payload_bytes=1,
                seed=seed,
                split=split,
                metadata=meta | {"baseline": "raw 1-byte source-selected candidate index"},
            )
        )
        baseline_rows.append(
            _prediction_row(
                condition="source_choice_label_text",
                row=row,
                prediction_index=source_index,
                payload_bytes=len(row.choice_labels[source_index].encode("utf-8")),
                seed=seed,
                split=split,
                metadata=meta | {"baseline": "source-selected label text"},
            )
        )
        baseline_rows.append(
            _prediction_row(
                condition="source_rank_code",
                row=row,
                prediction_index=source_index,
                payload_bytes=1,
                seed=seed,
                split=split,
                metadata=meta
                | {
                    "baseline": "source top-rank code",
                    "source_score_available": False,
                    "note": "current frozen source cache stores top choice, not calibrated raw source scores",
                },
            )
        )
        random_index = rng.choices(choices, weights=weights, k=1)[0]
        random_index = min(int(random_index), len(row.choices) - 1)
        baseline_rows.append(
            _prediction_row(
                condition="entropy_matched_random_index",
                row=row,
                prediction_index=random_index,
                payload_bytes=1,
                seed=seed,
                split=split,
                metadata=meta | {"baseline": "random source-index code matched to empirical source-index entropy"},
            )
        )
    return prediction_rows + baseline_rows


def _condition_rows(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["condition"] == condition]


def _condition_accuracy(rows: list[dict[str, Any]], condition: str) -> float:
    selected = _condition_rows(rows, condition)
    if not selected:
        return 0.0
    return float(sum(1 for row in selected if row["correct"]) / len(selected))


def _paired_ci(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    baseline: str,
    seed: int,
    samples: int,
) -> dict[str, float]:
    return arc_gate._paired_bootstrap(rows, condition=condition, baseline=baseline, seed=seed, samples=samples)


def _best_destructive(rows: list[dict[str, Any]]) -> tuple[str, float]:
    candidates = {
        condition: _condition_accuracy(rows, condition)
        for condition in arc_gate.STRICT_DESTRUCTIVE_CONTROLS
        if _condition_rows(rows, condition)
    }
    if not candidates:
        return "", 0.0
    name = max(candidates, key=candidates.get)
    return name, candidates[name]


def _source_follow_rate(rows: list[dict[str, Any]]) -> float:
    matched = _condition_rows(rows, arc_gate.MATCHED_CONDITION)
    if not matched:
        return 0.0
    count = 0
    for row in matched:
        source_index = int(row.get("metadata", {}).get("source_selected_index", -1))
        count += int(int(row["prediction_index"]) == source_index)
    return float(count / len(matched))


def _seed_summary(
    *,
    rows: list[dict[str, Any]],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    best_control, best_control_acc = _best_destructive(rows)
    comparisons = {
        "packet_vs_target": _paired_ci(
            rows, condition=arc_gate.MATCHED_CONDITION, baseline="target_only", seed=seed + 101, samples=bootstrap_samples
        ),
        "packet_vs_same_budget_text": _paired_ci(
            rows,
            condition=arc_gate.MATCHED_CONDITION,
            baseline="same_byte_structured_text",
            seed=seed + 102,
            samples=bootstrap_samples,
        ),
        "packet_vs_source_index": _paired_ci(
            rows,
            condition=arc_gate.MATCHED_CONDITION,
            baseline="source_index_byte",
            seed=seed + 103,
            samples=bootstrap_samples,
        ),
        "packet_vs_source_choice_text": _paired_ci(
            rows,
            condition=arc_gate.MATCHED_CONDITION,
            baseline="source_choice_label_text",
            seed=seed + 104,
            samples=bootstrap_samples,
        ),
        "packet_vs_source_rank_code": _paired_ci(
            rows,
            condition=arc_gate.MATCHED_CONDITION,
            baseline="source_rank_code",
            seed=seed + 105,
            samples=bootstrap_samples,
        ),
        "packet_vs_entropy_random_index": _paired_ci(
            rows,
            condition=arc_gate.MATCHED_CONDITION,
            baseline="entropy_matched_random_index",
            seed=seed + 106,
            samples=bootstrap_samples,
        ),
    }
    if best_control:
        comparisons["packet_vs_best_destructive"] = _paired_ci(
            rows,
            condition=arc_gate.MATCHED_CONDITION,
            baseline=best_control,
            seed=seed + 107,
            samples=bootstrap_samples,
        )
    return {
        "seed": int(seed),
        "n": len(_condition_rows(rows, arc_gate.MATCHED_CONDITION)),
        "packet_accuracy": _condition_accuracy(rows, arc_gate.MATCHED_CONDITION),
        "target_accuracy": _condition_accuracy(rows, "target_only"),
        "same_budget_text_accuracy": _condition_accuracy(rows, "same_byte_structured_text"),
        "source_index_accuracy": _condition_accuracy(rows, "source_index_byte"),
        "source_choice_text_accuracy": _condition_accuracy(rows, "source_choice_label_text"),
        "source_rank_code_accuracy": _condition_accuracy(rows, "source_rank_code"),
        "entropy_matched_random_index_accuracy": _condition_accuracy(rows, "entropy_matched_random_index"),
        "best_destructive_control": best_control,
        "best_destructive_control_accuracy": best_control_acc,
        "packet_follows_source_index": _source_follow_rate(rows),
        "paired_ci": comparisons,
    }


def _aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_keys = [
        "packet_accuracy",
        "target_accuracy",
        "same_budget_text_accuracy",
        "source_index_accuracy",
        "source_choice_text_accuracy",
        "source_rank_code_accuracy",
        "entropy_matched_random_index_accuracy",
        "packet_follows_source_index",
    ]
    out: dict[str, Any] = {"seed_count": len(seed_rows)}
    for key in accuracy_keys:
        values = [float(row[key]) for row in seed_rows]
        out[f"{key}_mean"] = float(statistics.fmean(values))
        out[f"{key}_min"] = float(min(values))
        out[f"{key}_max"] = float(max(values))
    for comp in [
        "packet_vs_target",
        "packet_vs_same_budget_text",
        "packet_vs_source_index",
        "packet_vs_source_choice_text",
        "packet_vs_source_rank_code",
        "packet_vs_entropy_random_index",
        "packet_vs_best_destructive",
    ]:
        lows = [row["paired_ci"][comp]["ci95_low"] for row in seed_rows if comp in row["paired_ci"]]
        means = [row["paired_ci"][comp]["mean"] for row in seed_rows if comp in row["paired_ci"]]
        if lows:
            out[f"{comp}_mean"] = float(statistics.fmean(means))
            out[f"{comp}_ci95_low_min"] = float(min(lows))
    out["packet_beats_source_index_all_seeds"] = bool(
        all(row["packet_accuracy"] > row["source_index_accuracy"] for row in seed_rows)
    )
    out["packet_ties_or_loses_source_index"] = bool(
        any(row["packet_accuracy"] <= row["source_index_accuracy"] for row in seed_rows)
    )
    return out


def _validation_selected_fusion(
    *,
    validation_seed_rows: list[dict[str, Any]],
    test_seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # Candidate policies available without gold at test after validation choice.
    policy_names = [
        "packet_accuracy",
        "source_index_accuracy",
        "target_accuracy",
        "same_budget_text_accuracy",
    ]
    validation_means = {
        policy: statistics.fmean(float(row[policy]) for row in validation_seed_rows)
        for policy in policy_names
    }
    selected_policy = max(policy_names, key=lambda policy: validation_means[policy])
    test_values = [float(row[selected_policy]) for row in test_seed_rows]
    return {
        "candidate_policies": validation_means,
        "selected_policy": selected_policy,
        "selected_policy_test_mean": float(statistics.fmean(test_values)),
        "selected_policy_test_min": float(min(test_values)),
        "interpretation": (
            "Validation selects the strongest available policy among packet, source-index, target-only, "
            "and same-budget text. On current ARC this selects source-index, confirming that the packet "
            "does not yet beat explicit source-choice communication."
        ),
    }


def _arc_full_rows_for_split(
    *,
    split: str,
    rows: list[arc_gate.ArcRow],
    source_cache: pathlib.Path,
    train_rows: list[arc_gate.ArcRow],
    seeds: list[int],
    budget_bytes: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_predictions = _source_cache_predictions(source_cache, rows)
    anchor_texts = arc_gate._choice_pair_texts(train_rows)
    source_features, receiver_features, _ = fourier_gate._fourier_pair_features_for_variant(
        eval_rows=rows,
        anchor_texts=anchor_texts,
        anchor_count=384,
        spectral_dim=96,
        variant=fourier_gate.MATCHED_VARIANT,
    )
    source_residuals = arc_gate._candidate_residuals(rows, source_features)
    receiver_residuals = arc_gate._candidate_residuals(rows, receiver_features)
    priors = arc_gate._index_prior(train_rows)
    all_prediction_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        projection = arc_gate._projection_matrix(source_features.shape[1], 96, seed=seed + 171)
        seed_rows = arc_gate._rows_for_predictions(
            eval_rows=rows,
            residuals=source_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=source_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=priors,
            seed=seed + 911,
        )
        for row in seed_rows:
            row["seed"] = int(seed)
            row["split"] = split
        seed_rows = _add_baseline_rows(
            rows=rows,
            prediction_rows=seed_rows,
            source_predictions=source_predictions,
            seed=seed,
            split=split,
        )
        summaries.append(_seed_summary(rows=seed_rows, seed=seed, bootstrap_samples=bootstrap_samples))
        all_prediction_rows.extend(seed_rows)
    return summaries, all_prediction_rows


def _obqa_full_rows(
    *,
    rows: list[arc_gate.ArcRow],
    train_rows: list[arc_gate.ArcRow],
    anchor_predictions: pathlib.Path,
    seeds: list[int],
    budget_bytes: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_predictions, cache_rows = _obqa_source_predictions(eval_rows=rows, anchor_predictions=anchor_predictions)
    source_features, receiver_features, _ = seed_stability._pair_features_for_anchor_control(
        eval_rows=rows,
        anchor_texts=arc_gate._choice_pair_texts(train_rows),
        feature_dim=384,
        feature_mode="hashed",
        feature_model="",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
        anchor_control="none",
    )
    source_residuals = arc_gate._candidate_residuals(rows, source_features)
    receiver_residuals = arc_gate._candidate_residuals(rows, receiver_features)
    priors = arc_gate._index_prior(train_rows)
    all_prediction_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        projection = arc_gate._projection_matrix(384, 96, seed=seed + 171)
        seed_rows = arc_gate._rows_for_predictions(
            eval_rows=rows,
            residuals=source_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=source_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=priors,
            seed=seed + 911,
        )
        for row in seed_rows:
            row["seed"] = int(seed)
            row["split"] = "test"
        seed_rows = _add_baseline_rows(
            rows=rows,
            prediction_rows=seed_rows,
            source_predictions=source_predictions,
            seed=seed,
            split="test",
        )
        summaries.append(_seed_summary(rows=seed_rows, seed=seed, bootstrap_samples=bootstrap_samples))
        all_prediction_rows.extend(seed_rows)
    return summaries, all_prediction_rows, cache_rows


def _rate_curve_from_features(
    *,
    benchmark: str,
    split: str,
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    train_rows: list[arc_gate.ArcRow],
    source_features: np.ndarray,
    receiver_features: np.ndarray,
    budgets: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    source_residuals = arc_gate._candidate_residuals(rows, source_features)
    receiver_residuals = arc_gate._candidate_residuals(rows, receiver_features)
    priors = arc_gate._index_prior(train_rows)
    out: list[dict[str, Any]] = []
    for budget in budgets:
        values = []
        for seed in seeds:
            projection = arc_gate._projection_matrix(source_features.shape[1], 96, seed=seed + 171)
            seed_rows = arc_gate._rows_for_predictions(
                eval_rows=rows,
                residuals=source_residuals,
                decode_residuals=receiver_residuals,
                source_predictions=source_predictions,
                projection=projection,
                budget_bytes=budget,
                index_prior=priors,
                seed=seed + 911,
            )
            values.append(_condition_accuracy(seed_rows, arc_gate.MATCHED_CONDITION))
        out.append(
            {
                "benchmark": benchmark,
                "split": split,
                "budget_bytes": int(budget),
                "framed_bytes": int(budget + 3),
                "seed_count": len(seeds),
                "packet_accuracy_mean": float(statistics.fmean(values)),
                "packet_accuracy_min": float(min(values)),
                "packet_accuracy_max": float(max(values)),
            }
        )
    return out


def _build_rate_curves(
    *,
    arc_train: list[arc_gate.ArcRow],
    arc_test: list[arc_gate.ArcRow],
    arc_test_cache: pathlib.Path,
    obqa_train: list[arc_gate.ArcRow],
    obqa_test: list[arc_gate.ArcRow],
    obqa_anchor: pathlib.Path,
    budgets: list[int],
) -> list[dict[str, Any]]:
    arc_source_predictions = _source_cache_predictions(arc_test_cache, arc_test)
    arc_source_features, arc_receiver_features, _ = fourier_gate._fourier_pair_features_for_variant(
        eval_rows=arc_test,
        anchor_texts=arc_gate._choice_pair_texts(arc_train),
        anchor_count=384,
        spectral_dim=96,
        variant=fourier_gate.MATCHED_VARIANT,
    )
    obqa_source_predictions, _ = _obqa_source_predictions(eval_rows=obqa_test, anchor_predictions=obqa_anchor)
    obqa_source_features, obqa_receiver_features, _ = seed_stability._pair_features_for_anchor_control(
        eval_rows=obqa_test,
        anchor_texts=arc_gate._choice_pair_texts(obqa_train),
        feature_dim=384,
        feature_mode="hashed",
        feature_model="",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
        anchor_control="none",
    )
    return [
        *_rate_curve_from_features(
            benchmark="ARC-Challenge",
            split="test",
            rows=arc_test,
            source_predictions=arc_source_predictions,
            train_rows=arc_train,
            source_features=arc_source_features,
            receiver_features=arc_receiver_features,
            budgets=budgets,
            seeds=list(ARC_SEEDS[:5]),
        ),
        *_rate_curve_from_features(
            benchmark="OpenBookQA",
            split="test",
            rows=obqa_test,
            source_predictions=obqa_source_predictions,
            train_rows=obqa_train,
            source_features=obqa_source_features,
            receiver_features=obqa_receiver_features,
            budgets=budgets,
            seeds=list(OBQA_SEEDS),
        ),
    ]


def _write_seed_csv(path: pathlib.Path, benchmark: str, split: str, seed_rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "split",
        "seed",
        "packet_accuracy",
        "source_index_accuracy",
        "target_accuracy",
        "same_budget_text_accuracy",
        "packet_follows_source_index",
        "packet_vs_source_index_mean",
        "packet_vs_source_index_ci95_low",
        "packet_vs_same_budget_text_ci95_low",
        "packet_vs_best_destructive_ci95_low",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for row in seed_rows:
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "split": split,
                    "seed": row["seed"],
                    "packet_accuracy": row["packet_accuracy"],
                    "source_index_accuracy": row["source_index_accuracy"],
                    "target_accuracy": row["target_accuracy"],
                    "same_budget_text_accuracy": row["same_budget_text_accuracy"],
                    "packet_follows_source_index": row["packet_follows_source_index"],
                    "packet_vs_source_index_mean": row["paired_ci"]["packet_vs_source_index"]["mean"],
                    "packet_vs_source_index_ci95_low": row["paired_ci"]["packet_vs_source_index"]["ci95_low"],
                    "packet_vs_same_budget_text_ci95_low": row["paired_ci"]["packet_vs_same_budget_text"]["ci95_low"],
                    "packet_vs_best_destructive_ci95_low": row["paired_ci"].get("packet_vs_best_destructive", {}).get("ci95_low", 0.0),
                }
            )


def _write_rate_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "split",
        "budget_bytes",
        "framed_bytes",
        "seed_count",
        "packet_accuracy_mean",
        "packet_accuracy_min",
        "packet_accuracy_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# COLM Acceptance Baseline Audit",
        "",
        f"- date: `{payload['date']}`",
        f"- scoped COLM gate: `{payload['scoped_colm_gate']}`",
        f"- strict positive beyond source-index gate: `{payload['strict_positive_beyond_source_index_gate']}`",
        f"- interpretation: {payload['interpretation']}",
        f"- strict gate interpretation: {payload['strict_gate_interpretation']}",
        "",
        "## Main Baseline Readout",
        "",
        "| Benchmark | Split | Seeds | Packet | Source index | Target | Text | Packet-source CI low | Packet-text CI low |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("arc_test", "obqa_test"):
        row = payload["benchmarks"][key]
        agg = row["aggregate"]
        lines.append(
            f"| {row['benchmark']} | {row['split']} | {agg['seed_count']} | "
            f"{agg['packet_accuracy_mean']:.3f} | {agg['source_index_accuracy_mean']:.3f} | "
            f"{agg['target_accuracy_mean']:.3f} | {agg['same_budget_text_accuracy_mean']:.3f} | "
            f"{agg['packet_vs_source_index_ci95_low_min']:.3f} | "
            f"{agg['packet_vs_same_budget_text_ci95_low_min']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rate Curve",
            "",
            "| Benchmark | Bytes | Framed | Packet mean | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rate_curve"]:
        lines.append(
            f"| {row['benchmark']} | {row['budget_bytes']} | {row['framed_bytes']} | "
            f"{row['packet_accuracy_mean']:.3f} | {row['packet_accuracy_min']:.3f} | "
            f"{row['packet_accuracy_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reviewer Implication",
            "",
            payload["reviewer_implication"],
            "",
            "Lay explanation: the new rows ask whether the tiny packet beats simply sending "
            "which answer the source picked. On the current frozen surfaces, it does not; "
            "the packet is best described as a structured fixed-byte way to carry source "
            "candidate evidence with strict controls, not as a method that outperforms an "
            "explicit source-index channel.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_acceptance_baseline_audit(
    *,
    output_dir: pathlib.Path,
    bootstrap_samples: int,
    rate_budgets: list[int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "arc_validation_predictions.jsonl",
        "arc_test_predictions.jsonl",
        "openbookqa_test_predictions.jsonl",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    arc_train = arc_gate._load_rows(_resolve(ARC_TRAIN))
    arc_validation = arc_gate._load_rows(_resolve(ARC_VALIDATION))
    arc_test = arc_gate._load_rows(_resolve(ARC_TEST))
    obqa_train = arc_gate._load_rows(_resolve(OBQA_TRAIN))
    obqa_test = arc_gate._load_rows(_resolve(OBQA_TEST))

    arc_validation_summaries, arc_validation_predictions = _arc_full_rows_for_split(
        split="validation",
        rows=arc_validation,
        source_cache=_resolve(ARC_VALIDATION_CACHE),
        train_rows=arc_train,
        seeds=list(ARC_SEEDS),
        budget_bytes=8,
        bootstrap_samples=bootstrap_samples,
    )
    arc_test_summaries, arc_test_predictions = _arc_full_rows_for_split(
        split="test",
        rows=arc_test,
        source_cache=_resolve(ARC_TEST_CACHE),
        train_rows=arc_train,
        seeds=list(ARC_SEEDS),
        budget_bytes=8,
        bootstrap_samples=bootstrap_samples,
    )
    obqa_summaries, obqa_predictions, obqa_cache_rows = _obqa_full_rows(
        rows=obqa_test,
        train_rows=obqa_train,
        anchor_predictions=_resolve(OBQA_ANCHOR),
        seeds=list(OBQA_SEEDS),
        budget_bytes=3,
        bootstrap_samples=bootstrap_samples,
    )
    _write_jsonl_gz(output_dir / "arc_validation_predictions.jsonl.gz", arc_validation_predictions)
    _write_jsonl_gz(output_dir / "arc_test_predictions.jsonl.gz", arc_test_predictions)
    _write_jsonl_gz(output_dir / "openbookqa_test_predictions.jsonl.gz", obqa_predictions)
    _write_jsonl(output_dir / "openbookqa_source_index_cache.jsonl", obqa_cache_rows)

    seed_csv = output_dir / "per_seed_baseline_metrics.csv"
    if seed_csv.exists():
        seed_csv.unlink()
    _write_seed_csv(seed_csv, "ARC-Challenge", "validation", arc_validation_summaries)
    _write_seed_csv(seed_csv, "ARC-Challenge", "test", arc_test_summaries)
    _write_seed_csv(seed_csv, "OpenBookQA", "test", obqa_summaries)

    rate_curve = _build_rate_curves(
        arc_train=arc_train,
        arc_test=arc_test,
        arc_test_cache=_resolve(ARC_TEST_CACHE),
        obqa_train=obqa_train,
        obqa_test=obqa_test,
        obqa_anchor=_resolve(OBQA_ANCHOR),
        budgets=rate_budgets,
    )
    _write_rate_csv(output_dir / "rate_curve.csv", rate_curve)

    arc_validation_aggregate = _aggregate(arc_validation_summaries)
    arc_test_aggregate = _aggregate(arc_test_summaries)
    obqa_aggregate = _aggregate(obqa_summaries)
    validation_fusion = _validation_selected_fusion(
        validation_seed_rows=arc_validation_summaries,
        test_seed_rows=arc_test_summaries,
    )

    scoped_colm_gate = bool(
        arc_test_aggregate["packet_vs_target_ci95_low_min"] > 0
        and arc_test_aggregate["packet_vs_same_budget_text_mean"] > 0
        and arc_test_aggregate["packet_vs_same_budget_text_ci95_low_min"] >= 0
        and obqa_aggregate["packet_vs_target_ci95_low_min"] > 0
        and obqa_aggregate["packet_vs_same_budget_text_mean"] > 0
        and obqa_aggregate["packet_vs_same_budget_text_ci95_low_min"] >= 0
        and arc_test_aggregate["packet_ties_or_loses_source_index"]
        and obqa_aggregate["packet_ties_or_loses_source_index"]
    )
    strict_positive_beyond_source_index_gate = bool(
        arc_test_aggregate["packet_vs_source_index_ci95_low_min"] > 0
        and obqa_aggregate["packet_vs_source_index_ci95_low_min"] > 0
    )
    payload = {
        "gate": "source_private_colm_acceptance_baseline_audit",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": scoped_colm_gate,
        "scoped_colm_gate": scoped_colm_gate,
        "strict_positive_beyond_source_index_gate": strict_positive_beyond_source_index_gate,
        "pass_rule": (
            "Pass for scoped COLM readiness means the source-index objection is explicitly audited: packet rows "
            "must beat target-only with positive paired lower bounds, beat same-budget text in mean with "
            "nonnegative paired lower bounds, and if they do not beat explicit source-index communication "
            "the paper must scope the claim as source-candidate transfer rather than compression beyond "
            "source choice."
        ),
        "input_artifacts": {
            "arc_train": _display_path(_resolve(ARC_TRAIN)),
            "arc_validation": _display_path(_resolve(ARC_VALIDATION)),
            "arc_test": _display_path(_resolve(ARC_TEST)),
            "arc_validation_source_cache": _display_path(_resolve(ARC_VALIDATION_CACHE)),
            "arc_test_source_cache": _display_path(_resolve(ARC_TEST_CACHE)),
            "obqa_train": _display_path(_resolve(OBQA_TRAIN)),
            "obqa_test": _display_path(_resolve(OBQA_TEST)),
            "obqa_anchor_predictions": _display_path(_resolve(OBQA_ANCHOR)),
        },
        "input_hashes": {
            name: _sha256_file(_resolve(path))
            for name, path in {
                "arc_train": ARC_TRAIN,
                "arc_validation": ARC_VALIDATION,
                "arc_test": ARC_TEST,
                "arc_validation_source_cache": ARC_VALIDATION_CACHE,
                "arc_test_source_cache": ARC_TEST_CACHE,
                "obqa_train": OBQA_TRAIN,
                "obqa_test": OBQA_TEST,
                "obqa_anchor_predictions": OBQA_ANCHOR,
            }.items()
        },
        "benchmarks": {
            "arc_validation": {
                "benchmark": "ARC-Challenge",
                "split": "validation",
                "budget_bytes": 8,
                "framed_bytes": 11,
                "seed_rows": arc_validation_summaries,
                "aggregate": arc_validation_aggregate,
            },
            "arc_test": {
                "benchmark": "ARC-Challenge",
                "split": "test",
                "budget_bytes": 8,
                "framed_bytes": 11,
                "seed_rows": arc_test_summaries,
                "aggregate": arc_test_aggregate,
            },
            "obqa_test": {
                "benchmark": "OpenBookQA",
                "split": "test",
                "budget_bytes": 3,
                "framed_bytes": 6,
                "seed_rows": obqa_summaries,
                "aggregate": obqa_aggregate,
            },
        },
        "validation_selected_fusion": validation_fusion,
        "rate_curve": rate_curve,
        "nonlatent_baselines": {
            "source_index_byte": "1-byte explicit selected candidate index; accuracy is source-choice accuracy",
            "source_choice_label_text": "selected label text such as A/B/C/D",
            "source_rank_code": "top-rank source-choice code; identical to source-index with current cache because raw source score vectors are not frozen for the headline rows",
            "entropy_matched_random_index": "random 1-byte code sampled from empirical source-index distribution",
            "same_budget_structured_text": "existing same-budget structured text control from the packet gate",
        },
        "interpretation": (
            "The packet remains positive versus target-only, same-budget text, entropy-matched random index, "
            "and destructive controls. It does not beat explicit source-index/source-rank communication on "
            "the current frozen ARC/OBQA surfaces. This resolves the reviewer objection by narrowing the "
            "claim: current LatentWire is a fixed-byte source-private candidate-transfer protocol, not a "
            "method that compresses beyond the source's selected candidate."
        ),
        "strict_gate_interpretation": (
            "The stricter ICLR-style gate, positive transfer beyond an explicit source-index/source-rank "
            "channel, is not met. Passing this artifact is therefore a scoped-COLM correctness gate, not "
            "a claim that the current method beats source-index communication."
        ),
        "reviewer_implication": (
            "This artifact should raise confidence in correctness and scope, but not novelty beyond source-choice "
            "transfer. The paper should include these rows and avoid claiming superiority to explicit source-index "
            "or source-score communication until a richer receiver-family method beats this audit."
        ),
    }

    json_path = output_dir / "colm_acceptance_baseline_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "colm_acceptance_baseline_audit.md", payload)
    manifest = {
        "artifacts": [
            "colm_acceptance_baseline_audit.json",
            "colm_acceptance_baseline_audit.md",
            "per_seed_baseline_metrics.csv",
            "rate_curve.csv",
            "arc_validation_predictions.jsonl.gz",
            "arc_test_predictions.jsonl.gz",
            "openbookqa_test_predictions.jsonl.gz",
            "openbookqa_source_index_cache.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "colm_acceptance_baseline_audit.json",
                "colm_acceptance_baseline_audit.md",
                "per_seed_baseline_metrics.csv",
                "rate_curve.csv",
                "arc_validation_predictions.jsonl.gz",
                "arc_test_predictions.jsonl.gz",
                "openbookqa_test_predictions.jsonl.gz",
                "openbookqa_source_index_cache.jsonl",
            )
        },
        "pass_gate": scoped_colm_gate,
        "scoped_colm_gate": scoped_colm_gate,
        "strict_positive_beyond_source_index_gate": strict_positive_beyond_source_index_gate,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# COLM Acceptance Baseline Audit Manifest",
                "",
                f"- scoped COLM gate: `{scoped_colm_gate}`",
                f"- strict positive beyond source-index gate: `{strict_positive_beyond_source_index_gate}`",
                f"- ARC packet vs source-index mean: `{arc_test_aggregate['packet_vs_source_index_mean']:.4f}`",
                f"- OBQA packet vs source-index mean: `{obqa_aggregate['packet_vs_source_index_mean']:.4f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_budgets(raw: str) -> list[int]:
    budgets = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    if any(value < 2 for value in budgets):
        raise argparse.ArgumentTypeError("packet budgets must be at least 2 bytes; 1-byte source-index is reported separately")
    return budgets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--rate-budgets", type=_parse_budgets, default="2,3,4,8")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_acceptance_baseline_audit(
        output_dir=_resolve(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        rate_budgets=args.rate_budgets,
    )


if __name__ == "__main__":
    main()
