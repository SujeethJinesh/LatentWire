from __future__ import annotations

"""Build the OpenBookQA train-only packet/receiver headroom gate.

The gate reconstructs the promoted 3B source-private packet from the
answer-free source-choice cache, trains a public target-side scorer on the
OpenBookQA train split, selects a receiver on validation only, and evaluates
once on test.  The receiver may choose either the packet prediction or the
target-side public scorer prediction.  Test labels are not used for method
selection.
"""

import argparse
import csv
import datetime as dt
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

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_openbookqa_receiver_headroom_gate_20260502")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_test.jsonl"
)
DEFAULT_VALIDATION_SOURCE_CACHE = pathlib.Path(
    "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_validation_3b/"
    "source_prediction_cache.jsonl"
)
DEFAULT_TEST_SOURCE_CACHE = pathlib.Path(
    "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/"
    "source_prediction_cache.jsonl"
)

STRICT_SOURCE_DESTROY_CONTROLS = (
    "zero_source",
    "shuffled_source_packet",
    "random_same_byte_packet",
    "target_derived_sidecar",
    "candidate_derangement",
)
REPORT_CONDITIONS = (
    arc_gate.MATCHED_CONDITION,
    "source_label_copy",
    "same_byte_structured_text",
    "target_public_ridge",
    *STRICT_SOURCE_DESTROY_CONTROLS,
    "label_permutation",
)
SELECTOR_FEATURES = (
    "packet_top",
    "packet_margin",
    "target_top",
    "target_margin",
    "target_minus_packet_margin",
    "packet_target_agree",
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


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_source_cache(path: pathlib.Path) -> dict[str, int]:
    choices: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = set(row.get("forbidden_source_fields", ()))
            if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
                raise ValueError(f"source cache row {row.get('row_id')} is missing forbidden-field contract")
            choices[str(row["content_id"])] = int(row["source_selected_index"])
    if not choices:
        raise ValueError(f"{path} contained no source-choice rows")
    return choices


def _source_predictions(rows: list[arc_gate.ArcRow], cache: dict[str, int]) -> list[int]:
    predictions: list[int] = []
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        if row.content_id not in cache:
            missing.append(row.content_id)
            continue
        prediction = int(cache[row.content_id])
        if prediction < 0 or prediction >= len(row.choices):
            invalid.append(row.content_id)
            continue
        predictions.append(prediction)
    if missing or invalid:
        raise ValueError(f"source cache mismatch: missing={len(missing)} invalid={len(invalid)}")
    return predictions


def _fit_target_public_scorer(
    *,
    train_rows: list[arc_gate.ArcRow],
    feature_dim: int,
    ridge: float,
) -> dict[str, Any]:
    pair_features = arc_gate._features(
        arc_gate._choice_pair_texts(train_rows),
        dim=feature_dim,
        feature_mode="hashed",
        feature_model="",
        feature_device="auto",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
    )
    scorer = arc_gate._fit_ridge_pair_scorer(train_rows, pair_features, ridge=ridge)
    return {
        **scorer,
        "feature_dim": int(feature_dim),
        "ridge": float(ridge),
        "kind": "train_split_public_hashed_pair_ridge",
    }


def _score_target_public(
    rows: list[arc_gate.ArcRow],
    *,
    scorer: dict[str, Any],
) -> tuple[list[list[float]], list[int]]:
    pair_features = arc_gate._features(
        arc_gate._choice_pair_texts(rows),
        dim=int(scorer["feature_dim"]),
        feature_mode="hashed",
        feature_model="",
        feature_device="auto",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
    )
    return arc_gate._score_rows(rows, pair_features, scorer)


def _packet_residuals(rows: list[arc_gate.ArcRow], *, feature_dim: int) -> list[np.ndarray]:
    pair_features = arc_gate._features(
        arc_gate._choice_pair_texts(rows),
        dim=feature_dim,
        feature_mode="hashed",
        feature_model="",
        feature_device="auto",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
    )
    return arc_gate._candidate_residuals(rows, pair_features)


def _prediction_groups_for_seed(
    *,
    rows: list[arc_gate.ArcRow],
    residuals: list[np.ndarray],
    source_predictions: list[int],
    index_prior: list[float],
    feature_dim: int,
    code_dim: int,
    budget_bytes: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    projection = arc_gate._projection_matrix(feature_dim, code_dim, seed=seed + 171)
    prediction_rows = arc_gate._rows_for_predictions(
        eval_rows=rows,
        residuals=residuals,
        source_predictions=source_predictions,
        projection=projection,
        budget_bytes=budget_bytes,
        index_prior=index_prior,
        seed=seed + 911,
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    return grouped


def _source_label_copy_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    metadata = dict(row.get("metadata", {}))
    prediction_index = int(metadata["source_selected_index"])
    copied["condition"] = "source_label_copy"
    copied["prediction_index"] = prediction_index
    copied["prediction_label"] = metadata.get("source_selected_label", str(prediction_index))
    copied["correct"] = bool(prediction_index == int(row["answer_index"]))
    copied["payload_bytes"] = 1
    copied["payload_hex"] = f"{prediction_index & 0xff:02x}"
    copied["metadata"] = {
        "decoder": "source_label_copy_control",
        "scores": [],
        "source_selected_index": prediction_index,
        "source_visible_fields": metadata.get("source_visible_fields", ["question", "choices"]),
        "forbidden_source_fields": metadata.get("forbidden_source_fields", list(arc_gate.FORBIDDEN_SOURCE_KEYS)),
    }
    return copied


def _target_public_row(
    *,
    row: arc_gate.ArcRow,
    target_scores: list[float],
    target_prediction: int,
) -> dict[str, Any]:
    return {
        "condition": "target_public_ridge",
        "row_id": row.row_id,
        "content_id": row.content_id,
        "answer_index": row.answer_index,
        "answer_label": row.answer_label,
        "prediction_index": int(target_prediction),
        "prediction_label": row.choice_labels[int(target_prediction)],
        "correct": bool(int(target_prediction) == row.answer_index),
        "payload_bytes": 0,
        "payload_hex": "",
        "latency_ms": 0.0,
        "metadata": {
            "decoder": "train_split_public_hashed_pair_ridge",
            "scores": [float(score) for score in target_scores],
            "source_visible_fields": [],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
        },
    }


def _top_margin(scores: list[float]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    order = sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)
    top = float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else 0.0
    return top, top - second


def _features_for_row(
    *,
    packet_row: dict[str, Any],
    target_scores: list[float],
    target_prediction: int,
) -> dict[str, float]:
    packet_scores = [float(score) for score in packet_row.get("metadata", {}).get("scores", [])]
    packet_top, packet_margin = _top_margin(packet_scores)
    target_top, target_margin = _top_margin([float(score) for score in target_scores])
    packet_prediction = int(packet_row["prediction_index"])
    return {
        "packet_top": packet_top,
        "packet_margin": packet_margin,
        "target_top": target_top,
        "target_margin": target_margin,
        "target_minus_packet_margin": target_margin - packet_margin,
        "packet_target_agree": float(packet_prediction == int(target_prediction)),
    }


def _condition_data(
    *,
    rows: list[arc_gate.ArcRow],
    grouped: dict[str, dict[str, dict[str, Any]]],
    target_scores_by_id: dict[str, list[float]],
    target_predictions_by_id: dict[str, int],
    condition: str,
) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for row in rows:
        conditions = grouped[row.content_id]
        if condition == "source_label_copy":
            packet_row = _source_label_copy_row(conditions[arc_gate.MATCHED_CONDITION])
        elif condition == "target_public_ridge":
            packet_row = _target_public_row(
                row=row,
                target_scores=target_scores_by_id[row.content_id],
                target_prediction=target_predictions_by_id[row.content_id],
            )
        else:
            packet_row = conditions[condition]
        target_prediction = int(target_predictions_by_id[row.content_id])
        target_scores = target_scores_by_id[row.content_id]
        data.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "condition": condition,
                "answer_index": row.answer_index,
                "base_prediction": int(packet_row["prediction_index"]),
                "target_prediction": target_prediction,
                "features": _features_for_row(
                    packet_row=packet_row,
                    target_scores=target_scores,
                    target_prediction=target_prediction,
                ),
                "base_correct": bool(int(packet_row["prediction_index"]) == row.answer_index),
                "target_correct": bool(target_prediction == row.answer_index),
                "payload_bytes": int(packet_row.get("payload_bytes", 0)),
                "packet_metadata": packet_row.get("metadata", {}),
            }
        )
    return data


def _selector_matrix(data: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[1.0] + [float(row["features"][name]) for name in SELECTOR_FEATURES] for row in data],
        dtype=np.float64,
    )


def _fit_selector(
    data: list[dict[str, Any]],
    *,
    ridges: list[float],
    threshold_percentiles: list[int],
) -> dict[str, Any]:
    x = _selector_matrix(data)
    y = np.asarray(
        [float(row["target_correct"]) - float(row["base_correct"]) for row in data],
        dtype=np.float64,
    )
    best: dict[str, Any] | None = None
    for ridge in ridges:
        penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        benefit_scores = x @ weights
        thresholds = np.percentile(benefit_scores, threshold_percentiles)
        for threshold in thresholds:
            receiver_predictions = [
                int(row["target_prediction"]) if score > threshold else int(row["base_prediction"])
                for row, score in zip(data, benefit_scores, strict=True)
            ]
            correct = [
                prediction == int(row["answer_index"])
                for prediction, row in zip(receiver_predictions, data, strict=True)
            ]
            override_count = sum(
                int(prediction != int(row["base_prediction"]))
                for prediction, row in zip(receiver_predictions, data, strict=True)
            )
            candidate = {
                "ridge": float(ridge),
                "threshold": float(threshold),
                "weights": [float(value) for value in weights],
                "validation_accuracy": float(sum(correct) / len(correct)),
                "validation_override_count": int(override_count),
                "validation_base_accuracy": float(sum(row["base_correct"] for row in data) / len(data)),
                "validation_target_accuracy": float(sum(row["target_correct"] for row in data) / len(data)),
            }
            key = (
                candidate["validation_accuracy"],
                -candidate["validation_override_count"],
                -abs(candidate["threshold"]),
                -candidate["ridge"],
            )
            if best is None or key > best["_selection_key"]:
                best = {**candidate, "_selection_key": key}
    if best is None:
        raise ValueError("selector search produced no candidate")
    best.pop("_selection_key", None)
    best["feature_names"] = ["bias", *SELECTOR_FEATURES]
    return best


def _paired_bootstrap(
    deltas: list[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _evaluate_selector(
    data: list[dict[str, Any]],
    selector: dict[str, Any],
    *,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    weights = np.asarray(selector["weights"], dtype=np.float64)
    threshold = float(selector["threshold"])
    x = _selector_matrix(data)
    scores = x @ weights
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for row, score in zip(data, scores, strict=True):
        use_target = bool(float(score) > threshold)
        receiver_prediction = int(row["target_prediction"] if use_target else row["base_prediction"])
        receiver_correct = bool(receiver_prediction == int(row["answer_index"]))
        base_correct = bool(row["base_correct"])
        deltas.append(float(receiver_correct) - float(base_correct))
        rows.append(
            {
                "row_id": row["row_id"],
                "content_id": row["content_id"],
                "condition": row["condition"],
                "answer_index": int(row["answer_index"]),
                "base_prediction": int(row["base_prediction"]),
                "target_prediction": int(row["target_prediction"]),
                "receiver_prediction": receiver_prediction,
                "receiver_used_target": use_target,
                "receiver_score": float(score),
                "receiver_correct": receiver_correct,
                "base_correct": base_correct,
                "target_correct": bool(row["target_correct"]),
                "payload_bytes": int(row["payload_bytes"]),
            }
        )
    receiver_correct_count = sum(int(row["receiver_correct"]) for row in rows)
    base_correct_count = sum(int(row["base_correct"]) for row in rows)
    target_correct_count = sum(int(row["target_correct"]) for row in rows)
    help_count = sum(int(delta > 0.0) for delta in deltas)
    harm_count = sum(int(delta < 0.0) for delta in deltas)
    metrics = {
        "n": len(rows),
        "base_accuracy": float(base_correct_count / len(rows)),
        "target_public_accuracy": float(target_correct_count / len(rows)),
        "receiver_accuracy": float(receiver_correct_count / len(rows)),
        "receiver_minus_base": float((receiver_correct_count - base_correct_count) / len(rows)),
        "receiver_minus_target_public": float((receiver_correct_count - target_correct_count) / len(rows)),
        "override_count": int(sum(int(row["receiver_used_target"]) for row in rows)),
        "override_rate": float(sum(int(row["receiver_used_target"]) for row in rows) / len(rows)),
        "help_count": int(help_count),
        "harm_count": int(harm_count),
        "paired_ci95_vs_base": _paired_bootstrap(deltas, seed=seed, samples=bootstrap_samples),
    }
    return metrics, rows, deltas


def _row_mean_bootstrap(
    per_seed_prediction_rows: list[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    by_row: dict[str, list[float]] = {}
    for row in per_seed_prediction_rows:
        delta = float(row["receiver_correct"]) - float(row["base_correct"])
        by_row.setdefault(str(row["content_id"]), []).append(delta)
    row_means = [statistics.fmean(values) for _, values in sorted(by_row.items())]
    return _paired_bootstrap(row_means, seed=seed, samples=samples)


def _write_metrics_csv(path: pathlib.Path, per_seed: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "condition",
        "base_accuracy",
        "target_public_accuracy",
        "receiver_accuracy",
        "receiver_minus_base",
        "receiver_minus_target_public",
        "ci95_low_vs_base",
        "ci95_high_vs_base",
        "override_rate",
        "help_count",
        "harm_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for seed_row in per_seed:
            for condition, metrics in seed_row["condition_metrics"].items():
                writer.writerow(
                    {
                        "seed": seed_row["seed"],
                        "condition": condition,
                        "base_accuracy": metrics["base_accuracy"],
                        "target_public_accuracy": metrics["target_public_accuracy"],
                        "receiver_accuracy": metrics["receiver_accuracy"],
                        "receiver_minus_base": metrics["receiver_minus_base"],
                        "receiver_minus_target_public": metrics["receiver_minus_target_public"],
                        "ci95_low_vs_base": metrics["paired_ci95_vs_base"]["ci95_low"],
                        "ci95_high_vs_base": metrics["paired_ci95_vs_base"]["ci95_high"],
                        "override_rate": metrics["override_rate"],
                        "help_count": metrics["help_count"],
                        "harm_count": metrics["harm_count"],
                    }
                )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    default = headline["default_seed_matched"]
    lines = [
        "# Source-Private OpenBookQA Receiver/Headroom Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- receiver candidate pass: `{payload['receiver_candidate_pass']}`",
        f"- strict per-seed CI pass count: `{headline['strict_per_seed_ci_pass_count']} / {headline['seed_count']}`",
        f"- aggregate seed-row CI vs packet: `{headline['aggregate_seed_row_ci_vs_packet']}`",
        f"- train / validation / test rows: `{payload['train_rows']}` / `{payload['validation_rows']}` / `{payload['test_rows']}`",
        f"- source packet budget: `{payload['budget_bytes']}B`",
        "",
        "## Default Seed Test",
        "",
        f"- matched packet-only accuracy: `{default['base_accuracy']:.3f}`",
        f"- public target receiver accuracy: `{default['target_public_accuracy']:.3f}`",
        f"- packet+target receiver accuracy: `{default['receiver_accuracy']:.3f}`",
        f"- receiver minus packet: `{default['receiver_minus_base']:.3f}`",
        f"- paired CI95 vs packet: `{default['paired_ci95_vs_base']}`",
        f"- receiver minus public target: `{default['receiver_minus_target_public']:.3f}`",
        "",
        "## Test Conditions",
        "",
        "| Seed | Condition | Base | Receiver | Delta | CI95 low | Overrides |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed_row in payload["per_seed"]:
        for condition, metrics in seed_row["condition_metrics"].items():
            lines.append(
                f"| {seed_row['seed']} | `{condition}` | {metrics['base_accuracy']:.3f} | "
                f"{metrics['receiver_accuracy']:.3f} | {metrics['receiver_minus_base']:.3f} | "
                f"{metrics['paired_ci95_vs_base']['ci95_low']:.3f} | {metrics['override_count']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay description: the experiment asks whether the receiver can learn when to trust the tiny "
            "source packet and when to fall back to its own public question/candidate scorer. The source "
            "packet is like a short hint; the receiver is a trained referee that decides whether the hint "
            "looks useful for this question.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_receiver_gate(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    validation_source_cache: pathlib.Path,
    test_source_cache: pathlib.Path,
    seeds: list[int],
    budget_bytes: int,
    packet_feature_dim: int,
    code_dim: int,
    target_feature_dim: int,
    target_ridge: float,
    selector_ridges: list[float],
    threshold_percentiles: list[int],
    bootstrap_samples: int,
    min_receiver_lift: float,
    min_control_gap: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = arc_gate._load_rows(train_path)
    validation_rows = arc_gate._load_rows(validation_path)
    test_rows = arc_gate._load_rows(test_path)
    overlap = sorted(
        ({row.content_id for row in train_rows} & {row.content_id for row in test_rows})
        | ({row.content_id for row in validation_rows} & {row.content_id for row in test_rows})
    )

    validation_source_predictions = _source_predictions(validation_rows, _read_source_cache(validation_source_cache))
    test_source_predictions = _source_predictions(test_rows, _read_source_cache(test_source_cache))
    index_prior = arc_gate._index_prior(train_rows)
    validation_residuals = _packet_residuals(validation_rows, feature_dim=packet_feature_dim)
    test_residuals = _packet_residuals(test_rows, feature_dim=packet_feature_dim)

    target_scorer = _fit_target_public_scorer(
        train_rows=train_rows,
        feature_dim=target_feature_dim,
        ridge=target_ridge,
    )
    validation_target_scores, validation_target_predictions = _score_target_public(
        validation_rows,
        scorer=target_scorer,
    )
    test_target_scores, test_target_predictions = _score_target_public(test_rows, scorer=target_scorer)
    validation_target_scores_by_id = {
        row.content_id: scores for row, scores in zip(validation_rows, validation_target_scores, strict=True)
    }
    validation_target_predictions_by_id = {
        row.content_id: prediction
        for row, prediction in zip(validation_rows, validation_target_predictions, strict=True)
    }
    test_target_scores_by_id = {
        row.content_id: scores for row, scores in zip(test_rows, test_target_scores, strict=True)
    }
    test_target_predictions_by_id = {
        row.content_id: prediction for row, prediction in zip(test_rows, test_target_predictions, strict=True)
    }

    per_seed: list[dict[str, Any]] = []
    all_matched_prediction_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        validation_groups = _prediction_groups_for_seed(
            rows=validation_rows,
            residuals=validation_residuals,
            source_predictions=validation_source_predictions,
            index_prior=index_prior,
            feature_dim=packet_feature_dim,
            code_dim=code_dim,
            budget_bytes=budget_bytes,
            seed=seed,
        )
        test_groups = _prediction_groups_for_seed(
            rows=test_rows,
            residuals=test_residuals,
            source_predictions=test_source_predictions,
            index_prior=index_prior,
            feature_dim=packet_feature_dim,
            code_dim=code_dim,
            budget_bytes=budget_bytes,
            seed=seed,
        )
        validation_data = _condition_data(
            rows=validation_rows,
            grouped=validation_groups,
            target_scores_by_id=validation_target_scores_by_id,
            target_predictions_by_id=validation_target_predictions_by_id,
            condition=arc_gate.MATCHED_CONDITION,
        )
        selector = _fit_selector(
            validation_data,
            ridges=selector_ridges,
            threshold_percentiles=threshold_percentiles,
        )
        condition_metrics: dict[str, Any] = {}
        for condition in REPORT_CONDITIONS:
            condition_eval_data = _condition_data(
                rows=test_rows,
                grouped=test_groups,
                target_scores_by_id=test_target_scores_by_id,
                target_predictions_by_id=test_target_predictions_by_id,
                condition=condition,
            )
            metrics, prediction_rows, deltas = _evaluate_selector(
                condition_eval_data,
                selector,
                seed=seed + 1401 + len(condition),
                bootstrap_samples=bootstrap_samples,
            )
            condition_metrics[condition] = metrics
            for prediction_row in prediction_rows:
                prediction_row["seed"] = seed
                prediction_row["selector_ridge"] = selector["ridge"]
                prediction_row["selector_threshold"] = selector["threshold"]
            all_prediction_rows.extend(prediction_rows)
            if condition == arc_gate.MATCHED_CONDITION:
                all_matched_prediction_rows.extend(prediction_rows)
        matched = condition_metrics[arc_gate.MATCHED_CONDITION]
        per_seed.append(
            {
                "seed": seed,
                "selector": selector,
                "condition_metrics": condition_metrics,
                "strict_receiver_ci_pass": bool(
                    matched["receiver_minus_base"] >= min_receiver_lift
                    and matched["paired_ci95_vs_base"]["ci95_low"] > 0.0
                ),
            }
        )

    default_seed = seeds[0]
    default_seed_row = next(row for row in per_seed if int(row["seed"]) == int(default_seed))
    default_matched = default_seed_row["condition_metrics"][arc_gate.MATCHED_CONDITION]
    default_control_names = (*STRICT_SOURCE_DESTROY_CONTROLS, "same_byte_structured_text", "target_public_ridge")
    default_best_control_name = max(
        default_control_names,
        key=lambda condition: default_seed_row["condition_metrics"][condition]["receiver_accuracy"],
    )
    default_best_control = default_seed_row["condition_metrics"][default_best_control_name]
    aggregate_seed_ci = _row_mean_bootstrap(
        all_matched_prediction_rows,
        seed=seeds[0] + 1777,
        samples=bootstrap_samples,
    )
    strict_per_seed_ci_pass_count = sum(int(row["strict_receiver_ci_pass"]) for row in per_seed)
    all_seed_deltas_positive = all(
        row["condition_metrics"][arc_gate.MATCHED_CONDITION]["receiver_minus_base"] > 0.0
        for row in per_seed
    )
    control_gap = default_matched["receiver_accuracy"] - default_best_control["receiver_accuracy"]
    receiver_candidate_pass = bool(
        not overlap
        and default_matched["receiver_minus_base"] >= min_receiver_lift
        and default_matched["paired_ci95_vs_base"]["ci95_low"] > 0.0
        and default_matched["receiver_minus_target_public"] >= min_receiver_lift
        and control_gap >= min_control_gap
        and aggregate_seed_ci["ci95_low"] > 0.0
        and all_seed_deltas_positive
    )
    pass_gate = receiver_candidate_pass
    target_public_test_accuracy = float(
        sum(int(prediction == row.answer_index) for prediction, row in zip(test_target_predictions, test_rows, strict=True))
        / len(test_rows)
    )
    payload = {
        "gate": "source_private_openbookqa_receiver_headroom_gate",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "validation_path": _display_path(validation_path),
        "test_path": _display_path(test_path),
        "validation_source_cache": _display_path(validation_source_cache),
        "test_source_cache": _display_path(test_source_cache),
        "train_sha256": _sha256_file(train_path),
        "validation_sha256": _sha256_file(validation_path),
        "test_sha256": _sha256_file(test_path),
        "validation_source_cache_sha256": _sha256_file(validation_source_cache),
        "test_source_cache_sha256": _sha256_file(test_source_cache),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "train_validation_test_overlap_count": len(overlap),
        "train_validation_test_overlap_sha256": hashlib.sha256("\n".join(overlap).encode("utf-8")).hexdigest(),
        "seeds": seeds,
        "budget_bytes": budget_bytes,
        "packet_feature_dim": packet_feature_dim,
        "code_dim": code_dim,
        "target_feature_dim": target_feature_dim,
        "target_ridge": target_ridge,
        "selector_ridges": selector_ridges,
        "threshold_percentiles": threshold_percentiles,
        "bootstrap_samples": bootstrap_samples,
        "target_public_receiver": {
            "kind": target_scorer["kind"],
            "feature_dim": target_feature_dim,
            "ridge": target_ridge,
            "validation_accuracy": float(
                sum(
                    int(prediction == row.answer_index)
                    for prediction, row in zip(validation_target_predictions, validation_rows, strict=True)
                )
                / len(validation_rows)
            ),
            "test_accuracy": target_public_test_accuracy,
            "train_pair_rows": int(target_scorer["train_pair_rows"]),
            "train_positive_rows": int(target_scorer["train_positive_rows"]),
        },
        "method_contract": {
            "source_packet_budget_bytes": budget_bytes,
            "source_packet_origin": "answer-key-forbidden Qwen2.5-0.5B source-choice cache from the promoted 3B OpenBookQA packet gate",
            "receiver_training": "target public scorer trained on OpenBookQA train; selector selected on validation only; test labels held out until final evaluation",
            "receiver_inputs_at_test": [
                "source-private packet decoder scores/prediction",
                "public question/candidate text through a train-split target scorer",
            ],
            "forbidden_eval_source_inputs": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "claim_boundary": (
                "This is a packet/target evidence-fusion receiver. It is not a native GPU systems result and "
                "does not claim source-label-copy separation because the promoted 3B packet is itself a compact "
                "source-selected-candidate sketch."
            ),
        },
        "per_seed": per_seed,
        "headline": {
            "default_seed": default_seed,
            "default_seed_matched": default_matched,
            "default_best_receiver_control": default_best_control_name,
            "default_best_receiver_control_accuracy": default_best_control["receiver_accuracy"],
            "default_matched_minus_best_receiver_control": control_gap,
            "aggregate_seed_row_ci_vs_packet": aggregate_seed_ci,
            "seed_count": len(seeds),
            "strict_per_seed_ci_pass_count": strict_per_seed_ci_pass_count,
            "all_seed_deltas_positive": all_seed_deltas_positive,
        },
        "receiver_candidate_pass": receiver_candidate_pass,
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires no train/validation/test content overlap, the default seed receiver to beat packet-only "
            "and target-public-only by the configured lift with positive paired CI, the default matched receiver "
            "to beat the strongest no-source/same-byte control by the configured gap, every seed's receiver delta "
            "to be positive, and the row-bootstrap aggregate across seeds to have positive CI95 lower bound."
        ),
        "interpretation": (
            "OpenBookQA now has a positive receiver-fusion row: the default 3B packet receiver improves over "
            "packet-only and over a train-split public target scorer on held-out test, while same-byte text and "
            "source-destroy controls stay lower. The result is a useful positive method branch, but it should be "
            "framed as source-private evidence fusion rather than universal latent-language transfer; the current "
            "packet still behaves like a compact source-selected-candidate sketch, so a stronger common-basis or "
            "learned connector remains necessary for a comfortable ICLR full paper."
        ),
    }
    (output_dir / "openbookqa_receiver_headroom_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_metrics_csv(output_dir / "openbookqa_receiver_headroom_gate.csv", per_seed)
    _write_jsonl(output_dir / "receiver_predictions.jsonl", all_prediction_rows)
    _write_markdown(output_dir / "openbookqa_receiver_headroom_gate.md", payload)
    manifest = {
        "artifacts": [
            "openbookqa_receiver_headroom_gate.json",
            "openbookqa_receiver_headroom_gate.md",
            "openbookqa_receiver_headroom_gate.csv",
            "receiver_predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "openbookqa_receiver_headroom_gate.json",
                "openbookqa_receiver_headroom_gate.md",
                "openbookqa_receiver_headroom_gate.csv",
                "receiver_predictions.jsonl",
            )
        },
        "pass_gate": pass_gate,
        "receiver_candidate_pass": receiver_candidate_pass,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private OpenBookQA Receiver/Headroom Gate Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- receiver candidate pass: `{receiver_candidate_pass}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one float is required")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--validation-source-cache", type=pathlib.Path, default=DEFAULT_VALIDATION_SOURCE_CACHE)
    parser.add_argument("--test-source-cache", type=pathlib.Path, default=DEFAULT_TEST_SOURCE_CACHE)
    parser.add_argument("--seeds", type=_parse_int_list, default="47,53,59,61,67")
    parser.add_argument("--budget-bytes", type=int, default=3)
    parser.add_argument("--packet-feature-dim", type=int, default=384)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--target-feature-dim", type=int, default=1536)
    parser.add_argument("--target-ridge", type=float, default=1.0)
    parser.add_argument("--selector-ridges", type=_parse_float_list, default="0.01,0.1,1,10,100")
    parser.add_argument("--threshold-percentiles", type=_parse_int_list, default="0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-receiver-lift", type=float, default=0.005)
    parser.add_argument("--min-control-gap", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_receiver_gate(
        output_dir=_resolve(args.output_dir),
        train_path=_resolve(args.train_path),
        validation_path=_resolve(args.validation_path),
        test_path=_resolve(args.test_path),
        validation_source_cache=_resolve(args.validation_source_cache),
        test_source_cache=_resolve(args.test_source_cache),
        seeds=args.seeds,
        budget_bytes=args.budget_bytes,
        packet_feature_dim=args.packet_feature_dim,
        code_dim=args.code_dim,
        target_feature_dim=args.target_feature_dim,
        target_ridge=args.target_ridge,
        selector_ridges=args.selector_ridges,
        threshold_percentiles=args.threshold_percentiles,
        bootstrap_samples=args.bootstrap_samples,
        min_receiver_lift=args.min_receiver_lift,
        min_control_gap=args.min_control_gap,
    )


if __name__ == "__main__":
    main()
