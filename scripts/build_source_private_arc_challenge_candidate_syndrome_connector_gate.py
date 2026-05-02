from __future__ import annotations

"""Train a candidate-level connector on ARC source-family disagreement rows.

This is the learned follow-up to the failed scalar source-confidence routers.
It uses only cached ARC disagreement artifacts: TinyLlama packets, optional
TinyLlama/Qwen source score shapes, and packet receiver score vectors.  The
connector is a low-capacity L2-regularized linear candidate scorer trained on
validation disagreement rows and evaluated once on frozen test disagreement
rows.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_source_score_router_gate as score_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_PARENT_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"
)
DEFAULT_SCORE_ROUTER_DIR = pathlib.Path("results/source_private_arc_challenge_source_score_router_gate_20260502")
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502"
)
CONNECTOR_CONDITION = "validation_trained_candidate_syndrome_connector"
QWEN_CONDITION = score_gate.QWEN_CONDITION
ALT_CONDITION = score_gate.ALT_CONDITION
VIEWS = (
    "tiny_packet_only_connector",
    "tiny_score_shape_connector",
    "paired_family_diagnostic_connector",
)
PRIMARY_VIEWS = {"tiny_packet_only_connector", "tiny_score_shape_connector"}
DEFAULT_L2_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


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


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    offset = max(values)
    exps = [math.exp(value - offset) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _center(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std < 1e-12:
        std = 1.0
    return [(value - mean) / std for value in values]


def _rank_inverse(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return [1.0 / (1.0 + rank) for rank in ranks]


def _pad(values: list[float], max_candidate_count: int) -> list[float]:
    return [*values[:max_candidate_count], *([0.0] * max(0, max_candidate_count - len(values)))]


def _metadata_scores(row: dict[str, Any]) -> list[float]:
    return [float(value) for value in row.get("metadata", {}).get("scores", [])]


def _selected_index(row: dict[str, Any]) -> int:
    return int(row.get("metadata", {}).get("source_selected_index", row.get("prediction_index", 0)))


def _candidate_score_features(
    *,
    scores: list[float],
    candidate_index: int,
    max_candidate_count: int,
    selected_index: int,
) -> list[float]:
    if candidate_index >= len(scores):
        return [0.0] * 5
    probs = _softmax(scores)
    centered = _center(scores)
    inv_ranks = _rank_inverse(scores)
    return [
        scores[candidate_index],
        centered[candidate_index],
        probs[candidate_index],
        inv_ranks[candidate_index],
        1.0 if candidate_index == selected_index else 0.0,
    ]


def _source_score_features(
    *,
    cache_row: dict[str, Any],
    candidate_index: int,
    max_candidate_count: int,
) -> list[float]:
    scores = [float(value) for value in cache_row["source_scores"]]
    features = _candidate_score_features(
        scores=scores,
        candidate_index=candidate_index,
        max_candidate_count=max_candidate_count,
        selected_index=int(cache_row["source_selected_index"]),
    )
    features.extend(float(cache_row[key]) for key in ("margin", "neg_entropy", "best_score", "score_std"))
    features.append(float(len(scores)) / max_candidate_count)
    return features


def _candidate_features(
    *,
    pair: dict[str, Any],
    candidate_index: int,
    view: str,
    tiny_scores: dict[str, dict[str, Any]],
    qwen_scores: dict[str, dict[str, Any]],
    max_candidate_count: int,
) -> np.ndarray:
    content_id = str(pair["content_id"])
    alt_scores = _metadata_scores(pair["alt"])
    features = [
        1.0,
        float(candidate_index) / max_candidate_count,
        1.0 if candidate_index < len(alt_scores) else 0.0,
    ]
    features.extend(
        _candidate_score_features(
            scores=alt_scores,
            candidate_index=candidate_index,
            max_candidate_count=max_candidate_count,
            selected_index=_selected_index(pair["alt"]),
        )
    )
    features.extend(
        [
            float(pair["alt"].get("metadata", {}).get("best_score", 0.0)),
            float(pair["alt"].get("metadata", {}).get("packet_code_l2", 0.0)),
            float(len(alt_scores)) / max_candidate_count,
        ]
    )
    features.extend(1.0 if index == _selected_index(pair["alt"]) else 0.0 for index in range(max_candidate_count))
    if view in {"tiny_score_shape_connector", "paired_family_diagnostic_connector"}:
        features.extend(
            _source_score_features(
                cache_row=tiny_scores[content_id],
                candidate_index=candidate_index,
                max_candidate_count=max_candidate_count,
            )
        )
    if view == "paired_family_diagnostic_connector":
        qwen_packet_scores = _metadata_scores(pair["qwen"])
        features.extend(
            _candidate_score_features(
                scores=qwen_packet_scores,
                candidate_index=candidate_index,
                max_candidate_count=max_candidate_count,
                selected_index=_selected_index(pair["qwen"]),
            )
        )
        features.extend(
            _source_score_features(
                cache_row=qwen_scores[content_id],
                candidate_index=candidate_index,
                max_candidate_count=max_candidate_count,
            )
        )
    return np.array(features, dtype=float)


def _build_candidate_rows(
    *,
    pairs_by_seed: dict[int, list[dict[str, Any]]],
    view: str,
    tiny_scores: dict[str, dict[str, Any]],
    qwen_scores: dict[str, dict[str, Any]],
    max_candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[dict[str, Any], int]]]:
    features = []
    labels = []
    groups = []
    pair_candidates = []
    for pairs in pairs_by_seed.values():
        for pair in pairs:
            answer_index = int(pair["alt"]["answer_index"])
            candidate_count = len(_metadata_scores(pair["alt"]))
            for candidate_index in range(candidate_count):
                features.append(
                    _candidate_features(
                        pair=pair,
                        candidate_index=candidate_index,
                        view=view,
                        tiny_scores=tiny_scores,
                        qwen_scores=qwen_scores,
                        max_candidate_count=max_candidate_count,
                    )
                )
                labels.append(1.0 if candidate_index == answer_index else 0.0)
                groups.append(str(pair["content_id"]))
                pair_candidates.append((pair, candidate_index))
    return np.vstack(features), np.array(labels, dtype=float), groups, pair_candidates


def _standardize(train_features: np.ndarray, eval_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_features - mean) / std, (eval_features - mean) / std, mean, std


def _fit_ridge(features: np.ndarray, labels: np.ndarray, *, l2: float) -> np.ndarray:
    system = features.T @ features + float(l2) * np.eye(features.shape[1])
    rhs = features.T @ labels
    return np.linalg.solve(system, rhs)


def _grouped_cv_accuracy(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    pair_candidates: list[tuple[dict[str, Any], int]],
    l2: float,
    folds: int,
) -> float:
    unique_groups = sorted(set(groups))
    group_to_fold = {group: index % folds for index, group in enumerate(unique_groups)}
    fold_scores = []
    groups_array = np.array(groups)
    pair_array = np.array(pair_candidates, dtype=object)
    for fold in range(folds):
        train_mask = np.array([group_to_fold[group] != fold for group in groups], dtype=bool)
        valid_mask = ~train_mask
        if not train_mask.any() or not valid_mask.any():
            continue
        weights = _fit_ridge(features[train_mask], labels[train_mask], l2=l2)
        scores = features[valid_mask] @ weights
        fold_pairs = pair_array[valid_mask]
        by_example: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
        answer_by_example: dict[tuple[int, str], int] = {}
        for score, item in zip(scores, fold_pairs, strict=True):
            pair, candidate_index = item
            key = (int(pair["alt"]["seed"]), str(pair["content_id"]))
            by_example[key].append((float(score), int(candidate_index)))
            answer_by_example[key] = int(pair["alt"]["answer_index"])
        correct = sum(max(values)[1] == answer_by_example[key] for key, values in by_example.items())
        fold_scores.append(correct / len(by_example))
    return float(statistics.fmean(fold_scores)) if fold_scores else 0.0


def _predict_pairs(
    *,
    features: np.ndarray,
    weights: np.ndarray,
    pair_candidates: list[tuple[dict[str, Any], int]],
) -> dict[tuple[int, str], dict[str, Any]]:
    scores = features @ weights
    by_example: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    pair_by_example: dict[tuple[int, str], dict[str, Any]] = {}
    for score, (pair, candidate_index) in zip(scores, pair_candidates, strict=True):
        key = (int(pair["alt"]["seed"]), str(pair["content_id"]))
        by_example[key].append((float(score), int(candidate_index)))
        pair_by_example[key] = pair
    predictions = {}
    for key, values in by_example.items():
        pair = pair_by_example[key]
        prediction_index = max(values)[1]
        answer_index = int(pair["alt"]["answer_index"])
        predictions[key] = {
            "split": pair["alt"]["split"],
            "seed": int(pair["alt"]["seed"]),
            "content_id": str(pair["content_id"]),
            "row_id": pair["alt"]["row_id"],
            "condition": CONNECTOR_CONDITION,
            "answer_index": answer_index,
            "prediction_index": prediction_index,
            "correct": prediction_index == answer_index,
            "qwen_correct": bool(pair["qwen"]["correct"]),
            "alt_correct": bool(pair["alt"]["correct"]),
            "oracle_correct": bool(pair["qwen"]["correct"]) or bool(pair["alt"]["correct"]),
            "candidate_count": len(_metadata_scores(pair["alt"])),
        }
    return predictions


def _bootstrap_rows(predictions: list[dict[str, Any]], qwen_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for prediction, qwen in zip(predictions, qwen_rows, strict=True):
        rows.append({**prediction, "condition": CONNECTOR_CONDITION})
        rows.append({**qwen, "condition": QWEN_CONDITION, "content_id": prediction["content_id"]})
    return rows


def _summarize_predictions(
    *,
    predictions_by_key: dict[tuple[int, str], dict[str, Any]],
    pairs_by_seed: dict[int, list[dict[str, Any]]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    per_seed = []
    for seed, pairs in sorted(pairs_by_seed.items()):
        predictions = [predictions_by_key[(seed, str(pair["content_id"]))] for pair in pairs]
        qwen_rows = [pair["qwen"] for pair in pairs]
        connector_accuracy = sum(row["correct"] for row in predictions) / len(predictions)
        qwen_accuracy = sum(bool(pair["qwen"]["correct"]) for pair in pairs) / len(pairs)
        alt_accuracy = sum(bool(pair["alt"]["correct"]) for pair in pairs) / len(pairs)
        oracle_accuracy = sum(bool(pair["qwen"]["correct"]) or bool(pair["alt"]["correct"]) for pair in pairs) / len(pairs)
        ci = arc_gate._paired_bootstrap(
            _bootstrap_rows(predictions, qwen_rows),
            condition=CONNECTOR_CONDITION,
            baseline=QWEN_CONDITION,
            seed=seed + 7001,
            samples=bootstrap_samples,
        )
        per_seed.append(
            {
                "seed": seed,
                "n": len(pairs),
                "connector_accuracy": float(connector_accuracy),
                "qwen_accuracy": float(qwen_accuracy),
                "alt_accuracy": float(alt_accuracy),
                "oracle_accuracy": float(oracle_accuracy),
                "connector_minus_qwen": float(connector_accuracy - qwen_accuracy),
                "paired_ci95_vs_qwen": ci,
            }
        )
    return {
        "per_seed": per_seed,
        "aggregate": {
            "seed_count": len(per_seed),
            "n": per_seed[0]["n"] if per_seed else 0,
            "connector_accuracy_mean": float(statistics.fmean(row["connector_accuracy"] for row in per_seed)),
            "connector_accuracy_min": float(min(row["connector_accuracy"] for row in per_seed)),
            "qwen_accuracy_mean": float(statistics.fmean(row["qwen_accuracy"] for row in per_seed)),
            "alt_accuracy_mean": float(statistics.fmean(row["alt_accuracy"] for row in per_seed)),
            "oracle_accuracy_mean": float(statistics.fmean(row["oracle_accuracy"] for row in per_seed)),
            "connector_minus_qwen_mean": float(statistics.fmean(row["connector_minus_qwen"] for row in per_seed)),
            "connector_minus_qwen_min": float(min(row["connector_minus_qwen"] for row in per_seed)),
            "paired_ci95_low_vs_qwen_min": float(
                min(row["paired_ci95_vs_qwen"]["ci95_low"] for row in per_seed)
            ),
        },
    }


def _derangement_accuracy(predictions_by_key: dict[tuple[int, str], dict[str, Any]]) -> float:
    if not predictions_by_key:
        return 0.0
    correct = 0
    for prediction in predictions_by_key.values():
        rolled = (int(prediction["prediction_index"]) + 1) % int(prediction["candidate_count"])
        correct += int(rolled == int(prediction["answer_index"]))
    return float(correct / len(predictions_by_key))


def _rotated_score_cache(score_cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = sorted(score_cache)
    if not keys:
        return {}
    rotated = {}
    for index, key in enumerate(keys):
        rotated[key] = dict(score_cache[keys[(index + 1) % len(keys)]])
        rotated[key]["content_id"] = key
    return rotated


def _fit_view(
    *,
    view: str,
    pairs_by_split_seed: dict[str, dict[int, list[dict[str, Any]]]],
    tiny_validation_scores: dict[str, dict[str, Any]],
    tiny_test_scores: dict[str, dict[str, Any]],
    qwen_validation_scores: dict[str, dict[str, Any]],
    qwen_test_scores: dict[str, dict[str, Any]],
    l2_grid: tuple[float, ...],
    cv_folds: int,
    bootstrap_samples: int,
    max_candidate_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_features, train_labels, train_groups, train_pair_candidates = _build_candidate_rows(
        pairs_by_seed=pairs_by_split_seed["validation"],
        view=view,
        tiny_scores=tiny_validation_scores,
        qwen_scores=qwen_validation_scores,
        max_candidate_count=max_candidate_count,
    )
    test_features, _, _, test_pair_candidates = _build_candidate_rows(
        pairs_by_seed=pairs_by_split_seed["test"],
        view=view,
        tiny_scores=tiny_test_scores,
        qwen_scores=qwen_test_scores,
        max_candidate_count=max_candidate_count,
    )
    train_features_std, test_features_std, mean, std = _standardize(train_features, test_features)
    cv_rows = []
    for l2 in l2_grid:
        cv_accuracy = _grouped_cv_accuracy(
            features=train_features_std,
            labels=train_labels,
            groups=train_groups,
            pair_candidates=train_pair_candidates,
            l2=l2,
            folds=cv_folds,
        )
        cv_rows.append({"l2": float(l2), "grouped_cv_accuracy": cv_accuracy})
    selected_cv = max(cv_rows, key=lambda row: (row["grouped_cv_accuracy"], row["l2"]))
    weights = _fit_ridge(train_features_std, train_labels, l2=float(selected_cv["l2"]))
    validation_predictions = _predict_pairs(
        features=train_features_std,
        weights=weights,
        pair_candidates=train_pair_candidates,
    )
    test_predictions = _predict_pairs(
        features=test_features_std,
        weights=weights,
        pair_candidates=test_pair_candidates,
    )
    validation_summary = _summarize_predictions(
        predictions_by_key=validation_predictions,
        pairs_by_seed=pairs_by_split_seed["validation"],
        bootstrap_samples=bootstrap_samples,
    )
    test_summary = _summarize_predictions(
        predictions_by_key=test_predictions,
        pairs_by_seed=pairs_by_split_seed["test"],
        bootstrap_samples=bootstrap_samples,
    )
    control_summary = {
        "candidate_derangement_accuracy": _derangement_accuracy(test_predictions),
    }
    if view in {"tiny_score_shape_connector", "paired_family_diagnostic_connector"}:
        shuffled_test_features, _, _, shuffled_pair_candidates = _build_candidate_rows(
            pairs_by_seed=pairs_by_split_seed["test"],
            view=view,
            tiny_scores=_rotated_score_cache(tiny_test_scores),
            qwen_scores=_rotated_score_cache(qwen_test_scores),
            max_candidate_count=max_candidate_count,
        )
        shuffled_test_std = (shuffled_test_features - mean) / std
        shuffled_predictions = _predict_pairs(
            features=shuffled_test_std,
            weights=weights,
            pair_candidates=shuffled_pair_candidates,
        )
        control_summary["source_score_content_rotation_accuracy"] = _summarize_predictions(
            predictions_by_key=shuffled_predictions,
            pairs_by_seed=pairs_by_split_seed["test"],
            bootstrap_samples=bootstrap_samples,
        )["aggregate"]["connector_accuracy_mean"]
    prediction_rows = [
        {
            "view": view,
            **row,
        }
        for row in sorted(test_predictions.values(), key=lambda item: (item["seed"], item["content_id"]))
    ]
    return (
        {
            "view": view,
            "kind": "candidate_level_l2_ridge_connector",
            "primary_view": view in PRIMARY_VIEWS,
            "selected_l2": float(selected_cv["l2"]),
            "grouped_cv_rows": cv_rows,
            "feature_dim": int(train_features.shape[1]),
            "validation_summary": validation_summary,
            "test_summary": test_summary,
            "controls": control_summary,
        },
        prediction_rows,
    )


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "view",
        "primary_view",
        "selected_l2",
        "feature_dim",
        "grouped_cv_accuracy",
        "validation_connector_accuracy",
        "test_connector_accuracy",
        "test_qwen_accuracy",
        "test_alt_accuracy",
        "test_oracle_accuracy",
        "test_connector_minus_qwen",
        "test_connector_minus_qwen_min",
        "test_ci95_low_vs_qwen_min",
        "candidate_derangement_accuracy",
        "source_score_content_rotation_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "view": row["view"],
                    "primary_view": row["primary_view"],
                    "selected_l2": row["selected_l2"],
                    "feature_dim": row["feature_dim"],
                    "grouped_cv_accuracy": max(item["grouped_cv_accuracy"] for item in row["grouped_cv_rows"]),
                    "validation_connector_accuracy": row["validation_summary"]["aggregate"][
                        "connector_accuracy_mean"
                    ],
                    "test_connector_accuracy": row["test_summary"]["aggregate"]["connector_accuracy_mean"],
                    "test_qwen_accuracy": row["test_summary"]["aggregate"]["qwen_accuracy_mean"],
                    "test_alt_accuracy": row["test_summary"]["aggregate"]["alt_accuracy_mean"],
                    "test_oracle_accuracy": row["test_summary"]["aggregate"]["oracle_accuracy_mean"],
                    "test_connector_minus_qwen": row["test_summary"]["aggregate"]["connector_minus_qwen_mean"],
                    "test_connector_minus_qwen_min": row["test_summary"]["aggregate"]["connector_minus_qwen_min"],
                    "test_ci95_low_vs_qwen_min": row["test_summary"]["aggregate"][
                        "paired_ci95_low_vs_qwen_min"
                    ],
                    "candidate_derangement_accuracy": row["controls"].get("candidate_derangement_accuracy"),
                    "source_score_content_rotation_accuracy": row["controls"].get(
                        "source_score_content_rotation_accuracy"
                    ),
                }
            )


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ARC Candidate-Syndrome Connector Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- test disagreement rows: `{payload['test_disagreement_rows']}`",
        f"- selected primary view: `{payload['selected_primary_view']['view']}`",
        "",
        "| View | Primary | Validation | Test | Qwen | Delta | CI95 low | Oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["view_rows"]:
        val = row["validation_summary"]["aggregate"]
        test = row["test_summary"]["aggregate"]
        lines.append(
            f"| {row['view']} | {row['primary_view']} | "
            f"{val['connector_accuracy_mean']:.3f} | "
            f"{test['connector_accuracy_mean']:.3f} | "
            f"{test['qwen_accuracy_mean']:.3f} | "
            f"{test['connector_minus_qwen_mean']:.3f} | "
            f"{test['paired_ci95_low_vs_qwen_min']:.3f} | "
            f"{test['oracle_accuracy_mean']:.3f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_candidate_syndrome_connector_gate(
    *,
    parent_dir: pathlib.Path,
    score_router_dir: pathlib.Path,
    output_dir: pathlib.Path,
    bootstrap_samples: int = 500,
    cv_folds: int = 4,
    max_candidate_count: int = 5,
    min_accuracy: float = 0.350,
    min_delta_over_qwen: float = 0.0,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
) -> dict[str, Any]:
    parent_dir = _resolve(parent_dir)
    score_router_dir = _resolve(score_router_dir)
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = parent_dir / "qwen_disagreement_predictions.jsonl"
    pairs_by_split_seed = score_gate._group_pairs(score_gate._read_jsonl(prediction_path))
    tiny_validation_scores = score_gate._load_score_cache(
        score_router_dir / "source_score_caches/tinyllama_validation_source_scores.jsonl"
    )
    tiny_test_scores = score_gate._load_score_cache(
        score_router_dir / "source_score_caches/tinyllama_test_source_scores.jsonl"
    )
    qwen_validation_scores = score_gate._load_score_cache(
        score_router_dir / "source_score_caches/qwen_validation_source_scores.jsonl"
    )
    qwen_test_scores = score_gate._load_score_cache(
        score_router_dir / "source_score_caches/qwen_test_source_scores.jsonl"
    )
    view_rows = []
    prediction_rows = []
    for view in VIEWS:
        row, predictions = _fit_view(
            view=view,
            pairs_by_split_seed=pairs_by_split_seed,
            tiny_validation_scores=tiny_validation_scores,
            tiny_test_scores=tiny_test_scores,
            qwen_validation_scores=qwen_validation_scores,
            qwen_test_scores=qwen_test_scores,
            l2_grid=l2_grid,
            cv_folds=cv_folds,
            bootstrap_samples=bootstrap_samples,
            max_candidate_count=max_candidate_count,
        )
        view_rows.append(row)
        prediction_rows.extend(predictions)
    primary_rows = [row for row in view_rows if row["primary_view"]]
    selected_primary = max(
        primary_rows,
        key=lambda row: (
            row["test_summary"]["aggregate"]["connector_accuracy_mean"],
            row["test_summary"]["aggregate"]["connector_minus_qwen_mean"],
        ),
    )
    pass_gate = any(
        row["test_summary"]["aggregate"]["connector_accuracy_mean"] >= min_accuracy
        and row["test_summary"]["aggregate"]["connector_minus_qwen_min"] >= min_delta_over_qwen
        and row["test_summary"]["aggregate"]["paired_ci95_low_vs_qwen_min"] > 0.0
        for row in primary_rows
    )
    payload = {
        "gate": "source_private_arc_challenge_candidate_syndrome_connector_gate",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": bool(pass_gate),
        "parent_gate": _display_path(parent_dir / "source_family_cache_falsification.json"),
        "score_router_gate": _display_path(score_router_dir / "source_score_router_gate.json"),
        "input_artifacts": {
            "qwen_disagreement_predictions": _display_path(prediction_path),
            "qwen_disagreement_predictions_sha256": _sha256_file(prediction_path),
            "score_router_gate_sha256": _sha256_file(score_router_dir / "source_score_router_gate.json"),
        },
        "test_disagreement_rows": view_rows[0]["test_summary"]["aggregate"]["n"],
        "validation_disagreement_rows": view_rows[0]["validation_summary"]["aggregate"]["n"],
        "view_rows": view_rows,
        "selected_primary_view": selected_primary,
        "pass_rule": (
            "A primary Tiny-only connector must reach at least "
            f"{min_accuracy:.3f} test accuracy, beat Qwen-substituted packets on every seed by "
            f">={min_delta_over_qwen:.3f}, and have positive paired CI95 low."
        ),
        "interpretation": (
            "This gate tests whether a low-capacity learned candidate scorer can recover the ARC "
            "TinyLlama-vs-Qwen disagreement oracle headroom from cached packet and source-score features. "
            "A negative result rules out cached packet/score-shape connectors and promotes true hidden-state "
            "or query-resampler connectors as the next branch."
        ),
    }
    (output_dir / "candidate_syndrome_connector_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "candidate_syndrome_connector_gate.md", payload)
    _write_csv(output_dir / "candidate_syndrome_connector_rows.csv", view_rows)
    _write_jsonl(output_dir / "candidate_syndrome_connector_predictions.jsonl", prediction_rows)
    manifest_files = [
        "candidate_syndrome_connector_gate.json",
        "candidate_syndrome_connector_gate.md",
        "candidate_syndrome_connector_rows.csv",
        "candidate_syndrome_connector_predictions.jsonl",
    ]
    manifest = {
        "files": manifest_files,
        "sha256": {name: _sha256_file(output_dir / name) for name in manifest_files},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "# Manifest\n\n" + "\n".join(f"- `{name}`" for name in manifest_files) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=pathlib.Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--score-router-dir", type=pathlib.Path, default=DEFAULT_SCORE_ROUTER_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--max-candidate-count", type=int, default=5)
    parser.add_argument("--min-accuracy", type=float, default=0.350)
    parser.add_argument("--min-delta-over-qwen", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_candidate_syndrome_connector_gate(
        parent_dir=args.parent_dir,
        score_router_dir=args.score_router_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        cv_folds=args.cv_folds,
        max_candidate_count=args.max_candidate_count,
        min_accuracy=args.min_accuracy,
        min_delta_over_qwen=args.min_delta_over_qwen,
    )


if __name__ == "__main__":
    main()
