from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_switch_decomposition_20260502")
DEFAULT_TRAIN = top2.DEFAULT_TRAIN
DEFAULT_TRAIN_SCORE_CACHE = top2.DEFAULT_TRAIN_SCORE_CACHE
DEFAULT_TRAIN_HIDDEN_CACHE = top2.DEFAULT_TRAIN_HIDDEN_CACHE
DEFAULT_TRAIN_HIDDEN_ROWS = 512
DEFAULT_SELECTION_SEED = 1729
DEFAULT_DEV_FRACTION = 0.25
DEFAULT_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)
STRICT_DELTA = 0.02


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    name: str
    eval_path: pathlib.Path
    score_cache: pathlib.Path
    hidden_cache: pathlib.Path
    dense_reference_predictions: pathlib.Path | None = None
    hybrid_reference_predictions: pathlib.Path | None = None


DEFAULT_EVAL_SPECS = (
    EvalSpec(
        name="validation_first1024",
        eval_path=top2.DEFAULT_EVAL,
        score_cache=top2.DEFAULT_EVAL_SCORE_CACHE,
        hidden_cache=top2.DEFAULT_EVAL_HIDDEN_CACHE,
        dense_reference_predictions=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024/"
            "predictions.jsonl"
        ),
        hybrid_reference_predictions=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_bagged_gate_20260502_qwen05_train512_validation1024_hybrid_vote_on_score_agreement/"
            "predictions.jsonl"
        ),
    ),
    EvalSpec(
        name="terminal_tail_9216_10042",
        eval_path=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/"
            "hellaswag_validation_rows_9216_10042.jsonl"
        ),
        score_cache=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/"
            "source_eval_score_cache.json"
        ),
        hidden_cache=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/"
            "source_eval_hidden_cache.npz"
        ),
        dense_reference_predictions=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/"
            "bagged_gate/predictions.jsonl"
        ),
        hybrid_reference_predictions=pathlib.Path(
            "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260502_qwen05_train512_validation9216_10042_hybrid_vote_on_score_agreement/"
            "bagged_gate/predictions.jsonl"
        ),
    ),
)


def _prediction_accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    return top2._accuracy(rows, predictions)


def _top2_oracle_predictions(rows: list[arc_gate.ArcRow], scores: list[list[float]]) -> list[int]:
    predictions: list[int] = []
    for row, row_scores in zip(rows, scores, strict=True):
        ranked = top2._ranked_indices(row_scores)
        predictions.append(row.answer_index if row.answer_index in ranked[:2] else ranked[0])
    return predictions


def _score_feature_matrix(scores: list[list[float]]) -> np.ndarray:
    return np.asarray([top2._score_features(row_scores) for row_scores in scores], dtype=np.float64)


def _wrong_hidden_contrast_features(hidden: np.ndarray, scores: list[list[float]], *, layer_index: int) -> np.ndarray:
    return top2._hidden_contrast_features(np.roll(hidden, 1, axis=0), scores, layer_index=layer_index)


def _load_reference_predictions(path: pathlib.Path | None, *, rows: list[arc_gate.ArcRow]) -> list[int] | None:
    if path is None:
        return None
    path = top2._resolve(path)
    if not path.exists():
        return None
    predictions: list[int] = []
    row_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            row_ids.append(str(raw["row_id"]))
            predictions.append(int(raw["selected_prediction"]))
    expected = [row.row_id for row in rows]
    if row_ids != expected:
        raise ValueError(f"reference prediction row ids do not match eval rows: {path}")
    return predictions


def _fit_switch_view(
    *,
    view_name: str,
    train_features: np.ndarray,
    train_rows: list[arc_gate.ArcRow],
    train_scores: list[list[float]],
    fit_indices: list[int],
    dev_indices: list[int],
    ridges: tuple[float, ...],
    label_permutation_seed: int | None = None,
) -> dict[str, Any]:
    labels, trainable = top2._switch_labels(train_rows, train_scores)
    labels_for_fit = np.array(labels, copy=True)
    if label_permutation_seed is not None:
        rng = np.random.default_rng(label_permutation_seed)
        fit_trainable_indices = [
            index for index in fit_indices if bool(trainable[index])
        ]
        shuffled = labels_for_fit[np.asarray(fit_trainable_indices, dtype=np.int64)]
        rng.shuffle(shuffled)
        labels_for_fit[np.asarray(fit_trainable_indices, dtype=np.int64)] = shuffled

    fit_mask = np.zeros(len(train_rows), dtype=bool)
    fit_mask[np.asarray(fit_indices, dtype=np.int64)] = True
    dev_rows = top2._take_rows(train_rows, dev_indices)
    dev_scores = top2._take_scores(train_scores, dev_indices)
    dev_index_array = np.asarray(dev_indices, dtype=np.int64)
    selected: dict[str, Any] | None = None
    readouts: list[dict[str, Any]] = []
    for ridge in ridges:
        model = top2._fit_binary_ridge(
            train_features,
            labels_for_fit,
            trainable & fit_mask,
            ridge=ridge,
        )
        train_switch_scores = top2._score_binary_ridge(train_features, model)
        threshold = top2._select_threshold(
            rows=dev_rows,
            scores=dev_scores,
            switch_scores=train_switch_scores[dev_index_array],
        )
        train_predictions = top2._predictions_from_switch_scores(
            train_scores,
            train_switch_scores,
            float(threshold["threshold"]),
        )
        dev_predictions = top2._predictions_from_switch_scores(
            dev_scores,
            train_switch_scores[dev_index_array],
            float(threshold["threshold"]),
        )
        readout = {
            "view": view_name,
            "ridge": float(ridge),
            "threshold": float(threshold["threshold"]),
            "feature_dim": int(train_features.shape[1]),
            "fit_accuracy": top2._accuracy(
                top2._take_rows(train_rows, fit_indices),
                [train_predictions[index] for index in fit_indices],
            ),
            "internal_dev_accuracy": top2._accuracy(dev_rows, dev_predictions),
            "train_accuracy": top2._accuracy(train_rows, train_predictions),
            "train_switch_rate": float(np.mean(train_switch_scores > float(threshold["threshold"]))),
            "internal_dev_switch_rate": float(
                np.mean(train_switch_scores[dev_index_array] > float(threshold["threshold"]))
            ),
            "model_trainable_rows": int(model["trainable_rows"]),
            "model": model,
        }
        readouts.append({key: value for key, value in readout.items() if key != "model"})
        if selected is None or (
            readout["internal_dev_accuracy"],
            readout["fit_accuracy"],
            -readout["ridge"],
        ) > (
            selected["internal_dev_accuracy"],
            selected["fit_accuracy"],
            -selected["ridge"],
        ):
            selected = readout
    if selected is None:
        raise RuntimeError(f"no switch view selected for {view_name}")
    return {
        "view": view_name,
        "selected": selected,
        "candidate_readouts": readouts,
        "label_permutation_seed": label_permutation_seed,
    }


def _score_switch_model(model_row: dict[str, Any], features: np.ndarray, scores: list[list[float]]) -> tuple[list[int], np.ndarray]:
    selected = model_row["selected"]
    switch_scores = top2._score_binary_ridge(features, selected["model"])
    predictions = top2._predictions_from_switch_scores(scores, switch_scores, float(selected["threshold"]))
    return predictions, switch_scores


def _switch_metrics(
    *,
    rows: list[arc_gate.ArcRow],
    scores: list[list[float]],
    predictions: list[int],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("row and prediction counts do not match")
    top1: list[int] = []
    runner_up: list[int] = []
    for row_scores in scores:
        ranked = top2._ranked_indices(row_scores)
        top1.append(ranked[0])
        runner_up.append(ranked[1])
    switch_mask = [pred == second for pred, second in zip(predictions, runner_up, strict=True)]
    top1_gold = [row.answer_index == first for row, first in zip(rows, top1, strict=True)]
    top2_gold = [row.answer_index == second for row, second in zip(rows, runner_up, strict=True)]
    outside_top2_gold = [
        row.answer_index not in {first, second}
        for row, first, second in zip(rows, top1, runner_up, strict=True)
    ]
    correct = [row.answer_index == pred for row, pred in zip(rows, predictions, strict=True)]
    switch_count = int(sum(switch_mask))
    top2_gold_count = int(sum(top2_gold))
    top1_gold_count = int(sum(top1_gold))
    source_correct_count = top1_gold_count
    selected_correct_count = int(sum(correct))
    good_switch_count = int(sum(sw and gold for sw, gold in zip(switch_mask, top2_gold, strict=True)))
    false_switch_count = int(sum(sw and gold for sw, gold in zip(switch_mask, top1_gold, strict=True)))
    harmful_non_top2_switch_count = int(
        sum(sw and gold for sw, gold in zip(switch_mask, outside_top2_gold, strict=True))
    )
    outside_prediction_count = int(
        sum(
            pred not in {first, second}
            for pred, first, second in zip(predictions, top1, runner_up, strict=True)
        )
    )
    denominator = top2_gold_count if top2_gold_count else 0
    missed_switch_count = top2_gold_count - good_switch_count
    net_switch_gain_count = good_switch_count - false_switch_count
    recoverable_headroom = float(top2_gold_count / len(rows))
    net_accuracy_gain = float((selected_correct_count - source_correct_count) / len(rows))
    return {
        "accuracy": float(selected_correct_count / len(rows)),
        "correct": selected_correct_count,
        "rows": len(rows),
        "source_top1_accuracy": float(source_correct_count / len(rows)),
        "source_top2_oracle_accuracy": float((top1_gold_count + top2_gold_count) / len(rows)),
        "top1_gold_count": top1_gold_count,
        "top2_gold_count": top2_gold_count,
        "outside_top2_gold_count": int(sum(outside_top2_gold)),
        "gold_top1_rate": float(top1_gold_count / len(rows)),
        "gold_top2_rate": float(top2_gold_count / len(rows)),
        "gold_outside_top2_rate": float(sum(outside_top2_gold) / len(rows)),
        "top2_opportunity_rate": float(top2_gold_count / len(rows)),
        "outside_top2_gold_rate": float(sum(outside_top2_gold) / len(rows)),
        "recoverable_headroom_vs_source_top1": recoverable_headroom,
        "switch_count": switch_count,
        "switch_rate": float(switch_count / len(rows)),
        "switch_precision": float(good_switch_count / switch_count) if switch_count else None,
        "switch_recall_over_gold_top2": float(good_switch_count / denominator) if denominator else None,
        "false_switch_away_from_gold_top1_rate": (
            float(false_switch_count / top1_gold_count) if top1_gold_count else None
        ),
        "good_switch_count": good_switch_count,
        "false_switch_from_gold_top1_count": false_switch_count,
        "missed_switch_count": missed_switch_count,
        "net_switch_gain_count": net_switch_gain_count,
        "harmful_switch_from_outside_top2_gold_count": harmful_non_top2_switch_count,
        "outside_top2_prediction_count": outside_prediction_count,
        "outside_top2_prediction_rate": float(outside_prediction_count / len(rows)),
        "net_correct_gain_vs_source_top1": selected_correct_count - source_correct_count,
        "net_accuracy_gain_vs_source_top1": net_accuracy_gain,
        "headroom_capture_vs_source_top1": (
            float(net_accuracy_gain / recoverable_headroom) if recoverable_headroom > 0.0 else None
        ),
    }


def _always_switch_predictions(scores: list[list[float]]) -> list[int]:
    return [top2._ranked_indices(row_scores)[1] for row_scores in scores]


def _random_switch_predictions(
    scores: list[list[float]],
    *,
    switch_count: int,
    seed: int,
) -> list[int]:
    n = len(scores)
    switch_count = max(0, min(n, int(switch_count)))
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    switch_indices = set(indices[:switch_count])
    predictions: list[int] = []
    for index, row_scores in enumerate(scores):
        ranked = top2._ranked_indices(row_scores)
        predictions.append(ranked[1] if index in switch_indices else ranked[0])
    return predictions


def _candidate_row(
    *,
    eval_name: str,
    candidate_name: str,
    rows: list[arc_gate.ArcRow],
    scores: list[list[float]],
    predictions: list[int],
    best_label_predictions: list[int],
    score_only_predictions: list[int] | None,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = _switch_metrics(rows=rows, scores=scores, predictions=predictions)
    paired_label = top2._paired_ci_predictions(
        rows,
        predictions,
        best_label_predictions,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    row = {
        "eval_name": eval_name,
        "candidate": candidate_name,
        **metrics,
        "best_label_copy_accuracy": top2._accuracy(rows, best_label_predictions),
        "minus_best_label_copy": metrics["accuracy"] - top2._accuracy(rows, best_label_predictions),
        "paired_ci95_low_vs_best_label_copy": paired_label["ci95_low"],
        "paired_ci95_high_vs_best_label_copy": paired_label["ci95_high"],
    }
    if score_only_predictions is not None:
        paired_score = top2._paired_ci_predictions(
            rows,
            predictions,
            score_only_predictions,
            seed=bootstrap_seed + 100_000,
            samples=bootstrap_samples,
        )
        row.update(
            {
                "score_only_switch_accuracy": top2._accuracy(rows, score_only_predictions),
                "minus_score_only_switch": metrics["accuracy"] - top2._accuracy(rows, score_only_predictions),
                "paired_ci95_low_vs_score_only_switch": paired_score["ci95_low"],
                "paired_ci95_high_vs_score_only_switch": paired_score["ci95_high"],
            }
        )
    return row


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _manifest(output_dir: pathlib.Path, paths: list[pathlib.Path], *, gate: str, headline: dict[str, Any]) -> None:
    payload = {
        "gate": gate,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "headline": headline,
        "files": [
            {
                "path": top2._display_path(path),
                "sha256": top2._sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_decomposition(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = DEFAULT_TRAIN,
    train_score_cache: pathlib.Path = DEFAULT_TRAIN_SCORE_CACHE,
    train_hidden_cache: pathlib.Path = DEFAULT_TRAIN_HIDDEN_CACHE,
    train_hidden_rows: int = DEFAULT_TRAIN_HIDDEN_ROWS,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    hidden_layer_index: int = -1,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    bootstrap_samples: int = 500,
    run_date: str = "2026-05-02",
    eval_specs: tuple[EvalSpec, ...] = DEFAULT_EVAL_SPECS,
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = top2._resolve(train_path)
    train_score_cache = top2._resolve(train_score_cache)
    train_hidden_cache = top2._resolve(train_hidden_cache)
    all_train_rows = arc_gate._load_rows(train_path)
    train_rows = top2._select_train_rows(all_train_rows, count=train_hidden_rows, seed=selection_seed)
    fit_indices, dev_indices = top2._split_indices(
        len(train_rows),
        dev_fraction=dev_fraction,
        seed=selection_seed + 17,
    )
    train_scores, _, train_source_model = headroom._load_score_cache(train_score_cache, rows=train_rows)
    train_hidden, train_hidden_meta = top2._load_hidden_cache(train_hidden_cache, rows=train_rows)
    if hidden_layer_index < 0:
        hidden_layer_index = train_hidden.shape[2] + hidden_layer_index
    if hidden_layer_index < 0 or hidden_layer_index >= train_hidden.shape[2]:
        raise ValueError(f"hidden layer index out of range: {hidden_layer_index}")

    train_score_features = _score_feature_matrix(train_scores)
    train_hidden_contrast = top2._hidden_contrast_features(train_hidden, train_scores, layer_index=hidden_layer_index)
    train_hidden_score = np.concatenate([train_hidden_contrast, train_score_features], axis=1)

    switch_models = {
        "score_switch": _fit_switch_view(
            view_name="score_switch",
            train_features=train_score_features,
            train_rows=train_rows,
            train_scores=train_scores,
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            ridges=ridges,
        ),
        "hidden_contrast_switch": _fit_switch_view(
            view_name="hidden_contrast_switch",
            train_features=train_hidden_contrast,
            train_rows=train_rows,
            train_scores=train_scores,
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            ridges=ridges,
        ),
        "hidden_score_switch": _fit_switch_view(
            view_name="hidden_score_switch",
            train_features=train_hidden_score,
            train_rows=train_rows,
            train_scores=train_scores,
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            ridges=ridges,
        ),
        "label_permuted_hidden_score_switch": _fit_switch_view(
            view_name="label_permuted_hidden_score_switch",
            train_features=train_hidden_score,
            train_rows=train_rows,
            train_scores=train_scores,
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            ridges=ridges,
            label_permutation_seed=selection_seed + 9001,
        ),
    }
    selected_switch_name = max(
        ("score_switch", "hidden_contrast_switch", "hidden_score_switch"),
        key=lambda name: (
            switch_models[name]["selected"]["internal_dev_accuracy"],
            switch_models[name]["selected"]["fit_accuracy"],
            name.startswith("hidden"),
            name == "hidden_score_switch",
            -switch_models[name]["selected"]["ridge"],
        ),
    )

    offsets = score_repair._fit_choice_bias_offsets(
        top2._take_rows(train_rows, fit_indices),
        top2._take_scores(train_scores, fit_indices),
    )
    candidate_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    eval_payloads: dict[str, Any] = {}

    for eval_offset, spec in enumerate(eval_specs):
        eval_path = top2._resolve(spec.eval_path)
        eval_score_cache = top2._resolve(spec.score_cache)
        eval_hidden_cache = top2._resolve(spec.hidden_cache)
        eval_rows = arc_gate._load_rows(eval_path)
        eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
        eval_hidden, eval_hidden_meta = top2._load_hidden_cache(eval_hidden_cache, rows=eval_rows)
        eval_score_features = _score_feature_matrix(eval_scores)
        eval_hidden_contrast = top2._hidden_contrast_features(
            eval_hidden,
            eval_scores,
            layer_index=hidden_layer_index,
        )
        eval_hidden_score = np.concatenate([eval_hidden_contrast, eval_score_features], axis=1)
        eval_wrong_hidden_score = np.concatenate(
            [
                _wrong_hidden_contrast_features(eval_hidden, eval_scores, layer_index=hidden_layer_index),
                eval_score_features,
            ],
            axis=1,
        )
        source_label_predictions = top2._source_label_predictions(eval_scores)
        trained_label_predictions = [
            score_repair._predict_calibrated_label(row_scores, offsets) for row_scores in eval_scores
        ]
        source_label_accuracy = top2._accuracy(eval_rows, source_label_predictions)
        trained_label_accuracy = top2._accuracy(eval_rows, trained_label_predictions)
        best_label_predictions = (
            source_label_predictions if source_label_accuracy >= trained_label_accuracy else trained_label_predictions
        )
        top2_oracle_predictions = _top2_oracle_predictions(eval_rows, eval_scores)
        score_switch_predictions, score_switch_scores = _score_switch_model(
            switch_models["score_switch"],
            eval_score_features,
            eval_scores,
        )
        hidden_switch_predictions, hidden_switch_scores = _score_switch_model(
            switch_models["hidden_contrast_switch"],
            eval_hidden_contrast,
            eval_scores,
        )
        hidden_score_predictions, hidden_score_scores = _score_switch_model(
            switch_models["hidden_score_switch"],
            eval_hidden_score,
            eval_scores,
        )
        wrong_hidden_predictions, wrong_hidden_scores = _score_switch_model(
            switch_models["hidden_score_switch"],
            eval_wrong_hidden_score,
            eval_scores,
        )
        permuted_predictions, permuted_scores = _score_switch_model(
            switch_models["label_permuted_hidden_score_switch"],
            eval_hidden_score,
            eval_scores,
        )
        selected_switch_predictions = {
            "score_switch": score_switch_predictions,
            "hidden_contrast_switch": hidden_switch_predictions,
            "hidden_score_switch": hidden_score_predictions,
        }[selected_switch_name]
        selected_switch_count = _switch_metrics(
            rows=eval_rows,
            scores=eval_scores,
            predictions=selected_switch_predictions,
        )["switch_count"]
        always_switch_predictions = _always_switch_predictions(eval_scores)
        random_switch_predictions = _random_switch_predictions(
            eval_scores,
            switch_count=int(selected_switch_count),
            seed=selection_seed + 50_000 + eval_offset,
        )
        dense_reference = _load_reference_predictions(spec.dense_reference_predictions, rows=eval_rows)
        hybrid_reference = _load_reference_predictions(spec.hybrid_reference_predictions, rows=eval_rows)
        candidate_predictions: list[tuple[str, list[int], list[float] | None]] = [
            ("source_label_copy", source_label_predictions, None),
            ("trained_choice_bias_label_copy", trained_label_predictions, None),
            ("top2_oracle", top2_oracle_predictions, None),
            ("always_switch_to_top2", always_switch_predictions, None),
            ("random_switch_same_rate_as_selected", random_switch_predictions, None),
            ("score_switch", score_switch_predictions, [float(value) for value in score_switch_scores]),
            ("hidden_contrast_switch", hidden_switch_predictions, [float(value) for value in hidden_switch_scores]),
            ("hidden_score_switch", hidden_score_predictions, [float(value) for value in hidden_score_scores]),
            ("wrong_hidden_score_switch_control", wrong_hidden_predictions, [float(value) for value in wrong_hidden_scores]),
            (
                "label_permuted_hidden_score_switch_control",
                permuted_predictions,
                [float(value) for value in permuted_scores],
            ),
            ("selected_train_dev_switch", selected_switch_predictions, None),
        ]
        if dense_reference is not None:
            candidate_predictions.append(("dense_hidden_innovation_reference", dense_reference, None))
        if hybrid_reference is not None:
            candidate_predictions.append(("hybrid_hidden_vote_reference", hybrid_reference, None))

        rows_by_candidate: dict[str, dict[str, Any]] = {}
        for cand_offset, (candidate_name, predictions, switch_values) in enumerate(candidate_predictions):
            row = _candidate_row(
                eval_name=spec.name,
                candidate_name=candidate_name,
                rows=eval_rows,
                scores=eval_scores,
                predictions=predictions,
                best_label_predictions=best_label_predictions,
                score_only_predictions=score_switch_predictions
                if candidate_name not in {"score_switch", "source_label_copy", "trained_choice_bias_label_copy"}
                else None,
                bootstrap_seed=selection_seed + 10_000 * (eval_offset + 1) + cand_offset,
                bootstrap_samples=bootstrap_samples,
            )
            rows_by_candidate[candidate_name] = row
            candidate_rows.append(row)

        for index, row in enumerate(eval_rows):
            ranked = top2._ranked_indices(eval_scores[index])
            prediction_rows.append(
                {
                    "eval_name": spec.name,
                    "row_id": row.row_id,
                    "answer_index": row.answer_index,
                    "source_top1": ranked[0],
                    "source_top2": ranked[1],
                    "source_label_copy": source_label_predictions[index],
                    "trained_choice_bias_label_copy": trained_label_predictions[index],
                    "top2_oracle": top2_oracle_predictions[index],
                    "always_switch_to_top2": always_switch_predictions[index],
                    "random_switch_same_rate_as_selected": random_switch_predictions[index],
                    "score_switch": score_switch_predictions[index],
                    "score_switch_score": float(score_switch_scores[index]),
                    "hidden_contrast_switch": hidden_switch_predictions[index],
                    "hidden_contrast_switch_score": float(hidden_switch_scores[index]),
                    "hidden_score_switch": hidden_score_predictions[index],
                    "hidden_score_switch_score": float(hidden_score_scores[index]),
                    "wrong_hidden_score_switch_control": wrong_hidden_predictions[index],
                    "wrong_hidden_score_switch_score": float(wrong_hidden_scores[index]),
                    "label_permuted_hidden_score_switch_control": permuted_predictions[index],
                    "label_permuted_hidden_score_switch_score": float(permuted_scores[index]),
                    "selected_train_dev_switch": selected_switch_predictions[index],
                    "dense_hidden_innovation_reference": dense_reference[index] if dense_reference is not None else None,
                    "hybrid_hidden_vote_reference": hybrid_reference[index] if hybrid_reference is not None else None,
                }
            )

        hidden_score_row = rows_by_candidate["hidden_score_switch"]
        score_switch_row = rows_by_candidate["score_switch"]
        selected_row = rows_by_candidate["selected_train_dev_switch"]
        wrong_hidden_row = rows_by_candidate["wrong_hidden_score_switch_control"]
        permuted_row = rows_by_candidate["label_permuted_hidden_score_switch_control"]
        random_row = rows_by_candidate["random_switch_same_rate_as_selected"]
        eval_pass = bool(
            selected_switch_name in {"hidden_contrast_switch", "hidden_score_switch"}
            and selected_row["minus_best_label_copy"] >= STRICT_DELTA
            and selected_row["paired_ci95_low_vs_best_label_copy"] > 0.0
            and selected_row.get("minus_score_only_switch", 0.0) >= 0.01
            and selected_row["accuracy"] >= random_row["accuracy"] + 0.02
            and (
                selected_row["headroom_capture_vs_source_top1"] is not None
                and selected_row["headroom_capture_vs_source_top1"] >= 0.20
            )
            and (selected_row["switch_precision"] is not None and selected_row["switch_precision"] >= 0.60)
            and (
                selected_row["switch_recall_over_gold_top2"] is not None
                and selected_row["switch_recall_over_gold_top2"] >= 0.25
            )
            and (
                selected_row["false_switch_away_from_gold_top1_rate"] is not None
                and selected_row["false_switch_away_from_gold_top1_rate"] <= 0.15
            )
            and wrong_hidden_row["accuracy"] <= selected_row["accuracy"] - 0.01
            and permuted_row["accuracy"] <= selected_row["accuracy"] - 0.01
        )
        eval_payloads[spec.name] = {
            "eval_path": top2._display_path(eval_path),
            "eval_sha256": top2._sha256_file(eval_path),
            "eval_score_cache": top2._display_path(eval_score_cache),
            "eval_score_cache_sha256": top2._sha256_file(eval_score_cache),
            "eval_hidden_cache": top2._display_path(eval_hidden_cache),
            "eval_hidden_cache_sha256": top2._sha256_file(eval_hidden_cache),
            "eval_rows": len(eval_rows),
            "source_model": {
                "score_eval": eval_source_model,
                "hidden_eval": eval_hidden_meta,
            },
            "source_label_copy": rows_by_candidate["source_label_copy"],
            "trained_choice_bias_label_copy": rows_by_candidate["trained_choice_bias_label_copy"],
            "top2_oracle": rows_by_candidate["top2_oracle"],
            "always_switch_to_top2": rows_by_candidate["always_switch_to_top2"],
            "random_switch_same_rate_as_selected": random_row,
            "score_switch": score_switch_row,
            "hidden_contrast_switch": rows_by_candidate["hidden_contrast_switch"],
            "hidden_score_switch": hidden_score_row,
            "wrong_hidden_score_switch_control": wrong_hidden_row,
            "label_permuted_hidden_score_switch_control": permuted_row,
            "selected_train_dev_switch": selected_row,
            "dense_hidden_innovation_reference": rows_by_candidate.get("dense_hidden_innovation_reference"),
            "hybrid_hidden_vote_reference": rows_by_candidate.get("hybrid_hidden_vote_reference"),
            "pass_gate": eval_pass,
        }

    all_eval_pass = all(payload["pass_gate"] for payload in eval_payloads.values())
    first = eval_payloads[DEFAULT_EVAL_SPECS[0].name]
    tail = eval_payloads[DEFAULT_EVAL_SPECS[1].name]
    headline = {
        "selected_switch_view": selected_switch_name,
        "selected_switch_internal_dev_accuracy": switch_models[selected_switch_name]["selected"]["internal_dev_accuracy"],
        "selected_switch_ridge": switch_models[selected_switch_name]["selected"]["ridge"],
        "validation_first1024_selected_accuracy": first["selected_train_dev_switch"]["accuracy"],
        "validation_first1024_selected_minus_best_label_copy": first["selected_train_dev_switch"][
            "minus_best_label_copy"
        ],
        "validation_first1024_selected_switch_precision": first["selected_train_dev_switch"]["switch_precision"],
        "validation_first1024_top2_oracle_accuracy": first["top2_oracle"]["accuracy"],
        "terminal_tail_selected_accuracy": tail["selected_train_dev_switch"]["accuracy"],
        "terminal_tail_selected_minus_best_label_copy": tail["selected_train_dev_switch"]["minus_best_label_copy"],
        "terminal_tail_selected_switch_precision": tail["selected_train_dev_switch"]["switch_precision"],
        "terminal_tail_top2_oracle_accuracy": tail["top2_oracle"]["accuracy"],
        "switch_branch_pass_gate": all_eval_pass,
        "strict_delta_required": STRICT_DELTA,
    }
    pass_gate = bool(all_eval_pass)
    payload = {
        "gate": "source_private_hellaswag_switch_decomposition",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": {
            "selected_on_train_dev_only": True,
            "selected_switch_view_must_use_hidden_private_evidence": True,
            "selected_must_beat_best_label_copy_by": STRICT_DELTA,
            "paired_ci95_low_vs_best_label_copy_must_be_positive": True,
            "selected_must_beat_score_switch_by": 0.01,
            "selected_must_beat_random_same_rate_switch_by": 0.02,
            "switch_precision_must_be_at_least": 0.60,
            "gold_top2_recall_must_be_at_least": 0.25,
            "headroom_capture_must_be_at_least": 0.20,
            "false_switch_away_from_gold_top1_must_be_at_most": 0.15,
            "wrong_hidden_and_label_permuted_controls_must_lag_selected_by": 0.01,
            "both_validation_first1024_and_terminal_tail_must_pass": True,
        },
        "train_path": top2._display_path(train_path),
        "train_sha256": top2._sha256_file(train_path),
        "all_train_rows": len(all_train_rows),
        "scored_train_rows": len(train_rows),
        "internal_fit_rows": len(fit_indices),
        "internal_dev_rows": len(dev_indices),
        "train_score_cache": top2._display_path(train_score_cache),
        "train_score_cache_sha256": top2._sha256_file(train_score_cache),
        "train_hidden_cache": top2._display_path(train_hidden_cache),
        "train_hidden_cache_sha256": top2._sha256_file(train_hidden_cache),
        "selection_seed": selection_seed,
        "dev_fraction": dev_fraction,
        "hidden_layer_index": hidden_layer_index,
        "ridges": list(ridges),
        "source_model": {
            "score_train": train_source_model,
            "hidden_train": train_hidden_meta,
        },
        "switch_model_selection": {
            name: {
                key: value
                for key, value in model["selected"].items()
                if key != "model"
            }
            | {"label_permutation_seed": model["label_permutation_seed"]}
            for name, model in switch_models.items()
        },
        "candidate_readouts": [
            readout | {"view_family": name}
            for name, model in switch_models.items()
            for readout in model["candidate_readouts"]
        ],
        "headline": headline,
        "evals": eval_payloads,
        "decision": {
            "promoted": [
                "source top-2 oracle remains useful headroom evidence",
                "candidate-wise hidden-innovation packets remain the stronger HellaSwag method branch",
            ],
            "weakened": [
                "top-2 trust-or-switch does not yet qualify as an independent contribution unless this gate passes",
            ],
            "ruled_out_if_gate_false": [
                "claiming top-2 switch prediction as a main contribution",
                "framing the current switch policy as novel selective classification",
            ],
            "next_gate": (
                "If pass_gate is false, cut trust-or-switch from the contribution list and spend compute on "
                "candidate-wise hidden-innovation stability, cross-family falsification, and native systems rows."
            ),
        },
        "interpretation": (
            "This decomposition isolates whether HellaSwag gains come from a deployable top-2 switch decision. "
            "A switch-only method is publication-worthy only if hidden-private evidence chooses source top-2 "
            "with high precision, beats score-only switching, and survives wrong-hidden and label-permuted controls "
            "on both the passed validation-first1024 slice and the fragile terminal tail."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }

    json_path = output_dir / "hellaswag_switch_decomposition.json"
    md_path = output_dir / "hellaswag_switch_decomposition.md"
    candidate_path = output_dir / "candidate_rows.csv"
    prediction_path = output_dir / "predictions.jsonl"
    readout_path = output_dir / "switch_model_readouts.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HellaSwag Switch Decomposition",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected train-dev switch view: `{headline['selected_switch_view']}`",
        f"- validation[0:1024] selected accuracy: `{headline['validation_first1024_selected_accuracy']:.6f}`",
        f"- validation[0:1024] selected delta vs best label-copy: `{headline['validation_first1024_selected_minus_best_label_copy']:.6f}`",
        f"- validation[0:1024] selected switch precision: `{headline['validation_first1024_selected_switch_precision']}`",
        f"- validation[0:1024] top-2 oracle accuracy: `{headline['validation_first1024_top2_oracle_accuracy']:.6f}`",
        f"- terminal-tail selected accuracy: `{headline['terminal_tail_selected_accuracy']:.6f}`",
        f"- terminal-tail selected delta vs best label-copy: `{headline['terminal_tail_selected_minus_best_label_copy']:.6f}`",
        f"- terminal-tail selected switch precision: `{headline['terminal_tail_selected_switch_precision']}`",
        f"- terminal-tail top-2 oracle accuracy: `{headline['terminal_tail_top2_oracle_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(candidate_path, candidate_rows)
    _write_jsonl(prediction_path, prediction_rows)
    _write_jsonl(readout_path, payload["candidate_readouts"])
    _manifest(
        output_dir,
        [json_path, md_path, candidate_path, prediction_path, readout_path],
        gate=payload["gate"],
        headline=headline,
    )
    return payload


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HellaSwag top-2 switch decomposition gate.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--train-score-cache", type=pathlib.Path, default=DEFAULT_TRAIN_SCORE_CACHE)
    parser.add_argument("--train-hidden-cache", type=pathlib.Path, default=DEFAULT_TRAIN_HIDDEN_CACHE)
    parser.add_argument("--train-hidden-rows", type=int, default=DEFAULT_TRAIN_HIDDEN_ROWS)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--hidden-layer-index", type=int, default=-1)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default="2026-05-02")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_decomposition(
        output_dir=args.output_dir,
        train_path=args.train_path,
        train_score_cache=args.train_score_cache,
        train_hidden_cache=args.train_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        selection_seed=args.selection_seed,
        dev_fraction=args.dev_fraction,
        hidden_layer_index=args.hidden_layer_index,
        ridges=args.ridges,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
