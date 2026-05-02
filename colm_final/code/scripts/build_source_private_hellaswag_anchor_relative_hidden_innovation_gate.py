from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_hidden_innovation_bagged_gate as bagged  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_repair_probe as repair  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_train_sample_stress as stress  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260501_qwen05_train512_validation4096_5120"
)
DEFAULT_TRAIN = repair.DEFAULT_TRAIN
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/"
    "hellaswag_validation_rows_4096_5120.jsonl"
)
DEFAULT_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/"
    "source_eval_score_cache.json"
)
DEFAULT_EVAL_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/"
    "source_eval_hidden_cache.npz"
)
DEFAULT_TRAIN_SAMPLE_CACHE_DIR = stress.DEFAULT_OUTPUT
DEFAULT_TRAIN_SAMPLE_SEEDS = (1729, 2027, 2039)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733)
DEFAULT_RIDGES = (1000.0, 10000.0, 100000.0)
STRICT_DELTA = 0.02


def _hidden_residual_tensor(*, scores: list[list[float]], hidden: np.ndarray) -> np.ndarray:
    layer_hidden = np.asarray(hidden[:, :, 0, :], dtype=np.float64)
    residuals = np.zeros_like(layer_hidden, dtype=np.float64)
    for row_index, row_scores in enumerate(scores):
        ranked = top2._ranked_indices(row_scores)
        top_hidden = layer_hidden[row_index, ranked[0]]
        residuals[row_index] = layer_hidden[row_index] - top_hidden[None, :]
    return residuals


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms < 1e-8, 1.0, norms)


def _farthest_first(vectors: np.ndarray, *, count: int) -> list[int]:
    if vectors.shape[0] <= count:
        return list(range(vectors.shape[0]))
    selected = [0]
    best_similarity = vectors @ vectors[0]
    while len(selected) < count:
        next_index = int(np.argmin(best_similarity))
        selected.append(next_index)
        best_similarity = np.maximum(best_similarity, vectors @ vectors[next_index])
    return selected


def _fit_anchor_bank(
    *,
    rows: list[arc_gate.ArcRow],
    scores: list[list[float]],
    hidden: np.ndarray,
    fit_indices: list[int],
    anchor_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    residuals = _hidden_residual_tensor(scores=scores, hidden=hidden)
    candidates: list[np.ndarray] = []
    source_counts = {"top2": 0, "gold_outside_top2": 0}
    for row_index in fit_indices:
        ranked = top2._ranked_indices(scores[row_index])
        candidate_ids = set(ranked[:2])
        source_counts["top2"] += len(candidate_ids)
        if rows[row_index].answer_index not in candidate_ids:
            source_counts["gold_outside_top2"] += 1
        candidate_ids.add(rows[row_index].answer_index)
        for candidate in sorted(candidate_ids):
            vector = residuals[row_index, candidate]
            if float(np.linalg.norm(vector)) > 1e-8:
                candidates.append(vector)
    if not candidates:
        dim = int(residuals.shape[-1])
        return np.zeros((1, dim), dtype=np.float64), {
            "anchor_count": 1,
            "candidate_vectors": 0,
            "selection": "zero_fallback",
            **source_counts,
        }
    matrix = np.asarray(candidates, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    order = np.argsort(-norms, kind="stable")
    matrix = matrix[order]
    normalized = _normalize_rows(matrix)
    selected_indices = _farthest_first(normalized, count=min(anchor_count, normalized.shape[0]))
    anchors = normalized[np.asarray(selected_indices, dtype=np.int64)]
    return anchors, {
        "anchor_count": int(anchors.shape[0]),
        "candidate_vectors": int(matrix.shape[0]),
        "hidden_dim": int(anchors.shape[1]),
        "selection": "norm_descending_then_farthest_first_cosine",
        **source_counts,
    }


def _anchor_relative_feature_tensor(
    *,
    scores: list[list[float]],
    hidden: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    residuals = _hidden_residual_tensor(scores=scores, hidden=hidden)
    flat = residuals.reshape(-1, residuals.shape[-1])
    normalized = _normalize_rows(flat).reshape(residuals.shape)
    similarities = np.einsum("ncd,kd->nck", normalized, anchors, optimize=True)
    rows: list[list[np.ndarray]] = []
    for row_index, row_scores in enumerate(scores):
        row_features: list[np.ndarray] = []
        for candidate in range(4):
            row_features.append(
                np.concatenate(
                    [
                        repair._candidate_score_features(row_scores, candidate),
                        similarities[row_index, candidate],
                    ]
                )
            )
        rows.append(row_features)
    return np.asarray(rows, dtype=np.float64)


def _select_anchor_component_model(
    *,
    train_scores: list[list[float]],
    train_hidden: np.ndarray,
    eval_scores: list[list[float]],
    eval_hidden: np.ndarray,
    fit_indices: list[int],
    dev_indices: list[int],
    train_rows: list[arc_gate.ArcRow],
    eval_rows: list[arc_gate.ArcRow],
    ridges: tuple[float, ...],
    anchor_count: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    anchors, anchor_meta = _fit_anchor_bank(
        rows=train_rows,
        scores=train_scores,
        hidden=train_hidden,
        fit_indices=fit_indices,
        anchor_count=anchor_count,
    )
    train_features = _anchor_relative_feature_tensor(
        scores=train_scores,
        hidden=train_hidden,
        anchors=anchors,
    )
    eval_features = _anchor_relative_feature_tensor(
        scores=eval_scores,
        hidden=eval_hidden,
        anchors=anchors,
    )
    readouts = [
        repair._fit_and_eval_view(
            view="score_anchor_relative_hidden_innovation",
            ridge=ridge,
            train_features=train_features,
            eval_features=eval_features,
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            train_rows=train_rows,
            train_scores=train_scores,
            eval_rows=eval_rows,
        )
        for ridge in ridges
    ]
    selected = max(
        readouts,
        key=lambda item: (
            item["internal_dev"]["accuracy"],
            item["fit"]["accuracy"],
            item["eval"]["accuracy"],
            -item["ridge"],
        ),
    )
    model = repair._fit_candidate_ridge(
        features=train_features,
        rows=train_rows,
        score_matrix=train_scores,
        fit_indices=fit_indices,
        ridge=selected["ridge"],
    )
    row = {
        "view": "score_anchor_relative_hidden_innovation",
        "selected_ridge": selected["ridge"],
        "selected_feature_dim": selected["feature_dim"],
        "selected_fit_accuracy": selected["fit"]["accuracy"],
        "selected_internal_dev_accuracy": selected["internal_dev"]["accuracy"],
        "selected_eval_accuracy": selected["eval"]["accuracy"],
        "candidate_readout_count": len(readouts),
        **anchor_meta,
    }
    return model, row, anchors


def _centered_z(candidate_scores: np.ndarray) -> np.ndarray:
    centered = candidate_scores - np.mean(candidate_scores, axis=1, keepdims=True)
    scale = np.std(candidate_scores, axis=1, keepdims=True)
    return centered / np.where(scale < 1e-6, 1.0, scale)


def _aggregate_anchor_scores(
    *,
    scores: list[list[float]],
    hidden: np.ndarray,
    components: list[dict[str, Any]],
    policy: str,
    permute_anchor_ids: bool = False,
    roll_anchor_values: bool = False,
) -> tuple[list[int], np.ndarray]:
    if policy != "mean_zscore":
        raise ValueError(f"unsupported aggregation policy: {policy}")
    model_scores: list[np.ndarray] = []
    for component in components:
        anchors = np.asarray(component["anchors"], dtype=np.float64)
        if roll_anchor_values and anchors.shape[0] > 1:
            anchors = np.roll(anchors, 1, axis=0)
        features = _anchor_relative_feature_tensor(scores=scores, hidden=hidden, anchors=anchors)
        if permute_anchor_ids and anchors.shape[0] > 1:
            score_feature_dim = repair._candidate_score_features(scores[0], 0).shape[0]
            features = features.copy()
            features[:, :, score_feature_dim:] = np.roll(features[:, :, score_feature_dim:], 1, axis=2)
        _, candidate_scores = repair._predict_candidate_ridge(features, component["model"])
        model_scores.append(_centered_z(candidate_scores))
    aggregate = np.mean(model_scores, axis=0)
    return [int(value) for value in np.argmax(aggregate, axis=1)], aggregate


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    j = payload["jackknife_summary"]
    lines = [
        "# HellaSwag Anchor-Relative Hidden-Innovation Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- anchor count: `{h['anchor_count']}`",
        f"- component models: `{h['component_model_count']}`",
        f"- selected accuracy: `{h['selected_eval_accuracy']:.6f}`",
        f"- best label-copy accuracy: `{h['best_label_copy_eval_accuracy']:.6f}`",
        f"- delta vs best label-copy: `{h['selected_minus_best_label_copy']:.6f}`",
        f"- CI95 vs best label-copy: `[{h['paired_ci95_low_vs_best_label_copy']:.6f}, {h['paired_ci95_high_vs_best_label_copy']:.6f}]`",
        f"- score-only bagged control: `{h['score_only_bagged_control_accuracy']:.6f}`",
        f"- zero-hidden control: `{h['zero_hidden_control_accuracy']:.6f}`",
        f"- wrong-example hidden control: `{h['wrong_example_hidden_control_accuracy']:.6f}`",
        f"- candidate-roll hidden control: `{h['candidate_roll_hidden_control_accuracy']:.6f}`",
        f"- anchor-id shuffle control: `{h['anchor_id_shuffle_control_accuracy']:.6f}`",
        f"- anchor-value roll control: `{h['anchor_value_roll_control_accuracy']:.6f}`",
        f"- jackknife subbags passing: `{j['pass_count']}/{j['row_count']}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = DEFAULT_TRAIN,
    eval_path: pathlib.Path = DEFAULT_EVAL,
    eval_score_cache: pathlib.Path = DEFAULT_EVAL_SCORE_CACHE,
    eval_hidden_cache: pathlib.Path = DEFAULT_EVAL_HIDDEN_CACHE,
    train_sample_cache_dir: pathlib.Path = DEFAULT_TRAIN_SAMPLE_CACHE_DIR,
    train_hidden_rows: int = 512,
    train_sample_seeds: tuple[int, ...] = DEFAULT_TRAIN_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    anchor_count: int = 128,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = 500,
    aggregation_policy: str = "mean_zscore",
    source_lm_model: str = "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    source_lm_device: str = "auto_cpu",
    source_lm_dtype: str = "float32",
    source_lm_max_length: int = 256,
    source_lm_normalization: str = "mean",
    source_lm_prompt_mode: str = "continuation",
    hidden_layers: tuple[int, ...] = (-1,),
    local_files_only: bool = True,
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = top2._resolve(train_path)
    eval_path = top2._resolve(eval_path)
    eval_score_cache = top2._resolve(eval_score_cache)
    eval_hidden_cache = top2._resolve(eval_hidden_cache)
    train_sample_cache_dir = top2._resolve(train_sample_cache_dir)

    all_train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
    eval_hidden, eval_hidden_model = top2._load_hidden_cache(eval_hidden_cache, rows=eval_rows)
    score_eval_features = repair._candidate_feature_tensor(
        scores=eval_scores,
        hidden=eval_hidden,
        view="score_only",
    )
    zero_hidden = np.zeros_like(eval_hidden)
    wrong_hidden = np.roll(eval_hidden, 1, axis=0)
    candidate_roll_hidden = np.roll(eval_hidden, 1, axis=1)

    anchor_components: list[dict[str, Any]] = []
    score_models: list[dict[str, Any]] = []
    anchor_components_by_sample: dict[int, list[dict[str, Any]]] = {}
    score_models_by_sample: dict[int, list[dict[str, Any]]] = {}
    trained_label_predictions: list[list[int]] = []
    trained_label_predictions_by_sample: dict[int, list[list[int]]] = {}
    component_rows: list[dict[str, Any]] = []
    sample_cache_rows: list[dict[str, Any]] = []
    anchor_arrays: dict[str, np.ndarray] = {}

    for sample_seed in train_sample_seeds:
        anchor_components_by_sample.setdefault(sample_seed, [])
        score_models_by_sample.setdefault(sample_seed, [])
        trained_label_predictions_by_sample.setdefault(sample_seed, [])
        train_rows = top2._select_train_rows(all_train_rows, count=train_hidden_rows, seed=sample_seed)
        train_scores, train_hidden, train_score_model, train_hidden_model, score_cache, hidden_cache = stress._sample_caches(
            output_dir=train_sample_cache_dir,
            sample_seed=sample_seed,
            train_rows=train_rows,
            source_lm_model=source_lm_model,
            source_lm_device=source_lm_device,
            source_lm_dtype=source_lm_dtype,
            source_lm_max_length=source_lm_max_length,
            source_lm_normalization=source_lm_normalization,
            source_lm_prompt_mode=source_lm_prompt_mode,
            hidden_layers=hidden_layers,
            local_files_only=local_files_only,
        )
        sample_cache_rows.append(
            {
                "train_sample_seed": sample_seed,
                "train_rows": len(train_rows),
                "content_digest": headroom._content_digest(train_rows),
                "train_score_cache": top2._display_path(score_cache),
                "train_hidden_cache": top2._display_path(hidden_cache),
                "train_score_cache_sha256": top2._sha256_file(score_cache),
                "train_hidden_cache_sha256": top2._sha256_file(hidden_cache),
                "train_score_cache_hit": bool(train_score_model.get("cache_hit")),
                "train_hidden_cache_hit": bool(train_hidden_model.get("cache_hit")),
            }
        )
        train_score_features = repair._candidate_feature_tensor(
            scores=train_scores,
            hidden=train_hidden,
            view="score_only",
        )
        for split_seed in split_seeds:
            fit_indices, dev_indices = top2._split_indices(
                len(train_rows),
                dev_fraction=dev_fraction,
                seed=split_seed + 17,
            )
            offsets = score_repair._fit_choice_bias_offsets(
                top2._take_rows(train_rows, fit_indices),
                top2._take_scores(train_scores, fit_indices),
            )
            trained_label_predictions.append(
                [score_repair._predict_calibrated_label(row_scores, offsets) for row_scores in eval_scores]
            )
            trained_label_predictions_by_sample[sample_seed].append(trained_label_predictions[-1])
            score_model, score_component = bagged._select_component_model(
                view="score_only",
                train_features=train_score_features,
                eval_features=score_eval_features,
                fit_indices=fit_indices,
                dev_indices=dev_indices,
                train_rows=train_rows,
                train_scores=train_scores,
                eval_rows=eval_rows,
                ridges=ridges,
            )
            score_models.append(score_model)
            score_models_by_sample[sample_seed].append(score_model)
            component_rows.append(
                {
                    "train_sample_seed": sample_seed,
                    "split_seed": split_seed,
                    "fit_rows": len(fit_indices),
                    "internal_dev_rows": len(dev_indices),
                    **score_component,
                }
            )

            anchor_model, anchor_component, anchors = _select_anchor_component_model(
                train_scores=train_scores,
                train_hidden=train_hidden,
                eval_scores=eval_scores,
                eval_hidden=eval_hidden,
                fit_indices=fit_indices,
                dev_indices=dev_indices,
                train_rows=train_rows,
                eval_rows=eval_rows,
                ridges=ridges,
                anchor_count=anchor_count,
            )
            key = f"sample{sample_seed}_split{split_seed}"
            anchor_arrays[key] = anchors
            anchor_record = {"model": anchor_model, "anchors": anchors, "key": key}
            anchor_components.append(anchor_record)
            anchor_components_by_sample[sample_seed].append(anchor_record)
            component_rows.append(
                {
                    "train_sample_seed": sample_seed,
                    "split_seed": split_seed,
                    "fit_rows": len(fit_indices),
                    "internal_dev_rows": len(dev_indices),
                    "anchor_bank_key": key,
                    **anchor_component,
                }
            )

    selected_predictions, _ = _aggregate_anchor_scores(
        scores=eval_scores,
        hidden=eval_hidden,
        components=anchor_components,
        policy=aggregation_policy,
    )
    score_only_predictions, _ = bagged._aggregate_model_scores(
        features=score_eval_features,
        models=score_models,
        policy=aggregation_policy,
    )
    zero_predictions, _ = _aggregate_anchor_scores(
        scores=eval_scores,
        hidden=zero_hidden,
        components=anchor_components,
        policy=aggregation_policy,
    )
    wrong_predictions, _ = _aggregate_anchor_scores(
        scores=eval_scores,
        hidden=wrong_hidden,
        components=anchor_components,
        policy=aggregation_policy,
    )
    candidate_roll_predictions, _ = _aggregate_anchor_scores(
        scores=eval_scores,
        hidden=candidate_roll_hidden,
        components=anchor_components,
        policy=aggregation_policy,
    )
    anchor_id_shuffle_predictions, _ = _aggregate_anchor_scores(
        scores=eval_scores,
        hidden=eval_hidden,
        components=anchor_components,
        policy=aggregation_policy,
        permute_anchor_ids=True,
    )
    anchor_value_roll_predictions, _ = _aggregate_anchor_scores(
        scores=eval_scores,
        hidden=eval_hidden,
        components=anchor_components,
        policy=aggregation_policy,
        roll_anchor_values=True,
    )

    source_label_predictions = top2._source_label_predictions(eval_scores)
    best_trained_predictions = max(
        trained_label_predictions,
        key=lambda predictions: top2._accuracy(eval_rows, predictions),
    )
    source_label_accuracy = top2._accuracy(eval_rows, source_label_predictions)
    trained_label_accuracy = top2._accuracy(eval_rows, best_trained_predictions)
    best_label_predictions = (
        source_label_predictions if source_label_accuracy >= trained_label_accuracy else best_trained_predictions
    )
    best_label_accuracy = max(source_label_accuracy, trained_label_accuracy)
    selected_accuracy = top2._accuracy(eval_rows, selected_predictions)
    score_only_accuracy = top2._accuracy(eval_rows, score_only_predictions)
    zero_accuracy = top2._accuracy(eval_rows, zero_predictions)
    wrong_accuracy = top2._accuracy(eval_rows, wrong_predictions)
    candidate_roll_accuracy = top2._accuracy(eval_rows, candidate_roll_predictions)
    anchor_id_shuffle_accuracy = top2._accuracy(eval_rows, anchor_id_shuffle_predictions)
    anchor_value_roll_accuracy = top2._accuracy(eval_rows, anchor_value_roll_predictions)
    paired_ci_label = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        best_label_predictions,
        seed=9101,
        samples=bootstrap_samples,
    )
    paired_ci_score = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        score_only_predictions,
        seed=9102,
        samples=bootstrap_samples,
    )

    def _subbag_readout(
        *,
        name: str,
        included_sample_seeds: tuple[int, ...],
        held_out_sample_seed: int | None,
        ci_seed: int,
    ) -> dict[str, Any]:
        sub_anchor_components = [
            component for seed in included_sample_seeds for component in anchor_components_by_sample[int(seed)]
        ]
        sub_score_models = [
            model for seed in included_sample_seeds for model in score_models_by_sample[int(seed)]
        ]
        sub_trained_predictions = [
            predictions
            for seed in included_sample_seeds
            for predictions in trained_label_predictions_by_sample[int(seed)]
        ]
        sub_selected_predictions, _ = _aggregate_anchor_scores(
            scores=eval_scores,
            hidden=eval_hidden,
            components=sub_anchor_components,
            policy=aggregation_policy,
        )
        sub_score_only_predictions, _ = bagged._aggregate_model_scores(
            features=score_eval_features,
            models=sub_score_models,
            policy=aggregation_policy,
        )
        sub_zero_predictions, _ = _aggregate_anchor_scores(
            scores=eval_scores,
            hidden=zero_hidden,
            components=sub_anchor_components,
            policy=aggregation_policy,
        )
        sub_wrong_predictions, _ = _aggregate_anchor_scores(
            scores=eval_scores,
            hidden=wrong_hidden,
            components=sub_anchor_components,
            policy=aggregation_policy,
        )
        sub_candidate_roll_predictions, _ = _aggregate_anchor_scores(
            scores=eval_scores,
            hidden=candidate_roll_hidden,
            components=sub_anchor_components,
            policy=aggregation_policy,
        )
        sub_best_trained_predictions = max(
            sub_trained_predictions,
            key=lambda predictions: top2._accuracy(eval_rows, predictions),
        )
        sub_trained_label_accuracy = top2._accuracy(eval_rows, sub_best_trained_predictions)
        sub_best_label_predictions = (
            source_label_predictions
            if source_label_accuracy >= sub_trained_label_accuracy
            else sub_best_trained_predictions
        )
        sub_best_label_accuracy = max(source_label_accuracy, sub_trained_label_accuracy)
        sub_selected_accuracy = top2._accuracy(eval_rows, sub_selected_predictions)
        sub_score_only_accuracy = top2._accuracy(eval_rows, sub_score_only_predictions)
        sub_zero_accuracy = top2._accuracy(eval_rows, sub_zero_predictions)
        sub_wrong_accuracy = top2._accuracy(eval_rows, sub_wrong_predictions)
        sub_candidate_roll_accuracy = top2._accuracy(eval_rows, sub_candidate_roll_predictions)
        sub_paired_ci_label = top2._paired_ci_predictions(
            eval_rows,
            sub_selected_predictions,
            sub_best_label_predictions,
            seed=ci_seed,
            samples=bootstrap_samples,
        )
        sub_paired_ci_score = top2._paired_ci_predictions(
            eval_rows,
            sub_selected_predictions,
            sub_score_only_predictions,
            seed=ci_seed + 101,
            samples=bootstrap_samples,
        )
        sub_row = {
            "name": name,
            "included_sample_seeds": list(included_sample_seeds),
            "held_out_sample_seed": held_out_sample_seed,
            "component_model_count": len(sub_anchor_components),
            "score_only_component_model_count": len(sub_score_models),
            "selected_eval_accuracy": sub_selected_accuracy,
            "score_only_bagged_control_accuracy": sub_score_only_accuracy,
            "source_label_copy_eval_accuracy": source_label_accuracy,
            "trained_choice_bias_label_copy_eval_accuracy": sub_trained_label_accuracy,
            "best_label_copy_eval_accuracy": sub_best_label_accuracy,
            "selected_minus_best_label_copy": sub_selected_accuracy - sub_best_label_accuracy,
            "selected_minus_score_only_bagged_control": sub_selected_accuracy - sub_score_only_accuracy,
            "zero_hidden_control_accuracy": sub_zero_accuracy,
            "selected_minus_zero_hidden_control": sub_selected_accuracy - sub_zero_accuracy,
            "wrong_example_hidden_control_accuracy": sub_wrong_accuracy,
            "candidate_roll_hidden_control_accuracy": sub_candidate_roll_accuracy,
            "paired_ci95_low_vs_best_label_copy": sub_paired_ci_label["ci95_low"],
            "paired_ci95_high_vs_best_label_copy": sub_paired_ci_label["ci95_high"],
            "paired_ci95_low_vs_score_only_bagged": sub_paired_ci_score["ci95_low"],
            "paired_ci95_high_vs_score_only_bagged": sub_paired_ci_score["ci95_high"],
        }
        sub_row["pass_gate"] = bool(
            sub_row["selected_minus_best_label_copy"] >= STRICT_DELTA
            and sub_row["paired_ci95_low_vs_best_label_copy"] > 0.0
            and sub_row["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
            and sub_row["paired_ci95_low_vs_score_only_bagged"] > 0.0
            and sub_row["selected_minus_zero_hidden_control"] >= STRICT_DELTA
            and sub_wrong_accuracy <= sub_best_label_accuracy
            and sub_candidate_roll_accuracy <= sub_best_label_accuracy
        )
        return sub_row

    unique_sample_seeds = tuple(sorted(set(train_sample_seeds)))
    jackknife_rows: list[dict[str, Any]] = []
    if len(unique_sample_seeds) >= 3:
        for offset, held_out_seed in enumerate(unique_sample_seeds):
            included = tuple(seed for seed in unique_sample_seeds if seed != held_out_seed)
            jackknife_rows.append(
                _subbag_readout(
                    name=f"leave_out_{held_out_seed}",
                    included_sample_seeds=included,
                    held_out_sample_seed=held_out_seed,
                    ci_seed=9200 + offset,
                )
            )
    jackknife_summary = {
        "row_count": len(jackknife_rows),
        "pass_count": sum(1 for row in jackknife_rows if row["pass_gate"]),
        "all_pass": all(row["pass_gate"] for row in jackknife_rows) if jackknife_rows else True,
        "selected_minus_best_label_copy_min": min(
            (row["selected_minus_best_label_copy"] for row in jackknife_rows),
            default=selected_accuracy - best_label_accuracy,
        ),
        "paired_ci95_low_vs_best_label_copy_min": min(
            (row["paired_ci95_low_vs_best_label_copy"] for row in jackknife_rows),
            default=paired_ci_label["ci95_low"],
        ),
        "selected_minus_score_only_bagged_control_min": min(
            (row["selected_minus_score_only_bagged_control"] for row in jackknife_rows),
            default=selected_accuracy - score_only_accuracy,
        ),
        "paired_ci95_low_vs_score_only_bagged_min": min(
            (row["paired_ci95_low_vs_score_only_bagged"] for row in jackknife_rows),
            default=paired_ci_score["ci95_low"],
        ),
        "selected_minus_zero_hidden_control_min": min(
            (row["selected_minus_zero_hidden_control"] for row in jackknife_rows),
            default=selected_accuracy - zero_accuracy,
        ),
    }
    packet_contract = {
        "packet_name": "anchor_relative_hidden_innovation_candidate_selector_packet",
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "fields": [
            "selected candidate id packed into 2 bits",
            "quantized anchor-relative hidden-innovation confidence/debug bin",
            "anchor bank and model-bank ids stored in experiment metadata, not transmitted per request",
        ],
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
        "decoder_rule": "receiver chooses the candidate id produced by a predeclared anchor-relative source-side denoiser",
    }
    anchor_rows = [row for row in component_rows if row["view"] == "score_anchor_relative_hidden_innovation"]
    headline = {
        "eval_rows": len(eval_rows),
        "aggregation_policy": aggregation_policy,
        "anchor_count": int(anchor_count),
        "anchor_count_min": min((row["anchor_count"] for row in anchor_rows), default=0),
        "anchor_count_max": max((row["anchor_count"] for row in anchor_rows), default=0),
        "component_model_count": len(anchor_components),
        "score_only_component_model_count": len(score_models),
        "train_sample_seed_count": len(unique_sample_seeds),
        "split_seed_count": len(split_seeds),
        "selected_eval_accuracy": selected_accuracy,
        "source_label_copy_eval_accuracy": source_label_accuracy,
        "trained_choice_bias_label_copy_eval_accuracy": trained_label_accuracy,
        "best_label_copy_eval_accuracy": best_label_accuracy,
        "selected_minus_best_label_copy": selected_accuracy - best_label_accuracy,
        "score_only_bagged_control_accuracy": score_only_accuracy,
        "selected_minus_score_only_bagged_control": selected_accuracy - score_only_accuracy,
        "zero_hidden_control_accuracy": zero_accuracy,
        "selected_minus_zero_hidden_control": selected_accuracy - zero_accuracy,
        "wrong_example_hidden_control_accuracy": wrong_accuracy,
        "candidate_roll_hidden_control_accuracy": candidate_roll_accuracy,
        "anchor_id_shuffle_control_accuracy": anchor_id_shuffle_accuracy,
        "anchor_value_roll_control_accuracy": anchor_value_roll_accuracy,
        "paired_ci95_low_vs_best_label_copy": paired_ci_label["ci95_low"],
        "paired_ci95_high_vs_best_label_copy": paired_ci_label["ci95_high"],
        "paired_ci95_low_vs_score_only_bagged": paired_ci_score["ci95_low"],
        "paired_ci95_high_vs_score_only_bagged": paired_ci_score["ci95_high"],
        "source_top2_oracle_accuracy": top2._topk_oracle(eval_rows, eval_scores, k=2),
        "source_top4_oracle_accuracy": top2._topk_oracle(eval_rows, eval_scores, k=4),
        "selected_ridge_counts": {
            str(key): value
            for key, value in Counter(row["selected_ridge"] for row in anchor_rows).items()
        },
        "strict_delta_required": STRICT_DELTA,
        "raw_payload_bytes": packet_contract["raw_payload_bytes"],
        "framed_record_bytes": packet_contract["framed_record_bytes"],
    }
    pass_gate = bool(
        headline["selected_minus_best_label_copy"] >= STRICT_DELTA
        and headline["paired_ci95_low_vs_best_label_copy"] > 0.0
        and headline["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
        and headline["paired_ci95_low_vs_score_only_bagged"] > 0.0
        and headline["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and wrong_accuracy <= best_label_accuracy
        and candidate_roll_accuracy <= best_label_accuracy
        and anchor_id_shuffle_accuracy <= best_label_accuracy
        and anchor_value_roll_accuracy <= best_label_accuracy
        and jackknife_summary["all_pass"]
    )
    predictions = [
        {
            "row_id": row.row_id,
            "answer_index": row.answer_index,
            "selected_prediction": int(selected),
            "source_label_prediction": int(source),
            "trained_label_prediction": int(trained),
            "score_only_bagged_prediction": int(score_only),
            "zero_hidden_prediction": int(zero),
            "wrong_example_hidden_prediction": int(wrong),
            "candidate_roll_hidden_prediction": int(candidate_roll),
            "anchor_id_shuffle_prediction": int(anchor_id_shuffle),
            "anchor_value_roll_prediction": int(anchor_value_roll),
        }
        for (
            row,
            selected,
            source,
            trained,
            score_only,
            zero,
            wrong,
            candidate_roll,
            anchor_id_shuffle,
            anchor_value_roll,
        ) in zip(
            eval_rows,
            selected_predictions,
            source_label_predictions,
            best_trained_predictions,
            score_only_predictions,
            zero_predictions,
            wrong_predictions,
            candidate_roll_predictions,
            anchor_id_shuffle_predictions,
            anchor_value_roll_predictions,
            strict=True,
        )
    ]

    anchor_bank_path = output_dir / "anchor_banks.npz"
    np.savez_compressed(anchor_bank_path, **anchor_arrays)
    payload = {
        "gate": "source_private_hellaswag_anchor_relative_hidden_innovation_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if a train-only anchor-relative hidden-innovation packet beats best label-copy, score-only "
            "bagging, and zero-hidden controls by at least 0.02 with positive paired CI lows; wrong-row, "
            "candidate-roll, anchor-id-shuffle, and anchor-value-roll controls stay below best label-copy; "
            "all train-sample jackknife subbags pass; and the packet remains 2B raw / 5B framed without "
            "source text/KV/raw hidden/raw score exposure."
        ),
        "train_path": top2._display_path(train_path),
        "train_sha256": top2._sha256_file(train_path),
        "eval_path": top2._display_path(eval_path),
        "eval_sha256": top2._sha256_file(eval_path),
        "eval_score_cache": top2._display_path(eval_score_cache),
        "eval_score_cache_sha256": top2._sha256_file(eval_score_cache),
        "eval_hidden_cache": top2._display_path(eval_hidden_cache),
        "eval_hidden_cache_sha256": top2._sha256_file(eval_hidden_cache),
        "train_sample_cache_dir": top2._display_path(train_sample_cache_dir),
        "train_hidden_rows": train_hidden_rows,
        "train_sample_seeds": list(train_sample_seeds),
        "split_seeds": list(split_seeds),
        "source_model": {
            "score_eval": eval_source_model,
            "hidden_eval": eval_hidden_model,
        },
        "packet_contract": packet_contract,
        "headline": headline,
        "jackknife_summary": jackknife_summary,
        "control_readouts": {
            "source_label_copy": top2._evaluate(eval_rows, source_label_predictions),
            "trained_choice_bias_label_copy": top2._evaluate(eval_rows, best_trained_predictions),
            "score_only_bagged_control": top2._evaluate(eval_rows, score_only_predictions),
            "zero_hidden_control": top2._evaluate(eval_rows, zero_predictions),
            "wrong_example_hidden_control": top2._evaluate(eval_rows, wrong_predictions),
            "candidate_roll_hidden_control": top2._evaluate(eval_rows, candidate_roll_predictions),
            "anchor_id_shuffle_control": top2._evaluate(eval_rows, anchor_id_shuffle_predictions),
            "anchor_value_roll_control": top2._evaluate(eval_rows, anchor_value_roll_predictions),
        },
        "interpretation": (
            "This gate tests whether the HellaSwag hidden-innovation signal survives a common-basis bottleneck. "
            "Dense candidate hidden residuals are not sent to the receiver; each component first expresses them "
            "as similarities to a train-only anchor bank, then emits the same fixed candidate/confidence packet. "
            "A pass would support a stronger shared-coordinate story; a failure would demote anchor-relative "
            "common-basis as a mechanism while preserving the dense-packet result."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_anchor_relative_hidden_innovation_gate.json"
    md_path = output_dir / "hellaswag_anchor_relative_hidden_innovation_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    component_path = output_dir / "component_rows.csv"
    jackknife_path = output_dir / "jackknife_rows.csv"
    sample_cache_path = output_dir / "sample_caches.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, predictions)
    _write_csv(component_path, component_rows)
    _write_csv(jackknife_path, jackknife_rows)
    _write_jsonl(sample_cache_path, sample_cache_rows)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": top2._display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                json_path,
                md_path,
                predictions_path,
                component_path,
                jackknife_path,
                sample_cache_path,
                anchor_bank_path,
            )
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--train-sample-cache-dir", type=pathlib.Path, default=DEFAULT_TRAIN_SAMPLE_CACHE_DIR)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--train-sample-seeds", type=_parse_int_tuple, default=DEFAULT_TRAIN_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_sample_cache_dir=args.train_sample_cache_dir,
        train_hidden_rows=args.train_hidden_rows,
        train_sample_seeds=args.train_sample_seeds,
        split_seeds=args.split_seeds,
        anchor_count=args.anchor_count,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
