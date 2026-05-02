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

from scripts import build_source_private_hellaswag_hidden_innovation_repair_probe as repair  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_train_sample_stress as stress  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024"
)
DEFAULT_TRAIN_SAMPLE_CACHE_DIR = stress.DEFAULT_OUTPUT
DEFAULT_TRAIN_SAMPLE_SEEDS = (1729, 2027, 2039)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733)
DEFAULT_RIDGES = (1000.0, 10000.0, 100000.0)
STRICT_DELTA = 0.02


def _centered_z(candidate_scores: np.ndarray) -> np.ndarray:
    centered = candidate_scores - np.mean(candidate_scores, axis=1, keepdims=True)
    scale = np.std(candidate_scores, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return centered / scale


def _aggregate_model_scores(
    *,
    features: np.ndarray,
    models: list[dict[str, Any]],
    policy: str,
) -> tuple[list[int], np.ndarray]:
    if policy == "mean_zscore":
        scores = []
        for model in models:
            _, candidate_scores = repair._predict_candidate_ridge(features, model)
            scores.append(_centered_z(candidate_scores))
        aggregate_scores = np.mean(scores, axis=0)
        return [int(value) for value in np.argmax(aggregate_scores, axis=1)], aggregate_scores
    if policy == "vote":
        votes = np.zeros((features.shape[0], 4), dtype=np.float64)
        for model in models:
            predictions, _ = repair._predict_candidate_ridge(features, model)
            votes[np.arange(features.shape[0]), predictions] += 1.0
        return [int(value) for value in np.argmax(votes, axis=1)], votes
    raise ValueError(f"unsupported aggregation policy: {policy}")


def _hybrid_vote_on_score_agreement(
    *,
    mean_predictions: list[int],
    vote_predictions: list[int],
    score_mean_predictions: list[int],
) -> list[int]:
    return [
        int(vote) if int(mean) == int(score_mean) else int(mean)
        for mean, vote, score_mean in zip(
            mean_predictions,
            vote_predictions,
            score_mean_predictions,
            strict=True,
        )
    ]


def _control_predictions_for_policy(
    *,
    features: np.ndarray,
    models: list[dict[str, Any]],
    score_mean_predictions: list[int],
    aggregation_policy: str,
) -> list[int]:
    if aggregation_policy in {"mean_zscore", "vote"}:
        predictions, _ = _aggregate_model_scores(
            features=features,
            models=models,
            policy=aggregation_policy,
        )
        return predictions
    if aggregation_policy == "mean_zscore_vote_on_score_agreement":
        mean_predictions, _ = _aggregate_model_scores(
            features=features,
            models=models,
            policy="mean_zscore",
        )
        vote_predictions, _ = _aggregate_model_scores(
            features=features,
            models=models,
            policy="vote",
        )
        return _hybrid_vote_on_score_agreement(
            mean_predictions=mean_predictions,
            vote_predictions=vote_predictions,
            score_mean_predictions=score_mean_predictions,
        )
    raise ValueError(f"unsupported aggregation policy: {aggregation_policy}")


def _select_component_model(
    *,
    view: str,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: list[int],
    dev_indices: list[int],
    train_rows: list[arc_gate.ArcRow],
    train_scores: list[list[float]],
    eval_rows: list[arc_gate.ArcRow],
    ridges: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    readouts = [
        repair._fit_and_eval_view(
            view=view,
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
        "view": view,
        "selected_ridge": selected["ridge"],
        "selected_feature_dim": selected["feature_dim"],
        "selected_fit_accuracy": selected["fit"]["accuracy"],
        "selected_internal_dev_accuracy": selected["internal_dev"]["accuracy"],
        "selected_eval_accuracy": selected["eval"]["accuracy"],
        "candidate_readout_count": len(readouts),
    }
    return model, row


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    j = payload["jackknife_summary"]
    lines = [
        "# HellaSwag Hidden-Innovation Bagged Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- aggregation policy: `{h['aggregation_policy']}`",
        f"- component models: `{h['component_model_count']}`",
        f"- train sample seeds: `{h['train_sample_seed_count']}`",
        f"- new train sample seeds: `{h['new_train_sample_seed_count']}`",
        f"- eval accuracy: `{h['selected_eval_accuracy']:.6f}`",
        f"- best label-copy accuracy: `{h['best_label_copy_eval_accuracy']:.6f}`",
        f"- delta vs best label-copy: `{h['selected_minus_best_label_copy']:.6f}`",
        f"- CI95 vs best label-copy: `[{h['paired_ci95_low_vs_best_label_copy']:.6f}, {h['paired_ci95_high_vs_best_label_copy']:.6f}]`",
        f"- score-only bagged control accuracy: `{h['score_only_bagged_control_accuracy']:.6f}`",
        f"- zero-hidden control accuracy: `{h['zero_hidden_control_accuracy']:.6f}`",
        f"- wrong-example hidden control accuracy: `{h['wrong_example_hidden_control_accuracy']:.6f}`",
        f"- candidate-roll hidden control accuracy: `{h['candidate_roll_hidden_control_accuracy']:.6f}`",
        f"- jackknife subbags passing: `{j['pass_count']}/{j['row_count']}`",
        f"- jackknife min delta vs best label-copy: `{j['selected_minus_best_label_copy_min']:.6f}`",
        f"- jackknife min CI95 low vs best label-copy: `{j['paired_ci95_low_vs_best_label_copy_min']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_sample_cache_dir: pathlib.Path = DEFAULT_TRAIN_SAMPLE_CACHE_DIR,
    train_path: pathlib.Path = repair.DEFAULT_TRAIN,
    eval_path: pathlib.Path = repair.DEFAULT_EVAL,
    eval_score_cache: pathlib.Path = repair.DEFAULT_EVAL_SCORE_CACHE,
    eval_hidden_cache: pathlib.Path = repair.DEFAULT_EVAL_HIDDEN_CACHE,
    train_hidden_rows: int = 512,
    train_sample_seeds: tuple[int, ...] = DEFAULT_TRAIN_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = 500,
    source_lm_model: str = "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    source_lm_device: str = "auto_cpu",
    source_lm_dtype: str = "float32",
    source_lm_max_length: int = 256,
    source_lm_normalization: str = "mean",
    source_lm_prompt_mode: str = "continuation",
    hidden_layers: tuple[int, ...] = (-1,),
    local_files_only: bool = True,
    aggregation_policy: str = "mean_zscore",
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    train_sample_cache_dir = top2._resolve(train_sample_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = top2._resolve(train_path)
    eval_path = top2._resolve(eval_path)
    eval_score_cache = top2._resolve(eval_score_cache)
    eval_hidden_cache = top2._resolve(eval_hidden_cache)

    all_train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
    eval_hidden, eval_hidden_model = top2._load_hidden_cache(eval_hidden_cache, rows=eval_rows)

    hidden_eval_features = repair._candidate_feature_tensor(
        scores=eval_scores,
        hidden=eval_hidden,
        view="score_hidden_residual",
    )
    score_eval_features = repair._candidate_feature_tensor(
        scores=eval_scores,
        hidden=eval_hidden,
        view="score_only",
    )
    zero_eval_features = repair._candidate_feature_tensor(
        scores=eval_scores,
        hidden=np.zeros_like(eval_hidden),
        view="score_hidden_residual",
    )
    wrong_eval_features = repair._candidate_feature_tensor(
        scores=eval_scores,
        hidden=np.roll(eval_hidden, 1, axis=0),
        view="score_hidden_residual",
    )
    candidate_roll_eval_features = repair._candidate_feature_tensor(
        scores=eval_scores,
        hidden=np.roll(eval_hidden, 1, axis=1),
        view="score_hidden_residual",
    )

    hidden_models: list[dict[str, Any]] = []
    score_models: list[dict[str, Any]] = []
    hidden_models_by_sample: dict[int, list[dict[str, Any]]] = {}
    score_models_by_sample: dict[int, list[dict[str, Any]]] = {}
    trained_label_predictions: list[list[int]] = []
    trained_label_predictions_by_sample: dict[int, list[list[int]]] = {}
    component_rows: list[dict[str, Any]] = []
    sample_cache_rows: list[dict[str, Any]] = []

    for sample_seed in train_sample_seeds:
        hidden_models_by_sample.setdefault(sample_seed, [])
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
        train_feature_by_view = {
            "score_only": repair._candidate_feature_tensor(
                scores=train_scores,
                hidden=train_hidden,
                view="score_only",
            ),
            "score_hidden_residual": repair._candidate_feature_tensor(
                scores=train_scores,
                hidden=train_hidden,
                view="score_hidden_residual",
            ),
        }
        eval_feature_by_view = {
            "score_only": score_eval_features,
            "score_hidden_residual": hidden_eval_features,
        }
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
            for view in ("score_only", "score_hidden_residual"):
                model, component = _select_component_model(
                    view=view,
                    train_features=train_feature_by_view[view],
                    eval_features=eval_feature_by_view[view],
                    fit_indices=fit_indices,
                    dev_indices=dev_indices,
                    train_rows=train_rows,
                    train_scores=train_scores,
                    eval_rows=eval_rows,
                    ridges=ridges,
                )
                component_rows.append(
                    {
                        "train_sample_seed": sample_seed,
                        "split_seed": split_seed,
                        "fit_rows": len(fit_indices),
                        "internal_dev_rows": len(dev_indices),
                        **component,
                    }
                )
                if view == "score_hidden_residual":
                    hidden_models.append(model)
                    hidden_models_by_sample[sample_seed].append(model)
                else:
                    score_models.append(model)
                    score_models_by_sample[sample_seed].append(model)

    score_mean_predictions, _ = _aggregate_model_scores(
        features=score_eval_features,
        models=score_models,
        policy="mean_zscore",
    )
    score_vote_predictions, _ = _aggregate_model_scores(
        features=score_eval_features,
        models=score_models,
        policy="vote",
    )
    hidden_mean_predictions, hidden_mean_scores = _aggregate_model_scores(
        features=hidden_eval_features,
        models=hidden_models,
        policy="mean_zscore",
    )
    vote_predictions, hidden_vote_scores = _aggregate_model_scores(
        features=hidden_eval_features,
        models=hidden_models,
        policy="vote",
    )
    if aggregation_policy == "mean_zscore":
        selected_predictions = hidden_mean_predictions
        selected_scores = hidden_mean_scores
        score_only_predictions = score_mean_predictions
    elif aggregation_policy == "vote":
        selected_predictions = vote_predictions
        selected_scores = hidden_vote_scores
        score_only_predictions = score_vote_predictions
    elif aggregation_policy == "mean_zscore_vote_on_score_agreement":
        selected_predictions = _hybrid_vote_on_score_agreement(
            mean_predictions=hidden_mean_predictions,
            vote_predictions=vote_predictions,
            score_mean_predictions=score_mean_predictions,
        )
        selected_scores = hidden_mean_scores
        score_only_predictions = score_vote_predictions
    else:
        raise ValueError(f"unsupported aggregation policy: {aggregation_policy}")
    zero_predictions = _control_predictions_for_policy(
        features=zero_eval_features,
        models=hidden_models,
        score_mean_predictions=score_mean_predictions,
        aggregation_policy=aggregation_policy,
    )
    wrong_predictions = _control_predictions_for_policy(
        features=wrong_eval_features,
        models=hidden_models,
        score_mean_predictions=score_mean_predictions,
        aggregation_policy=aggregation_policy,
    )
    candidate_roll_predictions = _control_predictions_for_policy(
        features=candidate_roll_eval_features,
        models=hidden_models,
        score_mean_predictions=score_mean_predictions,
        aggregation_policy=aggregation_policy,
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
    paired_ci_label = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        best_label_predictions,
        seed=7001,
        samples=bootstrap_samples,
    )
    paired_ci_score = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        score_only_predictions,
        seed=7002,
        samples=bootstrap_samples,
    )

    def _subbag_readout(
        *,
        name: str,
        included_sample_seeds: tuple[int, ...],
        held_out_sample_seed: int | None,
        ci_seed: int,
    ) -> dict[str, Any]:
        sub_hidden_models = [
            model for seed in included_sample_seeds for model in hidden_models_by_sample[int(seed)]
        ]
        sub_score_models = [
            model for seed in included_sample_seeds for model in score_models_by_sample[int(seed)]
        ]
        sub_trained_predictions = [
            predictions
            for seed in included_sample_seeds
            for predictions in trained_label_predictions_by_sample[int(seed)]
        ]
        sub_score_mean_predictions, _ = _aggregate_model_scores(
            features=score_eval_features,
            models=sub_score_models,
            policy="mean_zscore",
        )
        sub_score_vote_predictions, _ = _aggregate_model_scores(
            features=score_eval_features,
            models=sub_score_models,
            policy="vote",
        )
        sub_hidden_mean_predictions, _ = _aggregate_model_scores(
            features=hidden_eval_features,
            models=sub_hidden_models,
            policy="mean_zscore",
        )
        sub_hidden_vote_predictions, _ = _aggregate_model_scores(
            features=hidden_eval_features,
            models=sub_hidden_models,
            policy="vote",
        )
        if aggregation_policy == "mean_zscore":
            sub_selected_predictions = sub_hidden_mean_predictions
            sub_score_only_predictions = sub_score_mean_predictions
        elif aggregation_policy == "vote":
            sub_selected_predictions = sub_hidden_vote_predictions
            sub_score_only_predictions = sub_score_vote_predictions
        elif aggregation_policy == "mean_zscore_vote_on_score_agreement":
            sub_selected_predictions = _hybrid_vote_on_score_agreement(
                mean_predictions=sub_hidden_mean_predictions,
                vote_predictions=sub_hidden_vote_predictions,
                score_mean_predictions=sub_score_mean_predictions,
            )
            sub_score_only_predictions = sub_score_vote_predictions
        else:
            raise ValueError(f"unsupported aggregation policy: {aggregation_policy}")
        sub_zero_predictions = _control_predictions_for_policy(
            features=zero_eval_features,
            models=sub_hidden_models,
            score_mean_predictions=sub_score_mean_predictions,
            aggregation_policy=aggregation_policy,
        )
        sub_wrong_predictions = _control_predictions_for_policy(
            features=wrong_eval_features,
            models=sub_hidden_models,
            score_mean_predictions=sub_score_mean_predictions,
            aggregation_policy=aggregation_policy,
        )
        sub_candidate_roll_predictions = _control_predictions_for_policy(
            features=candidate_roll_eval_features,
            models=sub_hidden_models,
            score_mean_predictions=sub_score_mean_predictions,
            aggregation_policy=aggregation_policy,
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
            "component_model_count": len(sub_hidden_models),
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
                    ci_seed=8100 + offset,
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
        "wrong_example_hidden_control_accuracy_max": max(
            (row["wrong_example_hidden_control_accuracy"] for row in jackknife_rows),
            default=wrong_accuracy,
        ),
        "candidate_roll_hidden_control_accuracy_max": max(
            (row["candidate_roll_hidden_control_accuracy"] for row in jackknife_rows),
            default=candidate_roll_accuracy,
        ),
    }

    sample_seed_count = len(set(train_sample_seeds))
    new_sample_seed_count = len({seed for seed in train_sample_seeds if seed != 1729})
    headline = {
        "aggregation_policy": aggregation_policy,
        "component_model_count": len(hidden_models),
        "score_only_component_model_count": len(score_models),
        "train_sample_seed_count": sample_seed_count,
        "new_train_sample_seed_count": new_sample_seed_count,
        "split_seed_count": len(split_seeds),
        "selected_eval_accuracy": selected_accuracy,
        "vote_eval_accuracy": top2._accuracy(eval_rows, vote_predictions),
        "source_label_copy_eval_accuracy": source_label_accuracy,
        "trained_choice_bias_label_copy_eval_accuracy": trained_label_accuracy,
        "best_label_copy_eval_accuracy": best_label_accuracy,
        "selected_minus_source_label_copy": selected_accuracy - source_label_accuracy,
        "selected_minus_trained_choice_bias_label_copy": selected_accuracy - trained_label_accuracy,
        "selected_minus_best_label_copy": selected_accuracy - best_label_accuracy,
        "score_only_bagged_control_accuracy": score_only_accuracy,
        "selected_minus_score_only_bagged_control": selected_accuracy - score_only_accuracy,
        "zero_hidden_control_accuracy": zero_accuracy,
        "selected_minus_zero_hidden_control": selected_accuracy - zero_accuracy,
        "wrong_example_hidden_control_accuracy": wrong_accuracy,
        "selected_minus_wrong_example_hidden_control": selected_accuracy - wrong_accuracy,
        "candidate_roll_hidden_control_accuracy": candidate_roll_accuracy,
        "selected_minus_candidate_roll_hidden_control": selected_accuracy - candidate_roll_accuracy,
        "paired_ci95_low_vs_best_label_copy": paired_ci_label["ci95_low"],
        "paired_ci95_high_vs_best_label_copy": paired_ci_label["ci95_high"],
        "paired_ci95_low_vs_score_only_bagged": paired_ci_score["ci95_low"],
        "paired_ci95_high_vs_score_only_bagged": paired_ci_score["ci95_high"],
        "source_top2_oracle_accuracy": top2._topk_oracle(eval_rows, eval_scores, k=2),
        "source_top4_oracle_accuracy": top2._topk_oracle(eval_rows, eval_scores, k=4),
        "selected_ridge_counts": {
            str(key): value
            for key, value in Counter(
                row["selected_ridge"] for row in component_rows if row["view"] == "score_hidden_residual"
            ).items()
        },
        "strict_delta_required": STRICT_DELTA,
    }
    pass_gate = bool(
        new_sample_seed_count >= 1
        and headline["selected_minus_best_label_copy"] >= STRICT_DELTA
        and headline["paired_ci95_low_vs_best_label_copy"] > 0.0
        and headline["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
        and headline["paired_ci95_low_vs_score_only_bagged"] > 0.0
        and headline["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and wrong_accuracy <= best_label_accuracy
        and candidate_roll_accuracy <= best_label_accuracy
        and jackknife_summary["all_pass"]
    )
    packet_contract = {
        "packet_name": "bagged_hidden_innovation_candidate_selector_packet",
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "fields": [
            "selected candidate id packed into 2 bits",
            "quantized bagged hidden-innovation confidence/debug bin",
            "predeclared model-bank id stored in experiment metadata, not transmitted per request",
        ],
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
        "decoder_rule": "receiver chooses the candidate id produced by the predeclared bagged source-side denoiser",
    }
    predictions = [
        {
            "row_id": row.row_id,
            "answer_index": row.answer_index,
            "selected_prediction": int(selected),
            "hidden_mean_prediction": int(hidden_mean),
            "vote_prediction": int(vote),
            "source_label_prediction": int(source),
            "trained_label_prediction": int(trained),
            "score_only_bagged_prediction": int(score_only),
            "score_mean_prediction": int(score_mean),
            "score_vote_prediction": int(score_vote),
            "zero_hidden_prediction": int(zero),
            "wrong_example_hidden_prediction": int(wrong),
            "candidate_roll_hidden_prediction": int(candidate_roll),
            "selected_margin": float(np.partition(selected_scores[index], -2)[-1] - np.partition(selected_scores[index], -2)[-2]),
        }
        for index, (
            row,
            selected,
            hidden_mean,
            vote,
            source,
            trained,
            score_only,
            score_mean,
            score_vote,
            zero,
            wrong,
            candidate_roll,
        ) in enumerate(
            zip(
                eval_rows,
                selected_predictions,
                hidden_mean_predictions,
                vote_predictions,
                source_label_predictions,
                best_trained_predictions,
                score_only_predictions,
                score_mean_predictions,
                score_vote_predictions,
                zero_predictions,
                wrong_predictions,
                candidate_roll_predictions,
                strict=True,
            )
        )
    ]

    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_bagged_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if the predeclared mean-zscore bag over source-side score+hidden-residual denoisers includes "
            "at least one fresh train-row sample seed and beats the best source/trained label-copy control, the "
            "bagged score-only control, and the zero-hidden control by at least 0.02 with paired CI95 low > 0; "
            "wrong-example and candidate-roll hidden controls must not beat label-copy."
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
        "ridges": list(ridges),
        "dev_fraction": dev_fraction,
        "source_model": {
            "score_eval": eval_source_model,
            "hidden_eval": eval_hidden_model,
            "source_lm_model": source_lm_model,
            "source_lm_device": source_lm_device,
            "source_lm_dtype": source_lm_dtype,
            "source_lm_max_length": source_lm_max_length,
            "source_lm_normalization": source_lm_normalization,
            "source_lm_prompt_mode": source_lm_prompt_mode,
        },
        "packet_contract": packet_contract,
        "headline": headline,
        "jackknife_summary": jackknife_summary,
        "jackknife_rows": jackknife_rows,
        "sample_caches": sample_cache_rows,
        "component_rows": component_rows,
        "control_readouts": {
            "source_label_copy": top2._evaluate(eval_rows, source_label_predictions),
            "trained_choice_bias_label_copy": top2._evaluate(eval_rows, best_trained_predictions),
            "score_only_bagged_control": top2._evaluate(eval_rows, score_only_predictions),
            "zero_hidden_control": top2._evaluate(eval_rows, zero_predictions),
            "wrong_example_hidden_control": top2._evaluate(eval_rows, wrong_predictions),
            "candidate_roll_hidden_control": top2._evaluate(eval_rows, candidate_roll_predictions),
        },
        "interpretation": (
            "The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample "
            "failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats "
            "the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split "
            "seeds is a stability method, not a larger communication channel: the receiver still sees only the "
            "selected candidate packet, while score-only and hidden-corruption controls test whether the win "
            "requires real source hidden innovation."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }

    json_path = output_dir / "hellaswag_hidden_innovation_bagged_gate.json"
    md_path = output_dir / "hellaswag_hidden_innovation_bagged_gate.md"
    rows_path = output_dir / "component_rows.csv"
    jackknife_path = output_dir / "jackknife_rows.csv"
    predictions_path = output_dir / "predictions.jsonl"
    sample_cache_path = output_dir / "sample_caches.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(rows_path, component_rows)
    _write_csv(jackknife_path, jackknife_rows)
    if not jackknife_path.exists():
        jackknife_path.write_text("", encoding="utf-8")
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(sample_cache_path, sample_cache_rows)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": top2._display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path, rows_path, jackknife_path, predictions_path, sample_cache_path)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-sample-cache-dir", type=pathlib.Path, default=DEFAULT_TRAIN_SAMPLE_CACHE_DIR)
    parser.add_argument("--train-path", type=pathlib.Path, default=repair.DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=repair.DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=repair.DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=repair.DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--train-sample-seeds", type=_parse_int_tuple, default=DEFAULT_TRAIN_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--aggregation-policy",
        choices=("mean_zscore", "vote", "mean_zscore_vote_on_score_agreement"),
        default="mean_zscore",
    )
    parser.add_argument("--source-lm-model", default="/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()

    payload = build_gate(
        output_dir=args.output_dir,
        train_sample_cache_dir=args.train_sample_cache_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        train_sample_seeds=args.train_sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        bootstrap_samples=args.bootstrap_samples,
        aggregation_policy=args.aggregation_policy,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=args.local_files_only,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
