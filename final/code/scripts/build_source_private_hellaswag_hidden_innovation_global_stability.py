from __future__ import annotations

"""Audit full-validation stability of the HellaSwag hidden-innovation packet."""

import argparse
import csv
import dataclasses
import datetime as dt
import json
import pathlib
import sys
import time
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
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502"
)
DEFAULT_TRAIN_SAMPLE_CACHE_DIR = stress.DEFAULT_OUTPUT
DEFAULT_TRAIN_SAMPLE_SEEDS = (1729, 2027, 2039)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733)
DEFAULT_RIDGES = (1000.0, 10000.0, 100000.0)
STRICT_DELTA = 0.02
POLICY_NAMES = ("mean_zscore", "vote", "hybrid_vote_on_score_agreement")


@dataclasses.dataclass(frozen=True)
class EvalSliceSpec:
    name: str
    start: int
    end: int
    eval_path: pathlib.Path
    score_cache: pathlib.Path
    hidden_cache: pathlib.Path


DEFAULT_EVAL_SPECS = (
    EvalSliceSpec(
        name="validation_0_1024",
        start=0,
        end=1024,
        eval_path=top2.DEFAULT_EVAL,
        score_cache=top2.DEFAULT_EVAL_SCORE_CACHE,
        hidden_cache=top2.DEFAULT_EVAL_HIDDEN_CACHE,
    ),
    *(
        EvalSliceSpec(
            name=f"validation_{start}_{start + 1024}",
            start=start,
            end=start + 1024,
            eval_path=pathlib.Path(
                f"results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation{start}_{start + 1024}/"
                f"hellaswag_validation_rows_{start}_{start + 1024}.jsonl"
            ),
            score_cache=pathlib.Path(
                f"results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation{start}_{start + 1024}/"
                "source_eval_score_cache.json"
            ),
            hidden_cache=pathlib.Path(
                f"results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation{start}_{start + 1024}/"
                "source_eval_hidden_cache.npz"
            ),
        )
        for start in range(1024, 9216, 1024)
    ),
    EvalSliceSpec(
        name="validation_9216_10042",
        start=9216,
        end=10042,
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
    ),
)


def _display_path(path: pathlib.Path) -> str:
    return top2._display_path(path)


def _load_eval_bundle(
    specs: tuple[EvalSliceSpec, ...],
) -> tuple[list[arc_gate.ArcRow], list[list[float]], np.ndarray, list[dict[str, Any]], list[str]]:
    all_rows: list[arc_gate.ArcRow] = []
    all_scores: list[list[float]] = []
    hidden_parts: list[np.ndarray] = []
    slice_rows: list[dict[str, Any]] = []
    source_models: list[str] = []
    for spec in specs:
        eval_path = top2._resolve(spec.eval_path)
        score_cache = top2._resolve(spec.score_cache)
        hidden_cache = top2._resolve(spec.hidden_cache)
        rows = arc_gate._load_rows(eval_path)
        scores, _, score_model = headroom._load_score_cache(score_cache, rows=rows)
        hidden, hidden_meta = top2._load_hidden_cache(hidden_cache, rows=rows)
        all_rows.extend(rows)
        all_scores.extend(scores)
        hidden_parts.append(hidden)
        source_models.append(str(score_model.get("model_path") or score_model.get("model") or "unknown"))
        slice_rows.append(
            {
                "name": spec.name,
                "start": spec.start,
                "end": spec.end,
                "rows": len(rows),
                "eval_path": _display_path(eval_path),
                "eval_sha256": top2._sha256_file(eval_path),
                "score_cache": _display_path(score_cache),
                "score_cache_sha256": top2._sha256_file(score_cache),
                "hidden_cache": _display_path(hidden_cache),
                "hidden_cache_sha256": top2._sha256_file(hidden_cache),
                "hidden_cache_meta": hidden_meta.get("cache_meta"),
            }
        )
    for left, right in zip(slice_rows, slice_rows[1:], strict=False):
        if int(left["end"]) != int(right["start"]):
            raise ValueError(f"non-contiguous eval specs: {left['name']} then {right['name']}")
    return all_rows, all_scores, np.concatenate(hidden_parts, axis=0), slice_rows, source_models


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    return top2._accuracy(rows, predictions)


def _best_label_predictions(
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    trained_predictions: list[list[int]],
) -> tuple[list[int], float, float, float, list[int]]:
    if not trained_predictions:
        raise ValueError("at least one trained label-copy prediction vector is required")
    source_accuracy = _accuracy(rows, source_predictions)
    best_trained = max(trained_predictions, key=lambda predictions: _accuracy(rows, predictions))
    trained_accuracy = _accuracy(rows, best_trained)
    if source_accuracy >= trained_accuracy:
        return source_predictions, source_accuracy, trained_accuracy, source_accuracy, best_trained
    return best_trained, source_accuracy, trained_accuracy, trained_accuracy, best_trained


def _policy_predictions(
    *,
    policy: str,
    hidden_mean_predictions: list[int],
    hidden_vote_predictions: list[int],
    score_mean_predictions: list[int],
) -> list[int]:
    if policy == "mean_zscore":
        return hidden_mean_predictions
    if policy == "vote":
        return hidden_vote_predictions
    if policy == "hybrid_vote_on_score_agreement":
        return bagged._hybrid_vote_on_score_agreement(
            mean_predictions=hidden_mean_predictions,
            vote_predictions=hidden_vote_predictions,
            score_mean_predictions=score_mean_predictions,
        )
    raise ValueError(f"unsupported policy: {policy}")


def _readout(
    *,
    rows: list[arc_gate.ArcRow],
    predictions: list[int],
    best_label_predictions: list[int],
    score_predictions: list[int],
    zero_predictions: list[int],
    wrong_predictions: list[int],
    candidate_roll_predictions: list[int],
    source_label_accuracy: float,
    trained_label_accuracy: float,
    best_label_accuracy: float,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    selected_accuracy = _accuracy(rows, predictions)
    score_accuracy = _accuracy(rows, score_predictions)
    zero_accuracy = _accuracy(rows, zero_predictions)
    wrong_accuracy = _accuracy(rows, wrong_predictions)
    candidate_roll_accuracy = _accuracy(rows, candidate_roll_predictions)
    correct = int(sum(row.answer_index == pred for row, pred in zip(rows, predictions, strict=True)))
    paired_label = top2._paired_ci_predictions(
        rows,
        predictions,
        best_label_predictions,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    paired_score = top2._paired_ci_predictions(
        rows,
        predictions,
        score_predictions,
        seed=bootstrap_seed + 101,
        samples=bootstrap_samples,
    )
    row = {
        "rows": len(rows),
        "correct": correct,
        "selected_eval_accuracy": selected_accuracy,
        "score_only_bagged_control_accuracy": score_accuracy,
        "source_label_copy_eval_accuracy": source_label_accuracy,
        "trained_choice_bias_label_copy_eval_accuracy": trained_label_accuracy,
        "best_label_copy_eval_accuracy": best_label_accuracy,
        "selected_minus_best_label_copy": selected_accuracy - best_label_accuracy,
        "selected_minus_score_only_bagged_control": selected_accuracy - score_accuracy,
        "zero_hidden_control_accuracy": zero_accuracy,
        "selected_minus_zero_hidden_control": selected_accuracy - zero_accuracy,
        "wrong_example_hidden_control_accuracy": wrong_accuracy,
        "selected_minus_wrong_example_hidden_control": selected_accuracy - wrong_accuracy,
        "candidate_roll_hidden_control_accuracy": candidate_roll_accuracy,
        "selected_minus_candidate_roll_hidden_control": selected_accuracy - candidate_roll_accuracy,
        "paired_ci95_low_vs_best_label_copy": paired_label["ci95_low"],
        "paired_ci95_high_vs_best_label_copy": paired_label["ci95_high"],
        "paired_ci95_low_vs_score_only_bagged": paired_score["ci95_low"],
        "paired_ci95_high_vs_score_only_bagged": paired_score["ci95_high"],
    }
    row["pass_gate"] = bool(
        row["selected_minus_best_label_copy"] >= STRICT_DELTA
        and row["paired_ci95_low_vs_best_label_copy"] > 0.0
        and row["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
        and row["paired_ci95_low_vs_score_only_bagged"] > 0.0
        and row["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and wrong_accuracy <= best_label_accuracy
        and candidate_roll_accuracy <= best_label_accuracy
    )
    return row


def _slice_policy_rows(
    *,
    rows: list[arc_gate.ArcRow],
    slice_specs: list[dict[str, Any]],
    predictions_by_policy: dict[str, list[int]],
    best_label_predictions: list[int],
    score_predictions_by_policy: dict[str, list[int]],
    zero_predictions_by_policy: dict[str, list[int]],
    wrong_predictions_by_policy: dict[str, list[int]],
    candidate_roll_predictions_by_policy: dict[str, list[int]],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    offset = 0
    for spec_index, spec in enumerate(slice_specs):
        count = int(spec["rows"])
        row_slice = rows[offset : offset + count]
        best_slice = best_label_predictions[offset : offset + count]
        source_slice_accuracy = _accuracy(
            row_slice,
            predictions_by_policy["source_label_copy"][offset : offset + count],
        )
        trained_slice_accuracy = _accuracy(
            row_slice,
            predictions_by_policy["trained_choice_bias_label_copy"][offset : offset + count],
        )
        best_slice_accuracy = _accuracy(row_slice, best_slice)
        for policy_index, policy in enumerate(POLICY_NAMES):
            readout = _readout(
                rows=row_slice,
                predictions=predictions_by_policy[policy][offset : offset + count],
                best_label_predictions=best_slice,
                score_predictions=score_predictions_by_policy[policy][offset : offset + count],
                zero_predictions=zero_predictions_by_policy[policy][offset : offset + count],
                wrong_predictions=wrong_predictions_by_policy[policy][offset : offset + count],
                candidate_roll_predictions=candidate_roll_predictions_by_policy[policy][offset : offset + count],
                source_label_accuracy=source_slice_accuracy,
                trained_label_accuracy=trained_slice_accuracy,
                best_label_accuracy=best_slice_accuracy,
                bootstrap_seed=31_000 + 100 * spec_index + policy_index,
                bootstrap_samples=bootstrap_samples,
            )
            output.append(
                {
                    "slice": spec["name"],
                    "start": spec["start"],
                    "end": spec["end"],
                    "rows": count,
                    "policy": policy,
                    **readout,
                }
            )
        offset += count
    return output


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


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_sample_cache_dir: pathlib.Path = DEFAULT_TRAIN_SAMPLE_CACHE_DIR,
    train_path: pathlib.Path = repair.DEFAULT_TRAIN,
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
    run_date: str = "2026-05-02",
    eval_specs: tuple[EvalSliceSpec, ...] = DEFAULT_EVAL_SPECS,
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    train_sample_cache_dir = top2._resolve(train_sample_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    eval_rows, eval_scores, eval_hidden, eval_slice_rows, source_models = _load_eval_bundle(eval_specs)
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

    train_path = top2._resolve(train_path)
    all_train_rows = arc_gate._load_rows(train_path)
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
                model, component = bagged._select_component_model(
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

    source_label_predictions = top2._source_label_predictions(eval_scores)
    (
        best_label_predictions,
        source_label_accuracy,
        trained_label_accuracy,
        best_label_accuracy,
        trained_choice_bias_label_predictions,
    ) = _best_label_predictions(
        eval_rows,
        source_label_predictions,
        trained_label_predictions,
    )
    score_mean_predictions, _ = bagged._aggregate_model_scores(
        features=score_eval_features,
        models=score_models,
        policy="mean_zscore",
    )
    score_vote_predictions, _ = bagged._aggregate_model_scores(
        features=score_eval_features,
        models=score_models,
        policy="vote",
    )
    hidden_mean_predictions, hidden_mean_scores = bagged._aggregate_model_scores(
        features=hidden_eval_features,
        models=hidden_models,
        policy="mean_zscore",
    )
    hidden_vote_predictions, hidden_vote_scores = bagged._aggregate_model_scores(
        features=hidden_eval_features,
        models=hidden_models,
        policy="vote",
    )
    policy_predictions = {
        "source_label_copy": source_label_predictions,
        "trained_choice_bias_label_copy": trained_choice_bias_label_predictions,
        "mean_zscore": hidden_mean_predictions,
        "vote": hidden_vote_predictions,
        "hybrid_vote_on_score_agreement": _policy_predictions(
            policy="hybrid_vote_on_score_agreement",
            hidden_mean_predictions=hidden_mean_predictions,
            hidden_vote_predictions=hidden_vote_predictions,
            score_mean_predictions=score_mean_predictions,
        ),
    }
    score_predictions_by_policy = {
        "mean_zscore": score_mean_predictions,
        "vote": score_vote_predictions,
        "hybrid_vote_on_score_agreement": score_vote_predictions,
    }
    zero_predictions_by_policy = {
        policy: bagged._control_predictions_for_policy(
            features=zero_eval_features,
            models=hidden_models,
            score_mean_predictions=score_mean_predictions,
            aggregation_policy="mean_zscore" if policy == "mean_zscore" else "vote"
            if policy == "vote"
            else "mean_zscore_vote_on_score_agreement",
        )
        for policy in POLICY_NAMES
    }
    wrong_predictions_by_policy = {
        policy: bagged._control_predictions_for_policy(
            features=wrong_eval_features,
            models=hidden_models,
            score_mean_predictions=score_mean_predictions,
            aggregation_policy="mean_zscore" if policy == "mean_zscore" else "vote"
            if policy == "vote"
            else "mean_zscore_vote_on_score_agreement",
        )
        for policy in POLICY_NAMES
    }
    candidate_roll_predictions_by_policy = {
        policy: bagged._control_predictions_for_policy(
            features=candidate_roll_eval_features,
            models=hidden_models,
            score_mean_predictions=score_mean_predictions,
            aggregation_policy="mean_zscore" if policy == "mean_zscore" else "vote"
            if policy == "vote"
            else "mean_zscore_vote_on_score_agreement",
        )
        for policy in POLICY_NAMES
    }

    policy_rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(POLICY_NAMES):
        policy_rows.append(
            {
                "policy": policy,
                **_readout(
                    rows=eval_rows,
                    predictions=policy_predictions[policy],
                    best_label_predictions=best_label_predictions,
                    score_predictions=score_predictions_by_policy[policy],
                    zero_predictions=zero_predictions_by_policy[policy],
                    wrong_predictions=wrong_predictions_by_policy[policy],
                    candidate_roll_predictions=candidate_roll_predictions_by_policy[policy],
                    source_label_accuracy=source_label_accuracy,
                    trained_label_accuracy=trained_label_accuracy,
                    best_label_accuracy=best_label_accuracy,
                    bootstrap_seed=71_000 + policy_index,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )

    subbag_rows: list[dict[str, Any]] = []
    unique_sample_seeds = tuple(sorted(set(train_sample_seeds)))
    for held_out_index, held_out_seed in enumerate(unique_sample_seeds):
        included_sample_seeds = tuple(seed for seed in unique_sample_seeds if seed != held_out_seed)
        sub_hidden_models = [model for seed in included_sample_seeds for model in hidden_models_by_sample[seed]]
        sub_score_models = [model for seed in included_sample_seeds for model in score_models_by_sample[seed]]
        sub_trained_predictions = [
            predictions for seed in included_sample_seeds for predictions in trained_label_predictions_by_sample[seed]
        ]
        (
            sub_best_label_predictions,
            sub_source_accuracy,
            sub_trained_accuracy,
            sub_best_accuracy,
            _sub_trained_choice_predictions,
        ) = _best_label_predictions(
            eval_rows,
            source_label_predictions,
            sub_trained_predictions,
        )
        sub_score_mean_predictions, _ = bagged._aggregate_model_scores(
            features=score_eval_features,
            models=sub_score_models,
            policy="mean_zscore",
        )
        sub_score_vote_predictions, _ = bagged._aggregate_model_scores(
            features=score_eval_features,
            models=sub_score_models,
            policy="vote",
        )
        sub_hidden_mean_predictions, _ = bagged._aggregate_model_scores(
            features=hidden_eval_features,
            models=sub_hidden_models,
            policy="mean_zscore",
        )
        sub_hidden_vote_predictions, _ = bagged._aggregate_model_scores(
            features=hidden_eval_features,
            models=sub_hidden_models,
            policy="vote",
        )
        sub_policy_predictions = {
            "mean_zscore": sub_hidden_mean_predictions,
            "vote": sub_hidden_vote_predictions,
            "hybrid_vote_on_score_agreement": _policy_predictions(
                policy="hybrid_vote_on_score_agreement",
                hidden_mean_predictions=sub_hidden_mean_predictions,
                hidden_vote_predictions=sub_hidden_vote_predictions,
                score_mean_predictions=sub_score_mean_predictions,
            ),
        }
        sub_score_predictions_by_policy = {
            "mean_zscore": sub_score_mean_predictions,
            "vote": sub_score_vote_predictions,
            "hybrid_vote_on_score_agreement": sub_score_vote_predictions,
        }
        sub_zero_predictions_by_policy = {
            policy: bagged._control_predictions_for_policy(
                features=zero_eval_features,
                models=sub_hidden_models,
                score_mean_predictions=sub_score_mean_predictions,
                aggregation_policy="mean_zscore" if policy == "mean_zscore" else "vote"
                if policy == "vote"
                else "mean_zscore_vote_on_score_agreement",
            )
            for policy in POLICY_NAMES
        }
        sub_wrong_predictions_by_policy = {
            policy: bagged._control_predictions_for_policy(
                features=wrong_eval_features,
                models=sub_hidden_models,
                score_mean_predictions=sub_score_mean_predictions,
                aggregation_policy="mean_zscore" if policy == "mean_zscore" else "vote"
                if policy == "vote"
                else "mean_zscore_vote_on_score_agreement",
            )
            for policy in POLICY_NAMES
        }
        sub_candidate_roll_predictions_by_policy = {
            policy: bagged._control_predictions_for_policy(
                features=candidate_roll_eval_features,
                models=sub_hidden_models,
                score_mean_predictions=sub_score_mean_predictions,
                aggregation_policy="mean_zscore" if policy == "mean_zscore" else "vote"
                if policy == "vote"
                else "mean_zscore_vote_on_score_agreement",
            )
            for policy in POLICY_NAMES
        }
        for policy_index, policy in enumerate(POLICY_NAMES):
            subbag_rows.append(
                {
                    "name": f"leave_out_{held_out_seed}",
                    "policy": policy,
                    "included_sample_seeds": list(included_sample_seeds),
                    "held_out_sample_seed": held_out_seed,
                    "component_model_count": len(sub_hidden_models),
                    "score_only_component_model_count": len(sub_score_models),
                    **_readout(
                        rows=eval_rows,
                        predictions=sub_policy_predictions[policy],
                        best_label_predictions=sub_best_label_predictions,
                        score_predictions=sub_score_predictions_by_policy[policy],
                        zero_predictions=sub_zero_predictions_by_policy[policy],
                        wrong_predictions=sub_wrong_predictions_by_policy[policy],
                        candidate_roll_predictions=sub_candidate_roll_predictions_by_policy[policy],
                        source_label_accuracy=sub_source_accuracy,
                        trained_label_accuracy=sub_trained_accuracy,
                        best_label_accuracy=sub_best_accuracy,
                        bootstrap_seed=81_000 + 100 * held_out_index + policy_index,
                        bootstrap_samples=bootstrap_samples,
                    ),
                }
            )

    slice_policy_rows = _slice_policy_rows(
        rows=eval_rows,
        slice_specs=eval_slice_rows,
        predictions_by_policy=policy_predictions,
        best_label_predictions=best_label_predictions,
        score_predictions_by_policy=score_predictions_by_policy,
        zero_predictions_by_policy=zero_predictions_by_policy,
        wrong_predictions_by_policy=wrong_predictions_by_policy,
        candidate_roll_predictions_by_policy=candidate_roll_predictions_by_policy,
        bootstrap_samples=bootstrap_samples,
    )

    mean_row = next(row for row in policy_rows if row["policy"] == "mean_zscore")
    hybrid_row = next(row for row in policy_rows if row["policy"] == "hybrid_vote_on_score_agreement")
    mean_subbags = [row for row in subbag_rows if row["policy"] == "mean_zscore"]
    hybrid_subbags = [row for row in subbag_rows if row["policy"] == "hybrid_vote_on_score_agreement"]
    mean_slice_rows = [row for row in slice_policy_rows if row["policy"] == "mean_zscore"]
    hybrid_slice_rows = [row for row in slice_policy_rows if row["policy"] == "hybrid_vote_on_score_agreement"]
    terminal_slice_name = str(eval_slice_rows[-1]["name"])
    mean_zscore_failed_slices = [str(row["slice"]) for row in mean_slice_rows if not row["pass_gate"]]
    hybrid_failed_slices = [str(row["slice"]) for row in hybrid_slice_rows if not row["pass_gate"]]
    headline = {
        "eval_rows": len(eval_rows),
        "eval_slice_count": len(eval_slice_rows),
        "train_sample_seed_count": len(unique_sample_seeds),
        "split_seed_count": len(split_seeds),
        "component_model_count": len(hidden_models),
        "score_only_component_model_count": len(score_models),
        "mean_zscore_accuracy": mean_row["selected_eval_accuracy"],
        "mean_zscore_minus_best_label_copy": mean_row["selected_minus_best_label_copy"],
        "mean_zscore_ci95_low_vs_best_label_copy": mean_row["paired_ci95_low_vs_best_label_copy"],
        "mean_zscore_subbag_pass_count": sum(1 for row in mean_subbags if row["pass_gate"]),
        "mean_zscore_slice_pass_count": sum(1 for row in mean_slice_rows if row["pass_gate"]),
        "mean_zscore_failed_slices": mean_zscore_failed_slices,
        "hybrid_accuracy": hybrid_row["selected_eval_accuracy"],
        "hybrid_minus_best_label_copy": hybrid_row["selected_minus_best_label_copy"],
        "hybrid_ci95_low_vs_best_label_copy": hybrid_row["paired_ci95_low_vs_best_label_copy"],
        "hybrid_subbag_pass_count": sum(1 for row in hybrid_subbags if row["pass_gate"]),
        "hybrid_slice_pass_count": sum(1 for row in hybrid_slice_rows if row["pass_gate"]),
        "hybrid_failed_slices": hybrid_failed_slices,
        "best_label_copy_accuracy": best_label_accuracy,
        "score_only_mean_accuracy": score_predictions_by_policy and _accuracy(eval_rows, score_mean_predictions),
        "source_private_packet_raw_bytes": 2,
        "source_private_packet_framed_bytes": 5,
        "terminal_tail_mean_zscore_slice_pass": bool(mean_slice_rows[-1]["pass_gate"]),
        "terminal_tail_hybrid_slice_pass": bool(hybrid_slice_rows[-1]["pass_gate"]),
        "strict_delta_required": STRICT_DELTA,
    }
    pass_gate = bool(
        mean_row["pass_gate"]
        and headline["mean_zscore_subbag_pass_count"] == len(mean_subbags)
        and set(mean_zscore_failed_slices).issubset({terminal_slice_name})
    )
    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_global_stability",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if the pre-existing mean-zscore candidate-wise hidden-innovation policy is globally positive "
            "over the full frozen HellaSwag validation split, beats best label-copy, score-only, and zero-hidden "
            "controls by at least 0.02 with paired CI95 lows > 0, all leave-one-train-sample subbags pass, and "
            "any slice-level failure is limited to the already-recorded terminal tail. This is a global-stability "
            "audit, not a claim that every local slice clears jackknife."
        ),
        "train_path": _display_path(train_path),
        "train_sha256": top2._sha256_file(train_path),
        "train_sample_cache_dir": _display_path(train_sample_cache_dir),
        "train_hidden_rows": train_hidden_rows,
        "train_sample_seeds": list(train_sample_seeds),
        "split_seeds": list(split_seeds),
        "ridges": list(ridges),
        "dev_fraction": dev_fraction,
        "eval_slices": eval_slice_rows,
        "source_models": sorted(set(source_models)),
        "packet_contract": {
            "packet_name": "bagged_hidden_innovation_candidate_selector_packet",
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "headline": headline,
        "policy_rows": policy_rows,
        "subbag_rows": subbag_rows,
        "slice_policy_rows": slice_policy_rows,
        "sample_caches": sample_cache_rows,
        "component_rows": component_rows,
        "interpretation": (
            "The terminal tail remains an important falsification surface because prior local scouts exposed "
            "tail fragility under alternative policies. This full-validation audit asks a stricter aggregate "
            "question: is the candidate-wise hidden-innovation packet globally positive and train-sample-stable "
            "when all frozen HellaSwag validation rows are pooled? A pass strengthens the method evidence while "
            "leaving cross-family and native-systems claims unresolved."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }

    json_path = output_dir / "hellaswag_hidden_innovation_global_stability.json"
    md_path = output_dir / "hellaswag_hidden_innovation_global_stability.md"
    policy_csv = output_dir / "policy_rows.csv"
    subbag_csv = output_dir / "subbag_rows.csv"
    slice_csv = output_dir / "slice_policy_rows.csv"
    sample_cache_jsonl = output_dir / "sample_caches.jsonl"
    component_jsonl = output_dir / "component_rows.jsonl"
    prediction_jsonl = output_dir / "predictions.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HellaSwag Hidden-Innovation Global Stability",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{headline['eval_rows']}`",
        f"- train sample seeds: `{headline['train_sample_seed_count']}`",
        f"- mean-zscore accuracy: `{headline['mean_zscore_accuracy']:.6f}`",
        f"- mean-zscore delta vs best label-copy: `{headline['mean_zscore_minus_best_label_copy']:.6f}`",
        f"- mean-zscore CI95 low vs best label-copy: `{headline['mean_zscore_ci95_low_vs_best_label_copy']:.6f}`",
        f"- mean-zscore subbags passing: `{headline['mean_zscore_subbag_pass_count']}/{len(mean_subbags)}`",
        f"- mean-zscore slices passing: `{headline['mean_zscore_slice_pass_count']}/{len(mean_slice_rows)}`",
        f"- hybrid accuracy: `{headline['hybrid_accuracy']:.6f}`",
        f"- hybrid delta vs best label-copy: `{headline['hybrid_minus_best_label_copy']:.6f}`",
        f"- hybrid subbags passing: `{headline['hybrid_subbag_pass_count']}/{len(hybrid_subbags)}`",
        f"- hybrid slices passing: `{headline['hybrid_slice_pass_count']}/{len(hybrid_slice_rows)}`",
        f"- terminal-tail mean-zscore slice pass: `{headline['terminal_tail_mean_zscore_slice_pass']}`",
        f"- terminal-tail hybrid slice pass: `{headline['terminal_tail_hybrid_slice_pass']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(policy_csv, policy_rows)
    _write_csv(subbag_csv, subbag_rows)
    _write_csv(slice_csv, slice_policy_rows)
    _write_jsonl(sample_cache_jsonl, sample_cache_rows)
    _write_jsonl(component_jsonl, component_rows)
    prediction_rows = [
        {
            "row_id": row.row_id,
            "answer_index": row.answer_index,
            "source_label_prediction": int(source_label),
            "trained_label_prediction": int(best_label),
            "score_mean_prediction": int(score_mean),
            "score_vote_prediction": int(score_vote),
            "mean_zscore_prediction": int(mean_pred),
            "vote_prediction": int(vote_pred),
            "hybrid_vote_on_score_agreement_prediction": int(hybrid_pred),
        }
        for row, source_label, best_label, score_mean, score_vote, mean_pred, vote_pred, hybrid_pred in zip(
            eval_rows,
            source_label_predictions,
            best_label_predictions,
            score_mean_predictions,
            score_vote_predictions,
            policy_predictions["mean_zscore"],
            policy_predictions["vote"],
            policy_predictions["hybrid_vote_on_score_agreement"],
            strict=True,
        )
    ]
    _write_jsonl(prediction_jsonl, prediction_rows)
    manifest_files = [
        json_path,
        md_path,
        policy_csv,
        subbag_csv,
        slice_csv,
        sample_cache_jsonl,
        component_jsonl,
        prediction_jsonl,
    ]
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in manifest_files
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
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--train-sample-seeds", type=_parse_int_tuple, default=DEFAULT_TRAIN_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--source-lm-model", default="/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_sample_cache_dir=args.train_sample_cache_dir,
        train_path=args.train_path,
        train_hidden_rows=args.train_hidden_rows,
        train_sample_seeds=args.train_sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        bootstrap_samples=args.bootstrap_samples,
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
