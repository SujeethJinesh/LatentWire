from __future__ import annotations

"""Official-train disagreement-prototype receiver for HellaSwag packets."""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_acceptance_gate as accept  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_headroom_decomposition as decomp  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_disagreement_prototype_receiver_gate_20260503")
DEFAULT_PROTO_COUNTS = (4, 8, 16, 32)
DEFAULT_VIEWS = ("score_only", "score_hidden_confidence")
STRICT_DELTA = 0.005
CONTROL_SEPARATION_DELTA = 0.003
STRICT_TARGET_DELTA = 0.02


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return official._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return official._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    return official._sha256_file(path)


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    return official._read_jsonl(path)


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


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one string is required")
    return result


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms < 1e-8, 1.0, norms)


def _standardize_from_fit(features: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features[fit_indices], axis=0)
    scale = np.std(features[fit_indices], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (features - mean) / scale, mean.astype(np.float64), scale.astype(np.float64)


def _fit_spherical_prototypes(
    points: np.ndarray,
    *,
    count: int,
    seed: int,
    iterations: int,
) -> np.ndarray:
    if len(points) == 0 or count <= 0:
        return np.zeros((0, points.shape[1]), dtype=np.float64)
    points = _normalize_rows(points.astype(np.float64))
    count = min(int(count), len(points))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(points)))
    chosen = [first]
    min_distance = 1.0 - (points @ points[first])
    while len(chosen) < count:
        next_index = int(np.argmax(min_distance))
        chosen.append(next_index)
        min_distance = np.minimum(min_distance, 1.0 - (points @ points[next_index]))
    centers = points[chosen].copy()
    for _ in range(max(1, int(iterations))):
        sims = points @ centers.T
        labels = np.argmax(sims, axis=1)
        updated = centers.copy()
        for center_index in range(count):
            members = points[labels == center_index]
            if len(members):
                updated[center_index] = np.mean(members, axis=0)
        centers = _normalize_rows(updated)
    return centers.astype(np.float64)


def _prototype_scores(
    features: np.ndarray,
    *,
    benefit: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    positive_count: int,
    negative_count: int,
    seed: int,
    iterations: int,
    aggregation: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    standardized, mean, scale = _standardize_from_fit(features, fit_indices)
    normalized = _normalize_rows(standardized)
    positive_fit = fit_indices[benefit[fit_indices] > 0.0]
    negative_fit = fit_indices[benefit[fit_indices] < 0.0]
    positive = _fit_spherical_prototypes(
        normalized[positive_fit],
        count=positive_count,
        seed=seed,
        iterations=iterations,
    )
    negative = _fit_spherical_prototypes(
        normalized[negative_fit],
        count=negative_count,
        seed=seed + 10_003,
        iterations=iterations,
    )
    query = normalized[query_indices]
    if len(positive):
        positive_sims = query @ positive.T
        if aggregation == "max":
            positive_score = np.max(positive_sims, axis=1)
        elif aggregation == "mean":
            positive_score = np.mean(positive_sims, axis=1)
        elif aggregation == "top2":
            width = min(2, positive_sims.shape[1])
            positive_score = np.mean(np.sort(positive_sims, axis=1)[:, -width:], axis=1)
        else:
            raise ValueError(f"unsupported aggregation: {aggregation}")
    else:
        positive_score = np.zeros(len(query_indices), dtype=np.float64)
    if len(negative):
        negative_sims = query @ negative.T
        if aggregation == "max":
            negative_score = np.max(negative_sims, axis=1)
        elif aggregation == "mean":
            negative_score = np.mean(negative_sims, axis=1)
        elif aggregation == "top2":
            width = min(2, negative_sims.shape[1])
            negative_score = np.mean(np.sort(negative_sims, axis=1)[:, -width:], axis=1)
        else:
            raise ValueError(f"unsupported aggregation: {aggregation}")
    else:
        negative_score = np.zeros(len(query_indices), dtype=np.float64)
    return (positive_score - negative_score).astype(np.float64), {
        "positive_fit_rows": int(len(positive_fit)),
        "negative_fit_rows": int(len(negative_fit)),
        "positive_prototypes": int(len(positive)),
        "negative_prototypes": int(len(negative)),
        "feature_mean_l2": float(np.linalg.norm(mean)),
        "feature_scale_min": float(np.min(scale)),
        "feature_scale_max": float(np.max(scale)),
    }


def _select_prototype_threshold(
    *,
    scores: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    answers: np.ndarray,
    dev_indices: np.ndarray,
) -> dict[str, Any]:
    return accept._select_threshold(
        scores=scores,
        packet_predictions=packet_predictions,
        alt_predictions=alt_predictions,
        answers=answers,
        dev_indices=dev_indices,
    )


def _roll_predictions(predictions: np.ndarray, *, width: int = 4) -> np.ndarray:
    return ((predictions.astype(np.int64) + 1) % int(width)).astype(np.int64)


def _run_prototype_receiver(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    train_benefit: np.ndarray,
    train_packet: np.ndarray,
    train_alt: np.ndarray,
    train_answers: np.ndarray,
    validation_packet: np.ndarray,
    validation_alt: np.ndarray,
    validation_answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    eval_indices: np.ndarray,
    positive_count: int,
    negative_count: int,
    seed: int,
    iterations: int,
    aggregation: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    target_predictions: np.ndarray,
) -> dict[str, Any]:
    dev_scores, proto_meta = _prototype_scores(
        train_features,
        benefit=train_benefit,
        fit_indices=fit_indices,
        query_indices=dev_indices,
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
        iterations=iterations,
        aggregation=aggregation,
    )
    train_scores = np.zeros(len(train_answers), dtype=np.float64)
    train_scores[dev_indices] = dev_scores
    selected = _select_prototype_threshold(
        scores=train_scores,
        packet_predictions=train_packet,
        alt_predictions=train_alt,
        answers=train_answers,
        dev_indices=dev_indices,
    )
    eval_scores, _ = _prototype_scores(
        np.concatenate([train_features, eval_features], axis=0),
        benefit=np.concatenate([train_benefit, np.zeros(len(validation_answers), dtype=np.float64)], axis=0),
        fit_indices=fit_indices,
        query_indices=np.arange(len(train_answers), len(train_answers) + len(validation_answers), dtype=np.int64),
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
        iterations=iterations,
        aggregation=aggregation,
    )
    use_alt = (eval_scores > selected["threshold"]) & (validation_alt != validation_packet)
    predictions = accept._task_predictions(
        packet_predictions=validation_packet,
        alt_predictions=validation_alt,
        use_alt=use_alt,
    )
    stats = accept._receiver_stats(
        predictions=predictions,
        packet_predictions=validation_packet,
        target_predictions=target_predictions,
        answers=validation_answers,
        indices=eval_indices,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    return {
        "method": "disagreement_prototype",
        "positive_prototype_budget": int(positive_count),
        "negative_prototype_budget": int(negative_count),
        "prototype_seed": int(seed),
        "prototype_iterations": int(iterations),
        "aggregation": aggregation,
        "threshold": selected["threshold"],
        "official_dev_accuracy": selected["dev_accuracy"],
        "official_dev_override_rate": selected["dev_override_rate"],
        "official_dev_help_count": selected["dev_help_count"],
        "official_dev_harm_count": selected["dev_harm_count"],
        "validation_override_rate": float(np.mean(use_alt)),
        "prototype_meta": proto_meta,
        "predictions": predictions,
        "use_alt": use_alt.astype(bool),
        **stats,
    }


def _contiguous_block_rows(
    *,
    predictions: np.ndarray,
    packet_predictions: np.ndarray,
    answers: np.ndarray,
    eval_indices: np.ndarray,
) -> list[dict[str, Any]]:
    return accept._contiguous_block_deltas(
        predictions=predictions,
        packet_predictions=packet_predictions,
        answers=answers,
        eval_indices=eval_indices,
    )


def _random_same_rate_control(
    *,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    target_predictions: np.ndarray,
    answers: np.ndarray,
    eval_indices: np.ndarray,
    override_rate: float,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    disagreement = alt_predictions != packet_predictions
    use_alt = (rng.random(len(answers)) < float(override_rate)) & disagreement
    predictions = accept._task_predictions(
        packet_predictions=packet_predictions,
        alt_predictions=alt_predictions,
        use_alt=use_alt,
    )
    return {
        "control": "random_same_override_rate",
        "validation_override_rate": float(np.mean(use_alt)),
        **accept._receiver_stats(
            predictions=predictions,
            packet_predictions=packet_predictions,
            target_predictions=target_predictions,
            answers=answers,
            indices=eval_indices,
            seed=seed + 17,
            samples=bootstrap_samples,
        ),
    }


def _load_calibration_and_validation(
    *,
    train_path: pathlib.Path,
    tiny_train_cache_dir: pathlib.Path,
    qwen_train_cache_dir: pathlib.Path,
    tiny_eval_packet_jsonl: pathlib.Path,
    qwen_eval_packet_jsonl: pathlib.Path,
    qwen_global_artifact: pathlib.Path,
    sample_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    ridges: tuple[float, ...],
    train_hidden_rows: int,
    dev_fraction: float,
    tiny_aggregation_policy: str,
) -> dict[str, Any]:
    all_train_rows = arc_gate._load_rows(_resolve(train_path))
    tiny_samples = {
        int(seed): official._load_sample_cache(
            cache_dir=tiny_train_cache_dir,
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for seed in sample_seeds
    }
    qwen_samples = {
        int(seed): official._load_sample_cache(
            cache_dir=qwen_train_cache_dir,
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for seed in sample_seeds
    }
    tiny_bank = official._fit_family_model_bank(
        samples=tiny_samples,
        split_seeds=split_seeds,
        ridges=ridges,
        dev_fraction=dev_fraction,
    )
    qwen_bank = official._fit_family_model_bank(
        samples=qwen_samples,
        split_seeds=split_seeds,
        ridges=ridges,
        dev_fraction=dev_fraction,
    )
    calibration = official._build_oob_calibration_rows(
        tiny_samples=tiny_samples,
        qwen_samples=qwen_samples,
        tiny_bank=tiny_bank,
        qwen_bank=qwen_bank,
        sample_seeds=sample_seeds,
        tiny_aggregation_policy=tiny_aggregation_policy,
    )
    fit_indices, dev_indices = official._official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )
    tiny_eval_rows = _read_jsonl(tiny_eval_packet_jsonl)
    qwen_eval_rows = _read_jsonl(qwen_eval_packet_jsonl)
    qwen_eval = accept._load_qwen_bundle(qwen_global_artifact)
    row_ids = [str(row["row_id"]) for row in tiny_eval_rows]
    if row_ids != [str(row["row_id"]) for row in qwen_eval_rows]:
        raise ValueError("validation TinyLlama/Qwen packet rows are not aligned")
    if row_ids != qwen_eval["row_ids"]:
        raise ValueError("validation packet rows and Qwen eval scores are not aligned")
    validation_answers = np.asarray([int(row["answer_index"]) for row in tiny_eval_rows], dtype=np.int64)
    validation_packet = np.asarray([int(row["selected_prediction"]) for row in tiny_eval_rows], dtype=np.int64)
    validation_margin = np.asarray(
        [float(row.get("selected_margin", 0.0)) for row in tiny_eval_rows],
        dtype=np.float64,
    )
    validation_alternatives = {
        "qwen_target_score": np.argmax(qwen_eval["scores"], axis=1).astype(np.int64),
        "mean_zscore_prediction": np.asarray(
            [int(row["mean_zscore_prediction"]) for row in qwen_eval_rows],
            dtype=np.int64,
        ),
        "hybrid_vote_on_score_agreement_prediction": np.asarray(
            [int(row["hybrid_vote_on_score_agreement_prediction"]) for row in qwen_eval_rows],
            dtype=np.int64,
        ),
    }
    train_alternatives = {
        "qwen_target_score": calibration["qwen_target"],
        "mean_zscore_prediction": calibration["qwen_mean"],
        "hybrid_vote_on_score_agreement_prediction": calibration["qwen_hybrid"],
    }
    return {
        "calibration": calibration,
        "fit_indices": fit_indices,
        "dev_indices": dev_indices,
        "validation_answers": validation_answers,
        "validation_packet": validation_packet,
        "validation_margin": validation_margin,
        "validation_alternatives": validation_alternatives,
        "validation_target": validation_alternatives["qwen_target_score"],
        "train_alternatives": train_alternatives,
        "qwen_eval": qwen_eval,
        "sample_cache_rows": [
            {
                "sample_seed": int(seed),
                "row_count": int(len(tiny_samples[int(seed)]["row_ids"])),
                "content_digest": tiny_samples[int(seed)]["content_digest"],
                "tiny_score_cache": tiny_samples[int(seed)]["score_path"],
                "tiny_score_cache_sha256": tiny_samples[int(seed)]["score_sha256"],
                "tiny_hidden_cache": tiny_samples[int(seed)]["hidden_path"],
                "tiny_hidden_cache_sha256": tiny_samples[int(seed)]["hidden_sha256"],
                "qwen_score_cache": qwen_samples[int(seed)]["score_path"],
                "qwen_score_cache_sha256": qwen_samples[int(seed)]["score_sha256"],
                "qwen_hidden_cache": qwen_samples[int(seed)]["hidden_path"],
                "qwen_hidden_cache_sha256": qwen_samples[int(seed)]["hidden_sha256"],
            }
            for seed in sample_seeds
        ],
        "component_rows": [
            *[row for seed in sample_seeds for row in tiny_bank[int(seed)]["component_rows"]],
            *[
                {"family": "Qwen2.5", **row}
                for seed in sample_seeds
                for row in qwen_bank[int(seed)]["component_rows"]
            ],
        ],
    }


def _prediction_flip_audit(
    *,
    predictions: np.ndarray,
    packet_predictions: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    selected_correct = predictions[indices] == answers[indices]
    packet_correct = packet_predictions[indices] == answers[indices]
    return {
        "same_prediction": int(np.sum(predictions[indices] == packet_predictions[indices])),
        "fixed_packet_error": int(np.sum(selected_correct & ~packet_correct)),
        "broke_packet_correct": int(np.sum(~selected_correct & packet_correct)),
        "changed_wrong_to_wrong": int(
            np.sum(
                (predictions[indices] != packet_predictions[indices])
                & ~selected_correct
                & ~packet_correct
            )
        ),
        "net_correct_vs_packet": int(np.sum(selected_correct & ~packet_correct) - np.sum(~selected_correct & packet_correct)),
        "eval_rows": int(len(indices)),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Disagreement-Prototype Receiver Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- scout pass gate: `{payload['scout_pass_gate']}`",
        f"- default pass gate: `{payload['predeclared_default_pass_gate']}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default receiver accuracy: `{h['default_eval_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- best scout receiver accuracy: `{h['best_scout_eval_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- best destructive/control delta vs packet-only: `{h['best_control_delta_vs_packet_only']:.6f}`",
        f"- target-or-packet oracle accuracy: `{h['target_or_packet_oracle_accuracy']:.6f}`",
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = official.DEFAULT_TRAIN_PATH,
    tiny_train_cache_dir: pathlib.Path = official.DEFAULT_TINY_TRAIN_CACHE_DIR,
    qwen_train_cache_dir: pathlib.Path = official.DEFAULT_QWEN_TRAIN_CACHE_DIR,
    tiny_eval_packet_jsonl: pathlib.Path = official.DEFAULT_TINY_EVAL_PACKET_JSONL,
    qwen_eval_packet_jsonl: pathlib.Path = official.DEFAULT_QWEN_EVAL_PACKET_JSONL,
    qwen_global_artifact: pathlib.Path = official.DEFAULT_QWEN_GLOBAL_ARTIFACT,
    sample_seeds: tuple[int, ...] = official.DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = official.DEFAULT_SPLIT_SEEDS,
    ridges: tuple[float, ...] = official.DEFAULT_RIDGES,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    prototype_counts: tuple[int, ...] = DEFAULT_PROTO_COUNTS,
    feature_views: tuple[str, ...] = DEFAULT_VIEWS,
    aggregations: tuple[str, ...] = ("max", "top2"),
    prototype_iterations: int = 12,
    bootstrap_samples: int = 1000,
    tiny_aggregation_policy: str = "mean_zscore",
    run_date: str = "2026-05-03",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_calibration_and_validation(
        train_path=train_path,
        tiny_train_cache_dir=tiny_train_cache_dir,
        qwen_train_cache_dir=qwen_train_cache_dir,
        tiny_eval_packet_jsonl=tiny_eval_packet_jsonl,
        qwen_eval_packet_jsonl=qwen_eval_packet_jsonl,
        qwen_global_artifact=qwen_global_artifact,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        ridges=ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
        tiny_aggregation_policy=tiny_aggregation_policy,
    )
    calibration = bundle["calibration"]
    fit_indices = bundle["fit_indices"]
    dev_indices = bundle["dev_indices"]
    validation_answers = bundle["validation_answers"]
    validation_packet = bundle["validation_packet"]
    validation_margin = bundle["validation_margin"]
    validation_alternatives = bundle["validation_alternatives"]
    validation_target = bundle["validation_target"]
    train_alternatives = bundle["train_alternatives"]
    qwen_eval = bundle["qwen_eval"]
    eval_indices = np.arange(len(validation_answers), dtype=np.int64)
    train_features: dict[tuple[str, str], np.ndarray] = {}
    eval_features: dict[tuple[str, str], np.ndarray] = {}
    for alt_name in train_alternatives:
        for view in feature_views:
            train_features[(alt_name, view)] = accept._feature_matrix(
                scores=calibration["qwen_scores"],
                hidden=calibration["qwen_hidden"],
                packet_predictions=calibration["tiny_packet"],
                packet_margins=calibration["tiny_margin"],
                alt_predictions=train_alternatives[alt_name],
                view=view,
            )
            eval_features[(alt_name, view)] = accept._feature_matrix(
                scores=qwen_eval["scores"],
                hidden=qwen_eval["hidden"],
                packet_predictions=validation_packet,
                packet_margins=validation_margin,
                alt_predictions=validation_alternatives[alt_name],
                view=view,
            )
    frontier_rows: list[dict[str, Any]] = []
    predictions_cache: dict[tuple[str, str, int, int, str], np.ndarray] = {}
    for alt_index, alt_name in enumerate(train_alternatives):
        benefit = accept._benefit_values(
            alt_predictions=train_alternatives[alt_name],
            packet_predictions=calibration["tiny_packet"],
            answers=calibration["answers"],
        )
        for view in feature_views:
            for positive_count in prototype_counts:
                for negative_count in prototype_counts:
                    for aggregation in aggregations:
                        result = _run_prototype_receiver(
                            train_features=train_features[(alt_name, view)],
                            eval_features=eval_features[(alt_name, view)],
                            train_benefit=benefit,
                            train_packet=calibration["tiny_packet"],
                            train_alt=train_alternatives[alt_name],
                            train_answers=calibration["answers"],
                            validation_packet=validation_packet,
                            validation_alt=validation_alternatives[alt_name],
                            validation_answers=validation_answers,
                            fit_indices=fit_indices,
                            dev_indices=dev_indices,
                            eval_indices=eval_indices,
                            positive_count=positive_count,
                            negative_count=negative_count,
                            seed=3100 + 101 * alt_index + 7 * positive_count + negative_count,
                            iterations=prototype_iterations,
                            aggregation=aggregation,
                            bootstrap_samples=bootstrap_samples,
                            bootstrap_seed=11_000 + 1009 * alt_index + 17 * positive_count + negative_count,
                            target_predictions=validation_target,
                        )
                        cache_key = (alt_name, view, positive_count, negative_count, aggregation)
                        predictions_cache[cache_key] = result.pop("predictions")
                        result.pop("use_alt")
                        frontier_rows.append(
                            {
                                "alternative": alt_name,
                                "feature_view": view,
                                **result,
                            }
                        )
    best_scout = max(
        frontier_rows,
        key=lambda row: (
            row["delta_vs_packet_only"],
            row["ci95_low_vs_packet_only"],
            row["official_dev_accuracy"],
            -row["validation_override_rate"],
        ),
    )
    default_candidates = [
        row
        for row in frontier_rows
        if row["alternative"] == "hybrid_vote_on_score_agreement_prediction"
        and row["feature_view"] == "score_hidden_confidence"
        and row["positive_prototype_budget"] == 16
        and row["negative_prototype_budget"] == 16
        and row["aggregation"] == "max"
    ]
    if not default_candidates:
        raise ValueError("missing predeclared default prototype row")
    default_row = max(default_candidates, key=lambda row: (row["official_dev_accuracy"], -row["validation_override_rate"]))
    default_key = (
        default_row["alternative"],
        default_row["feature_view"],
        default_row["positive_prototype_budget"],
        default_row["negative_prototype_budget"],
        default_row["aggregation"],
    )
    best_key = (
        best_scout["alternative"],
        best_scout["feature_view"],
        best_scout["positive_prototype_budget"],
        best_scout["negative_prototype_budget"],
        best_scout["aggregation"],
    )
    default_predictions = predictions_cache[default_key]
    best_predictions = predictions_cache[best_key]
    default_blocks = _contiguous_block_rows(
        predictions=default_predictions,
        packet_predictions=validation_packet,
        answers=validation_answers,
        eval_indices=eval_indices,
    )
    best_blocks = _contiguous_block_rows(
        predictions=best_predictions,
        packet_predictions=validation_packet,
        answers=validation_answers,
        eval_indices=eval_indices,
    )
    rng = np.random.default_rng(20260503)
    shuffled_answers = calibration["answers"].copy()
    rng.shuffle(shuffled_answers)
    label_shuffle_benefit = accept._benefit_values(
        alt_predictions=train_alternatives[default_row["alternative"]],
        packet_predictions=calibration["tiny_packet"],
        answers=shuffled_answers,
    )
    label_control = _run_prototype_receiver(
        train_features=train_features[(default_row["alternative"], default_row["feature_view"])],
        eval_features=eval_features[(default_row["alternative"], default_row["feature_view"])],
        train_benefit=label_shuffle_benefit,
        train_packet=calibration["tiny_packet"],
        train_alt=train_alternatives[default_row["alternative"]],
        train_answers=shuffled_answers,
        validation_packet=validation_packet,
        validation_alt=validation_alternatives[default_row["alternative"]],
        validation_answers=validation_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        eval_indices=eval_indices,
        positive_count=default_row["positive_prototype_budget"],
        negative_count=default_row["negative_prototype_budget"],
        seed=43_001,
        iterations=prototype_iterations,
        aggregation=default_row["aggregation"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=43_101,
        target_predictions=validation_target,
    )
    label_control.pop("predictions")
    label_control.pop("use_alt")
    rolled_train_alt = _roll_predictions(train_alternatives[default_row["alternative"]])
    rolled_eval_alt = _roll_predictions(validation_alternatives[default_row["alternative"]])
    rolled_benefit = accept._benefit_values(
        alt_predictions=rolled_train_alt,
        packet_predictions=calibration["tiny_packet"],
        answers=calibration["answers"],
    )
    rolled_train_features = accept._feature_matrix(
        scores=calibration["qwen_scores"],
        hidden=calibration["qwen_hidden"],
        packet_predictions=calibration["tiny_packet"],
        packet_margins=calibration["tiny_margin"],
        alt_predictions=rolled_train_alt,
        view=default_row["feature_view"],
    )
    rolled_eval_features = accept._feature_matrix(
        scores=qwen_eval["scores"],
        hidden=qwen_eval["hidden"],
        packet_predictions=validation_packet,
        packet_margins=validation_margin,
        alt_predictions=rolled_eval_alt,
        view=default_row["feature_view"],
    )
    rolled_control = _run_prototype_receiver(
        train_features=rolled_train_features,
        eval_features=rolled_eval_features,
        train_benefit=rolled_benefit,
        train_packet=calibration["tiny_packet"],
        train_alt=rolled_train_alt,
        train_answers=calibration["answers"],
        validation_packet=validation_packet,
        validation_alt=rolled_eval_alt,
        validation_answers=validation_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        eval_indices=eval_indices,
        positive_count=default_row["positive_prototype_budget"],
        negative_count=default_row["negative_prototype_budget"],
        seed=44_001,
        iterations=prototype_iterations,
        aggregation=default_row["aggregation"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=44_101,
        target_predictions=validation_target,
    )
    rolled_control.pop("predictions")
    rolled_control.pop("use_alt")
    random_control = _random_same_rate_control(
        packet_predictions=validation_packet,
        alt_predictions=validation_alternatives[default_row["alternative"]],
        target_predictions=validation_target,
        answers=validation_answers,
        eval_indices=eval_indices,
        override_rate=default_row["validation_override_rate"],
        seed=45_001,
        bootstrap_samples=bootstrap_samples,
    )
    control_rows = [
        {"control": "label_permutation_control", **label_control},
        {"control": "candidate_roll_alt_control", **rolled_control},
        random_control,
    ]
    best_control = max(control_rows, key=lambda row: row["delta_vs_packet_only"])
    target_or_packet_oracle = official._oracle_accuracy(
        [validation_packet, validation_alternatives["hybrid_vote_on_score_agreement_prediction"]],
        validation_answers,
    )
    default_flip_audit = _prediction_flip_audit(
        predictions=default_predictions,
        packet_predictions=validation_packet,
        answers=validation_answers,
        indices=eval_indices,
    )
    best_flip_audit = _prediction_flip_audit(
        predictions=best_predictions,
        packet_predictions=validation_packet,
        answers=validation_answers,
        indices=eval_indices,
    )
    predeclared_default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and default_row["delta_vs_target_only"] >= STRICT_TARGET_DELTA
        and default_row["ci95_low_vs_target_only"] > 0.0
        and all(row["delta_vs_packet_only"] > 0.0 for row in default_blocks)
        and default_row["delta_vs_packet_only"] - best_control["delta_vs_packet_only"] >= CONTROL_SEPARATION_DELTA
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
        and sum(row["delta_vs_packet_only"] > 0.0 for row in best_blocks) >= 4
        and best_scout["delta_vs_packet_only"] - best_control["delta_vs_packet_only"] >= CONTROL_SEPARATION_DELTA
    )
    pass_gate = bool(predeclared_default_pass_gate)
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "validation_rows": int(len(validation_answers)),
        "packet_only_accuracy": float(np.mean(validation_packet == validation_answers)),
        "qwen_target_accuracy": float(np.mean(validation_target == validation_answers)),
        "qwen_hybrid_accuracy": float(
            np.mean(validation_alternatives["hybrid_vote_on_score_agreement_prediction"] == validation_answers)
        ),
        "target_or_packet_oracle_accuracy": target_or_packet_oracle,
        "default_eval_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_override_rate": default_row["validation_override_rate"],
        "best_scout_eval_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_scout_alternative": best_scout["alternative"],
        "best_scout_feature_view": best_scout["feature_view"],
        "best_scout_positive_prototype_budget": best_scout["positive_prototype_budget"],
        "best_scout_negative_prototype_budget": best_scout["negative_prototype_budget"],
        "best_scout_aggregation": best_scout["aggregation"],
        "best_control": best_control["control"],
        "best_control_delta_vs_packet_only": best_control["delta_vs_packet_only"],
        "best_control_ci95_low_vs_packet_only": best_control["ci95_low_vs_packet_only"],
        "packet_raw_bytes": decomp.RAW_PACKET_BYTES,
        "packet_framed_bytes": decomp.FRAMED_PACKET_BYTES,
        "strict_delta_required": STRICT_DELTA,
        "native_gpu_claims_allowed": False,
    }
    lay_explanation = (
        "This gate looks for reusable disagreement shapes. On official train rows, it finds cases where "
        "Qwen's alternative answer would fix the TinyLlama packet and cases where Qwen would hurt it. "
        "Those groups become small prototypes. On validation, the receiver only overrides the TinyLlama "
        "packet when the current row looks closer to the helpful prototypes than to the harmful prototypes."
    )
    interpretation = (
        "This directly tests the next receiver/common-basis hypothesis from the ledger. A pass would mean "
        "that official-train disagreement prototypes can capture the TinyLlama/Qwen oracle headroom without "
        "validation-label tuning. A fail means the current HellaSwag receiver gap is not solved by shallow "
        "prototype geometry; the next live branch should move to an actual query-bottleneck/nonlinear "
        "connector or to a stronger source/benchmark surface."
    )
    payload = {
        "gate": "source_private_hellaswag_disagreement_prototype_receiver_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "predeclared_default_pass_gate": predeclared_default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "pass_rule": {
            "default_delta_vs_packet_only_at_least": STRICT_DELTA,
            "default_ci95_low_vs_packet_only_positive": True,
            "default_delta_vs_target_only_at_least": STRICT_TARGET_DELTA,
            "default_all_blocks_positive_vs_packet": True,
            "default_beats_best_control_by": CONTROL_SEPARATION_DELTA,
            "best_scout_is_diagnostic_only": True,
        },
        "headline": headline,
        "default_row": default_row,
        "best_scout_row": best_scout,
        "control_rows": control_rows,
        "default_block_rows": default_blocks,
        "best_scout_block_rows": best_blocks,
        "default_flip_audit": default_flip_audit,
        "best_scout_flip_audit": best_flip_audit,
        "frontier_rows": frontier_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": decomp.RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": decomp.FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(decomp.RAW_PACKET_BYTES * len(validation_answers)),
            "logical_validation_framed_record_bytes_total": int(decomp.FRAMED_PACKET_BYTES * len(validation_answers)),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "total_wall_time_s": float(time.perf_counter() - started),
        },
        "inputs": {
            "train_path": _display_path(train_path),
            "train_sha256": _sha256_file(train_path),
            "tiny_train_cache_dir": _display_path(tiny_train_cache_dir),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "tiny_eval_packet_jsonl": _display_path(tiny_eval_packet_jsonl),
            "tiny_eval_packet_jsonl_sha256": _sha256_file(tiny_eval_packet_jsonl),
            "qwen_eval_packet_jsonl": _display_path(qwen_eval_packet_jsonl),
            "qwen_eval_packet_jsonl_sha256": _sha256_file(qwen_eval_packet_jsonl),
            "qwen_global_artifact": _display_path(qwen_global_artifact),
            "qwen_global_artifact_sha256": _sha256_file(qwen_global_artifact),
            "sample_seeds": list(sample_seeds),
            "split_seeds": list(split_seeds),
            "prototype_counts": list(prototype_counts),
            "feature_views": list(feature_views),
            "aggregations": list(aggregations),
            "prototype_iterations": int(prototype_iterations),
        },
        "sample_cache_rows": bundle["sample_cache_rows"],
        "component_rows": bundle["component_rows"],
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_disagreement_prototype_receiver_gate.json"
    md_path = output_dir / "hellaswag_disagreement_prototype_receiver_gate.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path)
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=official.DEFAULT_TRAIN_PATH)
    parser.add_argument("--tiny-train-cache-dir", type=pathlib.Path, default=official.DEFAULT_TINY_TRAIN_CACHE_DIR)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=official.DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--tiny-eval-packet-jsonl", type=pathlib.Path, default=official.DEFAULT_TINY_EVAL_PACKET_JSONL)
    parser.add_argument("--qwen-eval-packet-jsonl", type=pathlib.Path, default=official.DEFAULT_QWEN_EVAL_PACKET_JSONL)
    parser.add_argument("--qwen-global-artifact", type=pathlib.Path, default=official.DEFAULT_QWEN_GLOBAL_ARTIFACT)
    parser.add_argument("--sample-seeds", type=_parse_int_tuple, default=official.DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=official.DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=official.DEFAULT_RIDGES)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--prototype-counts", type=_parse_int_tuple, default=DEFAULT_PROTO_COUNTS)
    parser.add_argument("--feature-views", type=_parse_str_tuple, default=DEFAULT_VIEWS)
    parser.add_argument("--aggregations", type=_parse_str_tuple, default=("max", "top2"))
    parser.add_argument("--prototype-iterations", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--tiny-aggregation-policy", default="mean_zscore")
    parser.add_argument("--run-date", default="2026-05-03")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        tiny_train_cache_dir=args.tiny_train_cache_dir,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        tiny_eval_packet_jsonl=args.tiny_eval_packet_jsonl,
        qwen_eval_packet_jsonl=args.qwen_eval_packet_jsonl,
        qwen_global_artifact=args.qwen_global_artifact,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        prototype_counts=args.prototype_counts,
        feature_views=args.feature_views,
        aggregations=args.aggregations,
        prototype_iterations=args.prototype_iterations,
        bootstrap_samples=args.bootstrap_samples,
        tiny_aggregation_policy=args.tiny_aggregation_policy,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
