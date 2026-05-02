from __future__ import annotations

"""Train-only disagreement-prototype receiver for HellaSwag packets."""

import argparse
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

from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_acceptance_gate as accept  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_headroom_decomposition as decomp  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_disagreement_prototype_receiver_20260502")
DEFAULT_TRAIN_PATH = official.DEFAULT_TRAIN_PATH
DEFAULT_TINY_TRAIN_CACHE_DIR = official.DEFAULT_TINY_TRAIN_CACHE_DIR
DEFAULT_QWEN_TRAIN_CACHE_DIR = official.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_TINY_EVAL_PACKET_JSONL = official.DEFAULT_TINY_EVAL_PACKET_JSONL
DEFAULT_TINY_EVAL_ARTIFACT = official.DEFAULT_TINY_EVAL_ARTIFACT
DEFAULT_QWEN_EVAL_PACKET_JSONL = official.DEFAULT_QWEN_EVAL_PACKET_JSONL
DEFAULT_QWEN_GLOBAL_ARTIFACT = official.DEFAULT_QWEN_GLOBAL_ARTIFACT
DEFAULT_SAMPLE_SEEDS = official.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = official.DEFAULT_SPLIT_SEEDS
DEFAULT_RIDGES = official.DEFAULT_RIDGES
DEFAULT_PROTOTYPE_COUNTS = (16, 32, 64, 128)
DEFAULT_NEIGHBOR_KS = (9, 25, 51)
DEFAULT_TOP_KS = (1, 3, 5, 9)
DEFAULT_CONTROL_SEED = 9901
STRICT_DELTA = 0.005
STRICT_TARGET_DELTA = 0.02
CONTROL_SEPARATION_DELTA = 0.003


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return official._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return official._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    return official._sha256_file(path)


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


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms < 1e-8, 1.0, norms)


def _standardize_from_fit(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_features[fit_indices], axis=0)
    scale = np.std(train_features[fit_indices], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train_normed = _normalize_rows((train_features - mean) / scale)
    eval_normed = _normalize_rows((eval_features - mean) / scale)
    return train_normed, eval_normed, mean, scale


def _farthest_first_indices(
    features: np.ndarray,
    pool_indices: np.ndarray,
    count: int,
) -> np.ndarray:
    if len(pool_indices) == 0:
        raise ValueError("cannot select prototypes from an empty pool")
    count = min(int(count), len(pool_indices))
    pool = features[pool_indices]
    center = np.mean(pool, axis=0, keepdims=True)
    first_local = int(np.argmax(np.sum((pool - center) ** 2, axis=1)))
    selected_local = [first_local]
    min_dist = np.sum((pool - pool[first_local]) ** 2, axis=1)
    while len(selected_local) < count:
        next_local = int(np.argmax(min_dist))
        selected_local.append(next_local)
        next_dist = np.sum((pool - pool[next_local]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, next_dist)
    return pool_indices[np.asarray(selected_local, dtype=np.int64)]


def _random_indices(pool_indices: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    if len(pool_indices) == 0:
        raise ValueError("cannot select prototypes from an empty pool")
    rng = np.random.default_rng(seed)
    count = min(int(count), len(pool_indices))
    return np.sort(rng.choice(pool_indices, size=count, replace=False).astype(np.int64))


def _prototype_statistics(
    *,
    train_features: np.ndarray,
    prototype_indices: np.ndarray,
    fit_indices: np.ndarray,
    benefit: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    answers: np.ndarray,
    neighbor_k: int,
) -> dict[str, np.ndarray]:
    fit = train_features[fit_indices]
    proto = train_features[prototype_indices]
    sims = proto @ fit.T
    k = min(int(neighbor_k), fit.shape[0])
    if k == fit.shape[0]:
        local = np.argsort(-sims, axis=1)[:, :k]
    else:
        local = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    local_fit_indices = fit_indices[local]
    local_sims = np.take_along_axis(sims, local, axis=1)
    weights = np.maximum(local_sims, 0.0) + 1e-3
    weight_sum = np.sum(weights, axis=1)
    packet_correct = (packet_predictions == answers).astype(np.float64)
    alt_correct = (alt_predictions == answers).astype(np.float64)
    disagreement = (packet_predictions != alt_predictions).astype(np.float64)

    def weighted(values: np.ndarray) -> np.ndarray:
        return np.sum(values[local_fit_indices] * weights, axis=1) / weight_sum

    return {
        "benefit_mean": weighted(benefit),
        "packet_correct_rate": weighted(packet_correct),
        "alt_correct_rate": weighted(alt_correct),
        "disagreement_rate": weighted(disagreement),
        "prototype_benefit": benefit[prototype_indices].astype(np.float64),
    }


def _score_from_prototypes(
    *,
    query_features: np.ndarray,
    train_features: np.ndarray,
    prototype_indices: np.ndarray,
    prototype_stats: dict[str, np.ndarray],
    top_k: int,
) -> dict[str, np.ndarray]:
    proto = train_features[prototype_indices]
    sims = query_features @ proto.T
    k = min(int(top_k), proto.shape[0])
    if k == proto.shape[0]:
        local = np.argsort(-sims, axis=1)[:, :k]
    else:
        local = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    local_sims = np.take_along_axis(sims, local, axis=1)
    weights = np.maximum(local_sims, 0.0) + 1e-3
    weight_sum = np.sum(weights, axis=1)

    def weighted(name: str) -> np.ndarray:
        values = prototype_stats[name][local]
        return np.sum(values * weights, axis=1) / weight_sum

    benefit_score = weighted("benefit_mean")
    return {
        "benefit_score": benefit_score.astype(np.float64),
        "top_similarity": np.max(local_sims, axis=1).astype(np.float64),
        "local_packet_correct_rate": weighted("packet_correct_rate").astype(np.float64),
        "local_alt_correct_rate": weighted("alt_correct_rate").astype(np.float64),
        "local_disagreement_rate": weighted("disagreement_rate").astype(np.float64),
    }


def _score_train_eval_prototypes(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: np.ndarray,
    packet_train: np.ndarray,
    alt_train: np.ndarray,
    answers_train: np.ndarray,
    prototype_count: int,
    neighbor_k: int,
    top_k: int,
    control_kind: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_work = np.asarray(train_features, dtype=np.float64)
    eval_work = np.asarray(eval_features, dtype=np.float64)
    if control_kind == "score_row_shuffle":
        train_work = train_work[rng.permutation(train_work.shape[0])]
        eval_work = eval_work[rng.permutation(eval_work.shape[0])]

    train_normed, eval_normed, _, _ = _standardize_from_fit(
        train_features=train_work,
        eval_features=eval_work,
        fit_indices=fit_indices,
    )
    benefit = accept._benefit_values(
        alt_predictions=alt_train,
        packet_predictions=packet_train,
        answers=answers_train,
    )
    if control_kind == "label_permutation":
        benefit = benefit[rng.permutation(len(benefit))]

    pool_indices = fit_indices[packet_train[fit_indices] != alt_train[fit_indices]]
    if len(pool_indices) == 0:
        pool_indices = fit_indices
    if control_kind == "random_prototypes":
        prototype_indices = _random_indices(pool_indices, prototype_count, seed=seed)
        prototype_kind = "random"
    else:
        prototype_indices = _farthest_first_indices(train_normed, pool_indices, prototype_count)
        prototype_kind = "farthest_disagreement"

    stats = _prototype_statistics(
        train_features=train_normed,
        prototype_indices=prototype_indices,
        fit_indices=fit_indices,
        benefit=benefit,
        packet_predictions=packet_train,
        alt_predictions=alt_train,
        answers=answers_train,
        neighbor_k=neighbor_k,
    )
    train_scores = _score_from_prototypes(
        query_features=train_normed,
        train_features=train_normed,
        prototype_indices=prototype_indices,
        prototype_stats=stats,
        top_k=top_k,
    )
    eval_scores = _score_from_prototypes(
        query_features=eval_normed,
        train_features=train_normed,
        prototype_indices=prototype_indices,
        prototype_stats=stats,
        top_k=top_k,
    )
    return {
        "prototype_kind": prototype_kind,
        "prototype_indices": prototype_indices,
        "prototype_pool_rows": int(len(pool_indices)),
        "actual_prototype_count": int(len(prototype_indices)),
        "train_scores": train_scores,
        "eval_scores": eval_scores,
    }


def _run_prototype_receiver(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    packet_train: np.ndarray,
    packet_eval: np.ndarray,
    alt_train: np.ndarray,
    alt_eval: np.ndarray,
    answers_train: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    prototype_count: int,
    neighbor_k: int,
    top_k: int,
    control_kind: str = "real",
    seed: int = DEFAULT_CONTROL_SEED,
) -> dict[str, Any]:
    scored = _score_train_eval_prototypes(
        train_features=train_features,
        eval_features=eval_features,
        fit_indices=fit_indices,
        packet_train=packet_train,
        alt_train=alt_train,
        answers_train=answers_train,
        prototype_count=prototype_count,
        neighbor_k=neighbor_k,
        top_k=top_k,
        control_kind=control_kind,
        seed=seed,
    )
    selected = accept._select_threshold(
        scores=scored["train_scores"]["benefit_score"],
        packet_predictions=packet_train,
        alt_predictions=alt_train,
        answers=answers_train,
        dev_indices=dev_indices,
    )
    use_alt_eval = (scored["eval_scores"]["benefit_score"] > selected["threshold"]) & (
        alt_eval != packet_eval
    )
    predictions = accept._task_predictions(
        packet_predictions=packet_eval,
        alt_predictions=alt_eval,
        use_alt=use_alt_eval,
    )
    return {
        "method": "disagreement_prototype_receiver",
        "control_kind": control_kind,
        "prototype_kind": scored["prototype_kind"],
        "prototype_count": int(prototype_count),
        "actual_prototype_count": scored["actual_prototype_count"],
        "prototype_pool_rows": scored["prototype_pool_rows"],
        "neighbor_k": int(neighbor_k),
        "top_k": int(top_k),
        "threshold": selected["threshold"],
        "dev_accuracy": selected["dev_accuracy"],
        "dev_override_rate": selected["dev_override_rate"],
        "dev_help_count": selected["dev_help_count"],
        "dev_harm_count": selected["dev_harm_count"],
        "eval_override_rate": float(np.mean(use_alt_eval)),
        "eval_mean_benefit_score": float(np.mean(scored["eval_scores"]["benefit_score"])),
        "eval_mean_top_similarity": float(np.mean(scored["eval_scores"]["top_similarity"])),
        "eval_mean_local_packet_correct_rate": float(
            np.mean(scored["eval_scores"]["local_packet_correct_rate"])
        ),
        "eval_mean_local_alt_correct_rate": float(np.mean(scored["eval_scores"]["local_alt_correct_rate"])),
        "predictions": predictions,
        "scores": scored["eval_scores"]["benefit_score"],
    }


def _load_official_train_calibration(
    *,
    train_path: pathlib.Path,
    tiny_train_cache_dir: pathlib.Path,
    qwen_train_cache_dir: pathlib.Path,
    sample_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    ridges: tuple[float, ...],
    train_hidden_rows: int,
    dev_fraction: float,
    tiny_aggregation_policy: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    all_train_rows = official.arc_gate._load_rows(_resolve(train_path))
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
    audit = {
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
        ]
    }
    return calibration, fit_indices, dev_indices, audit


def _load_validation_bundle(
    *,
    tiny_eval_packet_jsonl: pathlib.Path,
    qwen_eval_packet_jsonl: pathlib.Path,
    qwen_global_artifact: pathlib.Path,
) -> dict[str, Any]:
    tiny_rows = _read_jsonl(tiny_eval_packet_jsonl)
    qwen_rows = _read_jsonl(qwen_eval_packet_jsonl)
    qwen_eval = accept._load_qwen_bundle(qwen_global_artifact)
    row_ids = [str(row["row_id"]) for row in tiny_rows]
    if row_ids != [str(row["row_id"]) for row in qwen_rows]:
        raise ValueError("validation TinyLlama/Qwen packet rows are not aligned")
    if row_ids != qwen_eval["row_ids"]:
        raise ValueError("validation packet rows and Qwen eval scores are not aligned")
    answers = np.asarray([int(row["answer_index"]) for row in tiny_rows], dtype=np.int64)
    packet = np.asarray([int(row["selected_prediction"]) for row in tiny_rows], dtype=np.int64)
    packet_margin = np.asarray(
        [float(row.get("selected_margin", 0.0)) for row in tiny_rows],
        dtype=np.float64,
    )
    alternatives = {
        "qwen_target_score": np.argmax(qwen_eval["scores"], axis=1).astype(np.int64),
        "mean_zscore_prediction": np.asarray(
            [int(row["mean_zscore_prediction"]) for row in qwen_rows],
            dtype=np.int64,
        ),
        "hybrid_vote_on_score_agreement_prediction": np.asarray(
            [int(row["hybrid_vote_on_score_agreement_prediction"]) for row in qwen_rows],
            dtype=np.int64,
        ),
    }
    return {
        "rows": tiny_rows,
        "row_ids": row_ids,
        "answers": answers,
        "packet": packet,
        "packet_margin": packet_margin,
        "alternatives": alternatives,
        "qwen_scores": qwen_eval["scores"],
        "qwen_hidden": qwen_eval["hidden"],
        "qwen_slices": qwen_eval["slices"],
        "qwen_artifact_path": qwen_eval["artifact_path"],
        "qwen_artifact_sha256": qwen_eval["artifact_sha256"],
    }


def _oracle_accuracy(predictions: list[np.ndarray], answers: np.ndarray) -> float:
    correct = np.zeros_like(answers, dtype=bool)
    for item in predictions:
        correct |= item == answers
    return float(np.mean(correct))


def _baseline_accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Disagreement-Prototype Receiver",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- predeclared default pass gate: `{payload['predeclared_default_pass_gate']}`",
        f"- scout pass gate: `{payload['scout_pass_gate']}`",
        f"- control separation gate: `{payload['control_separation_gate']}`",
        f"- official train calibration rows: `{h['official_train_calibration_rows']}`",
        f"- validation rows: `{h['validation_rows']}`",
        f"- default eval accuracy: `{h['default_eval_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- best scout eval accuracy: `{h['best_scout_eval_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- full-validation Tiny+Qwen oracle: `{h['validation_tiny_qwen_hybrid_oracle_accuracy']:.6f}`",
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
    train_path: pathlib.Path = DEFAULT_TRAIN_PATH,
    tiny_train_cache_dir: pathlib.Path = DEFAULT_TINY_TRAIN_CACHE_DIR,
    qwen_train_cache_dir: pathlib.Path = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    tiny_eval_packet_jsonl: pathlib.Path = DEFAULT_TINY_EVAL_PACKET_JSONL,
    tiny_eval_artifact: pathlib.Path = DEFAULT_TINY_EVAL_ARTIFACT,
    qwen_eval_packet_jsonl: pathlib.Path = DEFAULT_QWEN_EVAL_PACKET_JSONL,
    qwen_global_artifact: pathlib.Path = DEFAULT_QWEN_GLOBAL_ARTIFACT,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    prototype_counts: tuple[int, ...] = DEFAULT_PROTOTYPE_COUNTS,
    neighbor_ks: tuple[int, ...] = DEFAULT_NEIGHBOR_KS,
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = 1000,
    tiny_aggregation_policy: str = "mean_zscore",
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_start = time.perf_counter()
    calibration, fit_indices, dev_indices, calibration_audit = _load_official_train_calibration(
        train_path=train_path,
        tiny_train_cache_dir=tiny_train_cache_dir,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        ridges=ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
        tiny_aggregation_policy=tiny_aggregation_policy,
    )
    calibration_wall_time_s = time.perf_counter() - calibration_start
    validation = _load_validation_bundle(
        tiny_eval_packet_jsonl=tiny_eval_packet_jsonl,
        qwen_eval_packet_jsonl=qwen_eval_packet_jsonl,
        qwen_global_artifact=qwen_global_artifact,
    )

    train_alternatives = {
        "qwen_target_score": calibration["qwen_target"],
        "mean_zscore_prediction": calibration["qwen_mean"],
        "hybrid_vote_on_score_agreement_prediction": calibration["qwen_hybrid"],
    }
    eval_alternatives = validation["alternatives"]

    feature_start = time.perf_counter()
    train_features: dict[tuple[str, str], np.ndarray] = {}
    eval_features: dict[tuple[str, str], np.ndarray] = {}
    for alt_name in train_alternatives:
        for view in ("score_only", "score_hidden_confidence"):
            train_features[(alt_name, view)] = accept._feature_matrix(
                scores=calibration["qwen_scores"],
                hidden=calibration["qwen_hidden"],
                packet_predictions=calibration["tiny_packet"],
                packet_margins=calibration["tiny_margin"],
                alt_predictions=train_alternatives[alt_name],
                view=view,
            )
            eval_features[(alt_name, view)] = accept._feature_matrix(
                scores=validation["qwen_scores"],
                hidden=validation["qwen_hidden"],
                packet_predictions=validation["packet"],
                packet_margins=validation["packet_margin"],
                alt_predictions=eval_alternatives[alt_name],
                view=view,
            )
    feature_build_wall_time_s = time.perf_counter() - feature_start

    frontier_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str, int, int, int], np.ndarray] = {}
    selector_start = time.perf_counter()
    eval_indices = np.arange(len(validation["answers"]), dtype=np.int64)
    for alt_index, alt_name in enumerate(train_alternatives):
        for view in ("score_only", "score_hidden_confidence"):
            for prototype_count in prototype_counts:
                for neighbor_k in neighbor_ks:
                    for top_k in top_ks:
                        receiver = _run_prototype_receiver(
                            train_features=train_features[(alt_name, view)],
                            eval_features=eval_features[(alt_name, view)],
                            packet_train=calibration["tiny_packet"],
                            packet_eval=validation["packet"],
                            alt_train=train_alternatives[alt_name],
                            alt_eval=eval_alternatives[alt_name],
                            answers_train=calibration["answers"],
                            fit_indices=fit_indices,
                            dev_indices=dev_indices,
                            prototype_count=int(prototype_count),
                            neighbor_k=int(neighbor_k),
                            top_k=int(top_k),
                            control_kind="real",
                            seed=DEFAULT_CONTROL_SEED + int(prototype_count) + int(neighbor_k) + int(top_k),
                        )
                        stats = accept._receiver_stats(
                            predictions=receiver["predictions"],
                            packet_predictions=validation["packet"],
                            target_predictions=eval_alternatives["qwen_target_score"],
                            answers=validation["answers"],
                            indices=eval_indices,
                            seed=10100
                            + 1009 * alt_index
                            + 101 * int(prototype_count)
                            + 11 * int(neighbor_k)
                            + int(top_k),
                            samples=bootstrap_samples,
                        )
                        key = (
                            alt_name,
                            view,
                            int(prototype_count),
                            int(neighbor_k),
                            int(top_k),
                        )
                        prediction_cache[key] = receiver["predictions"]
                        frontier_rows.append(
                            {
                                "alternative": alt_name,
                                "feature_view": view,
                                "method": receiver["method"],
                                "control_kind": receiver["control_kind"],
                                "prototype_kind": receiver["prototype_kind"],
                                "prototype_count": receiver["prototype_count"],
                                "actual_prototype_count": receiver["actual_prototype_count"],
                                "prototype_pool_rows": receiver["prototype_pool_rows"],
                                "neighbor_k": receiver["neighbor_k"],
                                "top_k": receiver["top_k"],
                                "threshold": receiver["threshold"],
                                "official_fit_rows": int(len(fit_indices)),
                                "official_dev_rows": int(len(dev_indices)),
                                "official_dev_accuracy": receiver["dev_accuracy"],
                                "official_dev_override_rate": receiver["dev_override_rate"],
                                "official_dev_help_count": receiver["dev_help_count"],
                                "official_dev_harm_count": receiver["dev_harm_count"],
                                "validation_override_rate": receiver["eval_override_rate"],
                                "validation_mean_top_similarity": receiver["eval_mean_top_similarity"],
                                "validation_mean_local_packet_correct_rate": receiver[
                                    "eval_mean_local_packet_correct_rate"
                                ],
                                "validation_mean_local_alt_correct_rate": receiver[
                                    "eval_mean_local_alt_correct_rate"
                                ],
                                **stats,
                            }
                        )
    selector_wall_time_s = time.perf_counter() - selector_start

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
        and row["prototype_count"] == 64
        and row["neighbor_k"] == 25
        and row["top_k"] == 5
    ]
    if not default_candidates:
        raise ValueError("missing predeclared default disagreement-prototype row")
    default_row = default_candidates[0]
    default_key = (
        default_row["alternative"],
        default_row["feature_view"],
        default_row["prototype_count"],
        default_row["neighbor_k"],
        default_row["top_k"],
    )
    default_predictions = prediction_cache[default_key]
    default_blocks = accept._contiguous_block_deltas(
        predictions=default_predictions,
        packet_predictions=validation["packet"],
        answers=validation["answers"],
        eval_indices=eval_indices,
    )

    control_rows: list[dict[str, Any]] = []
    for offset, control_kind in enumerate(("random_prototypes", "label_permutation", "score_row_shuffle")):
        receiver = _run_prototype_receiver(
            train_features=train_features[(default_row["alternative"], default_row["feature_view"])],
            eval_features=eval_features[(default_row["alternative"], default_row["feature_view"])],
            packet_train=calibration["tiny_packet"],
            packet_eval=validation["packet"],
            alt_train=train_alternatives[default_row["alternative"]],
            alt_eval=eval_alternatives[default_row["alternative"]],
            answers_train=calibration["answers"],
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            prototype_count=default_row["prototype_count"],
            neighbor_k=default_row["neighbor_k"],
            top_k=default_row["top_k"],
            control_kind=control_kind,
            seed=DEFAULT_CONTROL_SEED + 100 * offset,
        )
        stats = accept._receiver_stats(
            predictions=receiver["predictions"],
            packet_predictions=validation["packet"],
            target_predictions=eval_alternatives["qwen_target_score"],
            answers=validation["answers"],
            indices=eval_indices,
            seed=12100 + offset,
            samples=bootstrap_samples,
        )
        control_rows.append(
            {
                "name": control_kind,
                "method": receiver["method"],
                "prototype_count": receiver["prototype_count"],
                "neighbor_k": receiver["neighbor_k"],
                "top_k": receiver["top_k"],
                "official_dev_accuracy": receiver["dev_accuracy"],
                "official_dev_override_rate": receiver["dev_override_rate"],
                "validation_override_rate": receiver["eval_override_rate"],
                **stats,
            }
        )

    control_max_delta = max(row["delta_vs_packet_only"] for row in control_rows)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    predeclared_default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
    )
    target_transfer_gate = bool(
        default_row["delta_vs_target_only"] >= STRICT_TARGET_DELTA
        and default_row["ci95_low_vs_target_only"] > 0.0
    )
    block_stability_gate = bool(all(row["delta_vs_packet_only"] > 0.0 for row in default_blocks))
    pass_gate = bool(
        predeclared_default_pass_gate
        and target_transfer_gate
        and block_stability_gate
        and control_separation_gate
    )

    validation_oracle = _oracle_accuracy(
        [validation["packet"], eval_alternatives["hybrid_vote_on_score_agreement_prediction"]],
        validation["answers"],
    )
    train_oracle = _oracle_accuracy(
        [calibration["tiny_packet"], calibration["qwen_hybrid"]],
        calibration["answers"],
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "validation_rows": int(len(validation["answers"])),
        "sample_seeds": list(sample_seeds),
        "default_method": default_row["method"],
        "default_alternative": default_row["alternative"],
        "default_feature_view": default_row["feature_view"],
        "default_prototype_count": default_row["prototype_count"],
        "default_neighbor_k": default_row["neighbor_k"],
        "default_top_k": default_row["top_k"],
        "default_eval_accuracy": default_row["accuracy"],
        "default_packet_only_accuracy": default_row["packet_only_accuracy"],
        "default_target_only_accuracy": default_row["target_only_accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_target_only": default_row["delta_vs_target_only"],
        "default_ci95_low_vs_target_only": default_row["ci95_low_vs_target_only"],
        "best_scout_method": best_scout["method"],
        "best_scout_alternative": best_scout["alternative"],
        "best_scout_feature_view": best_scout["feature_view"],
        "best_scout_prototype_count": best_scout["prototype_count"],
        "best_scout_neighbor_k": best_scout["neighbor_k"],
        "best_scout_top_k": best_scout["top_k"],
        "best_scout_eval_accuracy": best_scout["accuracy"],
        "best_scout_packet_only_accuracy": best_scout["packet_only_accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "control_max_delta_vs_packet_only": control_max_delta,
        "official_train_tiny_packet_accuracy": _baseline_accuracy(
            calibration["tiny_packet"],
            calibration["answers"],
        ),
        "official_train_qwen_hybrid_accuracy": _baseline_accuracy(
            calibration["qwen_hybrid"],
            calibration["answers"],
        ),
        "official_train_tiny_qwen_hybrid_oracle_accuracy": train_oracle,
        "validation_tiny_packet_accuracy": _baseline_accuracy(validation["packet"], validation["answers"]),
        "validation_qwen_hybrid_accuracy": _baseline_accuracy(
            eval_alternatives["hybrid_vote_on_score_agreement_prediction"],
            validation["answers"],
        ),
        "validation_tiny_qwen_hybrid_oracle_accuracy": validation_oracle,
        "strict_delta_required": STRICT_DELTA,
        "control_separation_delta_required": CONTROL_SEPARATION_DELTA,
        "packet_raw_bytes": decomp.RAW_PACKET_BYTES,
        "packet_framed_bytes": decomp.FRAMED_PACKET_BYTES,
        "native_gpu_claims_allowed": False,
    }
    lay_explanation = (
        "The scalar receiver asked whether the packet or Qwen looked globally more confident. This "
        "experiment instead builds train-only prototypes of rows where TinyLlama and Qwen disagree. At "
        "validation time, it asks whether a row is near past disagreement types where Qwen helped more "
        "than it harmed. Random, label-permuted, and score-row-shuffled controls test whether any lift "
        "comes from real local disagreement structure."
    )
    interpretation = (
        "This is a common-basis probe, not a learned soft-prompt or KV-cache fusion method. If it passes, "
        "it promotes local disagreement prototypes as a receiver mechanism. If it fails, the evidence says "
        "the remaining Tiny/Qwen oracle headroom likely needs a richer sparse/crosscoder or learned "
        "query-bottleneck receiver rather than local prototype thresholds."
    )
    payload = {
        "gate": "source_private_hellaswag_disagreement_prototype_receiver",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "predeclared_default_pass_gate": predeclared_default_pass_gate,
        "target_transfer_gate": target_transfer_gate,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "pass_rule": (
            "Strict promotion requires the predeclared official-train disagreement-prototype receiver "
            "to beat TinyLlama packet-only on full validation by >=0.005 with positive paired CI95 low, "
            "beat Qwen target-only by >=0.02, remain positive across contiguous validation blocks, and "
            "separate from random/label-permuted/score-shuffled controls by >=0.003 delta."
        ),
        "headline": headline,
        "frontier_rows": frontier_rows,
        "control_rows": control_rows,
        "default_block_rows": default_blocks,
        "sample_cache_rows": calibration_audit["sample_cache_rows"],
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": decomp.RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": decomp.FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(
                decomp.RAW_PACKET_BYTES * len(validation["answers"])
            ),
            "logical_validation_framed_record_bytes_total": int(
                decomp.FRAMED_PACKET_BYTES * len(validation["answers"])
            ),
            "official_train_calibration_wall_time_s": float(calibration_wall_time_s),
            "feature_build_wall_time_s": float(feature_build_wall_time_s),
            "selector_wall_time_s": float(selector_wall_time_s),
            "total_wall_time_s": float(time.perf_counter() - started),
            "prototype_grid_rows": int(len(frontier_rows)),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
        },
        "inputs": {
            "train_path": _display_path(train_path),
            "train_sha256": _sha256_file(train_path),
            "tiny_train_cache_dir": _display_path(tiny_train_cache_dir),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "tiny_eval_packet_jsonl": _display_path(tiny_eval_packet_jsonl),
            "tiny_eval_packet_jsonl_sha256": _sha256_file(tiny_eval_packet_jsonl),
            "tiny_eval_artifact": _display_path(tiny_eval_artifact),
            "tiny_eval_artifact_sha256": _sha256_file(tiny_eval_artifact),
            "qwen_eval_packet_jsonl": _display_path(qwen_eval_packet_jsonl),
            "qwen_eval_packet_jsonl_sha256": _sha256_file(qwen_eval_packet_jsonl),
            "qwen_global_artifact": _display_path(qwen_global_artifact),
            "qwen_global_artifact_sha256": _sha256_file(qwen_global_artifact),
        },
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_disagreement_prototype_receiver.json"
    md_path = output_dir / "hellaswag_disagreement_prototype_receiver.md"
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
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--tiny-train-cache-dir", type=pathlib.Path, default=DEFAULT_TINY_TRAIN_CACHE_DIR)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--tiny-eval-packet-jsonl", type=pathlib.Path, default=DEFAULT_TINY_EVAL_PACKET_JSONL)
    parser.add_argument("--tiny-eval-artifact", type=pathlib.Path, default=DEFAULT_TINY_EVAL_ARTIFACT)
    parser.add_argument("--qwen-eval-packet-jsonl", type=pathlib.Path, default=DEFAULT_QWEN_EVAL_PACKET_JSONL)
    parser.add_argument("--qwen-global-artifact", type=pathlib.Path, default=DEFAULT_QWEN_GLOBAL_ARTIFACT)
    parser.add_argument("--sample-seeds", type=_parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--prototype-counts", type=_parse_int_tuple, default=DEFAULT_PROTOTYPE_COUNTS)
    parser.add_argument("--neighbor-ks", type=_parse_int_tuple, default=DEFAULT_NEIGHBOR_KS)
    parser.add_argument("--top-ks", type=_parse_int_tuple, default=DEFAULT_TOP_KS)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--tiny-aggregation-policy", default="mean_zscore")
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        tiny_train_cache_dir=args.tiny_train_cache_dir,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        tiny_eval_packet_jsonl=args.tiny_eval_packet_jsonl,
        tiny_eval_artifact=args.tiny_eval_artifact,
        qwen_eval_packet_jsonl=args.qwen_eval_packet_jsonl,
        qwen_global_artifact=args.qwen_global_artifact,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        prototype_counts=args.prototype_counts,
        neighbor_ks=args.neighbor_ks,
        top_ks=args.top_ks,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        bootstrap_samples=args.bootstrap_samples,
        tiny_aggregation_policy=args.tiny_aggregation_policy,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
