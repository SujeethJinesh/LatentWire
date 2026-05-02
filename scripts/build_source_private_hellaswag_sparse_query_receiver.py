from __future__ import annotations

"""Train-only sparse-query receiver for HellaSwag source-private packets."""

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

from scripts import build_source_private_hellaswag_disagreement_prototype_receiver as proto  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_acceptance_gate as accept  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_headroom_decomposition as decomp  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_sparse_query_receiver_20260502")
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
DEFAULT_RECEIVER_RIDGES = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
DEFAULT_QUERY_COUNTS = (4, 8, 16, 32, 64)
DEFAULT_ACTIVE_QUERIES = (0, 4, 8, 16)
DEFAULT_CONTROL_SEED = 12017
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


def _gather_candidate(hidden: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = np.arange(hidden.shape[0], dtype=np.int64)
    return hidden[rows, indices.astype(np.int64)]


def _candidate_residual_design(
    *,
    scores: np.ndarray,
    hidden: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0].astype(np.int64)
    top2 = order[:, 1].astype(np.int64)
    packet_h = _gather_candidate(hidden, packet_predictions).astype(np.float32)
    alt_h = _gather_candidate(hidden, alt_predictions).astype(np.float32)
    top1_h = _gather_candidate(hidden, top1).astype(np.float32)
    top2_h = _gather_candidate(hidden, top2).astype(np.float32)
    mean_h = np.mean(hidden, axis=1).astype(np.float32)
    return np.concatenate(
        [
            alt_h - packet_h,
            top1_h - packet_h,
            alt_h - top1_h,
            top1_h - top2_h,
            alt_h - mean_h,
            packet_h - mean_h,
        ],
        axis=1,
    ).astype(np.float32)


def _standardize_fit(
    *,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit = train_features[fit_indices].astype(np.float64)
    mean = np.mean(fit, axis=0).astype(np.float32)
    scale = np.std(fit, axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    train_std = ((train_features - mean) / scale).astype(np.float32)
    eval_std = ((eval_features - mean) / scale).astype(np.float32)
    return train_std, eval_std, mean, scale


def _normalize_basis(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.where(norms < 1e-8, 1.0, norms)).astype(np.float32)


def _build_sparse_query_basis(
    *,
    train_design: np.ndarray,
    benefit: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    max_query_count: int,
    seed: int,
    control_kind: str = "real",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = benefit.astype(np.float32).copy()
    if control_kind == "label_permutation":
        y = y[rng.permutation(len(y))]

    x_fit = train_design[fit_indices].astype(np.float32)
    y_fit = y[fit_indices].astype(np.float32)
    packet_correct = (packet_predictions == answers).astype(np.float32)
    alt_correct = (alt_predictions == answers).astype(np.float32)
    dirs: list[np.ndarray] = []

    if control_kind == "random_query_basis":
        random = rng.normal(size=(max_query_count, train_design.shape[1])).astype(np.float32)
        return _normalize_basis(random)

    benefit_dir = x_fit.T @ y_fit
    dirs.append(benefit_dir[None, :])
    pos = x_fit[y_fit > 0]
    neg = x_fit[y_fit < 0]
    if len(pos) and len(neg):
        dirs.append((np.mean(pos, axis=0) - np.mean(neg, axis=0))[None, :])
    alt_pos = train_design[fit_indices][alt_correct[fit_indices] > 0]
    alt_neg = train_design[fit_indices][alt_correct[fit_indices] <= 0]
    if len(alt_pos) and len(alt_neg):
        dirs.append((np.mean(alt_pos, axis=0) - np.mean(alt_neg, axis=0))[None, :])
    packet_pos = train_design[fit_indices][packet_correct[fit_indices] > 0]
    packet_neg = train_design[fit_indices][packet_correct[fit_indices] <= 0]
    if len(packet_pos) and len(packet_neg):
        dirs.append((np.mean(packet_neg, axis=0) - np.mean(packet_pos, axis=0))[None, :])

    random_width = max(max_query_count * 2, 8)
    random = rng.normal(size=(train_design.shape[1], random_width)).astype(np.float32)
    sketch = x_fit.T @ (x_fit @ random)
    q, _ = np.linalg.qr(sketch.astype(np.float64), mode="reduced")
    dirs.append(q[:, : max(1, max_query_count)].T.astype(np.float32))

    basis = _normalize_basis(np.vstack(dirs))
    if basis.shape[0] < max_query_count:
        random_extra = rng.normal(
            size=(max_query_count - basis.shape[0], train_design.shape[1])
        ).astype(np.float32)
        basis = np.vstack([basis, _normalize_basis(random_extra)])
    return basis[:max_query_count].astype(np.float32)


def _top_abs_sparse(values: np.ndarray, active_queries: int) -> np.ndarray:
    active_queries = int(active_queries)
    if active_queries <= 0 or active_queries >= values.shape[1]:
        return values.astype(np.float64)
    sparse = np.zeros_like(values, dtype=np.float32)
    indices = np.argpartition(np.abs(values), -active_queries, axis=1)[:, -active_queries:]
    rows = np.arange(values.shape[0])[:, None]
    sparse[rows, indices] = values[rows, indices]
    return sparse.astype(np.float64)


def _run_sparse_query_receiver(
    *,
    train_scalar_features: np.ndarray,
    eval_scalar_features: np.ndarray,
    train_query_features: np.ndarray,
    eval_query_features: np.ndarray,
    benefit: np.ndarray,
    packet_train: np.ndarray,
    packet_eval: np.ndarray,
    alt_train: np.ndarray,
    alt_eval: np.ndarray,
    answers_train: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    ridges: tuple[float, ...],
) -> dict[str, Any]:
    train_features = np.concatenate([train_scalar_features, train_query_features], axis=1).astype(
        np.float64
    )
    eval_features = np.concatenate([eval_scalar_features, eval_query_features], axis=1).astype(np.float64)
    candidates: list[dict[str, Any]] = []
    for model in accept._fit_benefit_ridge(
        features=train_features,
        benefit=benefit,
        fit_indices=fit_indices,
        ridges=ridges,
    ):
        train_scores = accept._score_benefit_ridge(model, train_features)
        selected = accept._select_threshold(
            scores=train_scores,
            packet_predictions=packet_train,
            alt_predictions=alt_train,
            answers=answers_train,
            dev_indices=dev_indices,
        )
        eval_scores = accept._score_benefit_ridge(model, eval_features)
        use_alt_eval = (eval_scores > selected["threshold"]) & (alt_eval != packet_eval)
        predictions = accept._task_predictions(
            packet_predictions=packet_eval,
            alt_predictions=alt_eval,
            use_alt=use_alt_eval,
        )
        candidates.append(
            {
                "method": "sparse_query_benefit_ridge",
                "ridge": model["ridge"],
                "threshold": selected["threshold"],
                "dev_accuracy": selected["dev_accuracy"],
                "dev_override_rate": selected["dev_override_rate"],
                "dev_help_count": selected["dev_help_count"],
                "dev_harm_count": selected["dev_harm_count"],
                "eval_override_rate": float(np.mean(use_alt_eval)),
                "eval_mean_query_score": float(np.mean(eval_scores)),
                "predictions": predictions,
                "scores": eval_scores,
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["dev_accuracy"],
            row["dev_help_count"] - row["dev_harm_count"],
            -row["dev_override_rate"],
            -row["ridge"],
        ),
    )


def _baseline_accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _oracle_accuracy(predictions: list[np.ndarray], answers: np.ndarray) -> float:
    correct = np.zeros_like(answers, dtype=bool)
    for item in predictions:
        correct |= item == answers
    return float(np.mean(correct))


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Sparse-Query Receiver",
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
    receiver_ridges: tuple[float, ...] = DEFAULT_RECEIVER_RIDGES,
    query_counts: tuple[int, ...] = DEFAULT_QUERY_COUNTS,
    active_queries: tuple[int, ...] = DEFAULT_ACTIVE_QUERIES,
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
    calibration, fit_indices, dev_indices, calibration_audit = proto._load_official_train_calibration(
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
    validation = proto._load_validation_bundle(
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
    train_hidden_design: dict[str, np.ndarray] = {}
    eval_hidden_design: dict[str, np.ndarray] = {}
    train_scalar: dict[tuple[str, str], np.ndarray] = {}
    eval_scalar: dict[tuple[str, str], np.ndarray] = {}
    standardized_hidden: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bases: dict[tuple[str, str], np.ndarray] = {}
    max_query_count = int(max(query_counts))
    for alt_name, alt_train in train_alternatives.items():
        alt_eval = eval_alternatives[alt_name]
        train_hidden_design[alt_name] = _candidate_residual_design(
            scores=calibration["qwen_scores"],
            hidden=calibration["qwen_hidden"],
            packet_predictions=calibration["tiny_packet"],
            alt_predictions=alt_train,
        )
        eval_hidden_design[alt_name] = _candidate_residual_design(
            scores=validation["qwen_scores"],
            hidden=validation["qwen_hidden"],
            packet_predictions=validation["packet"],
            alt_predictions=alt_eval,
        )
        train_std, eval_std, _, _ = _standardize_fit(
            train_features=train_hidden_design[alt_name],
            eval_features=eval_hidden_design[alt_name],
            fit_indices=fit_indices,
        )
        standardized_hidden[alt_name] = (train_std, eval_std)
        benefit = accept._benefit_values(
            alt_predictions=alt_train,
            packet_predictions=calibration["tiny_packet"],
            answers=calibration["answers"],
        )
        bases[(alt_name, "real")] = _build_sparse_query_basis(
            train_design=train_std,
            benefit=benefit,
            packet_predictions=calibration["tiny_packet"],
            alt_predictions=alt_train,
            answers=calibration["answers"],
            fit_indices=fit_indices,
            max_query_count=max_query_count,
            seed=DEFAULT_CONTROL_SEED,
        )
        for scalar_view in ("score_only", "score_hidden_confidence"):
            train_scalar[(alt_name, scalar_view)] = accept._feature_matrix(
                scores=calibration["qwen_scores"],
                hidden=calibration["qwen_hidden"],
                packet_predictions=calibration["tiny_packet"],
                packet_margins=calibration["tiny_margin"],
                alt_predictions=alt_train,
                view=scalar_view,
            )
            eval_scalar[(alt_name, scalar_view)] = accept._feature_matrix(
                scores=validation["qwen_scores"],
                hidden=validation["qwen_hidden"],
                packet_predictions=validation["packet"],
                packet_margins=validation["packet_margin"],
                alt_predictions=alt_eval,
                view=scalar_view,
            )
    feature_build_wall_time_s = time.perf_counter() - feature_start

    frontier_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
    selector_start = time.perf_counter()
    eval_indices = np.arange(len(validation["answers"]), dtype=np.int64)
    for alt_index, (alt_name, alt_train) in enumerate(train_alternatives.items()):
        alt_eval = eval_alternatives[alt_name]
        benefit = accept._benefit_values(
            alt_predictions=alt_train,
            packet_predictions=calibration["tiny_packet"],
            answers=calibration["answers"],
        )
        train_std, eval_std = standardized_hidden[alt_name]
        query_projection_train = train_std @ bases[(alt_name, "real")].T
        query_projection_eval = eval_std @ bases[(alt_name, "real")].T
        for scalar_index, scalar_view in enumerate(("score_only", "score_hidden_confidence")):
            for query_count in query_counts:
                for active_query_count in active_queries:
                    if int(active_query_count) > int(query_count):
                        continue
                    train_query = _top_abs_sparse(
                        query_projection_train[:, : int(query_count)],
                        int(active_query_count),
                    )
                    eval_query = _top_abs_sparse(
                        query_projection_eval[:, : int(query_count)],
                        int(active_query_count),
                    )
                    receiver = _run_sparse_query_receiver(
                        train_scalar_features=train_scalar[(alt_name, scalar_view)],
                        eval_scalar_features=eval_scalar[(alt_name, scalar_view)],
                        train_query_features=train_query,
                        eval_query_features=eval_query,
                        benefit=benefit,
                        packet_train=calibration["tiny_packet"],
                        packet_eval=validation["packet"],
                        alt_train=alt_train,
                        alt_eval=alt_eval,
                        answers_train=calibration["answers"],
                        fit_indices=fit_indices,
                        dev_indices=dev_indices,
                        ridges=receiver_ridges,
                    )
                    stats = accept._receiver_stats(
                        predictions=receiver["predictions"],
                        packet_predictions=validation["packet"],
                        target_predictions=eval_alternatives["qwen_target_score"],
                        answers=validation["answers"],
                        indices=eval_indices,
                        seed=13200
                        + 1009 * alt_index
                        + 101 * scalar_index
                        + 11 * int(query_count)
                        + int(active_query_count),
                        samples=bootstrap_samples,
                    )
                    key = (alt_name, scalar_view, int(query_count), int(active_query_count))
                    prediction_cache[key] = receiver["predictions"]
                    frontier_rows.append(
                        {
                            "alternative": alt_name,
                            "scalar_view": scalar_view,
                            "method": receiver["method"],
                            "query_basis": "supervised_plus_randomized_pca",
                            "query_count": int(query_count),
                            "active_queries": int(active_query_count),
                            "ridge": receiver["ridge"],
                            "threshold": receiver["threshold"],
                            "official_fit_rows": int(len(fit_indices)),
                            "official_dev_rows": int(len(dev_indices)),
                            "official_dev_accuracy": receiver["dev_accuracy"],
                            "official_dev_override_rate": receiver["dev_override_rate"],
                            "official_dev_help_count": receiver["dev_help_count"],
                            "official_dev_harm_count": receiver["dev_harm_count"],
                            "validation_override_rate": receiver["eval_override_rate"],
                            "validation_mean_query_score": receiver["eval_mean_query_score"],
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
        if row["alternative"] == "mean_zscore_prediction"
        and row["scalar_view"] == "score_hidden_confidence"
        and row["query_count"] == 16
        and row["active_queries"] == 4
    ]
    if not default_candidates:
        raise ValueError("missing predeclared sparse-query default row")
    default_row = default_candidates[0]
    default_key = (
        default_row["alternative"],
        default_row["scalar_view"],
        default_row["query_count"],
        default_row["active_queries"],
    )
    default_predictions = prediction_cache[default_key]
    default_blocks = accept._contiguous_block_deltas(
        predictions=default_predictions,
        packet_predictions=validation["packet"],
        answers=validation["answers"],
        eval_indices=eval_indices,
    )

    control_rows: list[dict[str, Any]] = []
    default_alt_name = default_row["alternative"]
    default_alt_train = train_alternatives[default_alt_name]
    default_alt_eval = eval_alternatives[default_alt_name]
    default_benefit = accept._benefit_values(
        alt_predictions=default_alt_train,
        packet_predictions=calibration["tiny_packet"],
        answers=calibration["answers"],
    )
    for offset, control_kind in enumerate(("random_query_basis", "label_permutation", "hidden_row_shuffle")):
        rng = np.random.default_rng(DEFAULT_CONTROL_SEED + 200 * offset)
        train_design = train_hidden_design[default_alt_name]
        eval_design = eval_hidden_design[default_alt_name]
        if control_kind == "hidden_row_shuffle":
            train_design = train_design[rng.permutation(len(train_design))]
            eval_design = eval_design[rng.permutation(len(eval_design))]
        train_std, eval_std, _, _ = _standardize_fit(
            train_features=train_design,
            eval_features=eval_design,
            fit_indices=fit_indices,
        )
        basis = _build_sparse_query_basis(
            train_design=train_std,
            benefit=default_benefit,
            packet_predictions=calibration["tiny_packet"],
            alt_predictions=default_alt_train,
            answers=calibration["answers"],
            fit_indices=fit_indices,
            max_query_count=int(default_row["query_count"]),
            seed=DEFAULT_CONTROL_SEED + 300 * offset,
            control_kind=control_kind,
        )
        train_query = _top_abs_sparse(
            train_std @ basis.T,
            int(default_row["active_queries"]),
        )
        eval_query = _top_abs_sparse(
            eval_std @ basis.T,
            int(default_row["active_queries"]),
        )
        benefit_for_fit = default_benefit
        if control_kind == "label_permutation":
            benefit_for_fit = default_benefit[rng.permutation(len(default_benefit))]
        receiver = _run_sparse_query_receiver(
            train_scalar_features=train_scalar[(default_alt_name, default_row["scalar_view"])],
            eval_scalar_features=eval_scalar[(default_alt_name, default_row["scalar_view"])],
            train_query_features=train_query,
            eval_query_features=eval_query,
            benefit=benefit_for_fit,
            packet_train=calibration["tiny_packet"],
            packet_eval=validation["packet"],
            alt_train=default_alt_train,
            alt_eval=default_alt_eval,
            answers_train=calibration["answers"],
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            ridges=receiver_ridges,
        )
        stats = accept._receiver_stats(
            predictions=receiver["predictions"],
            packet_predictions=validation["packet"],
            target_predictions=eval_alternatives["qwen_target_score"],
            answers=validation["answers"],
            indices=eval_indices,
            seed=15100 + offset,
            samples=bootstrap_samples,
        )
        control_rows.append(
            {
                "name": control_kind,
                "method": receiver["method"],
                "query_count": int(default_row["query_count"]),
                "active_queries": int(default_row["active_queries"]),
                "ridge": receiver["ridge"],
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
        "hidden_design_dimension": int(next(iter(train_hidden_design.values())).shape[1]),
        "default_method": default_row["method"],
        "default_alternative": default_row["alternative"],
        "default_scalar_view": default_row["scalar_view"],
        "default_query_basis": default_row["query_basis"],
        "default_query_count": default_row["query_count"],
        "default_active_queries": default_row["active_queries"],
        "default_eval_accuracy": default_row["accuracy"],
        "default_packet_only_accuracy": default_row["packet_only_accuracy"],
        "default_target_only_accuracy": default_row["target_only_accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_target_only": default_row["delta_vs_target_only"],
        "default_ci95_low_vs_target_only": default_row["ci95_low_vs_target_only"],
        "best_scout_method": best_scout["method"],
        "best_scout_alternative": best_scout["alternative"],
        "best_scout_scalar_view": best_scout["scalar_view"],
        "best_scout_query_basis": best_scout["query_basis"],
        "best_scout_query_count": best_scout["query_count"],
        "best_scout_active_queries": best_scout["active_queries"],
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
        "Previous receivers only looked at score summaries or local prototypes. This experiment "
        "looks at the full hidden-state difference between Qwen's candidate and the TinyLlama packet "
        "candidate, compresses that large residual into a tiny set of train-only query features, and "
        "learns when those features say Qwen should override the packet."
    )
    interpretation = (
        "This is a sparse-query common-basis probe, not C2C, prefix tuning, or KV-cache transfer. "
        "A pass would promote full candidate residual queries as the missing receiver mechanism. A "
        "failure would mean that even a low-rank view of the full Qwen candidate residuals cannot close "
        "the Tiny/Qwen oracle on the current official-train calibration surface."
    )
    payload = {
        "gate": "source_private_hellaswag_sparse_query_receiver",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "predeclared_default_pass_gate": predeclared_default_pass_gate,
        "target_transfer_gate": target_transfer_gate,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "pass_rule": (
            "Strict promotion requires the predeclared official-train sparse-query receiver to beat "
            "TinyLlama packet-only on full validation by >=0.005 with positive paired CI95 low, beat "
            "Qwen target-only by >=0.02, remain positive across contiguous validation blocks, and "
            "separate from random-query, label-permuted, and hidden-row-shuffled controls by >=0.003 delta."
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
            "frontier_rows": int(len(frontier_rows)),
            "hidden_design_dimension": int(headline["hidden_design_dimension"]),
            "max_query_count": int(max_query_count),
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
    json_path = output_dir / "hellaswag_sparse_query_receiver.json"
    md_path = output_dir / "hellaswag_sparse_query_receiver.md"
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
    parser.add_argument("--receiver-ridges", type=_parse_float_tuple, default=DEFAULT_RECEIVER_RIDGES)
    parser.add_argument("--query-counts", type=_parse_int_tuple, default=DEFAULT_QUERY_COUNTS)
    parser.add_argument("--active-queries", type=_parse_int_tuple, default=DEFAULT_ACTIVE_QUERIES)
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
        receiver_ridges=args.receiver_ridges,
        query_counts=args.query_counts,
        active_queries=args.active_queries,
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
