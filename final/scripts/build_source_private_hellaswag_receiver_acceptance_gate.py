from __future__ import annotations

"""Train-only receiver acceptance gate for HellaSwag source-private packets."""

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

from scripts import build_source_private_hellaswag_receiver_headroom_decomposition as decomp  # noqa: E402

DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_receiver_acceptance_gate_20260502")
DEFAULT_TINY_PACKET_JSONL = decomp.DEFAULT_TINY_PACKET_JSONL
DEFAULT_TINY_ARTIFACT = decomp.DEFAULT_TINY_ARTIFACT
DEFAULT_QWEN_PACKET_JSONL = decomp.DEFAULT_QWEN_PACKET_JSONL
DEFAULT_QWEN_GLOBAL_ARTIFACT = decomp.DEFAULT_QWEN_GLOBAL_ARTIFACT
DEFAULT_TINY_FIELD = decomp.DEFAULT_TINY_FIELD
DEFAULT_TINY_MARGIN_FIELD = decomp.DEFAULT_TINY_MARGIN_FIELD
DEFAULT_QWEN_FIELDS = decomp.DEFAULT_QWEN_FIELDS
DEFAULT_CONTROL_FIELDS = decomp.DEFAULT_CONTROL_FIELDS
DEFAULT_TRAIN_PREFIXES = (2048, 4096, 6144)
DEFAULT_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_K_VALUES = (5, 11, 21, 51, 101)
STRICT_DELTA = 0.005
STRICT_TARGET_DELTA = 0.02


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return decomp._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return decomp._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    return decomp._sha256_file(path)


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return decomp._read_json(path)


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    return decomp._read_jsonl(path)


def _accuracy(predictions: np.ndarray, answers: np.ndarray, indices: np.ndarray) -> float:
    return decomp._accuracy(predictions, answers, indices)


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    return decomp._paired_ci(
        selected=selected,
        baseline=baseline,
        answers=answers,
        indices=indices,
        seed=seed,
        samples=samples,
    )


def _read_prediction(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return decomp._read_prediction(rows, field)


def _parse_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


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


def _load_qwen_bundle(global_artifact: pathlib.Path | str) -> dict[str, Any]:
    artifact = _read_json(global_artifact)
    row_ids: list[str] = []
    scores: list[list[float]] = []
    predictions: list[int] = []
    hidden_parts: list[np.ndarray] = []
    slices: list[dict[str, Any]] = []
    for item in artifact.get("eval_slices", []):
        score_cache = _read_json(item["score_cache"])
        row_ids.extend(str(value) for value in score_cache["row_ids"])
        scores.extend(score_cache["source_scores"])
        predictions.extend(int(value) for value in score_cache["source_predictions"])
        hidden_path = _resolve(item["hidden_cache"])
        hidden = np.load(hidden_path)["features"][:, :, 0, :].astype(np.float32)
        hidden_parts.append(hidden)
        slices.append(
            {
                "name": item.get("name"),
                "start": item.get("start"),
                "end": item.get("end"),
                "rows": item.get("rows"),
                "score_cache": _display_path(item["score_cache"]),
                "score_cache_sha256": _sha256_file(item["score_cache"]),
                "hidden_cache": _display_path(hidden_path),
                "hidden_cache_sha256": _sha256_file(hidden_path),
                "hidden_cache_bytes": hidden_path.stat().st_size,
            }
        )
    if not scores:
        raise ValueError(f"no eval_slices found in {global_artifact}")
    return {
        "row_ids": row_ids,
        "scores": np.asarray(scores, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "hidden": np.concatenate(hidden_parts, axis=0).astype(np.float32),
        "slices": slices,
        "artifact_path": _display_path(global_artifact),
        "artifact_sha256": _sha256_file(global_artifact),
    }


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.zeros_like(order)
    for row_index in range(order.shape[0]):
        for rank, candidate in enumerate(order[row_index]):
            ranks[row_index, candidate] = rank
    return ranks


def _normalize_matrix(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms < 1e-8, 1.0, norms)


def _candidate_one_hot(indices: np.ndarray) -> np.ndarray:
    return np.eye(4, dtype=np.float64)[indices.astype(np.int64)]


def _hidden_confidence_features(
    *,
    scores: np.ndarray,
    hidden: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
) -> np.ndarray:
    row_ids = np.arange(scores.shape[0], dtype=np.int64)
    top1 = np.argmax(scores, axis=1).astype(np.int64)
    top2 = np.argsort(-scores, axis=1)[:, 1].astype(np.int64)
    norms = np.linalg.norm(hidden, axis=2)
    hidden_normed = hidden / np.where(norms[:, :, None] < 1e-8, 1.0, norms[:, :, None])

    def gather(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
        return values[row_ids, ids]

    packet_h = gather(hidden, packet_predictions)
    alt_h = gather(hidden, alt_predictions)
    top1_h = gather(hidden, top1)
    top2_h = gather(hidden, top2)
    packet_n = gather(hidden_normed, packet_predictions)
    alt_n = gather(hidden_normed, alt_predictions)
    top1_n = gather(hidden_normed, top1)
    top2_n = gather(hidden_normed, top2)

    cos_alt_packet = np.sum(alt_n * packet_n, axis=1)
    cos_top_packet = np.sum(top1_n * packet_n, axis=1)
    cos_alt_top = np.sum(alt_n * top1_n, axis=1)
    cos_top_top2 = np.sum(top1_n * top2_n, axis=1)
    dist_alt_packet = np.linalg.norm(alt_h - packet_h, axis=1)
    dist_top_packet = np.linalg.norm(top1_h - packet_h, axis=1)
    dist_top_top2 = np.linalg.norm(top1_h - top2_h, axis=1)
    packet_norm = gather(norms, packet_predictions)
    alt_norm = gather(norms, alt_predictions)
    top1_norm = gather(norms, top1)
    top2_norm = gather(norms, top2)
    return np.stack(
        [
            cos_alt_packet,
            cos_top_packet,
            cos_alt_top,
            cos_top_top2,
            dist_alt_packet,
            dist_top_packet,
            dist_top_top2,
            packet_norm,
            alt_norm,
            top1_norm,
            top2_norm,
            alt_norm - packet_norm,
            top1_norm - packet_norm,
        ],
        axis=1,
    ).astype(np.float64)


def _feature_matrix(
    *,
    scores: np.ndarray,
    hidden: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    alt_predictions: np.ndarray,
    view: str,
) -> np.ndarray:
    row_ids = np.arange(scores.shape[0], dtype=np.int64)
    target_predictions = np.argmax(scores, axis=1).astype(np.int64)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(scores, axis=1, keepdims=True)
    zscores = centered / np.where(scale < 1e-6, 1.0, scale)
    probs = _row_softmax(scores)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    ranks = _rank_positions(scores)
    top2_scores = np.sort(scores, axis=1)[:, -2]
    top1_scores = np.max(scores, axis=1)

    parts = [
        scores,
        zscores,
        probs,
        entropy[:, None],
        (top1_scores - top2_scores)[:, None],
        packet_margins[:, None],
        scores[row_ids, packet_predictions][:, None],
        scores[row_ids, alt_predictions][:, None],
        scores[row_ids, target_predictions][:, None],
        zscores[row_ids, packet_predictions][:, None],
        zscores[row_ids, alt_predictions][:, None],
        ranks[row_ids, packet_predictions][:, None].astype(np.float64),
        ranks[row_ids, alt_predictions][:, None].astype(np.float64),
        (packet_predictions == alt_predictions)[:, None].astype(np.float64),
        (packet_predictions == target_predictions)[:, None].astype(np.float64),
        _candidate_one_hot(packet_predictions),
        _candidate_one_hot(alt_predictions),
        _candidate_one_hot(target_predictions),
    ]
    if view == "score_hidden_confidence":
        parts.append(
            _hidden_confidence_features(
                scores=scores,
                hidden=hidden,
                packet_predictions=packet_predictions,
                alt_predictions=alt_predictions,
            )
        )
    elif view != "score_only":
        raise ValueError(f"unsupported feature view: {view}")
    return np.concatenate(parts, axis=1).astype(np.float64)


def _benefit_values(
    *,
    alt_predictions: np.ndarray,
    packet_predictions: np.ndarray,
    answers: np.ndarray,
) -> np.ndarray:
    alt_correct = (alt_predictions == answers).astype(np.float64)
    packet_correct = (packet_predictions == answers).astype(np.float64)
    return alt_correct - packet_correct


def _task_predictions(
    *,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    use_alt: np.ndarray,
) -> np.ndarray:
    return np.where(use_alt, alt_predictions, packet_predictions).astype(np.int64)


def _receiver_stats(
    *,
    predictions: np.ndarray,
    packet_predictions: np.ndarray,
    target_predictions: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    ci_packet = _paired_ci(
        selected=predictions,
        baseline=packet_predictions,
        answers=answers,
        indices=indices,
        seed=seed,
        samples=samples,
    )
    ci_target = _paired_ci(
        selected=predictions,
        baseline=target_predictions,
        answers=answers,
        indices=indices,
        seed=seed + 17,
        samples=samples,
    )
    selected_correct = predictions[indices] == answers[indices]
    packet_correct = packet_predictions[indices] == answers[indices]
    return {
        "accuracy": _accuracy(predictions, answers, indices),
        "packet_only_accuracy": _accuracy(packet_predictions, answers, indices),
        "target_only_accuracy": _accuracy(target_predictions, answers, indices),
        "delta_vs_packet_only": ci_packet["delta"],
        "ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "ci95_high_vs_packet_only": ci_packet["ci95_high"],
        "delta_vs_target_only": ci_target["delta"],
        "ci95_low_vs_target_only": ci_target["ci95_low"],
        "help_count": int(np.sum(selected_correct & ~packet_correct)),
        "harm_count": int(np.sum(~selected_correct & packet_correct)),
        "net_help": int(np.sum(selected_correct & ~packet_correct) - np.sum(~selected_correct & packet_correct)),
    }


def _standardize_fit(features: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features[fit_indices], axis=0)
    scale = np.std(features[fit_indices], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean, scale


def _fit_benefit_ridge(
    *,
    features: np.ndarray,
    benefit: np.ndarray,
    fit_indices: np.ndarray,
    ridges: tuple[float, ...],
) -> list[dict[str, Any]]:
    mean, scale = _standardize_fit(features, fit_indices)
    x_body = (features[fit_indices] - mean) / scale
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    y = benefit[fit_indices]
    weights = np.where(np.abs(y) > 0.0, 1.0, 0.25)
    models = []
    for ridge in ridges:
        weighted_x = x * weights[:, None]
        xtx = x.T @ weighted_x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        xtx[0, 0] -= float(ridge)
        beta = np.linalg.solve(xtx, weighted_x.T @ y)
        models.append({"ridge": float(ridge), "mean": mean, "scale": scale, "beta": beta})
    return models


def _score_benefit_ridge(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x_body = (features - model["mean"]) / model["scale"]
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    return (x @ model["beta"]).astype(np.float64)


def _select_threshold(
    *,
    scores: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    answers: np.ndarray,
    dev_indices: np.ndarray,
) -> dict[str, Any]:
    candidates = np.unique(np.quantile(scores[dev_indices], np.linspace(0.0, 1.0, 101)))
    candidates = np.unique(np.concatenate([candidates, np.asarray([0.0], dtype=np.float64)]))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        use_alt = (scores > float(threshold)) & (alt_predictions != packet_predictions)
        predictions = _task_predictions(
            packet_predictions=packet_predictions,
            alt_predictions=alt_predictions,
            use_alt=use_alt,
        )
        selected_correct = predictions[dev_indices] == answers[dev_indices]
        packet_correct = packet_predictions[dev_indices] == answers[dev_indices]
        row = {
            "threshold": float(threshold),
            "dev_accuracy": _accuracy(predictions, answers, dev_indices),
            "dev_override_rate": float(np.mean(use_alt[dev_indices])),
            "dev_help_count": int(np.sum(selected_correct & ~packet_correct)),
            "dev_harm_count": int(np.sum(~selected_correct & packet_correct)),
        }
        key = (
            row["dev_accuracy"],
            row["dev_help_count"] - row["dev_harm_count"],
            -row["dev_override_rate"],
            -abs(row["threshold"]),
        )
        if best is None or key > best["key"]:
            best = {**row, "key": key}
    assert best is not None
    best.pop("key")
    return best


def _run_ridge_receiver(
    *,
    features: np.ndarray,
    benefit: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    eval_indices: np.ndarray,
    ridges: tuple[float, ...],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for model in _fit_benefit_ridge(features=features, benefit=benefit, fit_indices=fit_indices, ridges=ridges):
        scores = _score_benefit_ridge(model, features)
        selected = _select_threshold(
            scores=scores,
            packet_predictions=packet_predictions,
            alt_predictions=alt_predictions,
            answers=answers,
            dev_indices=dev_indices,
        )
        use_alt = (scores > selected["threshold"]) & (alt_predictions != packet_predictions)
        predictions = _task_predictions(
            packet_predictions=packet_predictions,
            alt_predictions=alt_predictions,
            use_alt=use_alt,
        )
        candidates.append(
            {
                "method": "benefit_ridge",
                "ridge": model["ridge"],
                "threshold": selected["threshold"],
                "dev_accuracy": selected["dev_accuracy"],
                "dev_override_rate": selected["dev_override_rate"],
                "dev_help_count": selected["dev_help_count"],
                "dev_harm_count": selected["dev_harm_count"],
                "eval_override_rate": float(np.mean(use_alt[eval_indices])),
                "predictions": predictions,
                "scores": scores,
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


def _knn_scores(
    *,
    features: np.ndarray,
    benefit: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    mean, scale = _standardize_fit(features, fit_indices)
    fit = _normalize_matrix((features[fit_indices] - mean) / scale)
    query = _normalize_matrix((features[query_indices] - mean) / scale)
    sims = query @ fit.T
    k = min(int(k), fit.shape[0])
    if k == fit.shape[0]:
        top_ids = np.argsort(-sims, axis=1)[:, :k]
    else:
        top_ids = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, top_ids, axis=1)
    weights = np.maximum(top_sims, 0.0) + 1e-3
    values = benefit[fit_indices][top_ids]
    return (values * weights).sum(axis=1) / weights.sum(axis=1)


def _run_relative_knn_receiver(
    *,
    features: np.ndarray,
    benefit: np.ndarray,
    packet_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    eval_indices: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    all_indices = np.arange(features.shape[0], dtype=np.int64)
    for k in k_values:
        dev_scores = _knn_scores(
            features=features,
            benefit=benefit,
            fit_indices=fit_indices,
            query_indices=dev_indices,
            k=k,
        )
        eval_scores = _knn_scores(
            features=features,
            benefit=benefit,
            fit_indices=fit_indices,
            query_indices=eval_indices,
            k=k,
        )
        scores = np.zeros(features.shape[0], dtype=np.float64)
        scores[dev_indices] = dev_scores
        scores[eval_indices] = eval_scores
        selected = _select_threshold(
            scores=scores,
            packet_predictions=packet_predictions,
            alt_predictions=alt_predictions,
            answers=answers,
            dev_indices=dev_indices,
        )
        all_scores = np.zeros(features.shape[0], dtype=np.float64)
        all_scores[all_indices] = _knn_scores(
            features=features,
            benefit=benefit,
            fit_indices=fit_indices,
            query_indices=all_indices,
            k=k,
        )
        use_alt = (all_scores > selected["threshold"]) & (alt_predictions != packet_predictions)
        predictions = _task_predictions(
            packet_predictions=packet_predictions,
            alt_predictions=alt_predictions,
            use_alt=use_alt,
        )
        candidates.append(
            {
                "method": "relative_knn_benefit",
                "k": int(k),
                "threshold": selected["threshold"],
                "dev_accuracy": selected["dev_accuracy"],
                "dev_override_rate": selected["dev_override_rate"],
                "dev_help_count": selected["dev_help_count"],
                "dev_harm_count": selected["dev_harm_count"],
                "eval_override_rate": float(np.mean(use_alt[eval_indices])),
                "predictions": predictions,
                "scores": all_scores,
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["dev_accuracy"],
            row["dev_help_count"] - row["dev_harm_count"],
            -row["dev_override_rate"],
            -row["k"],
        ),
    )


def _contiguous_block_deltas(
    *,
    predictions: np.ndarray,
    packet_predictions: np.ndarray,
    answers: np.ndarray,
    eval_indices: np.ndarray,
    block_count: int = 5,
) -> list[dict[str, Any]]:
    blocks = np.array_split(eval_indices, block_count)
    rows = []
    for block_index, block in enumerate(blocks):
        if len(block) == 0:
            continue
        rows.append(
            {
                "block": block_index,
                "rows": int(len(block)),
                "start_index": int(block[0]),
                "end_index_exclusive": int(block[-1] + 1),
                "receiver_accuracy": _accuracy(predictions, answers, block),
                "packet_only_accuracy": _accuracy(packet_predictions, answers, block),
                "delta_vs_packet_only": float(
                    np.mean(predictions[block] == answers[block])
                    - np.mean(packet_predictions[block] == answers[block])
                ),
            }
        )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Receiver Acceptance Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- scout pass gate: `{payload['scout_pass_gate']}`",
        f"- predeclared default pass gate: `{payload['predeclared_default_pass_gate']}`",
        f"- best scout row: `{h['best_scout_method']}` / `{h['best_scout_alternative']}` / `{h['best_scout_view']}`",
        f"- best scout eval accuracy: `{h['best_scout_eval_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- default eval accuracy: `{h['default_eval_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
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
    tiny_packet_jsonl: pathlib.Path = DEFAULT_TINY_PACKET_JSONL,
    tiny_artifact: pathlib.Path = DEFAULT_TINY_ARTIFACT,
    qwen_packet_jsonl: pathlib.Path = DEFAULT_QWEN_PACKET_JSONL,
    qwen_global_artifact: pathlib.Path = DEFAULT_QWEN_GLOBAL_ARTIFACT,
    tiny_field: str = DEFAULT_TINY_FIELD,
    tiny_margin_field: str = DEFAULT_TINY_MARGIN_FIELD,
    qwen_fields: tuple[str, ...] = DEFAULT_QWEN_FIELDS,
    control_fields: tuple[str, ...] = DEFAULT_CONTROL_FIELDS,
    train_prefixes: tuple[int, ...] = DEFAULT_TRAIN_PREFIXES,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    bootstrap_samples: int = 1000,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    tiny_rows = _read_jsonl(tiny_packet_jsonl)
    qwen_rows = _read_jsonl(qwen_packet_jsonl)
    qwen = _load_qwen_bundle(qwen_global_artifact)
    row_ids = [str(row["row_id"]) for row in tiny_rows]
    if row_ids != [str(row["row_id"]) for row in qwen_rows]:
        raise ValueError("TinyLlama and Qwen prediction rows are not aligned")
    if row_ids != qwen["row_ids"]:
        raise ValueError("packet prediction rows and Qwen score rows are not aligned")
    answers = np.asarray([int(row["answer_index"]) for row in tiny_rows], dtype=np.int64)
    qwen_answers = np.asarray([int(row["answer_index"]) for row in qwen_rows], dtype=np.int64)
    if not np.array_equal(answers, qwen_answers):
        raise ValueError("TinyLlama and Qwen answer rows disagree")

    packet_predictions = _read_prediction(tiny_rows, tiny_field)
    packet_margins = np.asarray(
        [float(row.get(tiny_margin_field, 0.0)) for row in tiny_rows],
        dtype=np.float64,
    )
    target_predictions = np.argmax(qwen["scores"], axis=1).astype(np.int64)
    alternatives = {"qwen_target_score": target_predictions}
    alternatives.update({field: _read_prediction(qwen_rows, field) for field in qwen_fields})

    frontier_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[int, str, str, str], np.ndarray] = {}
    feature_build_start = time.perf_counter()
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    for alt_name, alt_predictions in alternatives.items():
        for view in ("score_only", "score_hidden_confidence"):
            feature_cache[(alt_name, view)] = _feature_matrix(
                scores=qwen["scores"],
                hidden=qwen["hidden"],
                packet_predictions=packet_predictions,
                packet_margins=packet_margins,
                alt_predictions=alt_predictions,
                view=view,
            )
    feature_build_wall_time_s = time.perf_counter() - feature_build_start
    selector_start = time.perf_counter()

    for train_prefix in train_prefixes:
        train_prefix = int(train_prefix)
        if train_prefix <= 4 or train_prefix >= len(tiny_rows):
            raise ValueError("train_prefixes must leave heldout eval rows")
        dev_start = max(1, int(round(train_prefix * 0.75)))
        fit_indices = np.arange(0, dev_start, dtype=np.int64)
        dev_indices = np.arange(dev_start, train_prefix, dtype=np.int64)
        eval_indices = np.arange(train_prefix, len(tiny_rows), dtype=np.int64)
        for alt_index, (alt_name, alt_predictions) in enumerate(alternatives.items()):
            benefit = _benefit_values(
                alt_predictions=alt_predictions,
                packet_predictions=packet_predictions,
                answers=answers,
            )
            for view in ("score_only", "score_hidden_confidence"):
                features = feature_cache[(alt_name, view)]
                receivers = [
                    _run_ridge_receiver(
                        features=features,
                        benefit=benefit,
                        packet_predictions=packet_predictions,
                        alt_predictions=alt_predictions,
                        answers=answers,
                        fit_indices=fit_indices,
                        dev_indices=dev_indices,
                        eval_indices=eval_indices,
                        ridges=ridges,
                    ),
                    _run_relative_knn_receiver(
                        features=features,
                        benefit=benefit,
                        packet_predictions=packet_predictions,
                        alt_predictions=alt_predictions,
                        answers=answers,
                        fit_indices=fit_indices,
                        dev_indices=dev_indices,
                        eval_indices=eval_indices,
                        k_values=k_values,
                    ),
                ]
                for method_index, receiver in enumerate(receivers):
                    predictions = receiver["predictions"]
                    stats = _receiver_stats(
                        predictions=predictions,
                        packet_predictions=packet_predictions,
                        target_predictions=target_predictions,
                        answers=answers,
                        indices=eval_indices,
                        seed=8100 + train_prefix + 31 * alt_index + method_index,
                        samples=bootstrap_samples,
                    )
                    key = (train_prefix, alt_name, view, receiver["method"])
                    prediction_cache[key] = predictions
                    frontier_rows.append(
                        {
                            "train_prefix_rows": train_prefix,
                            "fit_rows": int(len(fit_indices)),
                            "dev_rows": int(len(dev_indices)),
                            "eval_rows": int(len(eval_indices)),
                            "alternative": alt_name,
                            "feature_view": view,
                            "method": receiver["method"],
                            "hyperparameter": {
                                "ridge": receiver.get("ridge"),
                                "k": receiver.get("k"),
                                "threshold": receiver["threshold"],
                            },
                            "dev_accuracy": receiver["dev_accuracy"],
                            "dev_override_rate": receiver["dev_override_rate"],
                            "dev_help_count": receiver["dev_help_count"],
                            "dev_harm_count": receiver["dev_harm_count"],
                            "eval_override_rate": receiver["eval_override_rate"],
                            **stats,
                        }
                    )

    selector_wall_time_s = time.perf_counter() - selector_start
    best_scout = max(
        frontier_rows,
        key=lambda row: (
            row["delta_vs_packet_only"],
            row["ci95_low_vs_packet_only"],
            row["dev_accuracy"],
            -row["eval_override_rate"],
        ),
    )
    default_candidates = [
        row
        for row in frontier_rows
        if row["train_prefix_rows"] == train_prefixes[0]
        and row["alternative"] == "qwen_target_score"
        and row["feature_view"] == "score_hidden_confidence"
        and row["method"] == "benefit_ridge"
    ]
    if not default_candidates:
        raise ValueError("predeclared default receiver row was not built")
    default_row = default_candidates[0]
    default_predictions = prediction_cache[
        (
            default_row["train_prefix_rows"],
            default_row["alternative"],
            default_row["feature_view"],
            default_row["method"],
        )
    ]
    default_eval_indices = np.arange(default_row["train_prefix_rows"], len(tiny_rows), dtype=np.int64)
    default_blocks = _contiguous_block_deltas(
        predictions=default_predictions,
        packet_predictions=packet_predictions,
        answers=answers,
        eval_indices=default_eval_indices,
    )

    control_rows: list[dict[str, Any]] = []
    for offset, field in enumerate(control_fields):
        if field not in tiny_rows[0]:
            continue
        control_packet = _read_prediction(tiny_rows, field)
        control_stats = _receiver_stats(
            predictions=control_packet,
            packet_predictions=packet_predictions,
            target_predictions=target_predictions,
            answers=answers,
            indices=default_eval_indices,
            seed=8500 + offset,
            samples=bootstrap_samples,
        )
        control_rows.append({"name": field, **control_stats})

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
    pass_gate = bool(predeclared_default_pass_gate and target_transfer_gate and block_stability_gate)

    headline = {
        "row_count": len(tiny_rows),
        "train_prefixes": list(train_prefixes),
        "best_scout_train_prefix_rows": best_scout["train_prefix_rows"],
        "best_scout_method": best_scout["method"],
        "best_scout_alternative": best_scout["alternative"],
        "best_scout_view": best_scout["feature_view"],
        "best_scout_eval_accuracy": best_scout["accuracy"],
        "best_scout_packet_only_accuracy": best_scout["packet_only_accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "default_method": default_row["method"],
        "default_alternative": default_row["alternative"],
        "default_feature_view": default_row["feature_view"],
        "default_train_prefix_rows": default_row["train_prefix_rows"],
        "default_eval_accuracy": default_row["accuracy"],
        "default_packet_only_accuracy": default_row["packet_only_accuracy"],
        "default_target_only_accuracy": default_row["target_only_accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_target_only": default_row["delta_vs_target_only"],
        "default_ci95_low_vs_target_only": default_row["ci95_low_vs_target_only"],
        "strict_delta_required": STRICT_DELTA,
        "packet_raw_bytes": decomp.RAW_PACKET_BYTES,
        "packet_framed_bytes": decomp.FRAMED_PACKET_BYTES,
        "native_gpu_claims_allowed": False,
    }
    lay_explanation = (
        "This experiment asks whether a receiver can learn when to ignore the TinyLlama packet "
        "and use Qwen's own candidate instead. It trains only on an early prefix, chooses the "
        "override rule on a prefix dev split, and scores the frozen rule on later heldout rows. "
        "The ridge row is a learned error predictor; the relative-kNN row is a nearest-anchor "
        "common-basis test."
    )
    interpretation = (
        "This gate tests the most direct receiver-improvement path after the oracle headroom card. "
        "A pass would promote a train-only receiver that beats packet-only. A fail weakens simple "
        "selective prediction and relative-neighbor receiver families, pushing the next branch "
        "toward richer common-basis supervision, official-train receiver calibration, or a learned "
        "query-bottleneck receiver rather than more confidence thresholds."
    )
    payload = {
        "gate": "source_private_hellaswag_receiver_acceptance_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "predeclared_default_pass_gate": predeclared_default_pass_gate,
        "target_transfer_gate": target_transfer_gate,
        "block_stability_gate": block_stability_gate,
        "pass_rule": (
            "Strict promotion requires the predeclared default train-prefix receiver to beat "
            "packet-only by >=0.005 with positive paired CI95 low, beat target-only by >=0.02 "
            "with positive paired CI95 low, and stay positive across contiguous heldout blocks. "
            "The best-scout row is diagnostic only because it is selected after seeing the "
            "frontier table."
        ),
        "headline": headline,
        "frontier_rows": frontier_rows,
        "default_block_rows": default_blocks,
        "control_rows": control_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": decomp.RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": decomp.FRAMED_PACKET_BYTES,
            "logical_raw_payload_bytes_total": int(decomp.RAW_PACKET_BYTES * len(tiny_rows)),
            "logical_framed_record_bytes_total": int(decomp.FRAMED_PACKET_BYTES * len(tiny_rows)),
            "batch64_packed_bytes_per_request": decomp.FRAMED_PACKET_BYTES,
            "feature_build_wall_time_s": float(feature_build_wall_time_s),
            "selector_wall_time_s": float(selector_wall_time_s),
            "total_wall_time_s": float(time.perf_counter() - wall_start),
            "selector_examples_per_second": float(
                len(frontier_rows) * len(tiny_rows) / max(selector_wall_time_s, 1e-12)
            ),
            "qwen_hidden_cache_bytes_total": int(sum(item["hidden_cache_bytes"] for item in qwen["slices"])),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
        },
        "inputs": {
            "tiny_packet_jsonl": _display_path(tiny_packet_jsonl),
            "tiny_packet_jsonl_sha256": _sha256_file(tiny_packet_jsonl),
            "tiny_artifact": _display_path(tiny_artifact),
            "tiny_artifact_sha256": _sha256_file(tiny_artifact),
            "qwen_packet_jsonl": _display_path(qwen_packet_jsonl),
            "qwen_packet_jsonl_sha256": _sha256_file(qwen_packet_jsonl),
            "qwen_global_artifact": qwen["artifact_path"],
            "qwen_global_artifact_sha256": qwen["artifact_sha256"],
            "qwen_slices": qwen["slices"],
        },
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hellaswag_receiver_acceptance_gate.json"
    md_path = output_dir / "hellaswag_receiver_acceptance_gate.md"
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
    parser.add_argument("--tiny-packet-jsonl", type=pathlib.Path, default=DEFAULT_TINY_PACKET_JSONL)
    parser.add_argument("--tiny-artifact", type=pathlib.Path, default=DEFAULT_TINY_ARTIFACT)
    parser.add_argument("--qwen-packet-jsonl", type=pathlib.Path, default=DEFAULT_QWEN_PACKET_JSONL)
    parser.add_argument("--qwen-global-artifact", type=pathlib.Path, default=DEFAULT_QWEN_GLOBAL_ARTIFACT)
    parser.add_argument("--tiny-field", default=DEFAULT_TINY_FIELD)
    parser.add_argument("--tiny-margin-field", default=DEFAULT_TINY_MARGIN_FIELD)
    parser.add_argument("--qwen-fields", type=_parse_tuple, default=DEFAULT_QWEN_FIELDS)
    parser.add_argument("--control-fields", type=_parse_tuple, default=DEFAULT_CONTROL_FIELDS)
    parser.add_argument("--train-prefixes", type=_parse_int_tuple, default=DEFAULT_TRAIN_PREFIXES)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--k-values", type=_parse_int_tuple, default=DEFAULT_K_VALUES)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        tiny_packet_jsonl=args.tiny_packet_jsonl,
        tiny_artifact=args.tiny_artifact,
        qwen_packet_jsonl=args.qwen_packet_jsonl,
        qwen_global_artifact=args.qwen_global_artifact,
        tiny_field=args.tiny_field,
        tiny_margin_field=args.tiny_margin_field,
        qwen_fields=args.qwen_fields,
        control_fields=args.control_fields,
        train_prefixes=args.train_prefixes,
        ridges=args.ridges,
        k_values=args.k_values,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
