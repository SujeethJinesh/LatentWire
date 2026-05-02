from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_anchor_relative_hidden_innovation_gate as anchor_gate  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_bagged_gate as bagged  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_repair_probe as repair  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_train_sample_stress as stress  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_anchor_variant_scout_20260501_qwen05_validation4096_5120"
)
DEFAULT_TRAIN = anchor_gate.DEFAULT_TRAIN
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
DEFAULT_SOURCE_LM_MODEL = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str
    dim: int = 0
    bins: int = 0
    seed: int = 0


DEFAULT_VARIANTS = (
    VariantSpec(name="cosine_full", kind="cosine_full"),
    VariantSpec(name="signed_topk_hash16x64", kind="signed_topk_hash", dim=16, bins=64, seed=811),
    VariantSpec(name="cosine_topk16", kind="cosine_topk", dim=16),
    VariantSpec(name="rbf_topk16", kind="rbf_topk", dim=16),
    VariantSpec(name="spectral32", kind="spectral", dim=32),
    VariantSpec(name="qjl_sign32", kind="qjl_sign", dim=32, seed=1069),
)


def _anchor_order_controls_expected_effective(spec: VariantSpec) -> bool:
    return spec.kind not in {"cosine_topk", "rbf_topk"}


def _similarity_tensor(*, scores: list[list[float]], hidden: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    residuals = anchor_gate._hidden_residual_tensor(scores=scores, hidden=hidden)
    flat = residuals.reshape(-1, residuals.shape[-1])
    normalized = anchor_gate._normalize_rows(flat).reshape(residuals.shape)
    return np.einsum("ncd,kd->nck", normalized, anchors, optimize=True)


def _topk_stats(values: np.ndarray, *, k: int) -> np.ndarray:
    k = max(1, min(int(k), values.shape[-1]))
    top = np.sort(values, axis=-1)[..., -k:][..., ::-1]
    if k == 1:
        gap = np.zeros((*values.shape[:-1], 1), dtype=np.float64)
    else:
        gap = (top[..., :1] - top[..., 1:2]).astype(np.float64)
    stats = np.stack(
        [
            np.mean(values, axis=-1),
            np.std(values, axis=-1),
            np.max(values, axis=-1),
            np.min(values, axis=-1),
            np.mean(values > 0.0, axis=-1),
        ],
        axis=-1,
    )
    return np.concatenate([top, gap, stats], axis=-1)


def _signed_topk_hash_features(
    *,
    scores: list[list[float]],
    similarities: np.ndarray,
    k: int,
    bins: int,
    seed: int,
) -> np.ndarray:
    k = max(1, min(int(k), similarities.shape[-1]))
    bins = max(1, int(bins))
    rows: list[list[np.ndarray]] = []
    for row_index, row_scores in enumerate(scores):
        ranked = top2._ranked_indices(row_scores)
        top1 = ranked[0]
        pivot = ranked[1] if len(ranked) > 1 else top1
        pivot_similarities = similarities[row_index, pivot]
        top_ids = np.argsort(pivot_similarities, kind="stable")[-k:][::-1]
        row_features: list[np.ndarray] = []
        for candidate in range(4):
            values = similarities[row_index, candidate, top_ids]
            deltas = values - similarities[row_index, top1, top_ids]
            hashed = np.zeros(bins, dtype=np.float64)
            for anchor_id, delta in zip(top_ids, deltas, strict=True):
                hashed[(int(anchor_id) * 1315423911 + int(seed)) % bins] += float(delta)
            stats = np.asarray(
                [
                    float(np.mean(deltas)),
                    float(np.std(deltas)),
                    float(np.max(deltas)),
                    float(np.min(deltas)),
                    float(np.mean(deltas > 0.0)),
                ],
                dtype=np.float64,
            )
            row_features.append(np.concatenate([values, deltas, hashed, stats]))
        rows.append(row_features)
    return np.asarray(rows, dtype=np.float64)


def _spectral_basis(anchors: np.ndarray, *, dim: int) -> np.ndarray:
    if anchors.shape[0] <= 1:
        return np.ones((anchors.shape[0], 1), dtype=np.float64)
    similarities = np.clip(anchors @ anchors.T, -1.0, 1.0)
    distances = np.maximum(0.0, 1.0 - similarities)
    positive = distances[distances > 1e-8]
    sigma = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-distances / max(sigma, 1e-6))
    np.fill_diagonal(weights, 0.0)
    degree = np.sum(weights, axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.where(degree < 1e-8, 1.0, degree))
    laplacian = np.eye(weights.shape[0], dtype=np.float64) - (
        inv_sqrt_degree[:, None] * weights * inv_sqrt_degree[None, :]
    )
    _, vectors = np.linalg.eigh(laplacian)
    start = 1 if vectors.shape[1] > 1 else 0
    stop = min(vectors.shape[1], start + max(1, int(dim)))
    return vectors[:, start:stop].astype(np.float64)


def _qjl_matrix(anchor_count: int, *, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.choice((-1.0, 1.0), size=(anchor_count, max(1, int(dim))))
    return matrix.astype(np.float64) / np.sqrt(max(1, int(dim)))


def _variant_params(
    *,
    spec: VariantSpec,
    train_similarities: np.ndarray,
    anchors: np.ndarray,
    component_seed: int,
) -> dict[str, Any]:
    if spec.kind == "rbf_topk":
        distances = np.maximum(0.0, 1.0 - train_similarities.reshape(-1))
        positive = distances[distances > 1e-8]
        return {"sigma": float(np.median(positive)) if positive.size else 1.0}
    if spec.kind == "spectral":
        return {"basis": _spectral_basis(anchors, dim=spec.dim)}
    if spec.kind == "qjl_sign":
        return {"projection": _qjl_matrix(anchors.shape[0], dim=spec.dim, seed=spec.seed + component_seed)}
    return {}


def _variant_from_similarities(
    *,
    scores: list[list[float]],
    similarities: np.ndarray,
    spec: VariantSpec,
    params: dict[str, Any],
    permute_anchor_ids: bool = False,
) -> np.ndarray:
    sim = similarities
    if permute_anchor_ids and sim.shape[-1] > 1:
        sim = np.roll(sim, 1, axis=-1)
    if spec.kind == "cosine_full":
        transformed = sim
    elif spec.kind == "cosine_topk":
        transformed = _topk_stats(sim, k=spec.dim)
    elif spec.kind == "rbf_topk":
        sigma = float(params.get("sigma", 1.0))
        transformed = _topk_stats(np.exp(-np.maximum(0.0, 1.0 - sim) / max(sigma, 1e-6)), k=spec.dim)
    elif spec.kind == "signed_topk_hash":
        transformed = _signed_topk_hash_features(
            scores=scores,
            similarities=sim,
            k=spec.dim,
            bins=spec.bins,
            seed=spec.seed,
        )
    elif spec.kind == "spectral":
        basis = np.asarray(params["basis"], dtype=np.float64)
        transformed = sim @ basis
    elif spec.kind == "qjl_sign":
        projection = np.asarray(params["projection"], dtype=np.float64)
        transformed = np.sign(sim @ projection)
    else:
        raise ValueError(f"unknown variant kind: {spec.kind}")
    rows: list[list[np.ndarray]] = []
    for row_index, row_scores in enumerate(scores):
        row_features: list[np.ndarray] = []
        for candidate in range(4):
            row_features.append(
                np.concatenate(
                    [
                        repair._candidate_score_features(row_scores, candidate),
                        transformed[row_index, candidate],
                    ]
                )
            )
        rows.append(row_features)
    return np.asarray(rows, dtype=np.float64)


def _fit_component(
    *,
    spec: VariantSpec,
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
    component_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchors, anchor_meta = anchor_gate._fit_anchor_bank(
        rows=train_rows,
        scores=train_scores,
        hidden=train_hidden,
        fit_indices=fit_indices,
        anchor_count=anchor_count,
    )
    train_similarities = _similarity_tensor(scores=train_scores, hidden=train_hidden, anchors=anchors)
    eval_similarities = _similarity_tensor(scores=eval_scores, hidden=eval_hidden, anchors=anchors)
    params = _variant_params(
        spec=spec,
        train_similarities=train_similarities[np.asarray(fit_indices, dtype=np.int64)],
        anchors=anchors,
        component_seed=component_seed,
    )
    train_features = _variant_from_similarities(
        scores=train_scores,
        similarities=train_similarities,
        spec=spec,
        params=params,
    )
    eval_features = _variant_from_similarities(
        scores=eval_scores,
        similarities=eval_similarities,
        spec=spec,
        params=params,
    )
    readouts = [
        repair._fit_and_eval_view(
            view=spec.name,
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
    component = {
        "variant": spec.name,
        "kind": spec.kind,
        "dim": spec.dim,
        "bins": spec.bins,
        "selected_ridge": selected["ridge"],
        "selected_feature_dim": selected["feature_dim"],
        "selected_fit_accuracy": selected["fit"]["accuracy"],
        "selected_internal_dev_accuracy": selected["internal_dev"]["accuracy"],
        "selected_eval_accuracy": selected["eval"]["accuracy"],
        "candidate_readout_count": len(readouts),
        **anchor_meta,
    }
    serializable_params = {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return {
        "model": model,
        "anchors": anchors,
        "spec": spec,
        "params": params,
        "serializable_params": serializable_params,
    }, component


def _centered_z(candidate_scores: np.ndarray) -> np.ndarray:
    centered = candidate_scores - np.mean(candidate_scores, axis=1, keepdims=True)
    scale = np.std(candidate_scores, axis=1, keepdims=True)
    return centered / np.where(scale < 1e-6, 1.0, scale)


def _aggregate_variant_scores(
    *,
    scores: list[list[float]],
    hidden: np.ndarray,
    components: list[dict[str, Any]],
    permute_anchor_ids: bool = False,
    roll_anchor_values: bool = False,
) -> tuple[list[int], np.ndarray]:
    model_scores: list[np.ndarray] = []
    for component in components:
        anchors = np.asarray(component["anchors"], dtype=np.float64)
        if roll_anchor_values and anchors.shape[0] > 1:
            anchors = np.roll(anchors, 1, axis=0)
        similarities = _similarity_tensor(scores=scores, hidden=hidden, anchors=anchors)
        features = _variant_from_similarities(
            scores=scores,
            similarities=similarities,
            spec=component["spec"],
            params=component["params"],
            permute_anchor_ids=permute_anchor_ids,
        )
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
    lines = [
        "# HellaSwag Anchor Feature Variant Scout",
        "",
        f"- scout pass: `{payload['scout_pass']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- best variant: `{h['best_variant']}`",
        f"- best accuracy: `{h['best_selected_eval_accuracy']:.6f}`",
        f"- best label-copy accuracy: `{h['best_label_copy_eval_accuracy']:.6f}`",
        f"- delta vs best label-copy: `{h['best_selected_minus_best_label_copy']:.6f}`",
        f"- CI95 vs best label-copy: `[{h['best_ci95_low_vs_best_label_copy']:.6f}, {h['best_ci95_high_vs_best_label_copy']:.6f}]`",
        f"- score-only bagged control: `{h['score_only_bagged_control_accuracy']:.6f}`",
        f"- dense hidden-innovation reference: `{h['dense_hidden_reference_accuracy']}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_scout(
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
    variants: tuple[VariantSpec, ...] = DEFAULT_VARIANTS,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = 500,
    dense_hidden_reference_accuracy: float | None = 0.503125,
    source_lm_model: str = DEFAULT_SOURCE_LM_MODEL,
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

    components_by_variant: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in variants}
    score_models: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    sample_cache_rows: list[dict[str, Any]] = []
    trained_label_predictions: list[list[int]] = []

    for sample_seed in train_sample_seeds:
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
        for split_offset, split_seed in enumerate(split_seeds):
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
            component_rows.append(
                {
                    "train_sample_seed": sample_seed,
                    "split_seed": split_seed,
                    "fit_rows": len(fit_indices),
                    "internal_dev_rows": len(dev_indices),
                    **score_component,
                }
            )
            for variant_index, spec in enumerate(variants):
                component_seed = sample_seed * 1000 + split_seed * 10 + variant_index + split_offset
                model, component = _fit_component(
                    spec=spec,
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
                    component_seed=component_seed,
                )
                components_by_variant[spec.name].append(model)
                component_rows.append(
                    {
                        "train_sample_seed": sample_seed,
                        "split_seed": split_seed,
                        "fit_rows": len(fit_indices),
                        "internal_dev_rows": len(dev_indices),
                        **component,
                        **model["serializable_params"],
                    }
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
    score_only_predictions, _ = bagged._aggregate_model_scores(
        features=score_eval_features,
        models=score_models,
        policy="mean_zscore",
    )
    score_only_accuracy = top2._accuracy(eval_rows, score_only_predictions)

    variant_rows: list[dict[str, Any]] = []
    prediction_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for spec in variants:
        components = components_by_variant[spec.name]
        selected_predictions, _ = _aggregate_variant_scores(
            scores=eval_scores,
            hidden=eval_hidden,
            components=components,
        )
        zero_predictions, _ = _aggregate_variant_scores(
            scores=eval_scores,
            hidden=zero_hidden,
            components=components,
        )
        wrong_predictions, _ = _aggregate_variant_scores(
            scores=eval_scores,
            hidden=wrong_hidden,
            components=components,
        )
        candidate_roll_predictions, _ = _aggregate_variant_scores(
            scores=eval_scores,
            hidden=candidate_roll_hidden,
            components=components,
        )
        anchor_id_shuffle_predictions, _ = _aggregate_variant_scores(
            scores=eval_scores,
            hidden=eval_hidden,
            components=components,
            permute_anchor_ids=True,
        )
        anchor_value_roll_predictions, _ = _aggregate_variant_scores(
            scores=eval_scores,
            hidden=eval_hidden,
            components=components,
            roll_anchor_values=True,
        )
        paired_ci_label = top2._paired_ci_predictions(
            eval_rows,
            selected_predictions,
            best_label_predictions,
            seed=9300 + len(variant_rows),
            samples=bootstrap_samples,
        )
        paired_ci_score = top2._paired_ci_predictions(
            eval_rows,
            selected_predictions,
            score_only_predictions,
            seed=9400 + len(variant_rows),
            samples=bootstrap_samples,
        )
        selected_accuracy = top2._accuracy(eval_rows, selected_predictions)
        zero_accuracy = top2._accuracy(eval_rows, zero_predictions)
        wrong_accuracy = top2._accuracy(eval_rows, wrong_predictions)
        candidate_roll_accuracy = top2._accuracy(eval_rows, candidate_roll_predictions)
        anchor_id_shuffle_accuracy = top2._accuracy(eval_rows, anchor_id_shuffle_predictions)
        anchor_value_roll_accuracy = top2._accuracy(eval_rows, anchor_value_roll_predictions)
        row = {
            "variant": spec.name,
            "kind": spec.kind,
            "dim": spec.dim,
            "bins": spec.bins,
            "component_model_count": len(components),
            "selected_eval_accuracy": selected_accuracy,
            "source_label_copy_eval_accuracy": source_label_accuracy,
            "trained_choice_bias_label_copy_eval_accuracy": trained_label_accuracy,
            "best_label_copy_eval_accuracy": best_label_accuracy,
            "selected_minus_best_label_copy": selected_accuracy - best_label_accuracy,
            "paired_ci95_low_vs_best_label_copy": paired_ci_label["ci95_low"],
            "paired_ci95_high_vs_best_label_copy": paired_ci_label["ci95_high"],
            "score_only_bagged_control_accuracy": score_only_accuracy,
            "selected_minus_score_only_bagged_control": selected_accuracy - score_only_accuracy,
            "paired_ci95_low_vs_score_only_bagged": paired_ci_score["ci95_low"],
            "paired_ci95_high_vs_score_only_bagged": paired_ci_score["ci95_high"],
            "zero_hidden_control_accuracy": zero_accuracy,
            "selected_minus_zero_hidden_control": selected_accuracy - zero_accuracy,
            "wrong_example_hidden_control_accuracy": wrong_accuracy,
            "candidate_roll_hidden_control_accuracy": candidate_roll_accuracy,
            "anchor_id_shuffle_control_accuracy": anchor_id_shuffle_accuracy,
            "anchor_value_roll_control_accuracy": anchor_value_roll_accuracy,
            "selected_ridge_counts": json.dumps(
                {
                    str(key): value
                    for key, value in Counter(
                        component["model"]["ridge"] for component in components
                    ).items()
                },
                sort_keys=True,
            ),
        }
        anchor_order_controls_expected_effective = _anchor_order_controls_expected_effective(spec)
        anchor_order_controls_ok = (
            True
            if not anchor_order_controls_expected_effective
            else anchor_id_shuffle_accuracy <= best_label_accuracy + 0.005
            and anchor_value_roll_accuracy <= best_label_accuracy + 0.005
        )
        row["anchor_order_controls_expected_effective"] = anchor_order_controls_expected_effective
        row["anchor_order_controls_ok"] = bool(anchor_order_controls_ok)
        row["scout_pass_rule"] = bool(
            row["selected_minus_best_label_copy"] >= STRICT_DELTA
            and row["paired_ci95_low_vs_best_label_copy"] > 0.0
            and row["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
            and row["paired_ci95_low_vs_score_only_bagged"] > 0.0
            and row["selected_minus_zero_hidden_control"] >= STRICT_DELTA
            and wrong_accuracy <= best_label_accuracy
            and candidate_roll_accuracy <= best_label_accuracy
            and anchor_order_controls_ok
        )
        variant_rows.append(row)
        prediction_rows_by_variant[spec.name] = [
            {
                "row_id": eval_row.row_id,
                "answer_index": eval_row.answer_index,
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
                eval_row,
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

    best_variant = max(
        variant_rows,
        key=lambda row: (
            row["scout_pass_rule"],
            row["selected_minus_best_label_copy"],
            row["paired_ci95_low_vs_best_label_copy"],
            row["selected_minus_score_only_bagged_control"],
            row["selected_eval_accuracy"],
        ),
    )
    scout_pass = bool(best_variant["scout_pass_rule"])
    packet_contract = {
        "packet_name": "anchor_feature_variant_source_private_packet",
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
        "decoder_rule": "diagnostic scout; any passing variant must be rerun as a predeclared all-slice gate",
    }
    headline = {
        "eval_rows": len(eval_rows),
        "variant_count": len(variant_rows),
        "best_variant": best_variant["variant"],
        "best_selected_eval_accuracy": best_variant["selected_eval_accuracy"],
        "best_label_copy_eval_accuracy": best_label_accuracy,
        "best_selected_minus_best_label_copy": best_variant["selected_minus_best_label_copy"],
        "best_ci95_low_vs_best_label_copy": best_variant["paired_ci95_low_vs_best_label_copy"],
        "best_ci95_high_vs_best_label_copy": best_variant["paired_ci95_high_vs_best_label_copy"],
        "score_only_bagged_control_accuracy": score_only_accuracy,
        "best_selected_minus_score_only_bagged_control": best_variant[
            "selected_minus_score_only_bagged_control"
        ],
        "dense_hidden_reference_accuracy": dense_hidden_reference_accuracy,
        "best_delta_vs_dense_hidden_reference": (
            None
            if dense_hidden_reference_accuracy is None
            else best_variant["selected_eval_accuracy"] - dense_hidden_reference_accuracy
        ),
        "strict_delta_required": STRICT_DELTA,
        "raw_payload_bytes": packet_contract["raw_payload_bytes"],
        "framed_record_bytes": packet_contract["framed_record_bytes"],
    }
    payload = {
        "gate": "source_private_hellaswag_anchor_variant_scout",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scout_pass": scout_pass,
        "pass_rule": (
            "Scout-pass only if a train/dev-selected anchor feature variant beats best label-copy, score-only "
            "bagging, and zero-hidden controls by at least 0.02 with positive paired CI lows, while corrupted "
            "controls remain at or below label-copy. Anchor-order destructive controls are required only for "
            "order-sensitive variants; sorted top-k variants are intentionally anchor-order invariant. A scout "
            "pass is not a paper claim; the variant must then be predeclared and rerun over all five frozen "
            "slices."
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
        "variant_rows": variant_rows,
        "interpretation": (
            "This is a bounded rescue scout for the failed anchor-relative common-basis branch. It tests whether "
            "local-neighborhood, RBF, spectral/Fourier-like, or QJL sign-sketch coordinates can recover the dense "
            "hidden-innovation signal from existing cached source hiddens. Because variants are compared on the "
            "eval slice, any success only promotes a predeclared all-slice gate; a failure weakens anchor-feature "
            "rescues and pushes the next branch toward learned sparse/crosscoder-style bases."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_anchor_variant_scout.json"
    md_path = output_dir / "hellaswag_anchor_variant_scout.md"
    component_path = output_dir / "component_rows.csv"
    variant_path = output_dir / "variant_rows.csv"
    sample_cache_path = output_dir / "sample_caches.jsonl"
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(component_path, component_rows)
    _write_csv(variant_path, variant_rows)
    _write_jsonl(sample_cache_path, sample_cache_rows)
    for variant_name, rows in prediction_rows_by_variant.items():
        _write_jsonl(predictions_dir / f"{variant_name}.jsonl", rows)
    artifact_files = [
        json_path,
        md_path,
        component_path,
        variant_path,
        sample_cache_path,
        *sorted(predictions_dir.glob("*.jsonl")),
    ]
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": top2._display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in artifact_files
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
    payload = build_scout(
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
