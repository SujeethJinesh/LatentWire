from __future__ import annotations

"""Mac-local TinyLlama hidden-state source-code packet scout for HellaSwag."""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as hidden_summary  # noqa: E402
from scripts import build_source_private_hellaswag_learned_source_code_packet_gate as source_code  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048"
)
DEFAULT_EVAL_FULL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_SOURCE_MODEL = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/"
    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)
DEFAULT_PCA_DIMS = (4, 8, 16)
DEFAULT_CLUSTER_COUNTS = (4, 8, 16, 32, 64)
DEFAULT_RELIABILITY_BINS = (2, 4, 8, 16, 32, 64)
DEFAULT_RELIABILITY_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_DECODER_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_MAX_CODES = 256
STRICT_DELTA = 0.010
BEST_PRIOR_RECEIVER_SCOUT_ACCURACY = 0.620594
STRICT_PRIOR_SCOUT_DELTA = 0.005
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 1
FRAMED_PACKET_BYTES = 4
CANDIDATE_COUNT = 4


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return wz._resolve(path)


def _display_path(path: pathlib.Path | str) -> str:
    return wz._display_path(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _slice_jsonl(
    *,
    source_path: pathlib.Path,
    output_path: pathlib.Path,
    start: int,
    count: int,
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("slice start must be non-negative")
    if count <= 0:
        raise ValueError("slice count must be positive")
    selected: list[str] = []
    total_rows = 0
    source_path = _resolve(source_path)
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                total_rows += 1
            if start <= index < start + count and line.strip():
                selected.append(line.rstrip("\n"))
    if len(selected) != count:
        raise ValueError(f"requested {count} rows at {start}, got {len(selected)}")
    output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return {
        "source_path": _display_path(source_path),
        "slice_path": _display_path(output_path),
        "source_total_rows": total_rows,
        "slice_start": start,
        "slice_end_exclusive": start + count,
        "slice_rows": len(selected),
        "slice_sha256": _sha256_file(output_path),
    }


def _tiny_train_hidden_matrix(
    *,
    calibration_rows: list[dict[str, Any]],
    train_path: pathlib.Path,
    tiny_train_cache_dir: pathlib.Path,
    sample_seeds: tuple[int, ...],
    train_hidden_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_train_rows = official.arc_gate._load_rows(_resolve(train_path))
    hidden_by_row_id: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for seed in sample_seeds:
        sample = official._load_sample_cache(
            cache_dir=tiny_train_cache_dir,
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for row_id, hidden in zip(sample["row_ids"], sample["receiver_hidden"], strict=True):
            hidden_by_row_id.setdefault(str(row_id), np.asarray(hidden, dtype=np.float64))
        audit.append(
            {
                "sample_seed": int(seed),
                "row_count": int(len(sample["row_ids"])),
                "hidden_cache": sample["hidden_path"],
                "hidden_cache_sha256": sample["hidden_sha256"],
            }
        )
    missing = [str(row["row_id"]) for row in calibration_rows if str(row["row_id"]) not in hidden_by_row_id]
    if missing:
        raise ValueError(f"missing TinyLlama hidden rows for calibration rows: {len(missing)}")
    return (
        np.stack([hidden_by_row_id[str(row["row_id"])] for row in calibration_rows]).astype(np.float64),
        audit,
    )


def _hidden_source_feature_matrix(
    *,
    hidden: np.ndarray,
    scores: np.ndarray,
    packet: np.ndarray,
) -> np.ndarray:
    layer_hidden = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
    packet = packet.astype(np.int64)
    packet_hidden = layer_hidden[np.arange(len(packet)), packet]
    mean_hidden = np.mean(layer_hidden, axis=1)
    score_top = np.argmax(scores, axis=1).astype(np.int64)
    top_hidden = layer_hidden[np.arange(len(packet)), score_top]
    features = np.concatenate(
        [
            packet_hidden - mean_hidden,
            top_hidden - packet_hidden,
        ],
        axis=1,
    ).astype(np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms < 1e-8, 1.0, norms)


def _fit_pca(
    features: np.ndarray,
    fit_indices: np.ndarray,
    dims: int,
) -> dict[str, Any]:
    if dims <= 0:
        raise ValueError("PCA dimension must be positive")
    fit_features = features[fit_indices]
    mean = np.mean(fit_features, axis=0)
    centered = fit_features - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(int(dims), vt.shape[0])].astype(np.float64)
    return {
        "mean": mean.astype(np.float64),
        "components": components,
        "dims": int(components.shape[0]),
    }


def _project_pca(features: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    return (features - pca["mean"]) @ pca["components"].T


def _fit_kmeans(
    features: np.ndarray,
    fit_indices: np.ndarray,
    clusters: int,
    *,
    seed: int,
    iterations: int,
) -> np.ndarray:
    fit_features = features[fit_indices]
    if clusters < 2 or clusters > len(fit_features):
        raise ValueError("invalid cluster count")
    rng = np.random.default_rng(seed)
    centers = fit_features[rng.choice(len(fit_features), size=int(clusters), replace=False)].copy()
    for _ in range(int(iterations)):
        distances = np.sum((fit_features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        for cluster_index in range(int(clusters)):
            local = labels == cluster_index
            if np.any(local):
                centers[cluster_index] = np.mean(fit_features[local], axis=0)
    return centers.astype(np.float64)


def _nearest_center_codes(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)


def _fit_reliability_scorer(
    *,
    train_hidden_features: np.ndarray,
    train_packet: np.ndarray,
    train_answers: np.ndarray,
    fit_indices: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    labels = (train_packet.astype(np.int64) == train_answers.astype(np.int64)).astype(np.float64)
    mean = np.mean(train_hidden_features[fit_indices], axis=0)
    scale = np.std(train_hidden_features[fit_indices], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_body = (train_hidden_features - mean) / scale
    x = np.concatenate([np.ones((len(x_body), 1), dtype=np.float64), x_body], axis=1)
    fit = fit_indices.astype(np.int64)
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x[fit].T @ x[fit] + reg, x[fit].T @ labels[fit])
    return {
        "weights": weights.astype(np.float64),
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "ridge": float(ridge),
        "fit_label_mean": float(np.mean(labels[fit])),
    }


def _predict_reliability(features: np.ndarray, scorer: dict[str, Any]) -> np.ndarray:
    x_body = (features - scorer["mean"]) / scorer["scale"]
    x = np.concatenate([np.ones((len(x_body), 1), dtype=np.float64), x_body], axis=1)
    return (x @ scorer["weights"]).astype(np.float64)


def _fit_quantile_edges(values: np.ndarray, fit_indices: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("quantile encoder requires at least two bins")
    edges = np.quantile(values[fit_indices], np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges.astype(np.float64)


def _apply_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2).astype(
        np.int64
    )


def _encode_hidden_codes(
    *,
    config: dict[str, Any],
    train_hidden_features: np.ndarray,
    eval_hidden_features: np.ndarray,
    train_packet: np.ndarray,
    eval_packet: np.ndarray,
    train_answers: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    kind = str(config["kind"])
    if kind == "pca_kmeans_hidden_code":
        pca = _fit_pca(train_hidden_features, fit_indices, int(config["pca_dims"]))
        train_projected = _project_pca(train_hidden_features, pca)
        eval_projected = _project_pca(eval_hidden_features, pca)
        centers = _fit_kmeans(
            train_projected,
            fit_indices,
            int(config["clusters"]),
            seed=int(config["seed"]),
            iterations=int(config["iterations"]),
        )
        train_cluster = _nearest_center_codes(train_projected, centers)
        eval_cluster = _nearest_center_codes(eval_projected, centers)
        codebook_size = int(config["clusters"]) * CANDIDATE_COUNT
        return {
            "train_code": (train_cluster * CANDIDATE_COUNT + train_packet).astype(np.int64),
            "eval_code": (eval_cluster * CANDIDATE_COUNT + eval_packet).astype(np.int64),
            "codebook_size": codebook_size,
            "encoder_audit": {
                "kind": kind,
                "pca_dims": int(pca["dims"]),
                "clusters": int(config["clusters"]),
                "seed": int(config["seed"]),
                "iterations": int(config["iterations"]),
                "projected_center_shape": list(centers.shape),
            },
        }
    if kind == "reliability_quantile_hidden_code":
        bins = int(config["bins"])
        scorer = _fit_reliability_scorer(
            train_hidden_features=train_hidden_features,
            train_packet=train_packet,
            train_answers=train_answers,
            fit_indices=fit_indices,
            ridge=float(config["reliability_ridge"]),
        )
        train_score = _predict_reliability(train_hidden_features, scorer)
        eval_score = _predict_reliability(eval_hidden_features, scorer)
        edges = _fit_quantile_edges(train_score, fit_indices, bins)
        train_bin = _apply_edges(train_score, edges)
        eval_bin = _apply_edges(eval_score, edges)
        return {
            "train_code": (train_bin * CANDIDATE_COUNT + train_packet).astype(np.int64),
            "eval_code": (eval_bin * CANDIDATE_COUNT + eval_packet).astype(np.int64),
            "codebook_size": int(bins * CANDIDATE_COUNT),
            "encoder_audit": {
                "kind": kind,
                "bins": bins,
                "reliability_ridge": float(config["reliability_ridge"]),
                "edges": [float(item) for item in edges],
                "fit_label_mean": float(scorer["fit_label_mean"]),
                "train_score_mean": float(np.mean(train_score)),
                "train_score_std": float(np.std(train_score)),
                "eval_score_mean": float(np.mean(eval_score)),
                "eval_score_std": float(np.std(eval_score)),
            },
        }
    raise ValueError(f"unsupported encoder kind: {kind}")


def _evaluate_hidden_config(
    *,
    config: dict[str, Any],
    surfaces: dict[str, Any],
    train_hidden_features: np.ndarray,
    eval_hidden_features: np.ndarray,
    decoder_ridges: tuple[float, ...],
    bootstrap_samples: int,
    row_seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    dev_indices = surfaces["dev_indices"]
    encoded = _encode_hidden_codes(
        config=config,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
    )
    codebook_size = int(encoded["codebook_size"])
    train_features = source_code._candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=encoded["train_code"],
        codebook_size=codebook_size,
    )
    eval_features = source_code._candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=encoded["eval_code"],
        codebook_size=codebook_size,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for ridge in decoder_ridges:
        coef = wz._fit_candidate_decoder(
            train_features=train_features,
            train_answers=calibration["answers"],
            fit_indices=fit_indices,
            ridge=float(ridge),
        )
        train_predictions = wz._predict_candidate_decoder(train_features, coef)
        eval_predictions = wz._predict_candidate_decoder(eval_features, coef)
        predictions[(str(config["name"]), float(ridge))] = eval_predictions
        rows.append(
            wz._score_row(
                name="hidden_source_code_decoder",
                predictions=eval_predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=row_seed_offset + len(rows),
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(config["name"]),
                    "encoder_kind": str(config["kind"]),
                    "pca_dims": int(config.get("pca_dims", 0)),
                    "clusters": int(config.get("clusters", 0)),
                    "reliability_bins": int(config.get("bins", 0)),
                    "reliability_ridge": float(config.get("reliability_ridge", 0.0)),
                    "ridge": float(ridge),
                    "codebook_size": codebook_size,
                    "official_fit_accuracy": wz._accuracy(
                        train_predictions[fit_indices],
                        calibration["answers"][fit_indices],
                    ),
                    "official_dev_accuracy": wz._accuracy(
                        train_predictions[dev_indices],
                        calibration["answers"][dev_indices],
                    ),
                    "official_dev_delta_vs_packet": wz._accuracy(
                        train_predictions[dev_indices],
                        calibration["answers"][dev_indices],
                    )
                    - wz._accuracy(
                        calibration["tiny_packet"][dev_indices],
                        calibration["answers"][dev_indices],
                    ),
                    "eval_code_unique_count": int(len(np.unique(encoded["eval_code"]))),
                },
            )
        )
    return rows, predictions, encoded


def _control_rows(
    *,
    selected_config: dict[str, Any],
    selected_ridge: float,
    surfaces: dict[str, Any],
    train_hidden_features: np.ndarray,
    eval_hidden_features: np.ndarray,
    bootstrap_samples: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    calibration = surfaces["calibration"]
    validation = surfaces["validation"]
    fit_indices = surfaces["fit_indices"]
    encoded = _encode_hidden_codes(
        config=selected_config,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
    )
    codebook_size = int(encoded["codebook_size"])
    train_features = source_code._candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=encoded["train_code"],
        codebook_size=codebook_size,
    )
    coef = wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rng = np.random.default_rng(control_seed)
    shuffled_order = rng.permutation(len(validation["answers"]))
    shuffled_encoded = _encode_hidden_codes(
        config=selected_config,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features[shuffled_order],
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"][shuffled_order],
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
    )
    code_permutation = rng.permutation(codebook_size).astype(np.int64)
    random_code = rng.integers(0, codebook_size, size=len(validation["answers"]), dtype=np.int64)
    control_specs = [
        ("row_shuffle_hidden_code", encoded["eval_code"][rng.permutation(len(encoded["eval_code"]))]),
        ("hidden_feature_shuffle_before_encoding", np.mod(shuffled_encoded["eval_code"], codebook_size)),
        ("codebook_permutation_mismatch", code_permutation[encoded["eval_code"]]),
        ("random_same_byte_code", random_code),
        ("candidate_only_code", validation["packet"].astype(np.int64)),
        ("zero_source_code", np.zeros(len(validation["answers"]), dtype=np.int64)),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_code) in enumerate(control_specs):
        eval_features = source_code._candidate_decoder_features(
            qwen_scores=validation["qwen_scores"],
            qwen_target=validation["alternatives"]["qwen_target_score"],
            qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
            qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            source_code=np.mod(eval_code.astype(np.int64), codebook_size),
            codebook_size=codebook_size,
        )
        predictions = wz._predict_candidate_decoder(eval_features, coef)
        rows.append(
            wz._score_row(
                name=name,
                predictions=predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=20100 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(selected_config["name"]),
                    "ridge": float(selected_ridge),
                    "codebook_size": codebook_size,
                },
            )
        )
    candidate_train = source_code._candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=calibration["tiny_packet"],
        codebook_size=CANDIDATE_COUNT,
    )
    candidate_eval = source_code._candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=validation["packet"],
        codebook_size=CANDIDATE_COUNT,
    )
    candidate_coef = wz._fit_candidate_decoder(
        train_features=candidate_train,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
    )
    rows.append(
        wz._score_row(
            name="compact_candidate_only_decoder",
            predictions=wz._predict_candidate_decoder(candidate_eval, candidate_coef),
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=20180,
            bootstrap_samples=bootstrap_samples,
            extra={"encoder_name": "compact_candidate_only", "ridge": float(selected_ridge), "codebook_size": 4},
        )
    )
    rows.append(
        wz._score_row(
            name="packet_only",
            predictions=validation["packet"],
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=20181,
            bootstrap_samples=bootstrap_samples,
            extra={"encoder_name": "baseline", "ridge": 0.0, "codebook_size": 4},
        )
    )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Hidden-Code Packet Scout",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval slice: `{h['eval_slice_start']}:{h['eval_slice_end_exclusive']}`",
        f"- default encoder: `{h['default_encoder_name']}`",
        f"- default accuracy: `{h['default_accuracy']:.6f}`",
        f"- packet-only accuracy: `{h['packet_only_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- default CI95 low vs packet-only: `{h['default_ci95_low_vs_packet_only']:.6f}`",
        f"- best scout accuracy: `{h['best_scout_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
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
    eval_full_path: pathlib.Path = DEFAULT_EVAL_FULL,
    eval_slice_start: int = 1024,
    eval_slice_rows: int = 1024,
    eval_hidden_cache: pathlib.Path | None = None,
    pca_dims: tuple[int, ...] = DEFAULT_PCA_DIMS,
    cluster_counts: tuple[int, ...] = DEFAULT_CLUSTER_COUNTS,
    reliability_bins: tuple[int, ...] = DEFAULT_RELIABILITY_BINS,
    reliability_ridges: tuple[float, ...] = DEFAULT_RELIABILITY_RIDGES,
    decoder_ridges: tuple[float, ...] = DEFAULT_DECODER_RIDGES,
    kmeans_seed: int = 11,
    kmeans_iterations: int = 25,
    bootstrap_samples: int = 500,
    control_seed: int = 2017,
    source_lm_model: str = DEFAULT_SOURCE_MODEL,
    source_lm_device: str = "mps",
    source_lm_dtype: str = "float16",
    source_lm_max_length: int = 256,
    source_lm_prompt_mode: str = "continuation",
    hidden_layers: tuple[int, ...] = (-1,),
    local_files_only: bool = True,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    slice_path = output_dir / f"hellaswag_validation_rows_{eval_slice_start}_{eval_slice_start + eval_slice_rows}.jsonl"
    slice_meta = _slice_jsonl(
        source_path=eval_full_path,
        output_path=slice_path,
        start=eval_slice_start,
        count=eval_slice_rows,
    )
    eval_rows = arc_gate._load_rows(slice_path)
    hidden_npz = _resolve(eval_hidden_cache) if eval_hidden_cache else output_dir / "source_eval_hidden_cache.npz"
    hidden_meta = hidden_npz.with_suffix(".json")
    eval_hidden, eval_hidden_model = hidden_summary._source_hidden_features(
        eval_rows,
        npz_path=hidden_npz,
        meta_path=hidden_meta,
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        prompt_mode=source_lm_prompt_mode,
        layers=hidden_layers,
        local_files_only=local_files_only,
    )
    surfaces_full = wz._load_surfaces(
        train_path=wz.DEFAULT_TRAIN_PATH,
        tiny_train_cache_dir=wz.DEFAULT_TINY_TRAIN_CACHE_DIR,
        qwen_train_cache_dir=wz.DEFAULT_QWEN_TRAIN_CACHE_DIR,
        sample_seeds=wz.DEFAULT_SAMPLE_SEEDS,
        split_seeds=wz.DEFAULT_SPLIT_SEEDS,
        ridges=wz.DEFAULT_RIDGES,
        train_hidden_rows=512,
        dev_fraction=0.25,
        tiny_eval_packet_jsonl=wz.DEFAULT_TINY_EVAL_PACKET_JSONL,
        qwen_eval_packet_jsonl=wz.DEFAULT_QWEN_EVAL_PACKET_JSONL,
        qwen_global_artifact=wz.DEFAULT_QWEN_GLOBAL_ARTIFACT,
        tiny_eval_rows=wz.DEFAULT_TINY_EVAL_ROWS,
        tiny_eval_score_cache=wz.DEFAULT_TINY_EVAL_SCORE_CACHE,
        tiny_aggregation_policy="mean_zscore",
    )
    validation_full = surfaces_full["validation"]
    slice_row_ids = [row.row_id for row in eval_rows]
    validation_row_ids = validation_full["row_ids"][eval_slice_start : eval_slice_start + eval_slice_rows]
    if [str(row_id) for row_id in slice_row_ids] != [str(row_id) for row_id in validation_row_ids]:
        raise ValueError("eval hidden slice rows do not align with validation packet bundle")
    validation_slice = {
        **validation_full,
        "rows": validation_full["rows"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "row_ids": validation_row_ids,
        "answers": validation_full["answers"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "packet": validation_full["packet"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "packet_margin": validation_full["packet_margin"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "qwen_scores": validation_full["qwen_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "qwen_hidden": validation_full["qwen_hidden"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "alternatives": {
            key: value[eval_slice_start : eval_slice_start + eval_slice_rows]
            for key, value in validation_full["alternatives"].items()
        },
    }
    surfaces = {**surfaces_full, "validation": validation_slice}
    calibration = surfaces["calibration"]
    train_hidden, train_hidden_audit = _tiny_train_hidden_matrix(
        calibration_rows=calibration["rows"],
        train_path=wz.DEFAULT_TRAIN_PATH,
        tiny_train_cache_dir=wz.DEFAULT_TINY_TRAIN_CACHE_DIR,
        sample_seeds=wz.DEFAULT_SAMPLE_SEEDS,
        train_hidden_rows=512,
    )
    train_hidden_features = _hidden_source_feature_matrix(
        hidden=train_hidden,
        scores=surfaces["tiny_train_scores"],
        packet=calibration["tiny_packet"],
    )
    eval_hidden_features = _hidden_source_feature_matrix(
        hidden=eval_hidden,
        scores=surfaces["tiny_eval_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        packet=validation_slice["packet"],
    )
    configs: list[dict[str, Any]] = []
    for bins in reliability_bins:
        for reliability_ridge in reliability_ridges:
            if int(bins) * CANDIDATE_COUNT <= DEFAULT_MAX_CODES:
                configs.append(
                    {
                        "name": f"hidden_reliability_q{int(bins)}_ridge{float(reliability_ridge):g}",
                        "kind": "reliability_quantile_hidden_code",
                        "bins": int(bins),
                        "reliability_ridge": float(reliability_ridge),
                    }
                )
    for dims in pca_dims:
        for clusters in cluster_counts:
            if int(clusters) * CANDIDATE_COUNT <= DEFAULT_MAX_CODES:
                configs.append(
                    {
                        "name": f"hidden_pca{int(dims)}_kmeans{int(clusters)}",
                        "kind": "pca_kmeans_hidden_code",
                        "pca_dims": int(dims),
                        "clusters": int(clusters),
                        "seed": int(kmeans_seed),
                        "iterations": int(kmeans_iterations),
                    }
                )
    frontier_rows: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[str, float], np.ndarray] = {}
    encoded_by_name: dict[str, dict[str, Any]] = {}
    for config_index, config in enumerate(configs):
        rows, predictions, encoded = _evaluate_hidden_config(
            config=config,
            surfaces=surfaces,
            train_hidden_features=train_hidden_features,
            eval_hidden_features=eval_hidden_features,
            decoder_ridges=decoder_ridges,
            bootstrap_samples=bootstrap_samples,
            row_seed_offset=21000 + config_index * 100,
        )
        frontier_rows.extend(rows)
        predictions_by_key.update(predictions)
        encoded_by_name[str(config["name"])] = encoded | {"config": config}
    if not frontier_rows:
        raise ValueError("no hidden-code frontier rows were evaluated")
    default_row = max(
        frontier_rows,
        key=lambda row: (
            row["official_dev_accuracy"],
            row["official_dev_delta_vs_packet"],
            -row["codebook_size"],
            -math.log10(float(row["ridge"])),
        ),
    )
    best_scout = max(
        frontier_rows,
        key=lambda row: (
            row["delta_vs_packet_only"],
            row["ci95_low_vs_packet_only"],
            row["accuracy"],
            row["official_dev_accuracy"],
        ),
    )
    default_predictions = predictions_by_key[(str(default_row["encoder_name"]), float(default_row["ridge"]))]
    default_blocks = wz._block_rows(
        selected=default_predictions,
        packet=validation_slice["packet"],
        answers=validation_slice["answers"],
    )
    selected_config = encoded_by_name[str(default_row["encoder_name"])]["config"]
    control_rows = _control_rows(
        selected_config=selected_config,
        selected_ridge=float(default_row["ridge"]),
        surfaces=surfaces,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features,
        bootstrap_samples=bootstrap_samples,
        control_seed=control_seed,
    )
    control_max_delta = max(row["delta_vs_packet_only"] for row in control_rows if row["name"] != "packet_only")
    block_stability_gate = bool(sum(row["delta_vs_packet_only"] > 0.0 for row in default_blocks) >= 4)
    control_separation_gate = bool(
        default_row["delta_vs_packet_only"] - control_max_delta >= CONTROL_SEPARATION_DELTA
        and control_max_delta <= CONTROL_TOLERANCE
    )
    prior_receiver_scout_gate = bool(
        default_row["accuracy"] - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY >= STRICT_PRIOR_SCOUT_DELTA
    )
    pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and block_stability_gate
        and control_separation_gate
        and prior_receiver_scout_gate
    )
    packet_only_accuracy = wz._accuracy(validation_slice["packet"], validation_slice["answers"])
    headline = {
        "eval_slice_start": int(eval_slice_start),
        "eval_slice_end_exclusive": int(eval_slice_start + eval_slice_rows),
        "official_train_calibration_rows": int(len(calibration["answers"])),
        "official_train_fit_rows": int(len(surfaces["fit_indices"])),
        "official_train_dev_rows": int(len(surfaces["dev_indices"])),
        "validation_rows": int(len(validation_slice["answers"])),
        "packet_only_accuracy": packet_only_accuracy,
        "qwen_target_accuracy": wz._accuracy(
            validation_slice["alternatives"]["qwen_target_score"],
            validation_slice["answers"],
        ),
        "default_encoder_name": str(default_row["encoder_name"]),
        "default_codebook_size": int(default_row["codebook_size"]),
        "default_eval_code_unique_count": int(default_row["eval_code_unique_count"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_ridge": default_row["ridge"],
        "best_scout_encoder_name": str(best_scout["encoder_name"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_prior_receiver_scout_accuracy": BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "default_delta_vs_best_prior_receiver_scout": default_row["accuracy"]
        - BEST_PRIOR_RECEIVER_SCOUT_ACCURACY,
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "prior_receiver_scout_gate": prior_receiver_scout_gate,
        "raw_payload_bytes": RAW_PACKET_BYTES,
        "framed_record_bytes": FRAMED_PACKET_BYTES,
        "default_pass_gate": pass_gate,
        "scout_pass_gate": bool(
            best_scout["delta_vs_packet_only"] >= STRICT_DELTA
            and best_scout["ci95_low_vs_packet_only"] > 0.0
        ),
        "source_hidden_cache_hit": bool(eval_hidden_model.get("cache_hit")),
        "source_hidden_extraction_wall_time_s": float(eval_hidden_model.get("latency_s") or 0.0),
    }
    payload = {
        "gate": "source_private_hellaswag_hidden_code_packet_scout",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Scout pass if the official-train-dev-selected hidden-code packet beats compact packet-only "
            "by >=0.010 with positive paired CI95 low, beats the prior receiver scout by >=0.005, "
            "is positive on at least 4/5 contiguous blocks, and separates from destructive controls."
        ),
        "packet_contract": {
            "packet_name": "hidden_pca_kmeans_source_code_packet",
            "raw_payload_bytes": RAW_PACKET_BYTES,
            "framed_record_bytes": FRAMED_PACKET_BYTES,
            "max_codebook_size": DEFAULT_MAX_CODES,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "learned_discrete_source_hidden_code_transmitted": True,
            "decoder_uses_qwen_side_information": True,
        },
        "headline": headline,
        "frontier_rows": frontier_rows,
        "default_blocks": default_blocks,
        "control_rows": control_rows,
        "selected_encoder_audit": encoded_by_name[str(default_row["encoder_name"])]["encoder_audit"],
        "slice_metadata": slice_meta,
        "train_hidden_audit": train_hidden_audit,
        "eval_hidden_model": eval_hidden_model,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(len(validation_slice["answers"]) * RAW_PACKET_BYTES),
            "logical_validation_framed_record_bytes_total": int(len(validation_slice["answers"]) * FRAMED_PACKET_BYTES),
            "communication_object": "task_level_source_private_hidden_discrete_packet",
            "communication_objective": "downstream_candidate_decision_accuracy",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "total_wall_time_s": float(time.perf_counter() - started),
        },
        "inputs": {
            "eval_full_path": _display_path(eval_full_path),
            "eval_slice_path": _display_path(slice_path),
            "eval_hidden_cache": _display_path(hidden_npz),
            "eval_hidden_cache_sha256": _sha256_file(hidden_npz),
            "source_model": source_lm_model,
            "train_path": _display_path(wz.DEFAULT_TRAIN_PATH),
            "tiny_train_cache_dir": _display_path(wz.DEFAULT_TINY_TRAIN_CACHE_DIR),
            "qwen_global_artifact": _display_path(wz.DEFAULT_QWEN_GLOBAL_ARTIFACT),
        },
        "interpretation": (
            "This scout tests the next live branch after source-score codes failed: a source-hidden "
            "PCA/k-means discrete code plus a supervised train-only hidden reliability quantizer under "
            "the same one-byte packet contract. A pass would justify materializing full TinyLlama "
            "validation hidden caches; a fail means simple hidden-state codebooks are not enough and "
            "the next branch needs a stronger learned quantizer/crosscoder objective."
        ),
    }
    json_path = output_dir / "hellaswag_hidden_code_packet_scout.json"
    md_path = output_dir / "hellaswag_hidden_code_packet_scout.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "inputs": payload["inputs"],
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-full-path", type=pathlib.Path, default=DEFAULT_EVAL_FULL)
    parser.add_argument("--eval-slice-start", type=int, default=1024)
    parser.add_argument("--eval-slice-rows", type=int, default=1024)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path)
    parser.add_argument("--pca-dims", type=_parse_int_tuple, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--cluster-counts", type=_parse_int_tuple, default=DEFAULT_CLUSTER_COUNTS)
    parser.add_argument("--reliability-bins", type=_parse_int_tuple, default=DEFAULT_RELIABILITY_BINS)
    parser.add_argument("--reliability-ridges", type=_parse_float_tuple, default=DEFAULT_RELIABILITY_RIDGES)
    parser.add_argument("--decoder-ridges", type=_parse_float_tuple, default=DEFAULT_DECODER_RIDGES)
    parser.add_argument("--kmeans-seed", type=int, default=11)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=2017)
    parser.add_argument("--source-lm-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-lm-device", default="mps")
    parser.add_argument("--source-lm-dtype", default="float16")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-prompt-mode", default="continuation")
    parser.add_argument("--hidden-layers", type=_parse_int_tuple, default=(-1,))
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        eval_full_path=args.eval_full_path,
        eval_slice_start=args.eval_slice_start,
        eval_slice_rows=args.eval_slice_rows,
        eval_hidden_cache=args.eval_hidden_cache,
        pca_dims=args.pca_dims,
        cluster_counts=args.cluster_counts,
        reliability_bins=args.reliability_bins,
        reliability_ridges=args.reliability_ridges,
        decoder_ridges=args.decoder_ridges,
        kmeans_seed=args.kmeans_seed,
        kmeans_iterations=args.kmeans_iterations,
        bootstrap_samples=args.bootstrap_samples,
        control_seed=args.control_seed,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        hidden_layers=args.hidden_layers,
        local_files_only=not args.allow_downloads,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
