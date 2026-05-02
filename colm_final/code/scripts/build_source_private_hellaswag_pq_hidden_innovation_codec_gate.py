from __future__ import annotations

"""Product-quantized TinyLlama hidden-code scout for HellaSwag."""

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

from scripts import build_source_private_hellaswag_hidden_code_packet_scout as hidden_code  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as hidden_summary  # noqa: E402
from scripts import build_source_private_hellaswag_learned_source_code_packet_gate as source_code  # noqa: E402
from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as wz  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048"
)
DEFAULT_EVAL_FULL = hidden_code.DEFAULT_EVAL_FULL
DEFAULT_EVAL_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048/"
    "source_eval_hidden_cache.npz"
)
DEFAULT_SOURCE_MODEL = hidden_code.DEFAULT_SOURCE_MODEL
DEFAULT_PCA_DIMS = (8, 16, 32)
DEFAULT_SUBSPACES = (2, 3, 4)
DEFAULT_SUBCLUSTERS = (2, 4, 8)
DEFAULT_DECODER_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
STRICT_DELTA = 0.010
BEST_PRIOR_HIDDEN_SLICE_SCOUT_ACCURACY = 0.511719
STRICT_PRIOR_SCOUT_DELTA = 0.005
CONTROL_TOLERANCE = 0.002
CONTROL_SEPARATION_DELTA = 0.003
RAW_PACKET_BYTES = 1
FRAMED_PACKET_BYTES = 4
CANDIDATE_COUNT = 4
MAX_CODEBOOK_SIZE = 256


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
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one float is required")
    return parsed


def _standardize_fit(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_values[fit_indices], axis=0)
    scale = np.std(train_values[fit_indices], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train_values - mean) / scale, (eval_values - mean) / scale, mean, scale


def _orthogonal_rotation(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(int(dim), int(dim)))
    q, r = np.linalg.qr(random)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs[None, :]).astype(np.float64)


def _subspace_slices(dim: int, subspaces: int) -> list[slice]:
    if subspaces <= 0 or subspaces > dim:
        raise ValueError("subspaces must be in [1, dim]")
    bounds = np.linspace(0, int(dim), int(subspaces) + 1, dtype=np.int64)
    return [slice(int(bounds[index]), int(bounds[index + 1])) for index in range(int(subspaces))]


def _fit_product_quantizer(
    *,
    train_projected: np.ndarray,
    fit_indices: np.ndarray,
    subspaces: int,
    clusters: int,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    slices = _subspace_slices(train_projected.shape[1], int(subspaces))
    centers: list[np.ndarray] = []
    for index, item in enumerate(slices):
        centers.append(
            hidden_code._fit_kmeans(
                train_projected[:, item],
                fit_indices,
                int(clusters),
                seed=int(seed) + 1009 * index,
                iterations=int(iterations),
            )
        )
    return {
        "subspaces": int(subspaces),
        "clusters": int(clusters),
        "slices": [(int(item.start), int(item.stop)) for item in slices],
        "centers": centers,
    }


def _apply_product_quantizer(values: np.ndarray, quantizer: dict[str, Any]) -> np.ndarray:
    clusters = int(quantizer["clusters"])
    code = np.zeros(values.shape[0], dtype=np.int64)
    for item, centers in zip(quantizer["slices"], quantizer["centers"], strict=True):
        start, stop = item
        local_code = hidden_code._nearest_center_codes(values[:, int(start) : int(stop)], centers)
        code = code * clusters + local_code.astype(np.int64)
    return code.astype(np.int64)


def _encode_pq_hidden_codes(
    *,
    config: dict[str, Any],
    train_hidden_features: np.ndarray,
    eval_hidden_features: np.ndarray,
    train_packet: np.ndarray,
    eval_packet: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    pca = hidden_code._fit_pca(train_hidden_features, fit_indices, int(config["pca_dims"]))
    train_projected = hidden_code._project_pca(train_hidden_features, pca)
    eval_projected = hidden_code._project_pca(eval_hidden_features, pca)
    train_projected, eval_projected, mean, scale = _standardize_fit(
        train_projected,
        eval_projected,
        fit_indices,
    )
    rotation_kind = str(config["rotation"])
    rotation = None
    if rotation_kind == "orthogonal":
        rotation = _orthogonal_rotation(train_projected.shape[1], int(config["rotation_seed"]))
        train_projected = train_projected @ rotation
        eval_projected = eval_projected @ rotation
    elif rotation_kind != "identity":
        raise ValueError(f"unsupported rotation kind: {rotation_kind}")
    quantizer = _fit_product_quantizer(
        train_projected=train_projected,
        fit_indices=fit_indices,
        subspaces=int(config["subspaces"]),
        clusters=int(config["clusters"]),
        seed=int(config["seed"]),
        iterations=int(config["iterations"]),
    )
    train_subcode = _apply_product_quantizer(train_projected, quantizer)
    eval_subcode = _apply_product_quantizer(eval_projected, quantizer)
    codebook_size = int(config["clusters"]) ** int(config["subspaces"]) * CANDIDATE_COUNT
    if codebook_size > MAX_CODEBOOK_SIZE:
        raise ValueError(f"codebook size {codebook_size} exceeds one-byte max {MAX_CODEBOOK_SIZE}")
    return {
        "train_code": (train_subcode * CANDIDATE_COUNT + train_packet).astype(np.int64),
        "eval_code": (eval_subcode * CANDIDATE_COUNT + eval_packet).astype(np.int64),
        "codebook_size": codebook_size,
        "encoder_audit": {
            "kind": "product_quantized_hidden_code",
            "pca_dims": int(pca["dims"]),
            "subspaces": int(config["subspaces"]),
            "clusters": int(config["clusters"]),
            "rotation": rotation_kind,
            "rotation_seed": int(config["rotation_seed"]),
            "projected_mean": [float(item) for item in mean[: min(8, len(mean))]],
            "projected_scale": [float(item) for item in scale[: min(8, len(scale))]],
            "rotation_shape": list(rotation.shape) if rotation is not None else None,
            "center_shapes": [list(item.shape) for item in quantizer["centers"]],
        },
    }


def _decode_rows(
    *,
    train_code: np.ndarray,
    eval_code: np.ndarray,
    codebook_size: int,
    calibration: dict[str, Any],
    validation: dict[str, Any],
    fit_indices: np.ndarray,
    ridge: float,
    label_permutation_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = source_code._candidate_decoder_features(
        qwen_scores=calibration["qwen_scores"],
        qwen_target=calibration["qwen_target"],
        qwen_mean=calibration["qwen_mean"],
        qwen_hybrid=calibration["qwen_hybrid"],
        source_code=train_code,
        codebook_size=int(codebook_size),
    )
    eval_features = source_code._candidate_decoder_features(
        qwen_scores=validation["qwen_scores"],
        qwen_target=validation["alternatives"]["qwen_target_score"],
        qwen_mean=validation["alternatives"]["mean_zscore_prediction"],
        qwen_hybrid=validation["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
        source_code=eval_code,
        codebook_size=int(codebook_size),
    )
    coef = wz._fit_candidate_decoder(
        train_features=train_features,
        train_answers=calibration["answers"],
        fit_indices=fit_indices,
        ridge=float(ridge),
        label_permutation_seed=label_permutation_seed,
    )
    return wz._predict_candidate_decoder(train_features, coef), wz._predict_candidate_decoder(eval_features, coef)


def _evaluate_config(
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
    encoded = _encode_pq_hidden_codes(
        config=config,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        fit_indices=fit_indices,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for ridge_index, ridge in enumerate(decoder_ridges):
        train_predictions, eval_predictions = _decode_rows(
            train_code=encoded["train_code"],
            eval_code=encoded["eval_code"],
            codebook_size=int(encoded["codebook_size"]),
            calibration=calibration,
            validation=validation,
            fit_indices=fit_indices,
            ridge=float(ridge),
        )
        predictions[(str(config["name"]), float(ridge))] = eval_predictions
        rows.append(
            wz._score_row(
                name="pq_hidden_innovation_decoder",
                predictions=eval_predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=row_seed_offset + ridge_index,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(config["name"]),
                    "encoder_kind": "product_quantized_hidden_code",
                    "pca_dims": int(config["pca_dims"]),
                    "subspaces": int(config["subspaces"]),
                    "clusters": int(config["clusters"]),
                    "rotation": str(config["rotation"]),
                    "ridge": float(ridge),
                    "codebook_size": int(encoded["codebook_size"]),
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
    encoded = _encode_pq_hidden_codes(
        config=selected_config,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features,
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"],
        fit_indices=fit_indices,
    )
    codebook_size = int(encoded["codebook_size"])
    rng = np.random.default_rng(control_seed)
    shuffled_order = rng.permutation(len(validation["answers"]))
    shuffled_encoded = _encode_pq_hidden_codes(
        config=selected_config,
        train_hidden_features=train_hidden_features,
        eval_hidden_features=eval_hidden_features[shuffled_order],
        train_packet=calibration["tiny_packet"],
        eval_packet=validation["packet"][shuffled_order],
        fit_indices=fit_indices,
    )
    permutation = rng.permutation(codebook_size).astype(np.int64)
    control_specs = [
        ("row_shuffle_pq_code", encoded["eval_code"][rng.permutation(len(encoded["eval_code"]))], codebook_size),
        ("hidden_feature_shuffle_before_encoding", shuffled_encoded["eval_code"], codebook_size),
        ("codebook_permutation_mismatch", permutation[encoded["eval_code"]], codebook_size),
        (
            "random_same_byte_code",
            rng.integers(0, codebook_size, size=len(validation["answers"]), dtype=np.int64),
            codebook_size,
        ),
        ("candidate_only_code", validation["packet"].astype(np.int64), CANDIDATE_COUNT),
        ("zero_source_code", np.zeros(len(validation["answers"]), dtype=np.int64), 1),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (name, eval_code, local_codebook_size) in enumerate(control_specs):
        _, predictions = _decode_rows(
            train_code=encoded["train_code"] % int(local_codebook_size),
            eval_code=eval_code.astype(np.int64) % int(local_codebook_size),
            codebook_size=int(local_codebook_size),
            calibration=calibration,
            validation=validation,
            fit_indices=fit_indices,
            ridge=float(selected_ridge),
        )
        rows.append(
            wz._score_row(
                name=name,
                predictions=predictions,
                answers=validation["answers"],
                packet_predictions=validation["packet"],
                qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
                seed=50100 + offset,
                bootstrap_samples=bootstrap_samples,
                extra={
                    "encoder_name": str(selected_config["name"]),
                    "ridge": float(selected_ridge),
                    "codebook_size": int(local_codebook_size),
                },
            )
        )
    _, label_permutation_predictions = _decode_rows(
        train_code=encoded["train_code"],
        eval_code=encoded["eval_code"],
        codebook_size=codebook_size,
        calibration=calibration,
        validation=validation,
        fit_indices=fit_indices,
        ridge=float(selected_ridge),
        label_permutation_seed=control_seed + 77,
    )
    rows.append(
        wz._score_row(
            name="label_permutation_decoder",
            predictions=label_permutation_predictions,
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=50180,
            bootstrap_samples=bootstrap_samples,
            extra={
                "encoder_name": str(selected_config["name"]),
                "ridge": float(selected_ridge),
                "codebook_size": codebook_size,
            },
        )
    )
    rows.append(
        wz._score_row(
            name="packet_only",
            predictions=validation["packet"],
            answers=validation["answers"],
            packet_predictions=validation["packet"],
            qwen_target_predictions=validation["alternatives"]["qwen_target_score"],
            seed=50181,
            bootstrap_samples=bootstrap_samples,
            extra={"encoder_name": "baseline", "ridge": 0.0, "codebook_size": CANDIDATE_COUNT},
        )
    )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag PQ Hidden Innovation Codec Gate",
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
    eval_full_path: pathlib.Path = DEFAULT_EVAL_FULL,
    eval_slice_start: int = 1024,
    eval_slice_rows: int = 1024,
    eval_hidden_cache: pathlib.Path = DEFAULT_EVAL_HIDDEN_CACHE,
    pca_dims: tuple[int, ...] = DEFAULT_PCA_DIMS,
    subspaces: tuple[int, ...] = DEFAULT_SUBSPACES,
    subclusters: tuple[int, ...] = DEFAULT_SUBCLUSTERS,
    decoder_ridges: tuple[float, ...] = DEFAULT_DECODER_RIDGES,
    kmeans_seed: int = 71,
    kmeans_iterations: int = 25,
    rotation_seed: int = 411,
    bootstrap_samples: int = 500,
    control_seed: int = 5017,
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
    slice_meta = hidden_code._slice_jsonl(
        source_path=eval_full_path,
        output_path=slice_path,
        start=eval_slice_start,
        count=eval_slice_rows,
    )
    eval_rows = arc_gate._load_rows(slice_path)
    hidden_npz = _resolve(eval_hidden_cache)
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
    validation_row_ids = validation_full["row_ids"][eval_slice_start : eval_slice_start + eval_slice_rows]
    if [str(row.row_id) for row in eval_rows] != [str(row_id) for row_id in validation_row_ids]:
        raise ValueError("eval hidden slice rows do not align with validation packet bundle")
    validation_slice = {
        **validation_full,
        "rows": validation_full["rows"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "row_ids": validation_row_ids,
        "answers": validation_full["answers"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "packet": validation_full["packet"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "packet_margin": validation_full["packet_margin"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "qwen_scores": validation_full["qwen_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        "alternatives": {
            key: value[eval_slice_start : eval_slice_start + eval_slice_rows]
            for key, value in validation_full["alternatives"].items()
        },
    }
    surfaces = {**surfaces_full, "validation": validation_slice}
    calibration = surfaces["calibration"]
    train_hidden, train_hidden_audit = hidden_code._tiny_train_hidden_matrix(
        calibration_rows=calibration["rows"],
        train_path=wz.DEFAULT_TRAIN_PATH,
        tiny_train_cache_dir=wz.DEFAULT_TINY_TRAIN_CACHE_DIR,
        sample_seeds=wz.DEFAULT_SAMPLE_SEEDS,
        train_hidden_rows=512,
    )
    train_hidden_features = hidden_code._hidden_source_feature_matrix(
        hidden=train_hidden,
        scores=surfaces["tiny_train_scores"],
        packet=calibration["tiny_packet"],
    )
    eval_hidden_features = hidden_code._hidden_source_feature_matrix(
        hidden=eval_hidden,
        scores=surfaces["tiny_eval_scores"][eval_slice_start : eval_slice_start + eval_slice_rows],
        packet=validation_slice["packet"],
    )
    configs: list[dict[str, Any]] = []
    for dim in pca_dims:
        for subspace_count in subspaces:
            if int(subspace_count) > int(dim):
                continue
            for cluster_count in subclusters:
                codebook_size = int(cluster_count) ** int(subspace_count) * CANDIDATE_COUNT
                if codebook_size > MAX_CODEBOOK_SIZE:
                    continue
                for rotation in ("identity", "orthogonal"):
                    configs.append(
                        {
                            "name": (
                                f"pq_pca{int(dim)}_m{int(subspace_count)}_k{int(cluster_count)}"
                                f"_{rotation}"
                            ),
                            "pca_dims": int(dim),
                            "subspaces": int(subspace_count),
                            "clusters": int(cluster_count),
                            "rotation": rotation,
                            "rotation_seed": int(rotation_seed + int(dim) + 17 * int(subspace_count)),
                            "seed": int(kmeans_seed + int(dim) + 101 * int(subspace_count) + int(cluster_count)),
                            "iterations": int(kmeans_iterations),
                            "codebook_size": codebook_size,
                        }
                    )
    frontier_rows: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[str, float], np.ndarray] = {}
    encoded_by_name: dict[str, dict[str, Any]] = {}
    for index, config in enumerate(configs):
        rows, predictions, encoded = _evaluate_config(
            config=config,
            surfaces=surfaces,
            train_hidden_features=train_hidden_features,
            eval_hidden_features=eval_hidden_features,
            decoder_ridges=decoder_ridges,
            bootstrap_samples=bootstrap_samples,
            row_seed_offset=51000 + 100 * index,
        )
        frontier_rows.extend(rows)
        predictions_by_key.update(predictions)
        encoded_by_name[str(config["name"])] = encoded | {"config": config}
    if not frontier_rows:
        raise ValueError("no PQ hidden-code rows were evaluated")
    default_row = max(
        frontier_rows,
        key=lambda row: (
            row["official_dev_accuracy"],
            row["official_dev_delta_vs_packet"],
            -row["codebook_size"],
            row["rotation"] == "orthogonal",
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
    prior_scout_gate = bool(
        default_row["accuracy"] - BEST_PRIOR_HIDDEN_SLICE_SCOUT_ACCURACY >= STRICT_PRIOR_SCOUT_DELTA
    )
    default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
        and block_stability_gate
        and control_separation_gate
        and prior_scout_gate
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    pass_gate = bool(default_pass_gate)
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
        "qwen_hybrid_accuracy": wz._accuracy(
            validation_slice["alternatives"]["hybrid_vote_on_score_agreement_prediction"],
            validation_slice["answers"],
        ),
        "default_encoder_name": str(default_row["encoder_name"]),
        "default_codebook_size": int(default_row["codebook_size"]),
        "default_eval_code_unique_count": int(default_row["eval_code_unique_count"]),
        "default_accuracy": default_row["accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_best_prior_hidden_slice_scout": default_row["accuracy"]
        - BEST_PRIOR_HIDDEN_SLICE_SCOUT_ACCURACY,
        "default_ridge": default_row["ridge"],
        "best_scout_encoder_name": str(best_scout["encoder_name"]),
        "best_scout_accuracy": best_scout["accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "best_prior_hidden_slice_scout_accuracy": BEST_PRIOR_HIDDEN_SLICE_SCOUT_ACCURACY,
        "control_max_delta_vs_packet_only": control_max_delta,
        "block_stability_gate": block_stability_gate,
        "control_separation_gate": control_separation_gate,
        "prior_scout_gate": prior_scout_gate,
        "default_pass_gate": default_pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "raw_payload_bytes": RAW_PACKET_BYTES,
        "framed_record_bytes": FRAMED_PACKET_BYTES,
        "source_hidden_cache_hit": bool(eval_hidden_model.get("cache_hit")),
        "source_hidden_extraction_wall_time_s": float(eval_hidden_model.get("latency_s") or 0.0),
    }
    payload = {
        "gate": "source_private_hellaswag_pq_hidden_innovation_codec_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if the official-train-dev-selected PQ hidden code beats compact packet-only by "
            ">=0.010 with positive paired CI95 low, beats the prior hidden-code slice scout by "
            ">=0.005, is positive on at least 4/5 contiguous blocks, and separates from destructive "
            "controls."
        ),
        "packet_contract": {
            "packet_name": "product_quantized_hidden_innovation_code",
            "raw_payload_bytes": RAW_PACKET_BYTES,
            "framed_record_bytes": FRAMED_PACKET_BYTES,
            "max_codebook_size": MAX_CODEBOOK_SIZE,
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
            "logical_validation_framed_record_bytes_total": int(
                len(validation_slice["answers"]) * FRAMED_PACKET_BYTES
            ),
            "communication_object": "task_level_source_private_pq_hidden_code",
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
        "lay_explanation": (
            "The experiment asks whether TinyLlama can send more than just its answer choice without "
            "sending a hidden vector. It compresses the hidden-state residual into several tiny "
            "subcodes, packs those subcodes plus the answer id into one byte, and lets Qwen decode it "
            "with its own scores."
        ),
        "interpretation": (
            "This is the cached Mac-local product-quantization branch suggested by the hidden-code "
            "and TurboQuant/PQ literature. A pass would justify materializing more TinyLlama hidden "
            "validation caches. A failure means factorized source-hidden codebooks still do not add "
            "stable task information beyond the compact candidate packet on this HellaSwag slice."
        ),
    }
    json_path = output_dir / "hellaswag_pq_hidden_innovation_codec_gate.json"
    md_path = output_dir / "hellaswag_pq_hidden_innovation_codec_gate.md"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-full-path", type=pathlib.Path, default=DEFAULT_EVAL_FULL)
    parser.add_argument("--eval-slice-start", type=int, default=1024)
    parser.add_argument("--eval-slice-rows", type=int, default=1024)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--pca-dims", type=_parse_int_tuple, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--subspaces", type=_parse_int_tuple, default=DEFAULT_SUBSPACES)
    parser.add_argument("--subclusters", type=_parse_int_tuple, default=DEFAULT_SUBCLUSTERS)
    parser.add_argument("--decoder-ridges", type=_parse_float_tuple, default=DEFAULT_DECODER_RIDGES)
    parser.add_argument("--kmeans-seed", type=int, default=71)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--rotation-seed", type=int, default=411)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--control-seed", type=int, default=5017)
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
        subspaces=args.subspaces,
        subclusters=args.subclusters,
        decoder_ridges=args.decoder_ridges,
        kmeans_seed=args.kmeans_seed,
        kmeans_iterations=args.kmeans_iterations,
        rotation_seed=args.rotation_seed,
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
