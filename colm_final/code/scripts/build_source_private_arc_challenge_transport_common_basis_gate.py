from __future__ import annotations

"""ARC transport-style common-basis connector gate.

This gate reuses the cached TinyLlama hidden/query features from the previous
ARC hidden/query common-basis failure. It asks whether a more constrained
geometric connector can recover the Qwen-substituted packet on the strict
TinyLlama-vs-Qwen disagreement surface:

* local barycentric transport in source-feature space;
* QJL-style sign-projected barycentric transport;
* whitened orthogonal Procrustes alignment.

The communicated object remains the same 12B source-private packet. The
transport/procrustes maps are trained only on validation disagreement rows and
are evaluated once on the frozen test disagreement rows.
"""

import argparse
import csv
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

from scripts import build_source_private_arc_challenge_hidden_query_common_basis_gate as hq_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_transport_common_basis_gate_20260502_tinyllama_disagreement"
)
DEFAULT_HIDDEN_QUERY_DIR = pathlib.Path(
    "results/source_private_arc_challenge_hidden_query_common_basis_gate_20260502_tinyllama_disagreement"
)
DEFAULT_SOURCE_FAMILY_GATE_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"
)
DEFAULT_TRAIN_ANCHORS = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_SEEDS = (47, 53, 59, 61, 67)
DEFAULT_VIEWS = ("hidden_residual", "query_residual", "hidden_query_residual")

CSV_COLUMNS = (
    "method",
    "view",
    "param",
    "transform",
    "dev_accuracy_mean",
    "dev_qwen_substituted_accuracy_mean",
    "dev_cached_tiny_packet_accuracy_mean",
    "dev_matched_minus_qwen_substituted_mean",
    "dev_matched_minus_cached_tiny_packet_mean",
    "dev_paired_ci95_low_vs_qwen_substituted_min",
    "selected",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one string is required")
    return values


def _load_state(npz_path: pathlib.Path, meta_path: pathlib.Path) -> hq_gate.HiddenQueryState:
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing hidden/query cache: {npz_path} / {meta_path}")
    with np.load(npz_path) as data:
        hidden = np.asarray(data["hidden"], dtype=np.float64)
        query = np.asarray(data["query"], dtype=np.float64)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return hq_gate.HiddenQueryState(hidden=hidden, query=query, metadata=metadata)


def _standardize_fit(values: np.ndarray, fit_indices: np.ndarray) -> dict[str, np.ndarray]:
    fit = np.asarray(values[fit_indices], dtype=np.float64)
    mean = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean, "scale": scale}


def _standardize_apply(values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - stats["mean"]) / stats["scale"]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1e-12), out=np.zeros_like(values), where=norms > 0)


def _qjl_sign_transform(
    values: np.ndarray,
    *,
    output_dim: int,
    seed: int,
    matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix is None:
        rng = np.random.default_rng(seed)
        matrix = rng.choice((-1.0, 1.0), size=(values.shape[1], int(output_dim))) / np.sqrt(float(output_dim))
    projected = np.asarray(values, dtype=np.float64) @ matrix
    signs = np.where(projected >= 0.0, 1.0, -1.0)
    return _normalize_rows(signs), matrix


def _topk_softmax(similarities: np.ndarray, *, k: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    k = min(int(k), similarities.shape[1])
    if k <= 0:
        raise ValueError("k must be positive")
    top_indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    top_values = np.take_along_axis(similarities, top_indices, axis=1)
    scaled = top_values / max(float(temperature), 1e-6)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return top_indices, weights


def _fit_knn_transport(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    fit_indices: np.ndarray,
    k: int,
    temperature: float,
) -> dict[str, Any]:
    stats = _standardize_fit(source_features, fit_indices)
    x_fit = _normalize_rows(_standardize_apply(source_features[fit_indices], stats))
    y_fit = np.asarray(target_features[fit_indices], dtype=np.float64)
    return {
        "kind": "knn_transport",
        "stats": stats,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "k": int(k),
        "temperature": float(temperature),
        "fit_candidate_rows": int(len(fit_indices)),
    }


def _apply_knn_transport(source_features: np.ndarray, mapper: dict[str, Any]) -> np.ndarray:
    x = _normalize_rows(_standardize_apply(source_features, mapper["stats"]))
    similarities = x @ mapper["x_fit"].T
    top_indices, weights = _topk_softmax(
        similarities,
        k=int(mapper["k"]),
        temperature=float(mapper["temperature"]),
    )
    return np.einsum("nk,nkd->nd", weights, mapper["y_fit"][top_indices])


def _pca_scores(values: np.ndarray, *, dim: int) -> dict[str, Any]:
    mean = np.mean(values, axis=0)
    centered = values - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(int(dim), vt.shape[0])]
    scores = centered @ components.T
    scale = np.std(scores, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean, "components": components, "scale": scale, "scores": scores / scale}


def _fit_procrustes_transport(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    fit_indices: np.ndarray,
    dim: int,
) -> dict[str, Any]:
    source_stats = _standardize_fit(source_features, fit_indices)
    x_raw = _standardize_apply(source_features[fit_indices], source_stats)
    y_raw = np.asarray(target_features[fit_indices], dtype=np.float64)
    x_pca = _pca_scores(x_raw, dim=dim)
    y_pca = _pca_scores(y_raw, dim=dim)
    cross = x_pca["scores"].T @ y_pca["scores"]
    u, singular_values, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    return {
        "kind": "whitened_procrustes",
        "source_stats": source_stats,
        "x_mean": x_pca["mean"],
        "x_components": x_pca["components"],
        "x_scale": x_pca["scale"],
        "y_mean": y_pca["mean"],
        "y_components": y_pca["components"],
        "y_scale": y_pca["scale"],
        "rotation": rotation,
        "dim": int(rotation.shape[0]),
        "singular_values_top8": [float(value) for value in singular_values[:8]],
        "fit_candidate_rows": int(len(fit_indices)),
    }


def _apply_procrustes_transport(source_features: np.ndarray, mapper: dict[str, Any]) -> np.ndarray:
    x_std = _standardize_apply(source_features, mapper["source_stats"])
    x_scores = (x_std - mapper["x_mean"]) @ mapper["x_components"].T
    x_scores = x_scores / mapper["x_scale"]
    y_scores = x_scores @ mapper["rotation"]
    y_scores = y_scores * mapper["y_scale"]
    return y_scores @ mapper["y_components"] + mapper["y_mean"]


def _method_feature_matrix(
    *,
    raw_source_features: np.ndarray,
    transform: str,
    qjl_dim: int,
    qjl_seed: int,
    qjl_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if transform == "raw":
        return raw_source_features, None
    if transform == "qjl_sign":
        return _qjl_sign_transform(raw_source_features, output_dim=qjl_dim, seed=qjl_seed, matrix=qjl_matrix)
    raise ValueError(f"unknown transform {transform!r}")


def _score_frontier_row(dev_aggregate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(dev_aggregate["matched_minus_qwen_substituted_mean"]),
        float(dev_aggregate["paired_ci95_low_vs_qwen_substituted_min"]),
        float(dev_aggregate["matched_minus_cached_tiny_packet_mean"]),
        float(dev_aggregate["matched_accuracy_mean"]),
    )


def _write_frontier_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC Transport Common-Basis Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected method: `{h['selected_method']}`",
        f"- selected view: `{h['selected_view']}`",
        f"- selected transform: `{h['selected_transform']}`",
        f"- selected parameter: `{h['selected_param']}`",
        f"- validation disagreement rows: `{h['validation_disagreement_rows']}`",
        f"- test disagreement rows: `{h['test_disagreement_rows']}`",
        f"- test matched mean: `{h['test_matched_accuracy_mean']:.6f}`",
        f"- test Qwen-substituted mean: `{h['test_qwen_substituted_accuracy_mean']:.6f}`",
        f"- test cached Tiny packet mean: `{h['test_cached_tiny_packet_accuracy_mean']:.6f}`",
        f"- test delta vs Qwen-sub: `{h['test_matched_minus_qwen_substituted_mean']:.6f}`",
        f"- test CI95 low vs Qwen-sub min: `{h['test_paired_ci95_low_vs_qwen_substituted_min']:.6f}`",
        f"- candidate-roll control mean: `{h['candidate_roll_matched_accuracy_mean']:.6f}`",
        f"- spectral-permutation control mean: `{h['spectral_permutation_matched_accuracy_mean']:.6f}`",
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
    hidden_query_dir: pathlib.Path = DEFAULT_HIDDEN_QUERY_DIR,
    source_family_gate_dir: pathlib.Path = DEFAULT_SOURCE_FAMILY_GATE_DIR,
    train_anchor_path: pathlib.Path = DEFAULT_TRAIN_ANCHORS,
    validation_path: pathlib.Path = DEFAULT_VALIDATION,
    test_path: pathlib.Path = DEFAULT_TEST,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    source_views: tuple[str, ...] = DEFAULT_VIEWS,
    knn_values: tuple[int, ...] = (3, 7, 15, 31),
    procrustes_dims: tuple[int, ...] = (16, 32, 64, 96),
    qjl_dims: tuple[int, ...] = (64, 128),
    selection_seed: int = 18013,
    dev_fraction: float = 0.25,
    budget_bytes: int = 12,
    anchor_count: int = 384,
    spectral_dim: int = 96,
    code_dim: int = 96,
    bootstrap_samples: int = 500,
    min_lift_over_target: float = 0.02,
    min_gap_over_control: float = 0.02,
    min_gap_over_text: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_query_dir = _resolve(hidden_query_dir)
    source_family_gate_dir = _resolve(source_family_gate_dir)
    qwen_disagreement_path = source_family_gate_dir / "qwen_disagreement_predictions.jsonl"

    train_anchor_rows = arc_gate._load_rows(_resolve(train_anchor_path))
    validation_full = arc_gate._load_rows(_resolve(validation_path))
    test_full = arc_gate._load_rows(_resolve(test_path))
    selection_seed_value = int(seeds[0])
    validation_ids = hq_gate._load_disagreement_row_ids(
        path=qwen_disagreement_path,
        split="validation",
        seed=selection_seed_value,
        limit=None,
    )
    test_ids = hq_gate._load_disagreement_row_ids(
        path=qwen_disagreement_path,
        split="test",
        seed=selection_seed_value,
        limit=None,
    )
    validation_rows = hq_gate._filter_rows_by_ids(validation_full, validation_ids)
    test_rows = hq_gate._filter_rows_by_ids(test_full, test_ids)
    overlap = sorted({row.content_id for row in validation_rows} & {row.content_id for row in test_rows})

    validation_state = _load_state(
        hidden_query_dir / "tinyllama_validation_disagreement_hidden_query_cache.npz",
        hidden_query_dir / "tinyllama_validation_disagreement_hidden_query_cache.json",
    )
    test_state = _load_state(
        hidden_query_dir / "tinyllama_test_disagreement_hidden_query_cache.npz",
        hidden_query_dir / "tinyllama_test_disagreement_hidden_query_cache.json",
    )
    validation_source_predictions = hq_gate._source_predictions(
        validation_rows,
        hq_gate._read_source_predictions(source_family_gate_dir / "tinyllama_validation" / "source_prediction_cache.jsonl"),
    )
    test_source_predictions = hq_gate._source_predictions(
        test_rows,
        hq_gate._read_source_predictions(source_family_gate_dir / "tinyllama_test" / "source_prediction_cache.jsonl"),
    )

    anchor_texts = arc_gate._choice_pair_texts(train_anchor_rows)
    validation_receiver_features, validation_basis_meta = hq_gate._target_spectra(
        rows=validation_rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
    )
    test_receiver_features, test_basis_meta = hq_gate._target_spectra(
        rows=test_rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
    )
    validation_target_residual_flat = hq_gate._target_residual_flat(validation_rows, validation_receiver_features)
    test_index_prior = arc_gate._index_prior(train_anchor_rows)
    fit_rows, dev_rows = hq_gate._dev_row_split(validation_rows, dev_fraction=dev_fraction, seed=selection_seed)
    fit_flat = hq_gate._row_flat_indices(validation_rows, fit_rows)

    frontier_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for view in source_views:
        raw_validation = hq_gate._flat_source_view(rows=validation_rows, state=validation_state, view=view)
        raw_test = hq_gate._flat_source_view(rows=test_rows, state=test_state, view=view)
        transforms: list[tuple[str, int | None, np.ndarray | None]] = [("raw", None, None)]
        for qjl_dim in qjl_dims:
            transformed_validation, matrix = _method_feature_matrix(
                raw_source_features=raw_validation,
                transform="qjl_sign",
                qjl_dim=int(qjl_dim),
                qjl_seed=selection_seed + int(qjl_dim),
            )
            transforms.append((f"qjl_sign{qjl_dim}", int(qjl_dim), matrix))
            # Store transformed validation in the candidate loop by replacing raw source below.
            del transformed_validation
        for transform_name, qjl_dim, qjl_matrix in transforms:
            if transform_name == "raw":
                validation_source = raw_validation
                test_source = raw_test
                transform_label = "raw"
            else:
                validation_source, _ = _method_feature_matrix(
                    raw_source_features=raw_validation,
                    transform="qjl_sign",
                    qjl_dim=int(qjl_dim),
                    qjl_seed=selection_seed + int(qjl_dim),
                    qjl_matrix=qjl_matrix,
                )
                test_source, _ = _method_feature_matrix(
                    raw_source_features=raw_test,
                    transform="qjl_sign",
                    qjl_dim=int(qjl_dim),
                    qjl_seed=selection_seed + int(qjl_dim),
                    qjl_matrix=qjl_matrix,
                )
                transform_label = str(transform_name)

            for k in knn_values:
                mapper = _fit_knn_transport(
                    source_features=validation_source,
                    target_features=validation_target_residual_flat,
                    fit_indices=fit_flat,
                    k=int(k),
                    temperature=0.10,
                )
                mapped_validation = _apply_knn_transport(validation_source, mapper)
                dev_subset = hq_gate._subset_rows_and_features(
                    rows=validation_rows,
                    source_predictions=validation_source_predictions,
                    receiver_features=validation_receiver_features,
                    mapped_features=mapped_validation,
                    row_indices=dev_rows,
                )
                _, dev_aggregate, _ = hq_gate._evaluate_features(
                    split="validation",
                    rows=dev_subset[0],
                    source_predictions=dev_subset[1],
                    mapped_features=dev_subset[3],
                    receiver_features=dev_subset[2],
                    qwen_disagreement_path=qwen_disagreement_path,
                    index_prior=test_index_prior,
                    seeds=seeds,
                    budget_bytes=budget_bytes,
                    code_dim=code_dim,
                    bootstrap_samples=bootstrap_samples,
                    min_lift_over_target=min_lift_over_target,
                    min_gap_over_control=min_gap_over_control,
                    min_gap_over_text=min_gap_over_text,
                    has_overlap=bool(overlap),
                    variant="dev_transport",
                )
                frontier = {
                    "method": "knn_barycentric_transport",
                    "view": view,
                    "param": f"k={int(k)},temp=0.10",
                    "transform": transform_label,
                    "dev_accuracy_mean": dev_aggregate["matched_accuracy_mean"],
                    "dev_qwen_substituted_accuracy_mean": dev_aggregate["qwen_substituted_accuracy_mean"],
                    "dev_cached_tiny_packet_accuracy_mean": dev_aggregate["cached_tiny_packet_accuracy_mean"],
                    "dev_matched_minus_qwen_substituted_mean": dev_aggregate[
                        "matched_minus_qwen_substituted_mean"
                    ],
                    "dev_matched_minus_cached_tiny_packet_mean": dev_aggregate[
                        "matched_minus_cached_tiny_packet_mean"
                    ],
                    "dev_paired_ci95_low_vs_qwen_substituted_min": dev_aggregate[
                        "paired_ci95_low_vs_qwen_substituted_min"
                    ],
                    "selected": False,
                }
                frontier_rows.append(frontier)
                candidates.append(
                    {
                        "frontier": frontier,
                        "view": view,
                        "transform": transform_label,
                        "qjl_dim": qjl_dim,
                        "qjl_matrix": qjl_matrix,
                        "method": "knn_barycentric_transport",
                        "k": int(k),
                        "sort_key": _score_frontier_row(dev_aggregate),
                    }
                )

        # Procrustes is already an orthogonal common-basis test; run it only on raw features.
        for dim in procrustes_dims:
            mapper = _fit_procrustes_transport(
                source_features=raw_validation,
                target_features=validation_target_residual_flat,
                fit_indices=fit_flat,
                dim=int(dim),
            )
            mapped_validation = _apply_procrustes_transport(raw_validation, mapper)
            dev_subset = hq_gate._subset_rows_and_features(
                rows=validation_rows,
                source_predictions=validation_source_predictions,
                receiver_features=validation_receiver_features,
                mapped_features=mapped_validation,
                row_indices=dev_rows,
            )
            _, dev_aggregate, _ = hq_gate._evaluate_features(
                split="validation",
                rows=dev_subset[0],
                source_predictions=dev_subset[1],
                mapped_features=dev_subset[3],
                receiver_features=dev_subset[2],
                qwen_disagreement_path=qwen_disagreement_path,
                index_prior=test_index_prior,
                seeds=seeds,
                budget_bytes=budget_bytes,
                code_dim=code_dim,
                bootstrap_samples=bootstrap_samples,
                min_lift_over_target=min_lift_over_target,
                min_gap_over_control=min_gap_over_control,
                min_gap_over_text=min_gap_over_text,
                has_overlap=bool(overlap),
                variant="dev_procrustes",
            )
            frontier = {
                "method": "whitened_procrustes",
                "view": view,
                "param": f"dim={int(dim)}",
                "transform": "raw",
                "dev_accuracy_mean": dev_aggregate["matched_accuracy_mean"],
                "dev_qwen_substituted_accuracy_mean": dev_aggregate["qwen_substituted_accuracy_mean"],
                "dev_cached_tiny_packet_accuracy_mean": dev_aggregate["cached_tiny_packet_accuracy_mean"],
                "dev_matched_minus_qwen_substituted_mean": dev_aggregate[
                    "matched_minus_qwen_substituted_mean"
                ],
                "dev_matched_minus_cached_tiny_packet_mean": dev_aggregate[
                    "matched_minus_cached_tiny_packet_mean"
                ],
                "dev_paired_ci95_low_vs_qwen_substituted_min": dev_aggregate[
                    "paired_ci95_low_vs_qwen_substituted_min"
                ],
                "selected": False,
            }
            frontier_rows.append(frontier)
            candidates.append(
                {
                    "frontier": frontier,
                    "view": view,
                    "transform": "raw",
                    "method": "whitened_procrustes",
                    "dim": int(dim),
                    "sort_key": _score_frontier_row(dev_aggregate),
                }
            )

    if not candidates:
        raise ValueError("no connector candidates were evaluated")
    selected = max(candidates, key=lambda candidate: candidate["sort_key"])
    selected["frontier"]["selected"] = True
    selected_view = str(selected["view"])
    raw_validation = hq_gate._flat_source_view(rows=validation_rows, state=validation_state, view=selected_view)
    raw_test = hq_gate._flat_source_view(rows=test_rows, state=test_state, view=selected_view)
    if selected["transform"] == "raw":
        final_validation_source = raw_validation
        final_test_source = raw_test
    else:
        final_validation_source, _ = _method_feature_matrix(
            raw_source_features=raw_validation,
            transform="qjl_sign",
            qjl_dim=int(selected["qjl_dim"]),
            qjl_seed=selection_seed + int(selected["qjl_dim"]),
            qjl_matrix=selected["qjl_matrix"],
        )
        final_test_source, _ = _method_feature_matrix(
            raw_source_features=raw_test,
            transform="qjl_sign",
            qjl_dim=int(selected["qjl_dim"]),
            qjl_seed=selection_seed + int(selected["qjl_dim"]),
            qjl_matrix=selected["qjl_matrix"],
        )
    all_validation_flat = np.arange(final_validation_source.shape[0], dtype=np.int64)
    if selected["method"] == "knn_barycentric_transport":
        final_mapper = _fit_knn_transport(
            source_features=final_validation_source,
            target_features=validation_target_residual_flat,
            fit_indices=all_validation_flat,
            k=int(selected["k"]),
            temperature=0.10,
        )
        selected_test_mapped = _apply_knn_transport(final_test_source, final_mapper)
        selected_param = f"k={int(selected['k'])},temp=0.10"
    else:
        final_mapper = _fit_procrustes_transport(
            source_features=final_validation_source,
            target_features=validation_target_residual_flat,
            fit_indices=all_validation_flat,
            dim=int(selected["dim"]),
        )
        selected_test_mapped = _apply_procrustes_transport(final_test_source, final_mapper)
        selected_param = f"dim={int(selected['dim'])}"

    test_per_seed, test_aggregate, _ = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=selected_test_mapped,
        receiver_features=test_receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=test_index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=bool(overlap),
        variant="matched_transport_common_basis",
    )
    candidate_roll_per_seed, candidate_roll_aggregate, _ = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=hq_gate._roll_candidate_features(test_rows, selected_test_mapped),
        receiver_features=test_receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=test_index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=bool(overlap),
        variant="candidate_roll_transport_control",
    )
    spectral_permutation_per_seed, spectral_permutation_aggregate, _ = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=hq_gate._permute_feature_columns(selected_test_mapped, seed=selection_seed + 73),
        receiver_features=test_receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=test_index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=bool(overlap),
        variant="spectral_permutation_transport_control",
    )

    pass_gate = bool(
        test_aggregate["all_seeds_pass"]
        and test_aggregate["matched_minus_qwen_substituted_mean"] >= 0.02
        and test_aggregate["paired_ci95_low_vs_qwen_substituted_min"] > 0.0
        and test_aggregate["matched_minus_cached_tiny_packet_mean"] >= 0.02
    )
    headline = {
        "pass_gate": pass_gate,
        "selected_method": selected["method"],
        "selected_view": selected_view,
        "selected_transform": selected["transform"],
        "selected_param": selected_param,
        "validation_disagreement_rows": len(validation_rows),
        "test_disagreement_rows": len(test_rows),
        "frontier_candidate_count": len(frontier_rows),
        "test_matched_accuracy_mean": test_aggregate["matched_accuracy_mean"],
        "test_qwen_substituted_accuracy_mean": test_aggregate["qwen_substituted_accuracy_mean"],
        "test_cached_tiny_packet_accuracy_mean": test_aggregate["cached_tiny_packet_accuracy_mean"],
        "test_matched_minus_qwen_substituted_mean": test_aggregate["matched_minus_qwen_substituted_mean"],
        "test_matched_minus_cached_tiny_packet_mean": test_aggregate["matched_minus_cached_tiny_packet_mean"],
        "test_paired_ci95_low_vs_qwen_substituted_min": test_aggregate[
            "paired_ci95_low_vs_qwen_substituted_min"
        ],
        "candidate_roll_matched_accuracy_mean": candidate_roll_aggregate["matched_accuracy_mean"],
        "spectral_permutation_matched_accuracy_mean": spectral_permutation_aggregate["matched_accuracy_mean"],
        "elapsed_s": float(time.perf_counter() - started),
    }
    payload = {
        "gate": "source_private_arc_challenge_transport_common_basis_gate",
        "pass_gate": pass_gate,
        "headline": headline,
        "frontier": frontier_rows,
        "test_per_seed": test_per_seed,
        "test_aggregate": test_aggregate,
        "candidate_roll_per_seed": candidate_roll_per_seed,
        "candidate_roll_aggregate": candidate_roll_aggregate,
        "spectral_permutation_per_seed": spectral_permutation_per_seed,
        "spectral_permutation_aggregate": spectral_permutation_aggregate,
        "basis_contract": {
            "budget_bytes": budget_bytes,
            "code_dim": code_dim,
            "anchor_count": anchor_count,
            "spectral_dim": spectral_dim,
            "validation_basis_metadata": validation_basis_meta,
            "test_basis_metadata": test_basis_meta,
        },
        "input_artifacts": {
            "hidden_query_dir": _display_path(hidden_query_dir),
            "source_family_gate_dir": _display_path(source_family_gate_dir),
            "qwen_disagreement_predictions": _display_path(qwen_disagreement_path),
            "qwen_disagreement_sha256": _sha256_file(qwen_disagreement_path),
        },
        "lay_explanation": (
            "This run keeps the same tiny ARC packet, but changes how TinyLlama's hidden/query "
            "vectors are translated into the public receiver coordinate system. Instead of a direct "
            "PCA/ridge fit, it tries local nearest-neighbor transport, sign-projected transport, and "
            "orthogonal Procrustes alignment. The key comparison is still against the stronger "
            "Qwen-substituted packet on the same disagreement rows."
        ),
        "interpretation": (
            "A pass would revive the TinyLlama hidden/query connector with a more principled common-basis "
            "map. A failure rules out another shallow Mac-local connector family and pushes the positive "
            "method gate back to a stronger true cross-family source or a trainable query/cache connector "
            "on NVIDIA."
        ),
    }
    _write_frontier_csv(output_dir / "transport_common_basis_frontier.csv", frontier_rows)
    (output_dir / "arc_challenge_transport_common_basis_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "arc_challenge_transport_common_basis_gate.md", payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "headline": headline,
        "files": [
            {
                "path": _display_path(output_dir / "arc_challenge_transport_common_basis_gate.json"),
                "sha256": _sha256_file(output_dir / "arc_challenge_transport_common_basis_gate.json"),
            },
            {
                "path": _display_path(output_dir / "arc_challenge_transport_common_basis_gate.md"),
                "sha256": _sha256_file(output_dir / "arc_challenge_transport_common_basis_gate.md"),
            },
            {
                "path": _display_path(output_dir / "transport_common_basis_frontier.csv"),
                "sha256": _sha256_file(output_dir / "transport_common_basis_frontier.csv"),
            },
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-query-dir", type=pathlib.Path, default=DEFAULT_HIDDEN_QUERY_DIR)
    parser.add_argument("--source-family-gate-dir", type=pathlib.Path, default=DEFAULT_SOURCE_FAMILY_GATE_DIR)
    parser.add_argument("--train-anchor-path", type=pathlib.Path, default=DEFAULT_TRAIN_ANCHORS)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--seeds", type=_parse_int_tuple, default="47,53,59,61,67")
    parser.add_argument("--source-views", type=_parse_str_tuple, default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--knn-values", type=_parse_int_tuple, default="3,7,15,31")
    parser.add_argument("--procrustes-dims", type=_parse_int_tuple, default="16,32,64,96")
    parser.add_argument("--qjl-dims", type=_parse_int_tuple, default="64,128")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        hidden_query_dir=args.hidden_query_dir,
        source_family_gate_dir=args.source_family_gate_dir,
        train_anchor_path=args.train_anchor_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        seeds=args.seeds,
        source_views=args.source_views,
        knn_values=args.knn_values,
        procrustes_dims=args.procrustes_dims,
        qjl_dims=args.qjl_dims,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "selected_method": payload["headline"]["selected_method"],
                "selected_view": payload["headline"]["selected_view"],
                "test_matched_accuracy_mean": payload["headline"]["test_matched_accuracy_mean"],
                "test_qwen_substituted_accuracy_mean": payload["headline"]["test_qwen_substituted_accuracy_mean"],
                "test_delta_vs_qwen": payload["headline"]["test_matched_minus_qwen_substituted_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
