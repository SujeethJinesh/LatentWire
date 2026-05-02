from __future__ import annotations

"""ARC nonlinear sparse-query cache-bottleneck gate.

This gate is the nonlinear follow-up to the negative ARC hidden/query
PCA/ridge and static transport gates. It reuses cached TinyLlama hidden/query
features on the TinyLlama-vs-Qwen disagreement rows, maps them through a
train-only random Fourier feature query bottleneck, sparsifies the active query
coordinates, decodes into the public ARC Fourier/anchor receiver basis, and
then emits the same 12B sparse signed packet as the existing ARC gates.
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
    "results/source_private_arc_challenge_sparse_query_cache_bottleneck_gate_20260502_tinyllama_disagreement"
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
DEFAULT_PCA_DIMS = (16, 32)
DEFAULT_RFF_COMPONENTS = (32, 64)
DEFAULT_ACTIVE_COMPONENTS = (4, 8, 16)
DEFAULT_GAMMAS = (0.5, 1.0)
DEFAULT_RIDGES = (10.0, 100.0, 1000.0)
CSV_COLUMNS = (
    "view",
    "pca_dim",
    "rff_components",
    "active_components",
    "gamma",
    "ridge",
    "dev_accuracy_mean",
    "dev_qwen_substituted_accuracy_mean",
    "dev_cached_tiny_packet_accuracy_mean",
    "dev_matched_minus_qwen_substituted_mean",
    "dev_matched_minus_cached_tiny_packet_mean",
    "dev_paired_ci95_low_vs_qwen_substituted_min",
    "fit_alignment_cosine_mean",
    "sparsity_fraction",
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


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one float is required")
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


def _hidden_query_cache_contract(hidden_query_dir: pathlib.Path) -> dict[str, Any]:
    payload_path = hidden_query_dir / "arc_challenge_hidden_query_common_basis_gate.json"
    if not payload_path.exists():
        return {}
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return dict(payload.get("source_cache_contract") or {})


def _hidden_query_cache_paths(hidden_query_dir: pathlib.Path, split: str) -> tuple[pathlib.Path, pathlib.Path]:
    contract = _hidden_query_cache_contract(hidden_query_dir)
    npz_key = f"{split}_hidden_query_cache_npz"
    meta_key = f"{split}_hidden_query_cache_meta"
    if contract.get(npz_key) and contract.get(meta_key):
        return _resolve(contract[npz_key]), _resolve(contract[meta_key])
    return (
        hidden_query_dir / f"tinyllama_{split}_disagreement_hidden_query_cache.npz",
        hidden_query_dir / f"tinyllama_{split}_disagreement_hidden_query_cache.json",
    )


def _fit_pca(values: np.ndarray, fit_indices: np.ndarray, dim: int) -> dict[str, Any]:
    fit = np.asarray(values[fit_indices], dtype=np.float64)
    mean = np.mean(fit, axis=0)
    centered = fit - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(int(dim), vt.shape[0])]
    fit_scores = centered @ components.T
    scale = np.std(fit_scores, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean, "components": components, "scale": scale, "dim": int(components.shape[0])}


def _project_pca(values: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - pca["mean"]) @ pca["components"].T) / pca["scale"]


def _rff_parameters(input_dim: int, components: int, *, seed: int, gamma: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(scale=float(gamma), size=(int(input_dim), int(components)))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=int(components))
    return {"omega": omega.astype(np.float64), "phase": phase.astype(np.float64)}


def _apply_rff(values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    omega = params["omega"]
    phase = params["phase"]
    return np.sqrt(2.0 / float(omega.shape[1])) * np.cos(np.asarray(values, dtype=np.float64) @ omega + phase)


def _topk_sparse(values: np.ndarray, active_components: int) -> np.ndarray:
    active = min(int(active_components), values.shape[1])
    if active <= 0:
        return np.zeros_like(values)
    if active >= values.shape[1]:
        return np.asarray(values, dtype=np.float64).copy()
    threshold_indices = np.argpartition(-np.abs(values), kth=active - 1, axis=1)[:, :active]
    output = np.zeros_like(values, dtype=np.float64)
    gathered = np.take_along_axis(values, threshold_indices, axis=1)
    np.put_along_axis(output, threshold_indices, gathered, axis=1)
    return output


def _fit_sparse_query_map(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    fit_indices: np.ndarray,
    pca_dim: int,
    rff_components: int,
    active_components: int,
    gamma: float,
    ridge: float,
    seed: int,
) -> dict[str, Any]:
    pca = _fit_pca(source_features, fit_indices, pca_dim)
    projected = _project_pca(source_features, pca)
    rff_params = _rff_parameters(int(pca["dim"]), rff_components, seed=seed, gamma=gamma)
    sparse = _topk_sparse(_apply_rff(projected, rff_params), active_components)
    fit_x = sparse[fit_indices]
    fit_y = np.asarray(target_features[fit_indices], dtype=np.float64)
    system = fit_x.T @ fit_x + float(ridge) * np.eye(fit_x.shape[1], dtype=np.float64)
    weights = np.linalg.solve(system, fit_x.T @ fit_y)
    prediction = fit_x @ weights
    cosine = hq_gate._rowwise_cosine(prediction, fit_y)
    return {
        "pca": pca,
        "rff": rff_params,
        "weights": weights,
        "pca_dim": int(pca["dim"]),
        "rff_components": int(rff_components),
        "active_components": int(active_components),
        "gamma": float(gamma),
        "ridge": float(ridge),
        "fit_alignment_cosine_mean": float(np.mean(cosine)),
        "fit_alignment_cosine_p10": float(np.percentile(cosine, 10)),
        "sparsity_fraction": float(np.mean(np.abs(sparse) > 0.0)),
    }


def _apply_sparse_query_map(source_features: np.ndarray, mapper: dict[str, Any]) -> np.ndarray:
    projected = _project_pca(source_features, mapper["pca"])
    sparse = _topk_sparse(_apply_rff(projected, mapper["rff"]), int(mapper["active_components"]))
    return sparse @ mapper["weights"]


def _score_frontier_row(dev_aggregate: dict[str, Any], fit_alignment: float) -> tuple[float, float, float, float, float]:
    return (
        float(dev_aggregate["matched_minus_qwen_substituted_mean"]),
        float(dev_aggregate["paired_ci95_low_vs_qwen_substituted_min"]),
        float(dev_aggregate["matched_minus_cached_tiny_packet_mean"]),
        float(dev_aggregate["matched_accuracy_mean"]),
        float(fit_alignment),
    )


def _write_frontier(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC Sparse-Query Cache-Bottleneck Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected view: `{h['selected_view']}`",
        f"- selected pca/rff/active/gamma/ridge: `{h['selected_pca_dim']}` / `{h['selected_rff_components']}` / `{h['selected_active_components']}` / `{h['selected_gamma']}` / `{h['selected_ridge']}`",
        f"- validation disagreement rows: `{h['validation_disagreement_rows']}`",
        f"- test disagreement rows: `{h['test_disagreement_rows']}`",
        f"- test matched mean: `{h['test_matched_accuracy_mean']:.6f}`",
        f"- test Qwen-substituted mean: `{h['test_qwen_substituted_accuracy_mean']:.6f}`",
        f"- test cached Tiny mean: `{h['test_cached_tiny_packet_accuracy_mean']:.6f}`",
        f"- test delta vs Qwen-sub: `{h['test_matched_minus_qwen_substituted_mean']:.6f}`",
        f"- test CI95 low vs Qwen-sub: `{h['test_paired_ci95_low_vs_qwen_substituted_min']:.6f}`",
        f"- candidate-roll control mean: `{h['candidate_roll_matched_accuracy_mean']:.6f}`",
        f"- content-rotation control mean: `{h['content_rotation_matched_accuracy_mean']:.6f}`",
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


def _rotate_blocks(rows: list[arc_gate.ArcRow], flat_features: np.ndarray) -> np.ndarray:
    blocks = hq_gate._flat_to_rows(rows, flat_features)
    if not blocks:
        return flat_features.copy()
    rotated = blocks[1:] + blocks[:1]
    return np.concatenate(rotated, axis=0)


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
    pca_dims: tuple[int, ...] = DEFAULT_PCA_DIMS,
    rff_components_values: tuple[int, ...] = DEFAULT_RFF_COMPONENTS,
    active_components_values: tuple[int, ...] = DEFAULT_ACTIVE_COMPONENTS,
    gammas: tuple[float, ...] = DEFAULT_GAMMAS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    selection_seed: int = 18013,
    dev_fraction: float = 0.25,
    train_disagreement_limit: int | None = None,
    test_disagreement_limit: int | None = None,
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
    validation_ids = hq_gate._load_disagreement_row_ids(
        path=qwen_disagreement_path,
        split="validation",
        seed=int(seeds[0]),
        limit=train_disagreement_limit,
    )
    test_ids = hq_gate._load_disagreement_row_ids(
        path=qwen_disagreement_path,
        split="test",
        seed=int(seeds[0]),
        limit=test_disagreement_limit,
    )
    validation_rows = hq_gate._filter_rows_by_ids(validation_full, validation_ids)
    test_rows = hq_gate._filter_rows_by_ids(test_full, test_ids)
    overlap = sorted({row.content_id for row in validation_rows} & {row.content_id for row in test_rows})
    validation_hidden_npz, validation_hidden_meta = _hidden_query_cache_paths(hidden_query_dir, "validation")
    test_hidden_npz, test_hidden_meta = _hidden_query_cache_paths(hidden_query_dir, "test")
    validation_state = _load_state(validation_hidden_npz, validation_hidden_meta)
    test_state = _load_state(test_hidden_npz, test_hidden_meta)
    source_contract = hq_gate._source_cache_contract(source_family_gate_dir, "auto")
    validation_source_predictions = hq_gate._source_predictions(
        validation_rows,
        hq_gate._read_source_predictions(pathlib.Path(source_contract["validation_source_cache"])),
    )
    test_source_predictions = hq_gate._source_predictions(
        test_rows,
        hq_gate._read_source_predictions(pathlib.Path(source_contract["test_source_cache"])),
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

    frontier: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for view in source_views:
        validation_source = hq_gate._flat_source_view(rows=validation_rows, state=validation_state, view=view)
        for pca_dim in pca_dims:
            for rff_components in rff_components_values:
                for active_components in active_components_values:
                    if int(active_components) > int(rff_components):
                        continue
                    for gamma in gammas:
                        for ridge in ridges:
                            mapper = _fit_sparse_query_map(
                                source_features=validation_source,
                                target_features=validation_target_residual_flat,
                                fit_indices=fit_flat,
                                pca_dim=int(pca_dim),
                                rff_components=int(rff_components),
                                active_components=int(active_components),
                                gamma=float(gamma),
                                ridge=float(ridge),
                                seed=selection_seed + int(pca_dim) * 31 + int(rff_components) * 7 + int(active_components),
                            )
                            mapped_validation = _apply_sparse_query_map(validation_source, mapper)
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
                                variant="dev_sparse_query_bottleneck",
                            )
                            row = {
                                "view": view,
                                "pca_dim": int(mapper["pca_dim"]),
                                "rff_components": int(mapper["rff_components"]),
                                "active_components": int(mapper["active_components"]),
                                "gamma": float(gamma),
                                "ridge": float(ridge),
                                "dev_accuracy_mean": dev_aggregate["matched_accuracy_mean"],
                                "dev_qwen_substituted_accuracy_mean": dev_aggregate[
                                    "qwen_substituted_accuracy_mean"
                                ],
                                "dev_cached_tiny_packet_accuracy_mean": dev_aggregate[
                                    "cached_tiny_packet_accuracy_mean"
                                ],
                                "dev_matched_minus_qwen_substituted_mean": dev_aggregate[
                                    "matched_minus_qwen_substituted_mean"
                                ],
                                "dev_matched_minus_cached_tiny_packet_mean": dev_aggregate[
                                    "matched_minus_cached_tiny_packet_mean"
                                ],
                                "dev_paired_ci95_low_vs_qwen_substituted_min": dev_aggregate[
                                    "paired_ci95_low_vs_qwen_substituted_min"
                                ],
                                "fit_alignment_cosine_mean": mapper["fit_alignment_cosine_mean"],
                                "fit_alignment_cosine_p10": mapper["fit_alignment_cosine_p10"],
                                "sparsity_fraction": mapper["sparsity_fraction"],
                                "selected": False,
                            }
                            frontier.append(row)
                            candidates.append(
                                {
                                    "row": row,
                                    "sort_key": _score_frontier_row(
                                        dev_aggregate,
                                        mapper["fit_alignment_cosine_mean"],
                                    ),
                                }
                            )
    if not candidates:
        raise ValueError("no sparse-query candidates evaluated")
    selected = max(candidates, key=lambda item: item["sort_key"])["row"]
    for row in frontier:
        row["selected"] = all(
            row[key] == selected[key]
            for key in ("view", "pca_dim", "rff_components", "active_components", "gamma", "ridge")
        )

    selected_view = str(selected["view"])
    validation_source = hq_gate._flat_source_view(rows=validation_rows, state=validation_state, view=selected_view)
    test_source = hq_gate._flat_source_view(rows=test_rows, state=test_state, view=selected_view)
    final_mapper = _fit_sparse_query_map(
        source_features=validation_source,
        target_features=validation_target_residual_flat,
        fit_indices=np.arange(validation_source.shape[0], dtype=np.int64),
        pca_dim=int(selected["pca_dim"]),
        rff_components=int(selected["rff_components"]),
        active_components=int(selected["active_components"]),
        gamma=float(selected["gamma"]),
        ridge=float(selected["ridge"]),
        seed=selection_seed
        + int(selected["pca_dim"]) * 31
        + int(selected["rff_components"]) * 7
        + int(selected["active_components"]),
    )
    selected_test_mapped = _apply_sparse_query_map(test_source, final_mapper)
    test_per_seed, test_aggregate, test_prediction_rows = hq_gate._evaluate_features(
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
        variant="matched_sparse_query_bottleneck",
    )
    candidate_roll_per_seed, candidate_roll_aggregate, candidate_roll_rows = hq_gate._evaluate_features(
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
        variant="candidate_roll_sparse_query_control",
    )
    content_rotation_per_seed, content_rotation_aggregate, content_rotation_rows = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=_apply_sparse_query_map(_rotate_blocks(test_rows, test_source), final_mapper),
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
        variant="content_rotation_sparse_query_control",
    )
    spectral_permutation_per_seed, spectral_permutation_aggregate, spectral_permutation_rows = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=selected_test_mapped,
        receiver_features=hq_gate._permute_feature_columns(test_receiver_features, seed=selection_seed + 83),
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
        variant="spectral_permutation_sparse_query_control",
    )
    pass_gate = bool(
        test_aggregate["matched_minus_qwen_substituted_mean"] >= 0.02
        and test_aggregate["matched_minus_cached_tiny_packet_mean"] >= 0.02
        and test_aggregate["paired_ci95_low_vs_qwen_substituted_min"] > 0.0
        and test_aggregate["paired_ci95_low_vs_cached_tiny_packet_min"] > 0.0
        and candidate_roll_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
        and content_rotation_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
        and spectral_permutation_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
    )
    headline = {
        "pass_gate": pass_gate,
        "selected_view": selected_view,
        "selected_pca_dim": int(selected["pca_dim"]),
        "selected_rff_components": int(selected["rff_components"]),
        "selected_active_components": int(selected["active_components"]),
        "selected_gamma": float(selected["gamma"]),
        "selected_ridge": float(selected["ridge"]),
        "validation_disagreement_rows": len(validation_rows),
        "test_disagreement_rows": len(test_rows),
        "frontier_candidate_count": len(frontier),
        "test_matched_accuracy_mean": test_aggregate["matched_accuracy_mean"],
        "test_qwen_substituted_accuracy_mean": test_aggregate["qwen_substituted_accuracy_mean"],
        "test_cached_tiny_packet_accuracy_mean": test_aggregate["cached_tiny_packet_accuracy_mean"],
        "test_matched_minus_qwen_substituted_mean": test_aggregate[
            "matched_minus_qwen_substituted_mean"
        ],
        "test_matched_minus_cached_tiny_packet_mean": test_aggregate[
            "matched_minus_cached_tiny_packet_mean"
        ],
        "test_paired_ci95_low_vs_qwen_substituted_min": test_aggregate[
            "paired_ci95_low_vs_qwen_substituted_min"
        ],
        "test_paired_ci95_low_vs_cached_tiny_packet_min": test_aggregate[
            "paired_ci95_low_vs_cached_tiny_packet_min"
        ],
        "candidate_roll_matched_accuracy_mean": candidate_roll_aggregate["matched_accuracy_mean"],
        "content_rotation_matched_accuracy_mean": content_rotation_aggregate["matched_accuracy_mean"],
        "spectral_permutation_matched_accuracy_mean": spectral_permutation_aggregate["matched_accuracy_mean"],
        "fit_alignment_cosine_mean": final_mapper["fit_alignment_cosine_mean"],
        "sparsity_fraction": final_mapper["sparsity_fraction"],
        "elapsed_s": float(time.perf_counter() - started),
    }
    payload = {
        "gate": "source_private_arc_challenge_sparse_query_cache_bottleneck_gate",
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires the selected nonlinear sparse-query packet to beat Qwen-substituted and cached "
            "Tiny packets by >=0.02 on frozen test, with positive paired CI95 lower bounds versus both. "
            "Candidate-roll, content-rotation, and receiver spectral-permutation controls must not exceed "
            "Qwen-substituted accuracy by more than 0.005."
        ),
        "headline": headline,
        "selected_frontier_row": selected,
        "frontier": frontier,
        "test_per_seed": test_per_seed,
        "test_aggregate": test_aggregate,
        "candidate_roll_per_seed": candidate_roll_per_seed,
        "candidate_roll_aggregate": candidate_roll_aggregate,
        "content_rotation_per_seed": content_rotation_per_seed,
        "content_rotation_aggregate": content_rotation_aggregate,
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
        "method_contract": {
            "source_family": source_contract["source_family"],
            "source_model": source_contract["source_model"],
            "source_inputs_at_eval": ["question", "choices"],
            "forbidden_eval_source_inputs": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "raw_hidden_query_transmitted": False,
            "source_text_transmitted": False,
            "source_kv_transmitted": False,
            "packet_format": f"{budget_bytes}-byte sparse signed packet emitted from train-only nonlinear sparse-query hidden/query features",
            "native_gpu_claims_allowed": False,
        },
        "input_artifacts": {
            "hidden_query_dir": _display_path(hidden_query_dir),
            "source_family_gate_dir": _display_path(source_family_gate_dir),
            "qwen_disagreement_predictions": _display_path(qwen_disagreement_path),
            "qwen_disagreement_sha256": _sha256_file(qwen_disagreement_path),
            "validation_hidden_query_cache_npz": _display_path(validation_hidden_npz),
            "validation_hidden_query_cache_meta": _display_path(validation_hidden_meta),
            "test_hidden_query_cache_npz": _display_path(test_hidden_npz),
            "test_hidden_query_cache_meta": _display_path(test_hidden_meta),
            "validation_source_cache": _display_path(source_contract["validation_source_cache"]),
            "test_source_cache": _display_path(source_contract["test_source_cache"]),
        },
        "lay_explanation": (
            f"This run gives {source_contract['source_family']} a small learned nonlinear bottleneck before it sends the same {budget_bytes}-byte "
            "ARC hint. The bottleneck acts like a set of sparse queries over the source model's hidden/query state, "
            "then translates those query activations into the public packet coordinate system. The hidden/query "
            "vectors themselves are not sent."
        ),
        "interpretation": (
            "A pass would revive the ARC hidden/query branch with a real nonlinear source-private connector. "
            "A failure weakens another Mac-local hidden/query connector family and leaves a stronger "
            "true cross-family source or larger learned query/cache connector on NVIDIA as the next live branch."
        ),
    }
    json_path = output_dir / "arc_challenge_sparse_query_cache_bottleneck_gate.json"
    md_path = output_dir / "arc_challenge_sparse_query_cache_bottleneck_gate.md"
    frontier_path = output_dir / "sparse_query_bottleneck_frontier.csv"
    predictions_path = output_dir / "test_predictions.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_frontier(frontier_path, frontier)
    hq_gate._write_jsonl(
        predictions_path,
        [*test_prediction_rows, *candidate_roll_rows, *content_rotation_rows, *spectral_permutation_rows],
    )
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "headline": headline,
        "files": [
            {"path": _display_path(json_path), "sha256": _sha256_file(json_path)},
            {"path": _display_path(md_path), "sha256": _sha256_file(md_path)},
            {"path": _display_path(frontier_path), "sha256": _sha256_file(frontier_path)},
            {"path": _display_path(predictions_path), "sha256": _sha256_file(predictions_path)},
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
    parser.add_argument("--seeds", type=_parse_int_tuple, default=DEFAULT_SEEDS)
    parser.add_argument("--source-views", type=_parse_str_tuple, default=DEFAULT_VIEWS)
    parser.add_argument("--pca-dims", type=_parse_int_tuple, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--rff-components", type=_parse_int_tuple, default=DEFAULT_RFF_COMPONENTS)
    parser.add_argument("--active-components", type=_parse_int_tuple, default=DEFAULT_ACTIVE_COMPONENTS)
    parser.add_argument("--gammas", type=_parse_float_tuple, default=DEFAULT_GAMMAS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--selection-seed", type=int, default=18013)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--train-disagreement-limit", type=int, default=None)
    parser.add_argument("--test-disagreement-limit", type=int, default=None)
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--anchor-count", type=int, default=384)
    parser.add_argument("--spectral-dim", type=int, default=96)
    parser.add_argument("--code-dim", type=int, default=96)
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
        pca_dims=args.pca_dims,
        rff_components_values=args.rff_components,
        active_components_values=args.active_components,
        gammas=args.gammas,
        ridges=args.ridges,
        selection_seed=int(args.selection_seed),
        dev_fraction=float(args.dev_fraction),
        train_disagreement_limit=args.train_disagreement_limit,
        test_disagreement_limit=args.test_disagreement_limit,
        budget_bytes=int(args.budget_bytes),
        anchor_count=int(args.anchor_count),
        spectral_dim=int(args.spectral_dim),
        code_dim=int(args.code_dim),
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "selected_view": payload["headline"]["selected_view"],
                "selected_pca_dim": payload["headline"]["selected_pca_dim"],
                "selected_rff_components": payload["headline"]["selected_rff_components"],
                "selected_active_components": payload["headline"]["selected_active_components"],
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
