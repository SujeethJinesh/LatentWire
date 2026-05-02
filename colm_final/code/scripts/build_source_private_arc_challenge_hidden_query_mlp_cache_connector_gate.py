from __future__ import annotations

"""ARC hidden/query MLP cache-to-packet connector gate.

This is the learned follow-up to the negative ARC hidden/query PCA/ridge,
transport, and sparse-query cache-bottleneck gates.  It reuses cached
TinyLlama hidden/query features, trains a tiny CPU MLP connector on validation
disagreement rows, decodes into the public ARC Fourier/anchor receiver basis,
and emits the same 12B sparse signed packet used by the existing ARC gates.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_hidden_query_common_basis_gate as hq_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_hidden_query_mlp_cache_connector_gate_20260502_tinyllama_disagreement"
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
DEFAULT_HIDDEN_DIMS = (16, 32)
DEFAULT_WEIGHT_DECAYS = (0.0, 0.001, 0.01)
BASELINE_ARTIFACTS = {
    "hidden_query_pca_ridge": pathlib.Path(
        "results/source_private_arc_challenge_hidden_query_common_basis_gate_20260502_tinyllama_disagreement/"
        "arc_challenge_hidden_query_common_basis_gate.json"
    ),
    "transport_common_basis": pathlib.Path(
        "results/source_private_arc_challenge_transport_common_basis_gate_20260502_tinyllama_disagreement/"
        "arc_challenge_transport_common_basis_gate.json"
    ),
    "sparse_query_cache_bottleneck": pathlib.Path(
        "results/source_private_arc_challenge_sparse_query_cache_bottleneck_gate_20260502_tinyllama_disagreement/"
        "arc_challenge_sparse_query_cache_bottleneck_gate.json"
    ),
    "candidate_syndrome_connector": pathlib.Path(
        "results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502/"
        "candidate_syndrome_connector_gate.json"
    ),
}
CSV_COLUMNS = (
    "view",
    "pca_dim",
    "hidden_dim",
    "weight_decay",
    "dev_accuracy_mean",
    "dev_qwen_substituted_accuracy_mean",
    "dev_cached_tiny_packet_accuracy_mean",
    "dev_matched_minus_qwen_substituted_mean",
    "dev_matched_minus_cached_tiny_packet_mean",
    "dev_paired_ci95_low_vs_qwen_substituted_min",
    "dev_paired_ci95_low_vs_cached_tiny_packet_min",
    "fit_alignment_cosine_mean",
    "fit_loss_final",
    "selected",
)


class _MlpConnector(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


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


def _fit_pca(values: np.ndarray, fit_indices: np.ndarray, dim: int) -> dict[str, Any]:
    fit = np.asarray(values[fit_indices], dtype=np.float64)
    mean = np.mean(fit, axis=0)
    centered = fit - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(int(dim), vt.shape[0])]
    fit_scores = centered @ components.T
    scale = np.std(fit_scores, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {
        "mean": mean,
        "components": components,
        "scale": scale,
        "dim": int(components.shape[0]),
        "singular_values_top8": [float(value) for value in singular_values[:8]],
    }


def _project_pca(values: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - pca["mean"]) @ pca["components"].T) / pca["scale"]


def _standardize_target(target: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = np.asarray(target[fit_indices], dtype=np.float64)
    mean = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (target - mean) / scale, mean, scale


def _state_arrays(model: _MlpConnector) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}


def _load_state_arrays(model: _MlpConnector, arrays: dict[str, np.ndarray]) -> None:
    state = {name: torch.as_tensor(value, dtype=torch.float32) for name, value in arrays.items()}
    model.load_state_dict(state)


def _fit_mlp_map(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    fit_indices: np.ndarray,
    pca_dim: int,
    hidden_dim: int,
    weight_decay: float,
    seed: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    pca = _fit_pca(source_features, fit_indices, pca_dim)
    projected = _project_pca(source_features, pca).astype(np.float32)
    target_std, target_mean, target_scale = _standardize_target(target_features, fit_indices)
    x_fit = torch.as_tensor(projected[fit_indices], dtype=torch.float32)
    y_fit = torch.as_tensor(target_std[fit_indices], dtype=torch.float32)
    model = _MlpConnector(int(pca["dim"]), int(hidden_dim), int(target_features.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_fit)
        loss = torch.nn.functional.mse_loss(prediction, y_fit)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        fit_prediction = model(x_fit).detach().cpu().numpy().astype(np.float64)
    fit_unstd = fit_prediction * target_scale + target_mean
    fit_y = np.asarray(target_features[fit_indices], dtype=np.float64)
    cosine = hq_gate._rowwise_cosine(fit_unstd, fit_y)
    return {
        "pca": pca,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "state": _state_arrays(model),
        "pca_dim": int(pca["dim"]),
        "hidden_dim": int(hidden_dim),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "lr": float(lr),
        "fit_loss_initial": float(loss_history[0]) if loss_history else float("nan"),
        "fit_loss_final": float(loss_history[-1]) if loss_history else float("nan"),
        "fit_alignment_cosine_mean": float(np.mean(cosine)),
        "fit_alignment_cosine_p10": float(np.percentile(cosine, 10)),
    }


def _apply_mlp_map(source_features: np.ndarray, mapper: dict[str, Any]) -> np.ndarray:
    projected = _project_pca(source_features, mapper["pca"]).astype(np.float32)
    model = _MlpConnector(int(mapper["pca_dim"]), int(mapper["hidden_dim"]), int(mapper["target_mean"].shape[0]))
    _load_state_arrays(model, mapper["state"])
    model.eval()
    with torch.inference_mode():
        predicted = model(torch.as_tensor(projected, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)
    return predicted * mapper["target_scale"] + mapper["target_mean"]


def _rotate_blocks(rows: list[arc_gate.ArcRow], flat_features: np.ndarray) -> np.ndarray:
    blocks = hq_gate._flat_to_rows(rows, flat_features)
    if not blocks:
        return flat_features.copy()
    rotated = blocks[1:] + blocks[:1]
    return np.concatenate(rotated, axis=0)


def _score_frontier_row(dev_aggregate: dict[str, Any], fit_alignment: float) -> tuple[float, float, float, float, float]:
    return (
        float(dev_aggregate["matched_minus_qwen_substituted_mean"]),
        float(dev_aggregate["paired_ci95_low_vs_qwen_substituted_min"]),
        float(dev_aggregate["matched_minus_cached_tiny_packet_mean"]),
        float(dev_aggregate["matched_accuracy_mean"]),
        float(fit_alignment),
    )


def _jsonable_mapper(mapper: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in mapper.items()
        if key
        not in {
            "pca",
            "target_mean",
            "target_scale",
            "state",
        }
    } | {
        "pca_dim": int(mapper["pca_dim"]),
        "hidden_dim": int(mapper["hidden_dim"]),
        "weight_decay": float(mapper["weight_decay"]),
        "pca_singular_values_top8": mapper["pca"]["singular_values_top8"],
    }


def _write_weights(path: pathlib.Path, mapper: dict[str, Any]) -> None:
    arrays = {
        "pca_mean": np.asarray(mapper["pca"]["mean"], dtype=np.float64),
        "pca_components": np.asarray(mapper["pca"]["components"], dtype=np.float64),
        "pca_scale": np.asarray(mapper["pca"]["scale"], dtype=np.float64),
        "target_mean": np.asarray(mapper["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(mapper["target_scale"], dtype=np.float64),
    }
    for name, value in mapper["state"].items():
        arrays[f"state__{name.replace('.', '__')}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _baseline_readouts(paths: dict[str, pathlib.Path]) -> dict[str, Any]:
    readouts: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = _resolve(path)
        if not resolved.exists():
            readouts[name] = {"available": False, "path": _display_path(path)}
            continue
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        headline = payload.get("headline", {})
        aggregate = payload.get("test_aggregate", {})
        selected = payload.get("selected_primary_view", {})
        readouts[name] = {
            "available": True,
            "path": _display_path(path),
            "pass_gate": bool(payload.get("pass_gate", False)),
            "test_matched_accuracy_mean": headline.get(
                "test_matched_accuracy_mean",
                aggregate.get("matched_accuracy_mean"),
            )
            or selected.get("test_summary", {}).get("aggregate", {}).get("connector_accuracy_mean"),
            "test_qwen_substituted_accuracy_mean": headline.get(
                "test_qwen_substituted_accuracy_mean",
                aggregate.get("qwen_substituted_accuracy_mean"),
            )
            or selected.get("test_summary", {}).get("aggregate", {}).get("qwen_accuracy_mean"),
            "test_delta_vs_qwen_substituted": headline.get(
                "test_matched_minus_qwen_substituted_mean",
                aggregate.get("matched_minus_qwen_substituted_mean"),
            )
            or selected.get("test_summary", {}).get("aggregate", {}).get("connector_minus_qwen_mean"),
            "test_ci95_low_vs_qwen_substituted": headline.get(
                "test_paired_ci95_low_vs_qwen_substituted_min",
                aggregate.get("paired_ci95_low_vs_qwen_substituted_min"),
            )
            or selected.get("test_summary", {}).get("aggregate", {}).get("paired_ci95_low_vs_qwen_min"),
        }
    return readouts


def _write_frontier(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _write_jsonl_gzip(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC Hidden/Query MLP Cache Connector Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected view: `{h['selected_view']}`",
        f"- selected pca/hidden/weight_decay: `{h['selected_pca_dim']}` / `{h['selected_hidden_dim']}` / `{h['selected_weight_decay']}`",
        f"- validation disagreement rows: `{h['validation_disagreement_rows']}`",
        f"- test disagreement rows: `{h['test_disagreement_rows']}`",
        f"- frontier candidates: `{h['frontier_candidate_count']}`",
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
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    weight_decays: tuple[float, ...] = DEFAULT_WEIGHT_DECAYS,
    selection_seed: int = 18013,
    dev_fraction: float = 0.25,
    epochs: int = 160,
    lr: float = 0.01,
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

    validation_source_predictions = hq_gate._source_predictions(
        validation_rows,
        hq_gate._read_source_predictions(source_family_gate_dir / "tinyllama_validation/source_prediction_cache.jsonl"),
    )
    test_source_predictions = hq_gate._source_predictions(
        test_rows,
        hq_gate._read_source_predictions(source_family_gate_dir / "tinyllama_test/source_prediction_cache.jsonl"),
    )
    validation_state = _load_state(
        hidden_query_dir / "tinyllama_validation_disagreement_hidden_query_cache.npz",
        hidden_query_dir / "tinyllama_validation_disagreement_hidden_query_cache.json",
    )
    test_state = _load_state(
        hidden_query_dir / "tinyllama_test_disagreement_hidden_query_cache.npz",
        hidden_query_dir / "tinyllama_test_disagreement_hidden_query_cache.json",
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
    selected: dict[str, Any] | None = None
    selected_key: tuple[str, int, int, float] | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    validation_source_by_view: dict[str, np.ndarray] = {}
    for view in source_views:
        validation_source = hq_gate._flat_source_view(rows=validation_rows, state=validation_state, view=view)
        validation_source_by_view[view] = validation_source
        for pca_dim in pca_dims:
            for hidden_dim in hidden_dims:
                for weight_decay in weight_decays:
                    mapper = _fit_mlp_map(
                        source_features=validation_source,
                        target_features=validation_target_residual_flat,
                        fit_indices=fit_flat,
                        pca_dim=int(pca_dim),
                        hidden_dim=int(hidden_dim),
                        weight_decay=float(weight_decay),
                        seed=selection_seed + int(pca_dim) * 31 + int(hidden_dim) * 17,
                        epochs=epochs,
                        lr=lr,
                    )
                    mapped_validation = _apply_mlp_map(validation_source, mapper)
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
                        variant="dev_candidate_hidden_query_mlp",
                    )
                    row = {
                        "view": view,
                        "pca_dim": int(mapper["pca_dim"]),
                        "hidden_dim": int(hidden_dim),
                        "weight_decay": float(weight_decay),
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
                        "dev_paired_ci95_low_vs_cached_tiny_packet_min": dev_aggregate[
                            "paired_ci95_low_vs_cached_tiny_packet_min"
                        ],
                        "fit_alignment_cosine_mean": mapper["fit_alignment_cosine_mean"],
                        "fit_loss_final": mapper["fit_loss_final"],
                        "selected": False,
                    }
                    score = _score_frontier_row(dev_aggregate, mapper["fit_alignment_cosine_mean"])
                    if best_score is None or score > best_score:
                        best_score = score
                        selected = row
                        selected_key = (view, int(mapper["pca_dim"]), int(hidden_dim), float(weight_decay))
                    frontier.append(row)
    if selected is None or selected_key is None:
        raise ValueError("no selected MLP connector candidate")
    for row in frontier:
        row["selected"] = (
            row["view"] == selected_key[0]
            and int(row["pca_dim"]) == int(selected_key[1])
            and int(row["hidden_dim"]) == int(selected_key[2])
            and float(row["weight_decay"]) == float(selected_key[3])
        )

    selected_view, selected_pca_dim, selected_hidden_dim, selected_weight_decay = selected_key
    selected_validation_source = validation_source_by_view[selected_view]
    selected_test_source = hq_gate._flat_source_view(rows=test_rows, state=test_state, view=selected_view)
    final_mapper = _fit_mlp_map(
        source_features=selected_validation_source,
        target_features=validation_target_residual_flat,
        fit_indices=np.arange(selected_validation_source.shape[0], dtype=np.int64),
        pca_dim=selected_pca_dim,
        hidden_dim=selected_hidden_dim,
        weight_decay=selected_weight_decay,
        seed=selection_seed + selected_pca_dim * 31 + selected_hidden_dim * 17 + 991,
        epochs=epochs,
        lr=lr,
    )
    selected_test_mapped = _apply_mlp_map(selected_test_source, final_mapper)

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
        variant="matched_hidden_query_mlp_cache_connector",
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
        variant="candidate_roll_hidden_query_mlp_control",
    )
    content_rotation_per_seed, content_rotation_aggregate, content_rotation_rows = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=_apply_mlp_map(_rotate_blocks(test_rows, selected_test_source), final_mapper),
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
        variant="content_rotation_hidden_query_mlp_control",
    )
    spectral_permutation_per_seed, spectral_permutation_aggregate, spectral_permutation_rows = hq_gate._evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=selected_test_mapped,
        receiver_features=hq_gate._permute_feature_columns(test_receiver_features, seed=selection_seed + 103),
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
        variant="spectral_permutation_hidden_query_mlp_control",
    )

    pass_gate = bool(
        test_aggregate["matched_minus_qwen_substituted_mean"] >= 0.02
        and test_aggregate["matched_minus_cached_tiny_packet_mean"] >= 0.02
        and test_aggregate["paired_ci95_low_vs_qwen_substituted_min"] > 0.0
        and test_aggregate["paired_ci95_low_vs_cached_tiny_packet_min"] > 0.0
        and candidate_roll_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
        and content_rotation_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
        and spectral_permutation_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
        and not overlap
    )
    headline = {
        "pass_gate": pass_gate,
        "selected_view": selected_view,
        "selected_pca_dim": selected_pca_dim,
        "selected_hidden_dim": selected_hidden_dim,
        "selected_weight_decay": selected_weight_decay,
        "validation_disagreement_rows": len(validation_rows),
        "test_disagreement_rows": len(test_rows),
        "frontier_candidate_count": len(frontier),
        "test_matched_accuracy_mean": test_aggregate["matched_accuracy_mean"],
        "test_matched_accuracy_min": test_aggregate["matched_accuracy_min"],
        "test_qwen_substituted_accuracy_mean": test_aggregate["qwen_substituted_accuracy_mean"],
        "test_cached_tiny_packet_accuracy_mean": test_aggregate["cached_tiny_packet_accuracy_mean"],
        "test_target_accuracy": test_aggregate["target_accuracy"],
        "test_same_byte_structured_text_accuracy": test_aggregate["same_byte_structured_text_accuracy"],
        "test_matched_minus_qwen_substituted_mean": test_aggregate["matched_minus_qwen_substituted_mean"],
        "test_matched_minus_cached_tiny_packet_mean": test_aggregate["matched_minus_cached_tiny_packet_mean"],
        "test_paired_ci95_low_vs_qwen_substituted_min": test_aggregate[
            "paired_ci95_low_vs_qwen_substituted_min"
        ],
        "test_paired_ci95_low_vs_cached_tiny_packet_min": test_aggregate[
            "paired_ci95_low_vs_cached_tiny_packet_min"
        ],
        "candidate_roll_matched_accuracy_mean": candidate_roll_aggregate["matched_accuracy_mean"],
        "content_rotation_matched_accuracy_mean": content_rotation_aggregate["matched_accuracy_mean"],
        "spectral_permutation_matched_accuracy_mean": spectral_permutation_aggregate["matched_accuracy_mean"],
        "final_fit_alignment_cosine_mean": final_mapper["fit_alignment_cosine_mean"],
        "final_fit_loss_final": final_mapper["fit_loss_final"],
        "train_test_content_overlap_count": len(overlap),
    }
    lay_explanation = (
        "This experiment tries a tiny learned translator instead of another hand-built geometric map. "
        "TinyLlama looks at the question and answer choices, the translator compresses each candidate's cached "
        "hidden/query vector into the same public packet coordinates, and only a 12-byte packet reaches the "
        "receiver. The hard comparison is whether that learned packet beats simply using Qwen's own packet on "
        "the same disagreement rows."
    )
    interpretation = (
        "This is a bounded Mac-local proxy for a query-bottleneck connector, not a full target-LM soft-prefix "
        "claim. A pass would promote the learned cache-to-packet branch for larger training and cross-family "
        "tests. A failure weakens the claim that the current TinyLlama hidden/query means contain a low-data "
        "connector into the ARC public packet basis, while leaving open tokenwise query/KV connectors that need "
        "new extraction infrastructure or NVIDIA runs."
    )
    payload = {
        "gate": "source_private_arc_challenge_hidden_query_mlp_cache_connector_gate",
        "date": "2026-05-02",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires the selected MLP cache connector to beat Qwen-substituted packets and cached "
            "TinyLlama packets by at least 0.02 on mean accuracy, with paired CI95 lower bound above zero "
            "against both. Candidate-roll, wrong-row/content-rotation, and receiver spectral-permutation "
            "controls must not exceed Qwen-substituted accuracy by more than 0.005."
        ),
        "headline": headline,
        "selected_dev_row": selected,
        "frontier_rows": frontier,
        "test_per_seed": test_per_seed,
        "test_aggregate": test_aggregate,
        "candidate_roll_per_seed": candidate_roll_per_seed,
        "candidate_roll_aggregate": candidate_roll_aggregate,
        "content_rotation_per_seed": content_rotation_per_seed,
        "content_rotation_aggregate": content_rotation_aggregate,
        "spectral_permutation_per_seed": spectral_permutation_per_seed,
        "spectral_permutation_aggregate": spectral_permutation_aggregate,
        "prior_mac_local_baselines": _baseline_readouts(BASELINE_ARTIFACTS),
        "fit_dev_split": {
            "selection_seed": selection_seed,
            "dev_fraction": dev_fraction,
            "fit_rows": int(len(fit_rows)),
            "dev_rows": int(len(dev_rows)),
            "fit_row_ids_sha256": hashlib.sha256(
                "\n".join(validation_rows[int(index)].row_id for index in fit_rows).encode("utf-8")
            ).hexdigest(),
            "dev_row_ids_sha256": hashlib.sha256(
                "\n".join(validation_rows[int(index)].row_id for index in dev_rows).encode("utf-8")
            ).hexdigest(),
        },
        "final_mapper": _jsonable_mapper(final_mapper),
        "basis_contract": {
            "target_basis": "public train-anchor relative coordinates followed by orthonormal low-frequency DCT-II",
            "anchor_count": anchor_count,
            "spectral_dim": spectral_dim,
            "code_dim": code_dim,
            "budget_bytes": budget_bytes,
            "validation_basis_metadata": validation_basis_meta,
            "test_basis_metadata": test_basis_meta,
        },
        "method_contract": {
            "source_model": validation_state.metadata.get("model_path"),
            "connector_kind": "train-only PCA + one-hidden-layer MLP cache-to-public-basis connector",
            "source_inputs_at_eval": ["question", "choices"],
            "forbidden_eval_source_inputs": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "source_hidden_query_raw_transmitted": False,
            "source_text_transmitted": False,
            "source_kv_transmitted": False,
            "packet_format": "12-byte sparse signed projection packet emitted from mapped source hidden/query features",
            "full_soft_prefix_claim": False,
            "native_gpu_claims_allowed": False,
        },
        "source_hidden_query_metadata": {
            "validation": validation_state.metadata,
            "test": test_state.metadata,
        },
        "inputs": {
            "hidden_query_dir": _display_path(hidden_query_dir),
            "source_family_gate_dir": _display_path(source_family_gate_dir),
            "qwen_disagreement_predictions": _display_path(qwen_disagreement_path),
            "qwen_disagreement_predictions_sha256": _sha256_file(qwen_disagreement_path),
            "train_anchor_path": _display_path(train_anchor_path),
            "train_anchor_sha256": _sha256_file(train_anchor_path),
            "validation_path": _display_path(validation_path),
            "validation_sha256": _sha256_file(validation_path),
            "test_path": _display_path(test_path),
            "test_sha256": _sha256_file(test_path),
        },
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": budget_bytes,
            "framed_record_bytes_per_request": budget_bytes + 3,
            "logical_test_raw_payload_bytes_total": int(budget_bytes * len(test_rows)),
            "logical_test_framed_record_bytes_total": int((budget_bytes + 3) * len(test_rows)),
            "total_wall_time_s": float(time.perf_counter() - started),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "native_gpu_claims_allowed": False,
        },
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }

    json_path = output_dir / "arc_challenge_hidden_query_mlp_cache_connector_gate.json"
    md_path = output_dir / "arc_challenge_hidden_query_mlp_cache_connector_gate.md"
    frontier_path = output_dir / "dev_frontier.csv"
    predictions_path = output_dir / "test_predictions.jsonl.gz"
    weights_path = output_dir / "selected_connector_weights.npz"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_frontier(frontier_path, frontier)
    _write_jsonl_gzip(
        predictions_path,
        [*test_prediction_rows, *candidate_roll_rows, *content_rotation_rows, *spectral_permutation_rows],
    )
    _write_weights(weights_path, final_mapper)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path, frontier_path, predictions_path, weights_path)
        ],
        "inputs": payload["inputs"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# ARC Hidden/Query MLP Cache Connector Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- selected view: `{selected_view}`",
                f"- test delta vs Qwen-sub: `{headline['test_matched_minus_qwen_substituted_mean']:.6f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> int:
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
    parser.add_argument("--hidden-dims", type=_parse_int_tuple, default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--weight-decays", type=_parse_float_tuple, default=DEFAULT_WEIGHT_DECAYS)
    parser.add_argument("--selection-seed", type=int, default=18013)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--anchor-count", type=int, default=384)
    parser.add_argument("--spectral-dim", type=int, default=96)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lift-over-target", type=float, default=0.02)
    parser.add_argument("--min-gap-over-control", type=float, default=0.02)
    parser.add_argument("--min-gap-over-text", type=float, default=0.0)
    args = parser.parse_args()
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
        hidden_dims=args.hidden_dims,
        weight_decays=args.weight_decays,
        selection_seed=args.selection_seed,
        dev_fraction=args.dev_fraction,
        epochs=args.epochs,
        lr=args.lr,
        budget_bytes=args.budget_bytes,
        anchor_count=args.anchor_count,
        spectral_dim=args.spectral_dim,
        code_dim=args.code_dim,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
