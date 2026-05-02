from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _constrained_nonself_index,
    _fit_ridge_encoder_for_view,
    _project_source,
    _remap_candidate_slots,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


@dataclass(frozen=True)
class GeometryProductCodebook:
    centroids: tuple[np.ndarray, ...]
    groups: tuple[np.ndarray, ...]
    variant: str
    utility_sum_by_group: tuple[float, ...]
    rotation: np.ndarray | None = None

    @property
    def subspaces(self) -> int:
        return len(self.centroids)


CONTROL_CONDITIONS = [
    "label_shuffled_ridge",
    "constrained_shuffled_source",
    "answer_masked_source",
    "permuted_codes",
    "random_same_byte",
]
CONDITIONS = ["target_only", "source", *CONTROL_CONDITIONS]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_index(example: Example) -> int:
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _dimension_utilities(
    train_rows: list[Example],
    *,
    encoder: np.ndarray,
    feature_dim: int,
    candidate_view: str,
) -> np.ndarray:
    utilities = np.zeros(feature_dim, dtype=np.float64)
    for example in train_rows:
        predicted = _project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched")
        candidates = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
        answer_index = _answer_index(example)
        gold_dist = np.abs(candidates[answer_index] - predicted)
        negative = np.delete(candidates, answer_index, axis=0)
        negative_dist = np.min(np.abs(negative - predicted[None, :]), axis=0)
        utilities += negative_dist - gold_dist
    return utilities.astype(np.float32)


def _groups_for_variant(
    *,
    variant: str,
    feature_dim: int,
    budget_bytes: int,
    utilities: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, ...]:
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    dims = np.arange(feature_dim, dtype=np.int64)
    if variant == "canonical":
        return tuple(group.astype(np.int64) for group in np.array_split(dims, budget_bytes))
    if variant == "utility_round_robin":
        order = np.argsort(-utilities, kind="stable")
        groups = [list() for _ in range(budget_bytes)]
        for rank, dim in enumerate(order):
            groups[rank % budget_bytes].append(int(dim))
        return tuple(np.asarray(sorted(group), dtype=np.int64) for group in groups)
    if variant == "utility_balanced":
        order = np.argsort(-utilities, kind="stable")
        target_size = int(np.ceil(feature_dim / budget_bytes))
        groups: list[list[int]] = [[] for _ in range(budget_bytes)]
        group_utility = np.zeros(budget_bytes, dtype=np.float64)
        shifted = np.maximum(utilities - float(np.min(utilities)), 0.0) + 1e-6
        for dim in order:
            allowed = [idx for idx, group in enumerate(groups) if len(group) < target_size]
            chosen = min(allowed, key=lambda idx: (group_utility[idx], len(groups[idx]), idx))
            groups[chosen].append(int(dim))
            group_utility[chosen] += float(shifted[int(dim)])
        return tuple(np.asarray(sorted(group), dtype=np.int64) for group in groups)
    if variant == "random_balanced":
        rng = np.random.default_rng(seed)
        order = rng.permutation(dims)
        return tuple(np.asarray(sorted(group), dtype=np.int64) for group in np.array_split(order, budget_bytes))
    if variant in {"opq_procrustes", "utility_opq_procrustes", "protected_hadamard", "utility_protected_hadamard"}:
        return tuple(group.astype(np.int64) for group in np.array_split(dims, budget_bytes))
    raise ValueError(f"unknown geometry variant {variant!r}")


def _fit_centroids_for_vectors(
    vectors: np.ndarray,
    *,
    groups: tuple[np.ndarray, ...],
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    centroids: list[np.ndarray] = []
    for group in groups:
        if len(group) == 0:
            raise ValueError("empty product-codebook group")
        part = vectors[:, group].astype(np.float32)
        cluster_count = min(256, max(2, part.shape[0]))
        init_indices = rng.choice(part.shape[0], size=cluster_count, replace=part.shape[0] < cluster_count)
        sub_centroids = part[init_indices].copy()
        for _ in range(iterations):
            distances = np.sum((part[:, None, :] - sub_centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(distances, axis=1)
            for centroid_index in range(cluster_count):
                mask = assignments == centroid_index
                if np.any(mask):
                    sub_centroids[centroid_index] = part[mask].mean(axis=0)
        centroids.append(sub_centroids.astype(np.float32))
    return tuple(centroids)


def _reconstruct_from_centroids(
    vectors: np.ndarray,
    *,
    centroids: tuple[np.ndarray, ...],
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    reconstructed = np.zeros_like(vectors, dtype=np.float32)
    for sub_centroids, group in zip(centroids, groups, strict=True):
        part = vectors[:, group].astype(np.float32)
        distances = np.sum((part[:, None, :] - sub_centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        reconstructed[:, group] = sub_centroids[assignments]
    return reconstructed


def _initial_rotation(
    *,
    variant: str,
    feature_dim: int,
    utilities: np.ndarray,
) -> np.ndarray:
    if variant != "utility_opq_procrustes":
        return np.eye(feature_dim, dtype=np.float32)
    order = np.argsort(-utilities, kind="stable")
    # A permutation is orthogonal and gives high-utility dimensions a stable,
    # contiguous starting layout before the Procrustes refinement.
    rotation = np.zeros((feature_dim, feature_dim), dtype=np.float32)
    for new_dim, old_dim in enumerate(order):
        rotation[int(old_dim), int(new_dim)] = 1.0
    return rotation


def _hadamard_matrix(feature_dim: int) -> np.ndarray:
    if feature_dim <= 0 or feature_dim & (feature_dim - 1):
        raise ValueError("protected_hadamard requires power-of-two feature_dim")
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < feature_dim:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]]).astype(np.float32)
    return (matrix / math.sqrt(feature_dim)).astype(np.float32)


def _protected_hadamard_rotation(
    *,
    feature_dim: int,
    seed: int,
    utilities: np.ndarray,
    utility_ordered: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _hadamard_matrix(feature_dim)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=feature_dim)
    rotation = signs[:, None] * base
    if utility_ordered:
        order = np.argsort(-utilities, kind="stable")
    else:
        order = rng.permutation(np.arange(feature_dim))
    protected = np.zeros_like(rotation)
    protected[order] = rotation
    return protected.astype(np.float32)


def _fit_geometry_codebook(
    train_rows: list[Example],
    *,
    encoder: np.ndarray,
    feature_dim: int,
    groups: tuple[np.ndarray, ...],
    variant: str,
    utilities: np.ndarray,
    seed: int,
    iterations: int = 12,
    opq_iterations: int = 0,
) -> GeometryProductCodebook:
    vectors = np.stack(
        [_project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched") for example in train_rows],
        axis=0,
    ).astype(np.float32)
    rotation: np.ndarray | None = None
    rotated_vectors = vectors
    if variant in {"protected_hadamard", "utility_protected_hadamard"}:
        rotation = _protected_hadamard_rotation(
            feature_dim=feature_dim,
            seed=seed,
            utilities=utilities,
            utility_ordered=variant == "utility_protected_hadamard",
        )
        rotated_vectors = (vectors @ rotation).astype(np.float32)
    if variant in {"opq_procrustes", "utility_opq_procrustes"}:
        rotation = _initial_rotation(variant=variant, feature_dim=feature_dim, utilities=utilities)
        for step in range(opq_iterations):
            rotated_vectors = (vectors @ rotation).astype(np.float32)
            step_centroids = _fit_centroids_for_vectors(
                rotated_vectors,
                groups=groups,
                seed=seed + step * 1009,
                iterations=iterations,
            )
            reconstructed = _reconstruct_from_centroids(rotated_vectors, centroids=step_centroids, groups=groups)
            cross = vectors.astype(np.float64).T @ reconstructed.astype(np.float64)
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rotation = (u @ vt).astype(np.float32)
        rotated_vectors = (vectors @ rotation).astype(np.float32)
    centroids = _fit_centroids_for_vectors(
        rotated_vectors,
        groups=groups,
        seed=seed + opq_iterations * 1009,
        iterations=iterations,
    )
    utility_sum_by_group = [float(np.sum(utilities[group])) for group in groups]
    return GeometryProductCodebook(
        centroids=centroids,
        groups=groups,
        variant=variant,
        utility_sum_by_group=tuple(utility_sum_by_group),
        rotation=rotation,
    )


def _geometry_packet(
    example: Example,
    *,
    encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    mode: str,
) -> bytes:
    predicted = _project_source(example, encoder=encoder, feature_dim=feature_dim, mode=mode)
    if codebook.rotation is not None:
        predicted = predicted @ codebook.rotation
    codes = np.zeros(codebook.subspaces, dtype=np.uint8)
    for subspace_index, (sub_centroids, group) in enumerate(zip(codebook.centroids, codebook.groups, strict=True)):
        part = predicted[group]
        distances = np.sum((sub_centroids - part[None, :]) ** 2, axis=1)
        codes[subspace_index] = int(np.argmin(distances))
    return codes.tobytes()


def _decode_geometry_packet(
    example: Example,
    payload: bytes | None,
    *,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    candidate_view: str,
) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    reconstructed = np.zeros(feature_dim, dtype=np.float32)
    raw_codes = np.frombuffer(payload[: codebook.subspaces], dtype=np.uint8)
    for subspace_index, code in enumerate(raw_codes):
        sub_centroids = codebook.centroids[subspace_index]
        centroid_index = int(code) % sub_centroids.shape[0]
        reconstructed[codebook.groups[subspace_index]] = sub_centroids[centroid_index]
    candidates = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
    if codebook.rotation is not None:
        candidates = candidates @ codebook.rotation
    distances = np.sum((candidates - reconstructed[None, :]) ** 2, axis=1)
    min_distance = float(np.min(distances))
    tied = np.flatnonzero(np.isclose(distances, min_distance, rtol=1e-6, atol=1e-8))
    prior = _prior_prediction(example)
    if any(example.candidates[int(idx)].label == prior for idx in tied):
        prediction = prior
    else:
        prediction = example.candidates[int(tied[0])].label
    return prediction, {
        "decoder": f"{codebook.variant}_product_codebook_l2",
        "min_l2": min_distance,
        "ties": [int(value) for value in tied.tolist()],
    }


def _permute_payload(payload: bytes) -> bytes:
    if len(payload) <= 1:
        return payload
    return payload[1:] + payload[:1]


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_rows: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    rng: random.Random,
) -> bytes | None:
    if condition == "target_only":
        return None
    if condition == "source":
        return _geometry_packet(example, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="matched")
    if condition == "label_shuffled_ridge":
        return _geometry_packet(
            example,
            encoder=label_shuffle_encoder,
            codebook=codebook,
            feature_dim=feature_dim,
            mode="matched",
        )
    if condition == "constrained_shuffled_source":
        other = eval_rows[_constrained_nonself_index(index, eval_rows)]
        return _geometry_packet(other, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="matched")
    if condition == "answer_masked_source":
        return _geometry_packet(example, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="answer_masked")
    if condition == "permuted_codes":
        payload = _geometry_packet(example, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="matched")
        return _permute_payload(payload)
    if condition == "random_same_byte":
        return rng.randbytes(codebook.subspaces)
    raise ValueError(f"unknown condition {condition!r}")


def _predict(
    *,
    condition: str,
    example: Example,
    eval_rows: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    candidate_view: str,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload = _payload_for_condition(
        condition=condition,
        example=example,
        eval_rows=eval_rows,
        index=index,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        feature_dim=feature_dim,
        rng=rng,
    )
    prediction, metadata = _decode_geometry_packet(
        example,
        payload,
        codebook=codebook,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "payload_hex": payload_hex,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": latency_ms,
        "metadata": metadata,
    }


def _summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    correct = [row["example_id"] for row in rows if row["correct"]]
    return {
        "correct": len(correct),
        "accuracy": len(correct) / len(rows),
        "correct_ids": correct,
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _paired_delta(
    rows: list[dict[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    ids = sorted({row["example_id"] for row in rows})
    by = {(row["condition"], row["example_id"]): bool(row["correct"]) for row in rows}
    diffs = np.array(
        [float(by[(condition_a, example_id)]) - float(by[(condition_b, example_id)]) for example_id in ids],
        dtype=np.float32,
    )
    point = float(np.mean(diffs)) if len(diffs) else 0.0
    if len(diffs) <= 1:
        return {"point": point, "ci95_low": point, "ci95_high": point}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float32)
    for idx in range(samples):
        indices = rng.integers(0, len(diffs), size=len(diffs))
        boot[idx] = float(np.mean(diffs[indices]))
    return {
        "point": point,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def run_geometry_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    remap_seeds: list[int],
    budgets: list[int],
    variants: list[str],
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    ridge: float,
    candidate_view: str,
    fit_intercept: bool,
    bootstrap_samples: int,
    opq_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for remap_seed in remap_seeds:
        train_rows = make_benchmark(
            examples=train_examples,
            candidates=candidates,
            seed=train_seed,
            family_set=train_family_set,
        )
        eval_rows = make_benchmark(
            examples=eval_examples,
            candidates=candidates,
            seed=eval_seed,
            family_set=eval_family_set,
        )
        train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_seed)
        eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_seed)
        encoder = _fit_ridge_encoder_for_view(
            train_rows,
            feature_dim=feature_dim,
            ridge=ridge,
            candidate_view=candidate_view,
            fit_intercept=fit_intercept,
        )
        label_shuffle_encoder = _fit_ridge_encoder_for_view(
            train_rows,
            feature_dim=feature_dim,
            ridge=ridge,
            candidate_view=candidate_view,
            fit_intercept=fit_intercept,
            label_shuffle_seed=train_seed * 5003 + eval_seed,
        )
        utilities = _dimension_utilities(
            train_rows,
            encoder=encoder,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
        )
        for budget in budgets:
            canonical_acc: float | None = None
            for variant in variants:
                run_dir = output_dir / f"remap_{remap_seed}" / f"budget_{budget}" / variant
                run_dir.mkdir(parents=True, exist_ok=True)
                run_dirs.append(str(run_dir))
                groups = _groups_for_variant(
                    variant=variant,
                    feature_dim=feature_dim,
                    budget_bytes=budget,
                    utilities=utilities,
                    seed=train_seed * 11003 + eval_seed * 97 + budget + remap_seed,
                )
                codebook = _fit_geometry_codebook(
                    train_rows,
                    encoder=encoder,
                    feature_dim=feature_dim,
                    groups=groups,
                    variant=variant,
                    utilities=utilities,
                    seed=train_seed * 9001 + eval_seed * 17 + budget,
                    opq_iterations=opq_iterations if variant in {"opq_procrustes", "utility_opq_procrustes"} else 0,
                )
                rng = random.Random(train_seed * 4001 + eval_seed + remap_seed * 13 + budget)
                by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
                for example_index, example in enumerate(eval_rows):
                    for condition in CONDITIONS:
                        by_condition[condition].append(
                            _predict(
                                condition=condition,
                                example=example,
                                eval_rows=eval_rows,
                                index=example_index,
                                encoder=encoder,
                                label_shuffle_encoder=label_shuffle_encoder,
                                codebook=codebook,
                                feature_dim=feature_dim,
                                candidate_view=candidate_view,
                                rng=rng,
                            )
                            | {
                                "example_id": example.example_id,
                                "family_name": example.family_name,
                                "variant": variant,
                                "budget_bytes": budget,
                                "remap_slot_seed": remap_seed,
                            }
                        )
                prediction_path = run_dir / "predictions.jsonl"
                prediction_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for condition in CONDITIONS
                        for row in by_condition[condition]
                    ),
                    encoding="utf-8",
                )
                condition_metrics = {
                    condition: _summarize_condition(condition_rows)
                    for condition, condition_rows in by_condition.items()
                }
                source = condition_metrics["source"]["accuracy"]
                target = condition_metrics["target_only"]["accuracy"]
                best_control_condition = max(CONTROL_CONDITIONS, key=lambda name: condition_metrics[name]["accuracy"])
                best_control = condition_metrics[best_control_condition]["accuracy"]
                controls_ok = all(condition_metrics[name]["accuracy"] <= target + 0.05 for name in CONTROL_CONDITIONS)
                paired_vs_target = _paired_delta(
                    [row for condition_rows in by_condition.values() for row in condition_rows],
                    condition_a="source",
                    condition_b="target_only",
                    samples=bootstrap_samples,
                    seed=20260430 + remap_seed + budget * 31,
                )
                row = {
                    "remap_slot_seed": remap_seed,
                    "budget_bytes": budget,
                    "variant": variant,
                    "n": eval_examples,
                    "source_accuracy": source,
                    "target_accuracy": target,
                    "best_control_condition": best_control_condition,
                    "best_control_accuracy": best_control,
                    "source_minus_target": source - target,
                    "source_minus_best_control": source - best_control,
                    "controls_ok": controls_ok,
                    "source_packet_pass": source >= target + 0.15 and controls_ok,
                    "p50_decode_latency_ms": condition_metrics["source"]["p50_latency_ms"],
                    "mean_payload_bytes": condition_metrics["source"]["mean_payload_bytes"],
                    "paired_source_vs_target": paired_vs_target,
                    "utility_sum_by_group": list(codebook.utility_sum_by_group),
                    "predictions_file": str(prediction_path.relative_to(output_dir)),
                    "metrics": condition_metrics,
                }
                if variant == "canonical":
                    canonical_acc = source
                    row["source_minus_canonical"] = 0.0
                elif canonical_acc is not None:
                    row["source_minus_canonical"] = source - canonical_acc
                rows.append(row)
                (run_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
                _write_variant_markdown(run_dir / "summary.md", row)

    promoted_rows = [
        row
        for row in rows
        if row["variant"] != "canonical"
        and row["source_packet_pass"]
        and row.get("source_minus_canonical", -1.0) >= 0.03
    ]
    source_pass_rows = [row for row in rows if row["source_packet_pass"]]
    payload = {
        "gate": "source_private_product_codebook_geometry_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "rows": len(rows),
            "source_pass_rows": len(source_pass_rows),
            "promoted_rows": len(promoted_rows),
            "max_source_accuracy": max((row["source_accuracy"] for row in rows), default=None),
            "max_noncanonical_minus_canonical": max(
                (row.get("source_minus_canonical", -1.0) for row in rows if row["variant"] != "canonical"),
                default=None,
            ),
            "promoted_variants": sorted({row["variant"] for row in promoted_rows}),
        },
        "pass_gate": bool(promoted_rows),
        "pass_rule": (
            "A non-canonical geometry variant must pass source controls and beat canonical contiguous PQ "
            "by at least +0.03 accuracy at the same remap, budget, and exact eval IDs."
        ),
        "interpretation": (
            "This gate tests whether changing PQ subspace geometry, rather than changing the receiver, "
            "improves source-private packet communication under the existing destructive controls."
        ),
    }
    (output_dir / "product_codebook_geometry_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_gate_markdown(output_dir / "product_codebook_geometry_gate.md", payload)
    manifest = {
        "artifacts": [
            "product_codebook_geometry_gate.json",
            "product_codebook_geometry_gate.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "product_codebook_geometry_gate.json": _sha256_file(output_dir / "product_codebook_geometry_gate.json"),
            "product_codebook_geometry_gate.md": _sha256_file(output_dir / "product_codebook_geometry_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Product-Codebook Geometry Gate Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_variant_markdown(path: pathlib.Path, row: dict[str, Any]) -> None:
    lines = [
        "# Product-Codebook Geometry Variant",
        "",
        f"- variant: `{row['variant']}`",
        f"- remap seed: `{row['remap_slot_seed']}`",
        f"- budget bytes: `{row['budget_bytes']}`",
        f"- source packet pass: `{row['source_packet_pass']}`",
        f"- source accuracy: `{row['source_accuracy']:.3f}`",
        f"- target accuracy: `{row['target_accuracy']:.3f}`",
        f"- best control: `{row['best_control_condition']}` at `{row['best_control_accuracy']:.3f}`",
        f"- source minus canonical: `{_fmt(row.get('source_minus_canonical'))}`",
        "",
        "| Condition | Accuracy | Correct | Mean bytes | p50 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, metrics in row["metrics"].items():
        lines.append(
            f"| {condition} | {metrics['accuracy']:.3f} | {metrics['correct']}/{row['n']} | "
            f"{metrics['mean_payload_bytes']:.2f} | {metrics['p50_latency_ms']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Product-Codebook Geometry Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{payload['headline']['rows']}`",
        f"- source pass rows: `{payload['headline']['source_pass_rows']}`",
        f"- promoted rows: `{payload['headline']['promoted_rows']}`",
        f"- max source accuracy: `{_fmt(payload['headline']['max_source_accuracy'])}`",
        f"- max noncanonical minus canonical: `{_fmt(payload['headline']['max_noncanonical_minus_canonical'])}`",
        "",
        "| Remap | Budget | Variant | Source | Target | Best control | Source-control | Source-canonical | Pass |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['variant']} | "
            f"{row['source_accuracy']:.3f} | {row['target_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | "
            f"{row['source_minus_best_control']:.3f} | {_fmt(row.get('source_minus_canonical'))} | "
            f"`{row['source_packet_pass']}` |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--variants",
        choices=[
            "canonical",
            "utility_round_robin",
            "utility_balanced",
            "random_balanced",
            "opq_procrustes",
            "utility_opq_procrustes",
            "protected_hadamard",
            "utility_protected_hadamard",
        ],
        nargs="+",
        default=["canonical", "utility_round_robin", "utility_balanced"],
    )
    parser.add_argument("--train-family-set", default="all")
    parser.add_argument("--eval-family-set", default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", default="slot")
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--opq-iterations", type=int, default=4)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_geometry_gate(
        output_dir=output_dir,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        remap_seeds=args.remap_seeds,
        budgets=args.budgets,
        variants=args.variants,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        ridge=args.ridge,
        candidate_view=args.candidate_view,
        fit_intercept=args.fit_intercept,
        bootstrap_samples=args.bootstrap_samples,
        opq_iterations=args.opq_iterations,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "command": " ".join(sys.argv),
            "args": vars(args) | {"output_dir": str(args.output_dir)},
            "python": sys.version,
            "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "script_sha256": _sha256_file(pathlib.Path(__file__)),
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.require_pass and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
