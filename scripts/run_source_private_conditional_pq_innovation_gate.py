from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_source_private_product_codebook_geometry_gate import (  # noqa: E402
    GeometryProductCodebook,
    _fit_centroids_for_vectors,
    _groups_for_variant,
    _protected_hadamard_rotation,
)
from scripts.run_source_private_candidate_embedding_receiver import _build_anchor_matrix  # noqa: E402
from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_masked_innovation_receiver import (  # noqa: E402
    _SEMANTIC_CANDIDATE_MATRIX_CACHE,
    _semantic_candidate_matrix,
    _source_innovation,
)
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _constrained_nonself_index,
    _remap_candidate_slots,
)
from scripts.run_source_private_tool_trace_learned_syndrome import (  # noqa: E402
    _augment,
    _normalize_rows,
    _token_count,
)


CONDITIONS = [
    "target_only",
    "source",
    "label_shuffled_encoder",
    "constrained_shuffled_source",
    "same_answer_slot_wrong_row_source",
    "answer_masked_source",
    "public_condition_only",
    "permuted_codes",
    "random_same_byte",
    "deranged_public_basis",
    "opaque_slot_basis",
]
CONTROL_CONDITIONS = [
    "label_shuffled_encoder",
    "constrained_shuffled_source",
    "same_answer_slot_wrong_row_source",
    "answer_masked_source",
    "public_condition_only",
    "permuted_codes",
    "random_same_byte",
    "deranged_public_basis",
    "opaque_slot_basis",
]
BASIS_VIEWS = ("shared_text", "anchor_relative", "semantic", "no_diag", "full", "diag_only", "slot")
CONDITIONING_MODES = ("none", "public_zscore")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_index(example: Example) -> int:
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == prior)


def _basis_candidates(
    example: Example,
    *,
    basis_view: str,
    feature_dim: int,
    anchor_matrix: np.ndarray | None,
) -> np.ndarray:
    if basis_view == "shared_text":
        matrix = _semantic_candidate_matrix(example, feature_dim=feature_dim)
    elif basis_view == "anchor_relative":
        if anchor_matrix is None:
            raise ValueError("anchor_relative basis requires anchor_matrix")
        base = _candidate_matrix_for_view(example, feature_dim, candidate_view="full")
        matrix = _normalize_rows(base @ anchor_matrix.T).astype(np.float32)
    elif basis_view in {"slot", "diag_only", "semantic", "no_diag", "full"}:
        base = _candidate_matrix_for_view(example, feature_dim, candidate_view=basis_view)
        if anchor_matrix is None:
            matrix = base
        else:
            matrix = _normalize_rows(base @ anchor_matrix.T).astype(np.float32)
    else:
        raise ValueError(f"unknown basis view {basis_view!r}")
    return matrix.astype(np.float32)


def _candidate_innovations_for_basis(
    example: Example,
    *,
    basis_view: str,
    feature_dim: int,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
) -> np.ndarray:
    matrix = _basis_candidates(example, basis_view=basis_view, feature_dim=feature_dim, anchor_matrix=anchor_matrix)
    prior = matrix[_prior_index(example)]
    deltas = matrix - prior[None, :]
    if 0 < target_topk < deltas.shape[1]:
        sparse = np.zeros_like(deltas)
        for row_index, delta in enumerate(deltas):
            keep = np.argpartition(np.abs(delta), -target_topk)[-target_topk:]
            sparse[row_index, keep] = delta[keep]
        deltas = sparse
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    return np.divide(deltas, np.maximum(norms, 1e-8)).astype(np.float32)


def _target_innovation_for_basis(
    example: Example,
    *,
    basis_view: str,
    feature_dim: int,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
) -> np.ndarray:
    return _candidate_innovations_for_basis(
        example,
        basis_view=basis_view,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
    )[_answer_index(example)]


def _public_condition_stats(
    example: Example,
    *,
    basis_view: str,
    feature_dim: int,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = _candidate_innovations_for_basis(
        example,
        basis_view=basis_view,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
    )
    center = np.mean(candidates, axis=0).astype(np.float32)
    scale = np.sqrt(np.mean((candidates - center[None, :]) ** 2, axis=0) + 1e-6).astype(np.float32)
    return center, scale


def _same_answer_slot_nonself_index(index: int, examples: list[Example]) -> int:
    current = examples[index]
    current_slot = _answer_index(current)
    n = len(examples)
    for offset in range(1, n):
        candidate_index = (index * 23 + 5 + offset) % n
        candidate = examples[candidate_index]
        if candidate_index != index and candidate.family_name != current.family_name and _answer_index(candidate) == current_slot:
            return candidate_index
    for offset in range(1, n):
        candidate_index = (index + offset) % n
        if candidate_index != index and _answer_index(examples[candidate_index]) == current_slot:
            return candidate_index
    return _constrained_nonself_index(index, examples)


def _condition_vector_to_public(
    vector: np.ndarray,
    example: Example,
    *,
    conditioning_mode: str,
    basis_view: str,
    feature_dim: int,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
) -> np.ndarray:
    if conditioning_mode == "none":
        return vector.astype(np.float32)
    if conditioning_mode != "public_zscore":
        raise ValueError(f"unknown conditioning mode {conditioning_mode!r}")
    center, scale = _public_condition_stats(
        example,
        basis_view=basis_view,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
    )
    conditioned = (vector - center) / scale
    norm = float(np.linalg.norm(conditioned))
    if norm > 1e-8:
        conditioned = conditioned / norm
    return conditioned.astype(np.float32)


def _condition_candidates_to_public(
    candidates: np.ndarray,
    example: Example,
    *,
    conditioning_mode: str,
    basis_view: str,
    feature_dim: int,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
) -> np.ndarray:
    if conditioning_mode == "none":
        return candidates.astype(np.float32)
    if conditioning_mode != "public_zscore":
        raise ValueError(f"unknown conditioning mode {conditioning_mode!r}")
    center, scale = _public_condition_stats(
        example,
        basis_view=basis_view,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
    )
    conditioned = (candidates - center[None, :]) / scale[None, :]
    norms = np.linalg.norm(conditioned, axis=1, keepdims=True)
    return np.divide(conditioned, np.maximum(norms, 1e-8)).astype(np.float32)


def _fit_conditional_encoder(
    train_rows: list[Example],
    *,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    source_topk: int,
    target_topk: int,
    ridge: float,
    mask_repeats: int,
    seed: int,
    fit_intercept: bool,
    label_shuffle_seed: int | None = None,
    constrained_label_shuffle: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    label_indices = list(range(len(train_rows)))
    if constrained_label_shuffle:
        label_indices = [_constrained_nonself_index(index, train_rows) for index in range(len(train_rows))]
    elif label_shuffle_seed is not None:
        shuffled = random.Random(label_shuffle_seed)
        shuffled.shuffle(label_indices)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for row_index, example in enumerate(train_rows):
        label_example = train_rows[label_indices[row_index]]
        source = _source_innovation(example, feature_dim=feature_dim, source_topk=source_topk, mode="matched")
        target = _target_innovation_for_basis(
            label_example,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        xs.append(source)
        ys.append(target)
        for _ in range(mask_repeats):
            source_mask = (rng.random(source.shape) >= 0.20).astype(np.float32)
            target_mask = (rng.random(target.shape) >= 0.10).astype(np.float32)
            xs.append(source * source_mask)
            ys.append(target * target_mask)
    x = np.stack(xs, axis=0).astype(np.float64)
    y = np.stack(ys, axis=0).astype(np.float64)
    if fit_intercept:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    else:
        x_aug = x
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    if fit_intercept:
        xtx[-1, -1] -= ridge
    return np.linalg.solve(xtx, x_aug.T @ y).astype(np.float32)


def _predict_conditional_innovation(
    example: Example,
    *,
    encoder: np.ndarray,
    feature_dim: int,
    source_topk: int,
    mode: str,
) -> np.ndarray:
    source = _source_innovation(example, feature_dim=feature_dim, source_topk=source_topk, mode=mode)
    if encoder.shape[0] == feature_dim:
        vector = source @ encoder
    elif encoder.shape[0] == feature_dim + 1:
        vector = _augment(source) @ encoder
    else:
        raise ValueError(f"encoder shape {encoder.shape} is incompatible with feature_dim={feature_dim}")
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector = vector / norm
    return vector.astype(np.float32)


def _dimension_utilities(
    train_rows: list[Example],
    *,
    encoder: np.ndarray,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
) -> np.ndarray:
    utilities = np.zeros(_representation_dim(feature_dim=feature_dim, anchor_matrix=anchor_matrix), dtype=np.float64)
    for example in train_rows:
        predicted = _predict_conditional_innovation(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        predicted = _condition_vector_to_public(
            predicted,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        candidates = _candidate_innovations_for_basis(
            example,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        candidates = _condition_candidates_to_public(
            candidates,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        answer_index = _answer_index(example)
        gold_dist = np.abs(candidates[answer_index] - predicted)
        negative = np.delete(candidates, answer_index, axis=0)
        negative_dist = np.min(np.abs(negative - predicted[None, :]), axis=0)
        utilities += negative_dist - gold_dist
    return utilities.astype(np.float32)


def _representation_dim(*, feature_dim: int, anchor_matrix: np.ndarray | None) -> int:
    return feature_dim if anchor_matrix is None else int(anchor_matrix.shape[0])


def _fit_innovation_codebook(
    train_rows: list[Example],
    *,
    encoder: np.ndarray,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
    groups: tuple[np.ndarray, ...],
    variant: str,
    utilities: np.ndarray,
    seed: int,
    iterations: int,
) -> GeometryProductCodebook:
    vectors = np.stack(
        [
            _condition_vector_to_public(
                _predict_conditional_innovation(
                    example,
                    encoder=encoder,
                    feature_dim=feature_dim,
                    source_topk=source_topk,
                    mode="matched",
                ),
                example,
                conditioning_mode=conditioning_mode,
                basis_view=basis_view,
                feature_dim=feature_dim,
                anchor_matrix=anchor_matrix,
                target_topk=target_topk,
            )
            for example in train_rows
        ],
        axis=0,
    ).astype(np.float32)
    rotation: np.ndarray | None = None
    rotated = vectors
    if variant in {"protected_hadamard", "utility_protected_hadamard"}:
        rotation = _protected_hadamard_rotation(
            feature_dim=vectors.shape[1],
            seed=seed,
            utilities=utilities,
            utility_ordered=variant == "utility_protected_hadamard",
        )
        rotated = (vectors @ rotation).astype(np.float32)
    centroids = _fit_centroids_for_vectors(rotated, groups=groups, seed=seed, iterations=iterations)
    utility_sum_by_group = tuple(float(np.sum(utilities[group])) for group in groups)
    return GeometryProductCodebook(
        centroids=centroids,
        groups=groups,
        variant=variant,
        utility_sum_by_group=utility_sum_by_group,
        rotation=rotation,
    )


def _packet_from_vector(vector: np.ndarray, *, codebook: GeometryProductCodebook) -> bytes:
    if codebook.rotation is not None:
        vector = vector @ codebook.rotation
    codes = np.zeros(codebook.subspaces, dtype=np.uint8)
    for subspace_index, (centroids, group) in enumerate(zip(codebook.centroids, codebook.groups, strict=True)):
        part = vector[group]
        distances = np.sum((centroids - part[None, :]) ** 2, axis=1)
        codes[subspace_index] = int(np.argmin(distances))
    return codes.tobytes()


def _decode_packet(
    example: Example,
    payload: bytes | None,
    *,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
    conditioning_mode: str,
    permutation: np.ndarray | None = None,
) -> tuple[str, dict[str, Any]]:
    prior = _prior_prediction(example)
    if not payload:
        return prior, {"decoder": "prior"}
    reconstructed = np.zeros(_representation_dim(feature_dim=feature_dim, anchor_matrix=anchor_matrix), dtype=np.float32)
    raw = np.frombuffer(payload[: codebook.subspaces], dtype=np.uint8)
    for subspace_index, code in enumerate(raw):
        centroids = codebook.centroids[subspace_index]
        reconstructed[codebook.groups[subspace_index]] = centroids[int(code) % centroids.shape[0]]
    candidates = _candidate_innovations_for_basis(
        example,
        basis_view=basis_view,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
    )
    candidates = _condition_candidates_to_public(
        candidates,
        example,
        conditioning_mode=conditioning_mode,
        basis_view=basis_view,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
    )
    if permutation is not None:
        candidates = candidates[permutation]
    if codebook.rotation is not None:
        candidates = (candidates @ codebook.rotation).astype(np.float32)
    distances = np.sum((candidates - reconstructed[None, :]) ** 2, axis=1)
    min_distance = float(np.min(distances))
    tied = np.flatnonzero(np.isclose(distances, min_distance, rtol=1e-6, atol=1e-8))
    if any(example.candidates[int(idx)].label == prior for idx in tied):
        prediction = prior
    else:
        prediction = example.candidates[int(tied[0])].label
    return prediction, {
        "decoder": f"{codebook.variant}_conditional_pq_innovation_l2",
        "basis_view": basis_view,
        "conditioning_mode": conditioning_mode,
        "min_l2": min_distance,
        "ties": [int(value) for value in tied.tolist()],
        "deranged_public_basis": permutation is not None,
    }


def _derangement(length: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(32):
        perm = rng.permutation(length)
        if np.all(perm != np.arange(length)):
            return perm.astype(np.int64)
    return np.roll(np.arange(length), 1).astype(np.int64)


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
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
    rng: random.Random,
) -> bytes | None:
    if condition == "target_only":
        return None
    if condition in {"source", "deranged_public_basis", "opaque_slot_basis"}:
        vector = _predict_conditional_innovation(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        return _packet_from_vector(vector, codebook=codebook)
    if condition == "label_shuffled_encoder":
        vector = _predict_conditional_innovation(
            example,
            encoder=label_shuffle_encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        return _packet_from_vector(vector, codebook=codebook)
    if condition == "constrained_shuffled_source":
        other = eval_rows[_constrained_nonself_index(index, eval_rows)]
        vector = _predict_conditional_innovation(
            other,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        return _packet_from_vector(vector, codebook=codebook)
    if condition == "same_answer_slot_wrong_row_source":
        other = eval_rows[_same_answer_slot_nonself_index(index, eval_rows)]
        vector = _predict_conditional_innovation(
            other,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        return _packet_from_vector(vector, codebook=codebook)
    if condition in {"answer_masked_source", "public_condition_only"}:
        vector = _predict_conditional_innovation(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="answer_masked",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        return _packet_from_vector(vector, codebook=codebook)
    if condition == "permuted_codes":
        vector = _predict_conditional_innovation(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        return _permute_payload(_packet_from_vector(vector, codebook=codebook))
    if condition == "random_same_byte":
        return rng.randbytes(codebook.subspaces)
    raise ValueError(f"unknown condition {condition!r}")


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_rows: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
    source_topk: int,
    conditioning_mode: str,
    seed: int,
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
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        rng=rng,
    )
    decode_basis = "slot" if condition == "opaque_slot_basis" else basis_view
    permutation = _derangement(len(example.candidates), seed=seed * 1009 + index) if condition == "deranged_public_basis" else None
    prediction, metadata = _decode_packet(
        example,
        payload,
        codebook=codebook,
        feature_dim=feature_dim,
        basis_view=decode_basis,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        permutation=permutation,
    )
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "example_id": example.example_id,
        "family_name": example.family_name,
        "answer_label": example.answer_label,
        "target_prior_label": _prior_prediction(example),
        "prediction": prediction,
        "correct": prediction == example.answer_label,
        "payload_hex": payload_hex,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "metadata": metadata,
    }


def _unquantized_accuracy(
    eval_rows: list[Example],
    *,
    encoder: np.ndarray,
    feature_dim: int,
    source_topk: int,
    target_topk: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    conditioning_mode: str,
) -> dict[str, Any]:
    correct_ids: list[str] = []
    for example in eval_rows:
        vector = _predict_conditional_innovation(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            source_topk=source_topk,
            mode="matched",
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        candidates = _candidate_innovations_for_basis(
            example,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        candidates = _condition_candidates_to_public(
            candidates,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        scores = candidates @ vector
        best = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best, rtol=1e-6, atol=1e-8))
        prior = _prior_prediction(example)
        if any(example.candidates[int(idx)].label == prior for idx in tied):
            prediction = prior
        else:
            prediction = example.candidates[int(tied[0])].label
        if prediction == example.answer_label:
            correct_ids.append(example.example_id)
    return {"correct": len(correct_ids), "accuracy": len(correct_ids) / len(eval_rows), "correct_ids": correct_ids}


def _oracle_accuracy(
    eval_rows: list[Example],
    *,
    feature_dim: int,
    target_topk: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    conditioning_mode: str,
) -> dict[str, Any]:
    correct_ids: list[str] = []
    for example in eval_rows:
        vector = _target_innovation_for_basis(
            example,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        vector = _condition_vector_to_public(
            vector,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        candidates = _candidate_innovations_for_basis(
            example,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        candidates = _condition_candidates_to_public(
            candidates,
            example,
            conditioning_mode=conditioning_mode,
            basis_view=basis_view,
            feature_dim=feature_dim,
            anchor_matrix=anchor_matrix,
            target_topk=target_topk,
        )
        prediction = example.candidates[int(np.argmax(candidates @ vector))].label
        if prediction == example.answer_label:
            correct_ids.append(example.example_id)
    return {"correct": len(correct_ids), "accuracy": len(correct_ids) / len(eval_rows), "correct_ids": correct_ids}


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_ids = [row["example_id"] for row in rows if row["correct"]]
    latencies = [float(row["latency_ms"]) for row in rows]
    payloads = [int(row["payload_bytes"]) for row in rows]
    tokens = [int(row["payload_tokens"]) for row in rows]
    return {
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(rows),
        "correct_ids": correct_ids,
        "mean_payload_bytes": statistics.fmean(payloads),
        "mean_payload_tokens": statistics.fmean(tokens),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    ids = sorted({row["example_id"] for row in rows})
    by = {(row["condition"], row["example_id"]): bool(row["correct"]) for row in rows}
    diffs = np.asarray(
        [float(by[(condition_a, example_id)]) - float(by[(condition_b, example_id)]) for example_id in ids],
        dtype=np.float32,
    )
    point = float(np.mean(diffs)) if len(diffs) else 0.0
    if len(diffs) <= 1:
        return {"point": point, "ci95_low": point, "ci95_high": point}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float32)
    for idx in range(samples):
        sampled = rng.integers(0, len(diffs), size=len(diffs))
        boot[idx] = float(np.mean(diffs[sampled]))
    return {
        "point": point,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _payload_uniqueness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [row for row in rows if row["condition"] == "source"]
    payloads = [row["payload_hex"] for row in source_rows]
    counts: dict[str, int] = {}
    for payload in payloads:
        counts[payload] = counts.get(payload, 0) + 1
    reused = {payload for payload, count in counts.items() if count > 1}
    reused_rows = [row for row in source_rows if row["payload_hex"] in reused]
    reused_correct = sum(1 for row in reused_rows if row["correct"])
    return {
        "unique_payloads": len(counts),
        "unique_payload_ratio": len(counts) / max(1, len(payloads)),
        "max_payload_frequency": max(counts.values()) if counts else 0,
        "reused_payload_examples": len(reused_rows),
        "reused_payload_accuracy": None if not reused_rows else reused_correct / len(reused_rows),
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    conditions: list[str],
    train_rows: list[Example],
    eval_rows: list[Example],
    unquantized: dict[str, Any],
    oracle: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    eval_ids = [row.example_id for row in eval_rows]
    train_ids = {row.example_id for row in train_rows}
    condition_ids = {
        condition: [row["example_id"] for row in rows if row["condition"] == condition]
        for condition in conditions
    }
    exact_id_parity = all(ids == eval_ids for ids in condition_ids.values())
    metrics = {
        condition: _metric([row for row in rows if row["condition"] == condition])
        for condition in conditions
    }
    target = metrics["target_only"]["accuracy"]
    source = metrics["source"]["accuracy"]
    best_control_condition = max(CONTROL_CONDITIONS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_condition]["accuracy"]
    controls_ok = all(metrics[condition]["accuracy"] <= target + 0.06 for condition in CONTROL_CONDITIONS)
    paired_vs_target = _paired_bootstrap(
        rows,
        condition_a="source",
        condition_b="target_only",
        samples=bootstrap_samples,
        seed=seed + 17,
    )
    paired_vs_best_control = _paired_bootstrap(
        rows,
        condition_a="source",
        condition_b=best_control_condition,
        samples=bootstrap_samples,
        seed=seed + 31,
    )
    train_eval_overlap = sorted(train_ids.intersection(eval_ids))
    payload_uniqueness = _payload_uniqueness(rows)
    payload_uniqueness_ok = (
        payload_uniqueness["unique_payload_ratio"] < 0.90
        or (
            payload_uniqueness["reused_payload_accuracy"] is not None
            and payload_uniqueness["reused_payload_accuracy"] >= target + 0.10
        )
    )
    pass_gate = (
        exact_id_parity
        and len(train_eval_overlap) == 0
        and source >= target + 0.15
        and source >= best_control + 0.15
        and controls_ok
        and paired_vs_best_control["ci95_low"] > 0.10
        and payload_uniqueness_ok
    )
    return {
        "n": len(eval_rows),
        "conditions": conditions,
        "exact_id_parity": exact_id_parity,
        "train_eval_id_intersection_count": len(train_eval_overlap),
        "train_eval_id_intersection_preview": train_eval_overlap[:10],
        "eval_id_sha256": hashlib.sha256("\n".join(eval_ids).encode("utf-8")).hexdigest(),
        "target_only_accuracy": target,
        "source_accuracy": source,
        "source_minus_target": source - target,
        "best_control_condition": best_control_condition,
        "best_control_accuracy": best_control,
        "source_minus_best_control": source - best_control,
        "controls_ok": controls_ok,
        "unquantized_predicted_accuracy": unquantized["accuracy"],
        "target_innovation_oracle_accuracy": oracle["accuracy"],
        "paired_bootstrap": {
            "source_vs_target": paired_vs_target,
            "source_vs_best_control": paired_vs_best_control,
        },
        "payload_uniqueness": payload_uniqueness,
        "payload_uniqueness_ok": payload_uniqueness_ok,
        "metrics": metrics,
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, "
            "source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low "
            "vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy "
            ">= target+0.10."
        ),
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Source-Private Conditional PQ Innovation Gate",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- examples: `{summary['n']}`",
        f"- train/eval ID overlap: `{summary['train_eval_id_intersection_count']}`",
        f"- basis view: `{payload['basis_view']}`",
        f"- conditioning mode: `{payload['conditioning_mode']}`",
        f"- variant: `{payload['variant']}`",
        f"- budget bytes: `{payload['budget_bytes']}`",
        f"- source accuracy: `{summary['source_accuracy']:.3f}`",
        f"- target accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
        f"- unquantized predicted accuracy: `{summary['unquantized_predicted_accuracy']:.3f}`",
        f"- target innovation oracle accuracy: `{summary['target_innovation_oracle_accuracy']:.3f}`",
        "",
        "| Condition | Accuracy | Mean bytes | p50 ms |",
        "|---|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        metric = summary["metrics"][condition]
        lines.append(
            f"| {condition} | {metric['accuracy']:.3f} | {metric['mean_payload_bytes']:.2f} | "
            f"{metric['p50_latency_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Payload uniqueness: `{summary['payload_uniqueness']}`",
            "",
            f"Pass rule: {summary['pass_rule']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_start_index: int,
    eval_start_index: int,
    train_family_set: str,
    eval_family_set: str,
    diagnostic_table_mode: str,
    candidates: int,
    feature_dim: int,
    anchor_count: int,
    basis_view: str,
    source_topk: int,
    target_topk: int,
    budget_bytes: int,
    variant: str,
    remap_slot_seed: int | None,
    ridge: float,
    fit_intercept: bool,
    mask_repeats: int,
    codebook_iterations: int,
    seed: int,
    bootstrap_samples: int,
    conditioning_mode: str = "none",
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _SEMANTIC_CANDIDATE_MATRIX_CACHE.clear()
    train_rows = make_benchmark(
        examples=train_examples,
        candidates=candidates,
        seed=train_seed,
        family_set=train_family_set,
        start_index=train_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    eval_rows = make_benchmark(
        examples=eval_examples,
        candidates=candidates,
        seed=eval_seed,
        family_set=eval_family_set,
        start_index=eval_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_slot_seed)
    eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_slot_seed)
    if basis_view == "anchor_relative":
        anchor_matrix: np.ndarray | None = _build_anchor_matrix(
            train_rows,
            feature_dim=feature_dim,
            anchor_count=anchor_count,
        )
    else:
        anchor_matrix = None
    representation_dim = _representation_dim(feature_dim=feature_dim, anchor_matrix=anchor_matrix)
    encoder = _fit_conditional_encoder(
        train_rows,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        ridge=ridge,
        fit_intercept=fit_intercept,
        mask_repeats=mask_repeats,
        seed=seed * 1009 + train_seed,
    )
    label_shuffle_encoder = _fit_conditional_encoder(
        train_rows,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        ridge=ridge,
        fit_intercept=fit_intercept,
        mask_repeats=mask_repeats,
        seed=seed * 1009 + train_seed,
        label_shuffle_seed=train_seed * 5003 + eval_seed,
        constrained_label_shuffle=True,
    )
    utilities = _dimension_utilities(
        train_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
    )
    groups = _groups_for_variant(
        variant=variant,
        feature_dim=representation_dim,
        budget_bytes=budget_bytes,
        utilities=utilities,
        seed=train_seed * 11003 + eval_seed * 97 + budget_bytes + (remap_slot_seed or 0),
    )
    codebook = _fit_innovation_codebook(
        train_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        groups=groups,
        variant=variant,
        utilities=utilities,
        seed=train_seed * 9001 + eval_seed * 17 + budget_bytes,
        iterations=codebook_iterations,
    )
    active_conditions = list(conditions or CONDITIONS)
    rng = random.Random(seed * 4001 + train_seed * 37 + eval_seed)
    rows: list[dict[str, Any]] = []
    for row_index, example in enumerate(eval_rows):
        for condition in active_conditions:
            rows.append(
                _predict_condition(
                    condition=condition,
                    example=example,
                    eval_rows=eval_rows,
                    index=row_index,
                    encoder=encoder,
                    label_shuffle_encoder=label_shuffle_encoder,
                    codebook=codebook,
                    feature_dim=feature_dim,
                    basis_view=basis_view,
                    anchor_matrix=anchor_matrix,
                    target_topk=target_topk,
                    source_topk=source_topk,
                    conditioning_mode=conditioning_mode,
                    seed=seed,
                    rng=rng,
                )
            )
    unquantized = _unquantized_accuracy(
        eval_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        source_topk=source_topk,
        target_topk=target_topk,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        conditioning_mode=conditioning_mode,
    )
    oracle = _oracle_accuracy(
        eval_rows,
        feature_dim=feature_dim,
        target_topk=target_topk,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        conditioning_mode=conditioning_mode,
    )
    summary = _summarize(
        rows,
        conditions=active_conditions,
        train_rows=train_rows,
        eval_rows=eval_rows,
        unquantized=unquantized,
        oracle=oracle,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    payload = {
        "gate": "source_private_conditional_pq_innovation",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_start_index": train_start_index,
        "eval_start_index": eval_start_index,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "diagnostic_table_mode": diagnostic_table_mode,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "anchor_count": anchor_count,
        "representation_dim": representation_dim,
        "basis_view": basis_view,
        "source_topk": source_topk,
        "target_topk": target_topk,
        "conditioning_mode": conditioning_mode,
        "budget_bytes": budget_bytes,
        "variant": variant,
        "remap_slot_seed": remap_slot_seed,
        "ridge": ridge,
        "fit_intercept": fit_intercept,
        "mask_repeats": mask_repeats,
        "codebook_iterations": codebook_iterations,
        "conditions": active_conditions,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "label_shuffle_encoder_sha256": hashlib.sha256(label_shuffle_encoder.tobytes()).hexdigest(),
        "codebook_sha256": hashlib.sha256(b"".join(centroid.tobytes() for centroid in codebook.centroids)).hexdigest(),
        "summary": summary,
        "pass_gate": summary["pass_gate"],
        "interpretation": (
            "This gate sends a rate-capped product-quantized conditional innovation: source matched "
            "features minus answer-masked source features are mapped into the target/public candidate "
            "innovation basis and decoded against candidate-minus-target-prior vectors. It directly tests "
            "whether a shared public basis fixes the disjoint-ID PQ collapse. The public_zscore conditioning "
            "mode additionally normalizes packet and candidate innovations by each row's public candidate "
            "geometry before quantization/decoding."
        ),
    }
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", "predictions.jsonl", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md", "predictions.jsonl"]
        },
        "pass_gate": summary["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Conditional PQ Innovation Gate Manifest",
                "",
                f"- pass gate: `{summary['pass_gate']}`",
                f"- basis view: `{basis_view}`",
                f"- conditioning mode: `{conditioning_mode}`",
                f"- train/eval overlap: `{summary['train_eval_id_intersection_count']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_conditional_pq_innovation_gate_20260430"),
    )
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=29)
    parser.add_argument("--train-start-index", type=int, default=10000)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="plausible_decoys")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--basis-view", choices=BASIS_VIEWS, default="shared_text")
    parser.add_argument("--source-topk", type=int, default=64)
    parser.add_argument("--target-topk", type=int, default=32)
    parser.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default="none")
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument(
        "--variant",
        choices=["canonical", "utility_balanced", "protected_hadamard", "utility_protected_hadamard"],
        default="utility_protected_hadamard",
    )
    parser.add_argument("--remap-slot-seed", type=int, default=101)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--mask-repeats", type=int, default=1)
    parser.add_argument("--codebook-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        diagnostic_table_mode=args.diagnostic_table_mode,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        anchor_count=args.anchor_count,
        basis_view=args.basis_view,
        source_topk=args.source_topk,
        target_topk=args.target_topk,
        conditioning_mode=args.conditioning_mode,
        budget_bytes=args.budget_bytes,
        variant=args.variant,
        remap_slot_seed=args.remap_slot_seed,
        ridge=args.ridge,
        fit_intercept=args.fit_intercept,
        mask_repeats=args.mask_repeats,
        codebook_iterations=args.codebook_iterations,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "source_accuracy": payload["summary"]["source_accuracy"],
                "target_accuracy": payload["summary"]["target_only_accuracy"],
                "best_control_accuracy": payload["summary"]["best_control_accuracy"],
                "unquantized_predicted_accuracy": payload["summary"]["unquantized_predicted_accuracy"],
                "oracle_accuracy": payload["summary"]["target_innovation_oracle_accuracy"],
                "output_dir": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
