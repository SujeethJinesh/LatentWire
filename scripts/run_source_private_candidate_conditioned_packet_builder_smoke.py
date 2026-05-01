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

from scripts import run_source_private_learned_synonym_dictionary_packet_gate as syn  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_conditioned_packet_builder_smoke_20260501")
BASE_MATCHED_CONDITION = "learned_synonym_dictionary_packet"
MATCHED_CONDITION = "candidate_conditioned_packet_builder"
ORACLE_CONDITION = "oracle_candidate_conditioned_packet"
STRICT_CONTROLS = tuple(syn.STRICT_SOURCE_DESTROYING_CONTROLS)
EVAL_CONDITIONS = ("target_only", BASE_MATCHED_CONDITION, MATCHED_CONDITION, *STRICT_CONTROLS, ORACLE_CONDITION)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_vector_from_atoms(atoms: dict[str, float]) -> np.ndarray:
    return syn._atom_vector(atoms)


def _source_vector(example: syn.Example, *, mode: str = "matched") -> np.ndarray:
    return _source_vector_from_atoms(syn._source_private_atoms(example.private_test_log, mode=mode))


def _candidate_vector(
    *,
    example: syn.Example,
    candidate_index: int,
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
) -> np.ndarray:
    candidate = example.candidates[candidate_index]
    text = syn._candidate_surface_text(candidate.patch_intent, candidate_atom_view=candidate_atom_view)
    return dictionary.predict_vector(text, apply_top_k=True)


def _candidate_matrix(
    *,
    example: syn.Example,
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
) -> np.ndarray:
    return np.stack(
        [
            _candidate_vector(
                example=example,
                candidate_index=idx,
                dictionary=dictionary,
                candidate_atom_view=candidate_atom_view,
            )
            for idx, _ in enumerate(example.candidates)
        ],
        axis=0,
    )


def _packet_builder_target_vector(
    *,
    example: syn.Example,
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
    target_mode: str,
) -> np.ndarray:
    candidate_matrix = _candidate_matrix(
        example=example,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
    )
    answer = candidate_matrix[syn._answer_index(example)]
    if target_mode == "answer_candidate":
        return answer
    if target_mode == "answer_minus_candidate_mean":
        return np.maximum(answer - candidate_matrix.mean(axis=0), 0.0)
    if target_mode == "answer_minus_prior":
        return np.maximum(answer - candidate_matrix[_prior_index(example)], 0.0)
    raise ValueError(f"unknown packet builder target mode {target_mode!r}")


def _fit_packet_builder(
    *,
    examples: list[syn.Example],
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
    ridge: float,
    target_mode: str = "answer_candidate",
) -> dict[str, Any]:
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    dim = len(syn.ATOM_ORDER)
    for example in examples:
        source = _source_vector(example, mode="matched")
        if not np.any(source):
            continue
        target = _packet_builder_target_vector(
            example=example,
            dictionary=dictionary,
            candidate_atom_view=candidate_atom_view,
            target_mode=target_mode,
        )
        source_rows.append(np.concatenate([np.ones(1, dtype=np.float64), source]))
        target_rows.append(target)
    if not source_rows:
        raise ValueError("no nonempty source rows available for packet builder fitting")
    x = np.stack(source_rows, axis=0).astype(np.float64)
    y = np.stack(target_rows, axis=0).astype(np.float64)
    source_matrix = x[:, 1:]
    xtx = x.T @ x + ridge * np.eye(dim + 1, dtype=np.float64)
    xtx[0, 0] -= ridge
    weights = np.linalg.solve(xtx, x.T @ y)
    predicted = np.maximum(x @ weights, 0.0)
    target_norms = np.linalg.norm(y, axis=1)
    prediction_norms = np.linalg.norm(predicted, axis=1)
    cosine = np.sum(predicted * y, axis=1) / np.maximum(target_norms * prediction_norms, 1e-12)
    mean_prediction = predicted.mean(axis=0)
    return {
        "weights": weights,
        "ridge": float(ridge),
        "packet_builder_target_mode": target_mode,
        "train_mean_source_vector": [float(value) for value in source_matrix.mean(axis=0)],
        "train_mean_prediction_vector": [float(value) for value in mean_prediction],
        "train_donor_source_vectors": [[float(value) for value in row] for row in source_matrix],
        "train_donor_mapped_vectors": [[float(value) for value in row] for row in predicted],
        "train_donor_example_ids": [
            str(example.example_id)
            for example in examples
            if np.any(_source_vector(example, mode="matched"))
        ],
        "train_donor_answer_indices": [
            int(syn._answer_index(example))
            for example in examples
            if np.any(_source_vector(example, mode="matched"))
        ],
        "train_examples": len(examples),
        "fit_rows": len(source_rows),
        "source_dim": dim,
        "target_dim": dim,
        "train_mean_target_l2": float(statistics.fmean(float(value) for value in target_norms)),
        "train_mean_prediction_l2": float(statistics.fmean(float(value) for value in prediction_norms)),
        "train_mean_cosine_to_target": float(statistics.fmean(float(value) for value in cosine)),
        "train_mean_cosine_to_answer_candidate": float(statistics.fmean(float(value) for value in cosine)),
        "train_p10_cosine_to_answer_candidate": float(syn._percentile([float(value) for value in cosine], 0.10)),
    }


def _predict_packet_vector(
    source_atoms: dict[str, float],
    builder: dict[str, Any],
    *,
    composition: str = "mapped",
    source_identity_weight: float = 0.0,
) -> np.ndarray:
    source = _source_vector_from_atoms(source_atoms)
    if not np.any(source):
        return np.zeros(len(syn.ATOM_ORDER), dtype=np.float64)
    weights = np.asarray(builder["weights"], dtype=np.float64)
    mapped = np.maximum(np.concatenate([np.ones(1, dtype=np.float64), source]) @ weights, 0.0)
    if composition.startswith("centered_"):
        center = np.asarray(builder.get("train_mean_prediction_vector", np.zeros_like(mapped)), dtype=np.float64)
        mapped = np.maximum(mapped - center, 0.0)
        composition = composition.removeprefix("centered_")
    if composition == "mapped":
        return mapped
    if composition == "add_source":
        return np.maximum(mapped + source_identity_weight * source, 0.0)
    if composition == "max_source":
        return np.maximum(mapped, source_identity_weight * source)
    raise ValueError(f"unknown packet builder composition {composition!r}")


def _atoms_from_packet_vector(vector: np.ndarray, *, budget_bytes: int, min_score: float) -> dict[str, float]:
    atom_budget = max(1, budget_bytes // 2)
    ranked = np.argsort(-vector)[:atom_budget]
    atoms: dict[str, float] = {}
    for atom_id in ranked:
        score = float(vector[int(atom_id)])
        if score >= min_score and score > 0.0:
            atoms[syn.ID_TO_ATOM[int(atom_id)]] = score
    return atoms


def _project_vector_to_candidate(
    vector: np.ndarray,
    *,
    example: syn.Example,
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_matrix = _candidate_matrix(
        example=example,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
    )
    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 0.0:
        return np.zeros(candidate_matrix.shape[1], dtype=np.float64), {
            "packet_projection": "zero_input",
            "packet_projection_scores": [0.0 for _ in example.candidates],
            "packet_projection_selected_index": None,
            "packet_projection_selected_label": None,
            "packet_projection_selected_is_prior": False,
            "packet_projection_selected_is_answer": False,
            "packet_projection_input_l2": vector_norm,
            "packet_projection_selected_l2": 0.0,
        }
    safe_candidates = np.divide(
        candidate_matrix,
        np.maximum(candidate_norms[:, None], 1e-12),
        out=np.zeros_like(candidate_matrix),
        where=candidate_norms[:, None] > 0,
    )
    safe_vector = vector / vector_norm
    scores = safe_candidates @ safe_vector
    selected_index = int(np.argmax(scores))
    return np.maximum(candidate_matrix[selected_index], 0.0), {
        "packet_projection": "nearest_public_candidate",
        "packet_projection_scores": [float(score) for score in scores],
        "packet_projection_selected_index": selected_index,
        "packet_projection_selected_label": example.candidates[selected_index].label,
        "packet_projection_selected_is_prior": selected_index == _prior_index(example),
        "packet_projection_selected_is_answer": selected_index == syn._answer_index(example),
        "packet_projection_input_l2": vector_norm,
        "packet_projection_selected_l2": float(np.linalg.norm(candidate_matrix[selected_index])),
    }


def _base_composition(composition: str) -> str:
    if composition.startswith("project_"):
        suffix = composition.removeprefix("project_")
        if suffix in {"mapped", "add_source", "max_source", "centered_mapped", "centered_add_source", "centered_max_source"}:
            return suffix
    return composition


def _full_atoms_from_vector(vector: np.ndarray, *, min_score: float = 0.0) -> dict[str, float]:
    return {
        syn.ID_TO_ATOM[int(atom_id)]: float(score)
        for atom_id, score in enumerate(vector)
        if float(score) > min_score
    }


def _score_packet_atoms(
    *,
    example: syn.Example,
    payload_atoms: dict[str, float],
    dictionary: syn.LearnedSynonymDictionary,
    null_dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
    decoder_score_mode: str,
    permuted_null_weight: float,
) -> tuple[list[float], dict[str, Any]]:
    if not payload_atoms:
        return [0.0 for _ in example.candidates], {"active_scores": [], "null_scores": []}
    if decoder_score_mode == "candidate_local_permuted_null_gap_residual_norm":
        active_scores, active_meta = syn._score_candidates(
            example=example,
            payload_atoms=payload_atoms,
            dictionary=dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode="candidate_local_residual_norm",
        )
        null_scores, null_meta = syn._score_candidates(
            example=example,
            payload_atoms=payload_atoms,
            dictionary=null_dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode="candidate_local_residual_norm",
        )
        scores = [
            float(active - permuted_null_weight * null)
            for active, null in zip(active_scores, null_scores, strict=True)
        ]
        return scores, {
            "active_scores": active_scores,
            "null_scores": null_scores,
            "active_payload_l2": active_meta.get("candidate_local_payload_l2", 0.0),
            "null_payload_l2": null_meta.get("candidate_local_payload_l2", 0.0),
        }
    scores, meta = syn._score_candidates(
        example=example,
        payload_atoms=payload_atoms,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
        decoder_score_mode=decoder_score_mode,
    )
    return scores, {"active_scores": scores, "null_scores": [], **meta}


def _score_margin(scores: list[float], target_index: int) -> float:
    if not scores:
        return 0.0
    other_scores = [score for idx, score in enumerate(scores) if idx != target_index]
    return float(scores[target_index] - max(other_scores or [0.0]))


def _top_index_with_prior_tie(example: syn.Example, scores: list[float]) -> int:
    if not scores:
        return _prior_index(example)
    best_score = max(scores)
    tied = [idx for idx, score in enumerate(scores) if abs(score - best_score) <= 1e-8]
    prior_index = _prior_index(example)
    return prior_index if prior_index in tied else tied[0]


def _nonoverlap_contrast_example(
    example: syn.Example,
    eval_examples: list[syn.Example],
    index: int,
    *,
    exclude_ids: set[str] | None = None,
) -> syn.Example:
    excluded = set(exclude_ids or set())
    excluded.add(example.example_id)
    current_atoms = set(syn._source_private_atoms(example.private_test_log, mode="matched"))
    current_answer = syn._answer_index(example)
    for offset in range(1, len(eval_examples)):
        other = eval_examples[(index + offset) % len(eval_examples)]
        if other.example_id in excluded:
            continue
        other_atoms = set(syn._source_private_atoms(other.private_test_log, mode="matched"))
        if syn._answer_index(other) != current_answer and not (current_atoms & other_atoms):
            return other
    for offset in range(1, len(eval_examples)):
        other = eval_examples[(index + offset) % len(eval_examples)]
        if other.example_id not in excluded and syn._answer_index(other) != current_answer:
            return other
    return syn._constrained_nonoverlap_example(example, eval_examples, index)


def _train_donor_contrast_vectors(
    *,
    example: syn.Example,
    builder: dict[str, Any],
    donor_count: int,
    source_identity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    source_vectors = np.asarray(builder.get("train_donor_source_vectors", []), dtype=np.float64)
    mapped_vectors = np.asarray(builder.get("train_donor_mapped_vectors", []), dtype=np.float64)
    dim = len(syn.ATOM_ORDER)
    if source_vectors.ndim != 2 or mapped_vectors.shape != source_vectors.shape or source_vectors.shape[0] == 0:
        zero = np.zeros(dim, dtype=np.float64)
        return zero, zero, zero, []

    donor_ids = [str(value) for value in builder.get("train_donor_example_ids", [])]
    donor_answers = [int(value) for value in builder.get("train_donor_answer_indices", [])]
    current_atoms = set(syn._source_private_atoms(example.private_test_log, mode="matched"))
    current_answer = int(syn._answer_index(example))
    seed_material = f"{example.example_id}|{source_vectors.shape[0]}|{donor_count}|train_donor_antishuffle"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    order = list(range(source_vectors.shape[0]))
    random.Random(seed).shuffle(order)

    selected: list[int] = []
    for strict in (True, False):
        for donor_index in order:
            if len(selected) >= max(1, donor_count):
                break
            if donor_index in selected:
                continue
            donor_id = donor_ids[donor_index] if donor_index < len(donor_ids) else str(donor_index)
            if donor_id == str(example.example_id):
                continue
            donor_answer = donor_answers[donor_index] if donor_index < len(donor_answers) else -1
            if donor_answer == current_answer:
                continue
            if strict:
                donor_atoms = {
                    syn.ID_TO_ATOM[int(atom_id)]
                    for atom_id, value in enumerate(source_vectors[donor_index])
                    if float(value) > 0.0
                }
                if current_atoms & donor_atoms:
                    continue
            selected.append(donor_index)
        if len(selected) >= max(1, donor_count):
            break

    if not selected:
        zero = np.zeros(dim, dtype=np.float64)
        return zero, zero, zero, []

    donor_source = source_vectors[selected]
    donor_mapped = mapped_vectors[selected]
    donor_proposal = np.maximum(donor_mapped + source_identity_weight * donor_source, 0.0)
    donor_strength = np.maximum(donor_proposal, donor_source)
    selected_ids = [
        donor_ids[donor_index] if donor_index < len(donor_ids) else str(donor_index)
        for donor_index in selected
    ]
    return (
        donor_source.mean(axis=0),
        donor_proposal.mean(axis=0),
        donor_strength.mean(axis=0),
        selected_ids,
    )


def _build_antishuffle_packet(
    *,
    source_atoms: dict[str, float],
    builder: dict[str, Any],
    composition: str,
    budget_bytes: int,
    packet_min_score: float,
    source_identity_weight: float,
    antishuffle_train_donors: int,
    antishuffle_donor_weight: float,
    antishuffle_null_weight: float,
    antishuffle_generic_weight: float,
    antishuffle_carrier_mode: str,
    example: syn.Example,
    eval_examples: list[syn.Example],
    index: int,
    dictionary: syn.LearnedSynonymDictionary,
    null_dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
    decoder_score_mode: str,
    permuted_null_weight: float,
    contrast_source_atoms: dict[str, float] | None = None,
    contrast_source_id: str | None = None,
    use_train_mean_contrast: bool = False,
    use_train_donor_contrast: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    source_vector = _source_vector_from_atoms(source_atoms)
    proposal_vector = _predict_packet_vector(
        source_atoms,
        builder,
        composition="add_source",
        source_identity_weight=source_identity_weight,
    )
    if not np.any(proposal_vector):
        return b"", {
            "packet_builder": "antishuffle_innovation",
            "packet_builder_composition": composition,
            "source_identity_weight": source_identity_weight,
            "antishuffle_train_donors": antishuffle_train_donors,
            "antishuffle_donor_weight": antishuffle_donor_weight,
            "antishuffle_null_weight": antishuffle_null_weight,
            "antishuffle_generic_weight": antishuffle_generic_weight,
            "antishuffle_carrier_mode": antishuffle_carrier_mode,
            "source_atoms": source_atoms,
            "contrast_source": contrast_source_id,
            "packet_atoms": {},
            "packet_atom_count": 0,
            "packet_vector_l2": 0.0,
            "source_vector_l2": float(np.linalg.norm(source_vector)),
            "antishuffle_target_index": None,
        }
    generic_source = np.asarray(builder.get("train_mean_source_vector", np.zeros_like(proposal_vector)), dtype=np.float64)
    generic_prediction = np.asarray(
        builder.get("train_mean_prediction_vector", np.zeros_like(proposal_vector)),
        dtype=np.float64,
    )
    generic_vector = generic_source + generic_prediction
    if use_train_mean_contrast:
        contrast_source_vector = generic_source
        contrast_vector = np.maximum(generic_prediction + source_identity_weight * generic_source, 0.0)
        donor_strength_vector = np.maximum(contrast_vector, contrast_source_vector)
        contrast_source_atoms = _full_atoms_from_vector(generic_source, min_score=0.0)
        contrast_source_id = "train_mean_source"
    elif use_train_donor_contrast:
        contrast_source_vector, contrast_vector, donor_strength_vector, donor_ids = _train_donor_contrast_vectors(
            example=example,
            builder=builder,
            donor_count=antishuffle_train_donors,
            source_identity_weight=source_identity_weight,
        )
        contrast_source_atoms = _full_atoms_from_vector(contrast_source_vector, min_score=0.0)
        contrast_source_id = "train_donor_mean:" + ",".join(donor_ids[:8])
    elif contrast_source_atoms is None:
        contrast = _nonoverlap_contrast_example(example, eval_examples, index)
        contrast_source_atoms = syn._source_private_atoms(contrast.private_test_log, mode="matched")
        contrast_source_id = contrast.example_id
        contrast_vector = _predict_packet_vector(
            contrast_source_atoms,
            builder,
            composition="add_source",
            source_identity_weight=source_identity_weight,
        )
        contrast_source_vector = _source_vector_from_atoms(contrast_source_atoms)
        donor_strength_vector = np.maximum(contrast_vector, contrast_source_vector)
    else:
        contrast_vector = _predict_packet_vector(
            contrast_source_atoms,
            builder,
            composition="add_source",
            source_identity_weight=source_identity_weight,
        )
        contrast_source_vector = _source_vector_from_atoms(contrast_source_atoms)
        donor_strength_vector = np.maximum(contrast_vector, contrast_source_vector)
    proposal_atoms = _full_atoms_from_vector(proposal_vector, min_score=packet_min_score)
    proposal_scores, _ = _score_packet_atoms(
        example=example,
        payload_atoms=proposal_atoms,
        dictionary=dictionary,
        null_dictionary=null_dictionary,
        candidate_atom_view=candidate_atom_view,
        decoder_score_mode=decoder_score_mode,
        permuted_null_weight=permuted_null_weight,
    )
    target_index = _top_index_with_prior_tie(example, proposal_scores)
    candidates: list[dict[str, Any]] = []
    for atom_id in range(len(syn.ATOM_ORDER)):
        proposal_value = float(max(proposal_vector[atom_id], 0.0))
        source_value = float(max(source_vector[atom_id], 0.0))
        if proposal_value <= packet_min_score and source_value <= packet_min_score:
            continue
        contrast_value = float(max(contrast_vector[atom_id], 0.0))
        contrast_source_value = float(max(contrast_source_vector[atom_id], 0.0))
        donor_strength_value = float(max(donor_strength_vector[atom_id], 0.0))
        atom_strength = max(proposal_value, source_value)
        atom = syn.ID_TO_ATOM[atom_id]
        atom_scores, score_meta = _score_packet_atoms(
            example=example,
            payload_atoms={atom: atom_strength},
            dictionary=dictionary,
            null_dictionary=null_dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode=decoder_score_mode,
            permuted_null_weight=permuted_null_weight,
        )
        active_scores = [float(value) for value in score_meta.get("active_scores", [])]
        null_scores = [float(value) for value in score_meta.get("null_scores", [])]
        receiver_margin = _score_margin(atom_scores, target_index)
        null_margin = max(0.0, _score_margin(null_scores, target_index)) if null_scores else 0.0
        proposal_specific = max(proposal_value - antishuffle_donor_weight * contrast_value, 0.0)
        source_specific = max(source_value - antishuffle_donor_weight * contrast_source_value, 0.0)
        source_carrier = source_identity_weight * source_specific
        if antishuffle_carrier_mode == "sum":
            carrier = proposal_specific + source_carrier
        elif antishuffle_carrier_mode == "min":
            carrier = min(proposal_specific, source_carrier)
        elif antishuffle_carrier_mode == "geomean":
            carrier = float(np.sqrt(max(proposal_specific, 0.0) * max(source_carrier, 0.0)))
        else:
            raise ValueError(f"unknown anti-shuffle carrier mode {antishuffle_carrier_mode!r}")
        generic_penalty = float(max(generic_vector[atom_id], 0.0))
        donor_gain_penalty = (
            antishuffle_donor_weight * donor_strength_value * max(receiver_margin, 0.0)
            if use_train_donor_contrast
            else 0.0
        )
        selection_value = (
            carrier * max(receiver_margin, 0.0)
            - donor_gain_penalty
            - antishuffle_null_weight * null_margin
            - antishuffle_generic_weight * generic_penalty
        )
        if selection_value <= 0.0:
            continue
        candidates.append(
            {
                "atom": atom,
                "atom_id": atom_id,
                "selection_value": float(selection_value),
                "packet_score": float(atom_strength),
                "receiver_margin": float(receiver_margin),
                "null_margin": float(null_margin),
                "proposal_value": proposal_value,
                "contrast_value": contrast_value,
                "source_value": source_value,
                "contrast_source_value": contrast_source_value,
                "donor_strength_value": donor_strength_value,
                "donor_gain_penalty": float(donor_gain_penalty),
            }
        )
    candidates.sort(key=lambda row: (-float(row["selection_value"]), int(row["atom_id"])))
    selected = candidates[: max(1, budget_bytes // 2)]
    if not selected:
        return b"", {
            "packet_builder": "antishuffle_innovation",
            "packet_builder_composition": composition,
            "source_identity_weight": source_identity_weight,
            "antishuffle_train_donors": antishuffle_train_donors,
            "antishuffle_donor_weight": antishuffle_donor_weight,
            "antishuffle_null_weight": antishuffle_null_weight,
            "antishuffle_generic_weight": antishuffle_generic_weight,
            "antishuffle_carrier_mode": antishuffle_carrier_mode,
            "source_atoms": source_atoms,
            "contrast_source": contrast_source_id,
            "packet_atoms": {},
            "packet_atom_count": 0,
            "packet_vector_l2": 0.0,
            "source_vector_l2": float(np.linalg.norm(source_vector)),
            "antishuffle_target_index": target_index,
            "antishuffle_target_is_answer": target_index == syn._answer_index(example),
            "antishuffle_candidates_considered": len(candidates),
        }
    max_score = max(float(row["packet_score"]) for row in selected)
    packet_atoms = {
        str(row["atom"]): float(row["packet_score"]) / max(max_score, 1e-12)
        for row in selected
        if float(row["packet_score"]) > 0.0
    }
    packet_vector = _source_vector_from_atoms(packet_atoms)
    return syn._encode_atoms(packet_atoms, budget_bytes=budget_bytes), {
        "packet_builder": "antishuffle_innovation",
        "packet_builder_composition": composition,
        "source_identity_weight": source_identity_weight,
        "antishuffle_train_donors": antishuffle_train_donors,
        "antishuffle_donor_weight": antishuffle_donor_weight,
        "antishuffle_null_weight": antishuffle_null_weight,
        "antishuffle_generic_weight": antishuffle_generic_weight,
        "antishuffle_carrier_mode": antishuffle_carrier_mode,
        "source_atoms": source_atoms,
        "contrast_source": contrast_source_id,
        "contrast_source_atoms": contrast_source_atoms,
        "packet_atoms": packet_atoms,
        "packet_atom_count": len(packet_atoms),
        "packet_vector_l2": float(np.linalg.norm(packet_vector)),
        "source_vector_l2": float(np.linalg.norm(source_vector)),
        "proposal_vector_l2": float(np.linalg.norm(proposal_vector)),
        "contrast_vector_l2": float(np.linalg.norm(contrast_vector)),
        "antishuffle_target_index": target_index,
        "antishuffle_target_is_answer": target_index == syn._answer_index(example),
        "antishuffle_target_is_prior": target_index == _prior_index(example),
        "antishuffle_proposal_scores": [float(score) for score in proposal_scores],
        "antishuffle_candidates_considered": len(candidates),
        "antishuffle_selected_atoms": selected,
    }


def _build_packet(
    *,
    source_atoms: dict[str, float],
    builder: dict[str, Any],
    budget_bytes: int,
    packet_min_score: float,
    composition: str = "mapped",
    source_identity_weight: float = 0.0,
    example: syn.Example | None = None,
    eval_examples: list[syn.Example] | None = None,
    index: int | None = None,
    dictionary: syn.LearnedSynonymDictionary | None = None,
    null_dictionary: syn.LearnedSynonymDictionary | None = None,
    candidate_atom_view: str | None = None,
    decoder_score_mode: str | None = None,
    permuted_null_weight: float = 0.75,
    antishuffle_train_donors: int = 12,
    antishuffle_donor_weight: float = 1.0,
    antishuffle_null_weight: float = 0.50,
    antishuffle_generic_weight: float = 0.25,
    antishuffle_carrier_mode: str = "sum",
    contrast_source_atoms: dict[str, float] | None = None,
    contrast_source_id: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if composition in {"antishuffle_innovation", "train_mean_antishuffle_innovation", "train_donor_antishuffle_innovation"}:
        if (
            example is None
            or eval_examples is None
            or index is None
            or dictionary is None
            or null_dictionary is None
            or candidate_atom_view is None
            or decoder_score_mode is None
        ):
            raise ValueError("antishuffle_innovation requires example, eval_examples, dictionary, and decoder state")
        return _build_antishuffle_packet(
            source_atoms=source_atoms,
            builder=builder,
            composition=composition,
            budget_bytes=budget_bytes,
            packet_min_score=packet_min_score,
            source_identity_weight=source_identity_weight,
            antishuffle_train_donors=antishuffle_train_donors,
            antishuffle_donor_weight=antishuffle_donor_weight,
            antishuffle_null_weight=antishuffle_null_weight,
            antishuffle_generic_weight=antishuffle_generic_weight,
            antishuffle_carrier_mode=antishuffle_carrier_mode,
            example=example,
            eval_examples=eval_examples,
            index=index,
            dictionary=dictionary,
            null_dictionary=null_dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode=decoder_score_mode,
            permuted_null_weight=permuted_null_weight,
            contrast_source_atoms=contrast_source_atoms,
            contrast_source_id=contrast_source_id,
            use_train_mean_contrast=composition == "train_mean_antishuffle_innovation",
            use_train_donor_contrast=composition == "train_donor_antishuffle_innovation",
        )
    source_vector = _source_vector_from_atoms(source_atoms)
    vector = _predict_packet_vector(
        source_atoms,
        builder,
        composition=_base_composition(composition),
        source_identity_weight=source_identity_weight,
    )
    projection_meta: dict[str, Any] = {}
    if composition.startswith("project_"):
        if example is None or dictionary is None or candidate_atom_view is None:
            raise ValueError("project_* packet-builder compositions require example, dictionary, and candidate_atom_view")
        vector, projection_meta = _project_vector_to_candidate(
            vector,
            example=example,
            dictionary=dictionary,
            candidate_atom_view=candidate_atom_view,
        )
    packet_atoms = _atoms_from_packet_vector(vector, budget_bytes=budget_bytes, min_score=packet_min_score)
    return syn._encode_atoms(packet_atoms, budget_bytes=budget_bytes), {
        "packet_builder": "source_to_candidate_ridge",
        "packet_builder_composition": composition,
        "source_identity_weight": source_identity_weight,
        "source_atoms": source_atoms,
        "packet_atoms": packet_atoms,
        "source_vector_l2": float(np.linalg.norm(source_vector)),
        "packet_vector_l2": float(np.linalg.norm(vector)),
        "packet_atom_count": len(packet_atoms),
        **projection_meta,
    }


def _oracle_packet(
    *,
    example: syn.Example,
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
    budget_bytes: int,
    packet_min_score: float,
) -> tuple[bytes, dict[str, Any]]:
    vector = _candidate_vector(
        example=example,
        candidate_index=syn._answer_index(example),
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
    )
    atoms = _atoms_from_packet_vector(np.maximum(vector, 0.0), budget_bytes=budget_bytes, min_score=packet_min_score)
    return syn._encode_atoms(atoms, budget_bytes=budget_bytes), {
        "packet_builder": "oracle_answer_candidate",
        "source": "oracle_answer_candidate",
        "packet_atoms": atoms,
        "packet_vector_l2": float(np.linalg.norm(vector)),
        "packet_atom_count": len(atoms),
    }


def _payload_for_condition(
    *,
    condition: str,
    example: syn.Example,
    eval_examples: list[syn.Example],
    index: int,
    budget_bytes: int,
    dictionary: syn.LearnedSynonymDictionary,
    null_dictionary: syn.LearnedSynonymDictionary,
    builder: dict[str, Any],
    candidate_atom_view: str,
    packet_min_score: float,
    packet_builder_composition: str,
    source_identity_weight: float,
    antishuffle_train_donors: int,
    antishuffle_donor_weight: float,
    antishuffle_null_weight: float,
    antishuffle_generic_weight: float,
    antishuffle_carrier_mode: str,
    decoder_score_mode: str,
    permuted_null_weight: float,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any], dict[str, Any]]:
    decode_kwargs: dict[str, Any] = {}

    def build(
        *,
        source_atoms: dict[str, float],
        contrast_source_atoms: dict[str, float] | None = None,
        contrast_source_id: str | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        return _build_packet(
            source_atoms=source_atoms,
            builder=builder,
            budget_bytes=budget_bytes,
            packet_min_score=packet_min_score,
            composition=packet_builder_composition,
            source_identity_weight=source_identity_weight,
            example=example,
            eval_examples=eval_examples,
            index=index,
            dictionary=dictionary,
            null_dictionary=null_dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode=decoder_score_mode,
            permuted_null_weight=permuted_null_weight,
            antishuffle_train_donors=antishuffle_train_donors,
            antishuffle_donor_weight=antishuffle_donor_weight,
            antishuffle_null_weight=antishuffle_null_weight,
            antishuffle_generic_weight=antishuffle_generic_weight,
            antishuffle_carrier_mode=antishuffle_carrier_mode,
            contrast_source_atoms=contrast_source_atoms,
            contrast_source_id=contrast_source_id,
        )

    if condition == "target_only":
        return None, {"source": "target_prior"}, decode_kwargs
    if condition == BASE_MATCHED_CONDITION:
        payload = syn._encode_atoms(
            syn._source_private_atoms(example.private_test_log, mode="matched"),
            budget_bytes=budget_bytes,
        )
        return payload, {"source": example.example_id, "packet_builder": "live_source_atoms"}, decode_kwargs
    if condition == MATCHED_CONDITION:
        payload, meta = build(
            source_atoms=syn._source_private_atoms(example.private_test_log, mode="matched"),
        )
        return payload, {"source": example.example_id, **meta}, decode_kwargs
    if condition == "zero_source":
        payload, meta = build(
            source_atoms={},
        )
        return payload, {"source": "zero", **meta}, decode_kwargs
    if condition == "shuffled_source":
        other = syn._constrained_nonoverlap_example(example, eval_examples, index)
        contrast = _nonoverlap_contrast_example(
            example,
            eval_examples,
            index,
            exclude_ids={other.example_id},
        )
        payload, meta = build(
            source_atoms=syn._source_private_atoms(other.private_test_log, mode="matched"),
            contrast_source_atoms=syn._source_private_atoms(contrast.private_test_log, mode="matched"),
            contrast_source_id=contrast.example_id,
        )
        return payload, {"source": other.example_id, **meta}, decode_kwargs
    if condition == "answer_masked_source":
        payload, meta = build(
            source_atoms={},
        )
        return payload, {"source": "answer_masked_strict", **meta}, decode_kwargs
    if condition == "public_only_sidecar":
        payload, meta = build(
            source_atoms={},
        )
        return payload, {"source": "public_only", **meta}, decode_kwargs
    if condition == "target_derived_sidecar":
        payload, meta = build(
            source_atoms={},
        )
        return payload, {"source": "target_prompt_only", **meta}, decode_kwargs
    if condition == "random_same_byte":
        return syn._random_packet(budget_bytes=budget_bytes, rng=rng), {"source": "random"}, decode_kwargs
    if condition == "answer_only_text":
        return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}, decode_kwargs
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_hidden_log"}, decode_kwargs
    if condition == "atom_id_derangement":
        payload, meta = build(
            source_atoms=syn._source_private_atoms(example.private_test_log, mode="matched"),
        )
        return payload, {"source": example.example_id, **meta}, {"derange": True}
    if condition == "private_random_source_atoms":
        randomized_atoms = syn._private_random_source_atoms(
            syn._source_private_atoms(example.private_test_log, mode="matched"),
            rng=rng,
        )
        payload, meta = build(
            source_atoms=randomized_atoms,
        )
        return payload, {"source": example.example_id, "control": "private_random_source_atoms", **meta}, decode_kwargs
    if condition == "permuted_teacher_receiver":
        payload, meta = build(
            source_atoms=syn._source_private_atoms(example.private_test_log, mode="matched"),
        )
        return payload, {"source": example.example_id, "control": "permuted_teacher_receiver", **meta}, decode_kwargs
    if condition == "top_atom_knockout":
        payload, meta = build(
            source_atoms=syn._source_private_atoms(example.private_test_log, mode="matched"),
        )
        return payload, {"source": example.example_id, **meta}, {"knockout": "top"}
    if condition == "private_random_knockout":
        payload, meta = build(
            source_atoms=syn._source_private_atoms(example.private_test_log, mode="matched"),
        )
        return payload, {"source": example.example_id, **meta}, {"knockout": "random"}
    if condition == ORACLE_CONDITION:
        payload, meta = _oracle_packet(
            example=example,
            dictionary=dictionary,
            candidate_atom_view=candidate_atom_view,
            budget_bytes=budget_bytes,
            packet_min_score=packet_min_score,
        )
        return payload, meta, decode_kwargs
    raise ValueError(f"unknown condition {condition!r}")


def _prior_index(example: syn.Example) -> int:
    prior = syn._prior_prediction(example)
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)


def _predict_condition(
    *,
    condition: str,
    example: syn.Example,
    eval_examples: list[syn.Example],
    index: int,
    budget_bytes: int,
    dictionary: syn.LearnedSynonymDictionary,
    permuted_teacher_dictionary: syn.LearnedSynonymDictionary,
    builder: dict[str, Any],
    candidate_atom_view: str,
    packet_min_score: float,
    packet_builder_composition: str,
    source_identity_weight: float,
    antishuffle_train_donors: int,
    antishuffle_donor_weight: float,
    antishuffle_null_weight: float,
    antishuffle_generic_weight: float,
    antishuffle_carrier_mode: str,
    decoder_score_mode: str,
    min_decision_score: float,
    permuted_null_weight: float = 0.75,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, meta, decode_kwargs = _payload_for_condition(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        budget_bytes=budget_bytes,
        dictionary=dictionary,
        null_dictionary=permuted_teacher_dictionary,
        builder=builder,
        candidate_atom_view=candidate_atom_view,
        packet_min_score=packet_min_score,
        packet_builder_composition=packet_builder_composition,
        source_identity_weight=source_identity_weight,
        antishuffle_train_donors=antishuffle_train_donors,
        antishuffle_donor_weight=antishuffle_donor_weight,
        antishuffle_null_weight=antishuffle_null_weight,
        antishuffle_generic_weight=antishuffle_generic_weight,
        antishuffle_carrier_mode=antishuffle_carrier_mode,
        decoder_score_mode=decoder_score_mode,
        permuted_null_weight=permuted_null_weight,
        rng=rng,
    )
    active_dictionary = permuted_teacher_dictionary if condition == "permuted_teacher_receiver" else dictionary
    null_dictionary = dictionary if condition == "permuted_teacher_receiver" else permuted_teacher_dictionary
    decision_threshold = 0.0 if condition == ORACLE_CONDITION else min_decision_score
    prediction, decode_meta = syn._predict_from_payload(
        example=example,
        payload=payload,
        budget_bytes=budget_bytes,
        dictionary=active_dictionary,
        null_dictionary=null_dictionary,
        candidate_atom_view=candidate_atom_view,
        decoder_score_mode=decoder_score_mode,
        min_decision_score=decision_threshold,
        permuted_null_weight=permuted_null_weight,
        rng=rng,
        **decode_kwargs,
    )
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "strict_correct": prediction == example.answer_label,
        "payload_bytes": len(payload or b""),
        "payload_tokens": syn._token_count(payload_hex),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "payload_hex": payload_hex,
        "answer_index": syn._answer_index(example),
        "prior_index": _prior_index(example),
        "metadata": {**meta, **decode_meta},
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": 0.0,
            "strict_accuracy": 0.0,
            "mean_payload_bytes": 0.0,
            "mean_payload_tokens": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "n": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
        "strict_accuracy": sum(1 for row in rows if row["strict_correct"]) / len(rows),
        "mean_payload_bytes": statistics.fmean(float(row["payload_bytes"]) for row in rows),
        "mean_payload_tokens": statistics.fmean(float(row["payload_tokens"]) for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    baseline: str,
    seed: int,
    samples: int,
) -> dict[str, float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = [
        float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"])
        for _, conditions in sorted(by_example.items())
        if condition in conditions and baseline in conditions
    ]
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(deltas),
        "ci95_low": syn._percentile(means, 0.025),
        "ci95_high": syn._percentile(means, 0.975),
    }


def _direction_summary(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    budget_bytes: int,
    seed: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_improvement_over_base: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in EVAL_CONDITIONS
    }
    metrics = {condition: _summarize(condition_rows) for condition, condition_rows in by_condition.items()}
    target = metrics["target_only"]["accuracy"]
    base = metrics[BASE_MATCHED_CONDITION]["accuracy"]
    matched = metrics[MATCHED_CONDITION]["accuracy"]
    best_control_name = max(STRICT_CONTROLS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    oracle = metrics[ORACLE_CONDITION]["accuracy"]
    controls_ok = all(metrics[condition]["accuracy"] <= target + 0.03 for condition in STRICT_CONTROLS)
    target_ci = _paired_bootstrap(
        rows,
        condition=MATCHED_CONDITION,
        baseline="target_only",
        seed=seed,
        samples=bootstrap_samples,
    )
    base_ci = _paired_bootstrap(
        rows,
        condition=MATCHED_CONDITION,
        baseline=BASE_MATCHED_CONDITION,
        seed=seed + 1,
        samples=bootstrap_samples,
    )
    control_ci = _paired_bootstrap(
        rows,
        condition=MATCHED_CONDITION,
        baseline=best_control_name,
        seed=seed + 2,
        samples=bootstrap_samples,
    )
    exact_ids = [row["example_id"] for row in by_condition["target_only"]]
    pass_gate = (
        matched >= target + min_lift_over_target
        and matched >= best_control + min_gap_over_control
        and matched >= base + min_improvement_over_base
        and controls_ok
        and target_ci["ci95_low"] > 0.05
        and oracle >= 0.80
        and len(exact_ids) == len(set(exact_ids))
    )
    return {
        "direction": direction,
        "budget_bytes": budget_bytes,
        "n": metrics["target_only"]["n"],
        "target_accuracy": target,
        "base_matched_accuracy": base,
        "candidate_conditioned_packet_accuracy": matched,
        "best_control_accuracy": best_control,
        "best_control_name": best_control_name,
        "oracle_candidate_conditioned_packet_accuracy": oracle,
        "candidate_minus_target": matched - target,
        "candidate_minus_base": matched - base,
        "candidate_minus_best_control": matched - best_control,
        "controls_ok": controls_ok,
        "paired_bootstrap_vs_target": target_ci,
        "paired_bootstrap_vs_base": base_ci,
        "paired_bootstrap_vs_best_control": control_ci,
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "exact_id_sha256": hashlib.sha256("\n".join(exact_ids).encode("utf-8")).hexdigest(),
        "pass_gate": pass_gate,
        "metrics": metrics,
    }


def _calibration_for_builder(
    *,
    mode: str,
    train_examples: list[syn.Example],
    eval_examples: list[syn.Example],
    calibration_count: int,
    seed: int,
) -> list[syn.Example]:
    if mode == "dictionary_calibration":
        return syn._calibration_examples(
            mode="all_public_eval_disjoint",
            train_examples=train_examples,
            eval_examples=eval_examples,
            calibration_count=calibration_count,
            seed=seed,
        )
    return syn._calibration_examples(
        mode=mode,
        train_examples=train_examples,
        eval_examples=eval_examples,
        calibration_count=calibration_count,
        seed=seed,
    )


def _public_eval_disjoint_pool(
    *,
    eval_examples: list[syn.Example],
    calibration_count: int,
    seed: int,
) -> list[syn.Example]:
    excluded_ids = {syn._qualified_example_id(example) for example in eval_examples}
    pool_size = max(calibration_count * 3, calibration_count + max(len(eval_examples) * 4, 512))
    pool = syn.make_benchmark(examples=pool_size, candidates=4, seed=seed, family_set="all")
    return [example for example in pool if syn._qualified_example_id(example) not in excluded_ids]


def _fit_packet_builder_bundle(
    *,
    mode: str,
    train_examples: list[syn.Example],
    eval_examples: list[syn.Example],
    calibration_count: int,
    seed: int,
    dictionary: syn.LearnedSynonymDictionary,
    candidate_atom_view: str,
    ridge: float,
    target_mode: str,
) -> dict[str, Any]:
    if mode == "leave_one_family_out_public":
        pool = _public_eval_disjoint_pool(
            eval_examples=eval_examples,
            calibration_count=calibration_count,
            seed=seed,
        )
        builders: dict[str, dict[str, Any]] = {}
        family_counts: dict[str, int] = {}
        for family_name in sorted({example.family_name for example in eval_examples}):
            rows = [example for example in pool if example.family_name != family_name][:calibration_count]
            builders[family_name] = _fit_packet_builder(
                examples=rows,
                dictionary=dictionary,
                candidate_atom_view=candidate_atom_view,
                ridge=ridge,
                target_mode=target_mode,
            )
            family_counts[family_name] = len(rows)
        return {
            "mode": mode,
            "builders": builders,
            "family_counts": family_counts,
            "public_pool_examples": len(pool),
            "global_rows": [],
        }

    rows = _calibration_for_builder(
        mode=mode,
        train_examples=train_examples,
        eval_examples=eval_examples,
        calibration_count=calibration_count,
        seed=seed,
    )
    builder = _fit_packet_builder(
        examples=rows,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
        ridge=ridge,
        target_mode=target_mode,
    )
    return {
        "mode": mode,
        "builder": builder,
        "global_rows": rows,
        "public_pool_examples": None,
        "family_counts": {},
    }


def _builder_for_example(bundle: dict[str, Any], example: syn.Example) -> dict[str, Any]:
    if bundle["mode"] == "leave_one_family_out_public":
        return bundle["builders"][example.family_name]
    return bundle["builder"]


def _builder_state_for_json(bundle: dict[str, Any]) -> dict[str, Any]:
    hidden_state_keys = {
        "weights",
        "train_donor_source_vectors",
        "train_donor_mapped_vectors",
        "train_donor_example_ids",
        "train_donor_answer_indices",
    }

    def public_state(builder: dict[str, Any]) -> dict[str, Any]:
        state = {key: value for key, value in builder.items() if key not in hidden_state_keys}
        state["train_donor_pool_size"] = len(builder.get("train_donor_source_vectors", []))
        return state

    if bundle["mode"] != "leave_one_family_out_public":
        return public_state(bundle["builder"])
    family_builders = {
        family_name: public_state(builder)
        for family_name, builder in sorted(bundle["builders"].items())
    }
    fit_rows = [state["fit_rows"] for state in family_builders.values()]
    cosine = [state["train_mean_cosine_to_answer_candidate"] for state in family_builders.values()]
    return {
        "mode": bundle["mode"],
        "family_builder_count": len(family_builders),
        "family_counts": bundle["family_counts"],
        "public_pool_examples": bundle["public_pool_examples"],
        "fit_rows_min": min(fit_rows) if fit_rows else 0,
        "fit_rows_max": max(fit_rows) if fit_rows else 0,
        "train_mean_cosine_min": min(cosine) if cosine else 0.0,
        "train_mean_cosine_max": max(cosine) if cosine else 0.0,
        "family_builders": family_builders,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _qualified_id_set(rows: list[syn.Example]) -> set[str]:
    return {syn._qualified_example_id(row) for row in rows}


def _write_direction_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate-Conditioned Packet Builder Direction",
        "",
        f"- direction: `{payload['direction']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval families: `{payload['train_family_set']} -> {payload['eval_family_set']}`",
        f"- candidate atom view: `{payload['candidate_atom_view']}`",
        f"- dictionary calibration: `{payload['candidate_calibration']}`",
        f"- packet builder calibration: `{payload['packet_builder_calibration']}`",
        f"- packet builder target mode: `{payload['packet_builder_target_mode']}`",
        f"- packet builder composition: `{payload['packet_builder_composition']}`",
        f"- source identity weight: `{payload['source_identity_weight']}`",
        f"- anti-shuffle train donors: `{payload['antishuffle_train_donors']}`",
        f"- anti-shuffle donor/null/generic weights: `{payload['antishuffle_donor_weight']}` / `{payload['antishuffle_null_weight']}` / `{payload['antishuffle_generic_weight']}`",
        f"- anti-shuffle carrier mode: `{payload['antishuffle_carrier_mode']}`",
        f"- decoder score mode: `{payload['decoder_score_mode']}`",
        f"- min decision score: `{payload['min_decision_score']}`",
        f"- packet builder train cosine: `{payload['packet_builder_state'].get('train_mean_cosine_to_answer_candidate', payload['packet_builder_state'].get('train_mean_cosine_min', 0.0)):.3f}`",
        "",
        "| Budget | Candidate packet | Base packet | Target | Best control | Pass |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for summary in payload["budget_summaries"]:
        lines.append(
            "| "
            f"{summary['budget_bytes']} | "
            f"{summary['candidate_conditioned_packet_accuracy']:.3f} | "
            f"{summary['base_matched_accuracy']:.3f} | "
            f"{summary['target_accuracy']:.3f} | "
            f"{summary['best_control_name']}={summary['best_control_accuracy']:.3f} | "
            f"{summary['pass_gate']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_direction(
    *,
    output_dir: pathlib.Path,
    direction: str,
    train_family_set: str,
    eval_family_set: str,
    train_start_index: int,
    eval_start_index: int,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    budgets: list[int],
    candidate_atom_view: str,
    calibration_atom_view: str,
    candidate_calibration: str,
    packet_builder_calibration: str,
    calibration_examples: int,
    packet_builder_examples: int,
    feature_dim: int,
    ridge: float,
    packet_builder_ridge: float,
    top_k: int,
    min_score: float,
    packet_min_score: float,
    packet_builder_target_mode: str,
    packet_builder_composition: str,
    source_identity_weight: float,
    antishuffle_train_donors: int,
    antishuffle_donor_weight: float,
    antishuffle_null_weight: float,
    antishuffle_generic_weight: float,
    antishuffle_carrier_mode: str,
    text_feature_mode: str,
    adapter_target_mode: str,
    decoder_score_mode: str,
    min_decision_score: float,
    permuted_null_weight: float = 0.75,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_improvement_over_base: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = syn.make_benchmark(
        examples=train_examples,
        candidates=4,
        seed=train_seed,
        family_set=train_family_set,
        start_index=train_start_index,
    )
    eval_rows = syn.make_benchmark(
        examples=eval_examples,
        candidates=4,
        seed=eval_seed,
        family_set=eval_family_set,
        start_index=eval_start_index,
    )
    train_ids = _qualified_id_set(train_rows)
    eval_ids = _qualified_id_set(eval_rows)
    train_eval_overlap = sorted(train_ids & eval_ids)
    calibration_rows = syn._calibration_examples(
        mode=candidate_calibration,
        train_examples=train_rows,
        eval_examples=eval_rows,
        calibration_count=calibration_examples,
        seed=train_seed + 101,
    )
    dictionary = syn._fit_dictionary(
        examples=calibration_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        adapter_target_mode=adapter_target_mode,
        receiver_mode="atom_ridge",
        contrastive_negative_sources=0,
        contrastive_rank=4,
        seed=train_seed + 211,
    )
    permuted_teacher_dictionary = syn._fit_ridge_dictionary(
        examples=calibration_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        adapter_target_mode="permuted_semantic_anchor_teacher",
    )
    builder_bundle = _fit_packet_builder_bundle(
        mode=packet_builder_calibration,
        train_examples=train_rows,
        eval_examples=eval_rows,
        calibration_count=packet_builder_examples,
        seed=train_seed + 313,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
        ridge=packet_builder_ridge,
        target_mode=packet_builder_target_mode,
    )
    builder_rows = builder_bundle["global_rows"]
    builder_json = _builder_state_for_json(builder_bundle)
    budget_summaries: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}
    for budget in budgets:
        rng = random.Random(train_seed * 1000003 + eval_seed * 1009 + budget)
        rows: list[dict[str, Any]] = []
        for row_index, example in enumerate(eval_rows):
            for condition in EVAL_CONDITIONS:
                rows.append(
                    _predict_condition(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        budget_bytes=budget,
                        dictionary=dictionary,
                        permuted_teacher_dictionary=permuted_teacher_dictionary,
                        builder=_builder_for_example(builder_bundle, example),
                        candidate_atom_view=candidate_atom_view,
                        packet_min_score=packet_min_score,
                        packet_builder_composition=packet_builder_composition,
                        source_identity_weight=source_identity_weight,
                        antishuffle_train_donors=antishuffle_train_donors,
                        antishuffle_donor_weight=antishuffle_donor_weight,
                        antishuffle_null_weight=antishuffle_null_weight,
                        antishuffle_generic_weight=antishuffle_generic_weight,
                        antishuffle_carrier_mode=antishuffle_carrier_mode,
                        decoder_score_mode=decoder_score_mode,
                        min_decision_score=min_decision_score,
                        permuted_null_weight=permuted_null_weight,
                        rng=rng,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        predictions_name = f"predictions_budget{budget}.jsonl"
        _write_jsonl(output_dir / predictions_name, rows)
        prediction_files[str(budget)] = predictions_name
        budget_summaries.append(
            _direction_summary(
                rows,
                direction=direction,
                budget_bytes=budget,
                seed=train_seed + eval_seed + budget,
                min_lift_over_target=min_lift_over_target,
                min_gap_over_control=min_gap_over_control,
                min_improvement_over_base=min_improvement_over_base,
                bootstrap_samples=bootstrap_samples,
            )
        )
    payload = {
        "gate": "source_private_candidate_conditioned_packet_builder_direction",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "direction": direction,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_start_index": train_start_index,
        "eval_start_index": eval_start_index,
        "train_exact_id_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode("utf-8")).hexdigest(),
        "eval_exact_id_sha256": hashlib.sha256("\n".join(sorted(eval_ids)).encode("utf-8")).hexdigest(),
        "train_eval_exact_id_overlap_count": len(train_eval_overlap),
        "train_eval_exact_id_overlap_sha256": hashlib.sha256(
            "\n".join(train_eval_overlap).encode("utf-8")
        ).hexdigest(),
        "budgets": budgets,
        "candidate_atom_view": candidate_atom_view,
        "calibration_atom_view": calibration_atom_view,
        "candidate_calibration": candidate_calibration,
        "packet_builder_calibration": packet_builder_calibration,
        "calibration_examples": len(calibration_rows),
        "packet_builder_examples": len(builder_rows) if builder_rows else packet_builder_examples,
        "feature_dim": feature_dim,
        "text_feature_mode": text_feature_mode,
        "adapter_target_mode": adapter_target_mode,
        "decoder_score_mode": decoder_score_mode,
        "permuted_null_weight": permuted_null_weight
        if decoder_score_mode == "candidate_local_permuted_null_gap_residual_norm"
        else None,
        "ridge": ridge,
        "packet_builder_ridge": packet_builder_ridge,
        "top_k": top_k,
        "min_score": min_score,
        "packet_min_score": packet_min_score,
        "packet_builder_target_mode": packet_builder_target_mode,
        "packet_builder_composition": packet_builder_composition,
        "source_identity_weight": source_identity_weight,
        "antishuffle_train_donors": antishuffle_train_donors,
        "antishuffle_donor_weight": antishuffle_donor_weight,
        "antishuffle_null_weight": antishuffle_null_weight,
        "antishuffle_generic_weight": antishuffle_generic_weight,
        "antishuffle_carrier_mode": antishuffle_carrier_mode,
        "min_decision_score": min_decision_score,
        "packet_builder_state": builder_json,
        "conditions": list(EVAL_CONDITIONS),
        "source_destroying_controls": list(STRICT_CONTROLS),
        "budget_summaries": budget_summaries,
        "prediction_files": prediction_files,
        "pass_gate": any(summary["pass_gate"] for summary in budget_summaries),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_direction_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", *prediction_files.values(), "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name) for name in ["summary.json", "summary.md", *prediction_files.values()]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Candidate-Conditioned Packet Builder Direction Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_gate_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate-Conditioned Packet Builder Smoke",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- candidate atom view: `{payload['candidate_atom_view']}`",
        f"- dictionary calibration: `{payload['candidate_calibration']}`",
        f"- packet builder calibration: `{payload['packet_builder_calibration']}`",
        f"- packet builder target mode: `{payload['packet_builder_target_mode']}`",
        f"- packet builder composition: `{payload['packet_builder_composition']}`",
        f"- source identity weight: `{payload['source_identity_weight']}`",
        f"- anti-shuffle train donors: `{payload['antishuffle_train_donors']}`",
        f"- anti-shuffle donor/null/generic weights: `{payload['antishuffle_donor_weight']}` / `{payload['antishuffle_null_weight']}` / `{payload['antishuffle_generic_weight']}`",
        f"- anti-shuffle carrier mode: `{payload['antishuffle_carrier_mode']}`",
        f"- decoder score mode: `{payload['decoder_score_mode']}`",
        f"- min decision score: `{payload['min_decision_score']}`",
        "",
        "| Direction | Budget | Candidate packet | Base packet | Target | Best control | Candidate-base | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['direction']} | "
            f"{row['budget_bytes']} | "
            f"{row['candidate_conditioned_packet_accuracy']:.3f} | "
            f"{row['base_matched_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | "
            f"{row['candidate_minus_base']:.3f} | "
            f"{row['pass_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This gate changes sender-side packet construction while holding the candidate-local receiver fixed. "
            "The sender learns a ridge map from source-private atom evidence to the receiver's candidate-atom basis, "
            "then transmits the top atoms as the same low-byte packet format used by the live method.",
            "",
            "A pass means the learned packet beats both the target prior and the live hand-built source-atom packet "
            "on at least one budget in both cross-family directions while strict source-destroying controls remain "
            "near the target prior.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    budgets: list[int],
    directions: list[str] | None,
    family_mode: str,
    train_examples: int,
    eval_examples: int,
    seed: int,
    candidate_atom_view: str,
    calibration_atom_view: str,
    candidate_calibration: str,
    packet_builder_calibration: str,
    calibration_examples: int,
    packet_builder_examples: int,
    feature_dim: int,
    ridge: float,
    packet_builder_ridge: float,
    top_k: int,
    min_score: float,
    packet_min_score: float,
    packet_builder_target_mode: str,
    packet_builder_composition: str,
    source_identity_weight: float,
    antishuffle_train_donors: int,
    antishuffle_donor_weight: float,
    antishuffle_null_weight: float,
    antishuffle_generic_weight: float,
    antishuffle_carrier_mode: str,
    text_feature_mode: str,
    adapter_target_mode: str,
    decoder_score_mode: str,
    min_decision_score: float,
    permuted_null_weight: float = 0.75,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_improvement_over_base: float,
    feature_model: str,
    feature_device: str,
    feature_dtype: str,
    feature_max_length: int,
    local_files_only: bool,
) -> dict[str, Any]:
    syn._HF_FEATURE_MODEL = feature_model
    syn._HF_FEATURE_DEVICE = feature_device
    syn._HF_FEATURE_DTYPE = feature_dtype
    syn._HF_FEATURE_MAX_LENGTH = feature_max_length
    syn._HF_FEATURE_LOCAL_FILES_ONLY = local_files_only
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("core_to_holdout", "core", "holdout", seed, seed + 1, 0, 0),
        ("holdout_to_core", "holdout", "core", seed + 1, seed, 0, 0),
        ("same_family_all", "all", "all", seed, seed + 2, 0, 0),
    ]
    if family_mode == "train_family_disjoint_validation":
        specs = [
            ("core_to_holdout", "core", "core", seed, seed + 7001, 0, train_examples),
            ("holdout_to_core", "holdout", "holdout", seed + 1, seed + 7002, 0, train_examples),
            ("same_family_all", "all", "all", seed, seed + 7003, 0, train_examples),
        ]
    elif family_mode != "cross_family":
        raise ValueError(f"unknown family_mode {family_mode!r}")
    if directions is not None:
        requested = set(directions)
        known = {direction for direction, *_ in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown directions {unknown}; expected a subset of {sorted(known)}")
        specs = [spec for spec in specs if spec[0] in requested]
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for direction, train_family, eval_family, train_seed, eval_seed, train_start_index, eval_start_index in specs:
        run_dir = output_dir / direction
        result = _run_direction(
            output_dir=run_dir,
            direction=direction,
            train_family_set=train_family,
            eval_family_set=eval_family,
            train_start_index=train_start_index,
            eval_start_index=eval_start_index,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_seed=train_seed,
            eval_seed=eval_seed,
            budgets=budgets,
            candidate_atom_view=candidate_atom_view,
            calibration_atom_view=calibration_atom_view,
            candidate_calibration=candidate_calibration,
            packet_builder_calibration=packet_builder_calibration,
            calibration_examples=calibration_examples,
            packet_builder_examples=packet_builder_examples,
            feature_dim=feature_dim,
            ridge=ridge,
            packet_builder_ridge=packet_builder_ridge,
            top_k=top_k,
            min_score=min_score,
            packet_min_score=packet_min_score,
            packet_builder_target_mode=packet_builder_target_mode,
            packet_builder_composition=packet_builder_composition,
            source_identity_weight=source_identity_weight,
            antishuffle_train_donors=antishuffle_train_donors,
            antishuffle_donor_weight=antishuffle_donor_weight,
            antishuffle_null_weight=antishuffle_null_weight,
            antishuffle_generic_weight=antishuffle_generic_weight,
            antishuffle_carrier_mode=antishuffle_carrier_mode,
            text_feature_mode=text_feature_mode,
            adapter_target_mode=adapter_target_mode,
            decoder_score_mode=decoder_score_mode,
            min_decision_score=min_decision_score,
            permuted_null_weight=permuted_null_weight,
            bootstrap_samples=bootstrap_samples,
            min_lift_over_target=min_lift_over_target,
            min_gap_over_control=min_gap_over_control,
            min_improvement_over_base=min_improvement_over_base,
        )
        try:
            run_dirs.append(str(run_dir.relative_to(ROOT)))
        except ValueError:
            run_dirs.append(str(run_dir))
        for summary in result["budget_summaries"]:
            rows.append(
                {
                    "direction": direction,
                    "budget_bytes": summary["budget_bytes"],
                    "n": summary["n"],
                    "target_accuracy": summary["target_accuracy"],
                    "base_matched_accuracy": summary["base_matched_accuracy"],
                    "candidate_conditioned_packet_accuracy": summary["candidate_conditioned_packet_accuracy"],
                    "best_control_accuracy": summary["best_control_accuracy"],
                    "best_control_name": summary["best_control_name"],
                    "oracle_candidate_conditioned_packet_accuracy": summary[
                        "oracle_candidate_conditioned_packet_accuracy"
                    ],
                    "candidate_minus_target": summary["candidate_minus_target"],
                    "candidate_minus_base": summary["candidate_minus_base"],
                    "candidate_minus_best_control": summary["candidate_minus_best_control"],
                    "paired_ci95_low_vs_target": summary["paired_bootstrap_vs_target"]["ci95_low"],
                    "paired_ci95_high_vs_target": summary["paired_bootstrap_vs_target"]["ci95_high"],
                    "paired_ci95_low_vs_base": summary["paired_bootstrap_vs_base"]["ci95_low"],
                    "paired_ci95_high_vs_base": summary["paired_bootstrap_vs_base"]["ci95_high"],
                    "controls_ok": summary["controls_ok"],
                    "pass_gate": summary["pass_gate"],
                }
            )
    direction_pass = {
        direction: any(row["pass_gate"] for row in rows if row["direction"] == direction)
        for direction, *_ in specs
    }
    cross_family_pass = (
        direction_pass.get("core_to_holdout", False)
        and direction_pass.get("holdout_to_core", False)
    )
    payload = {
        "gate": "source_private_candidate_conditioned_packet_builder_smoke",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "run_dirs": run_dirs,
        "budgets": budgets,
        "family_mode": family_mode,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "seed": seed,
        "candidate_atom_view": candidate_atom_view,
        "calibration_atom_view": calibration_atom_view,
        "candidate_calibration": candidate_calibration,
        "packet_builder_calibration": packet_builder_calibration,
        "calibration_examples": calibration_examples,
        "packet_builder_examples": packet_builder_examples,
        "feature_dim": feature_dim,
        "text_feature_mode": text_feature_mode,
        "adapter_target_mode": adapter_target_mode,
        "decoder_score_mode": decoder_score_mode,
        "permuted_null_weight": permuted_null_weight
        if decoder_score_mode == "candidate_local_permuted_null_gap_residual_norm"
        else None,
        "ridge": ridge,
        "packet_builder_ridge": packet_builder_ridge,
        "top_k": top_k,
        "min_score": min_score,
        "packet_min_score": packet_min_score,
        "packet_builder_target_mode": packet_builder_target_mode,
        "packet_builder_composition": packet_builder_composition,
        "source_identity_weight": source_identity_weight,
        "antishuffle_train_donors": antishuffle_train_donors,
        "antishuffle_donor_weight": antishuffle_donor_weight,
        "antishuffle_null_weight": antishuffle_null_weight,
        "antishuffle_generic_weight": antishuffle_generic_weight,
        "antishuffle_carrier_mode": antishuffle_carrier_mode,
        "min_decision_score": min_decision_score,
        "feature_model": feature_model if "hf_" in text_feature_mode else None,
        "feature_device": syn._resolve_torch_device(feature_device) if "hf_" in text_feature_mode else None,
        "feature_dtype": feature_dtype if "hf_" in text_feature_mode else None,
        "feature_max_length": feature_max_length if "hf_" in text_feature_mode else None,
        "rows": rows,
        "headline": {
            "direction_pass": direction_pass,
            "cross_family_pass": cross_family_pass,
            "pass_rows": sum(1 for row in rows if row["pass_gate"]),
            "max_candidate_conditioned_packet_accuracy": max(
                row["candidate_conditioned_packet_accuracy"] for row in rows
            ),
            "max_candidate_minus_base": max(row["candidate_minus_base"] for row in rows),
            "min_passing_ci95_low_vs_base": min(
                [row["paired_ci95_low_vs_base"] for row in rows if row["pass_gate"]] or [0.0]
            ),
        },
        "pass_gate": cross_family_pass,
        "pass_rule": (
            "Bidirectional pass requires at least one budget in core_to_holdout and holdout_to_core with the learned "
            "packet beating target by the configured lift, beating the best strict source-destroying control by the "
            "configured gap, beating the live source-atom packet by the configured base improvement, all strict "
            "controls within target+0.03, paired CI95 lower bound vs target >0.05, and oracle candidate packet >=0.80."
        ),
    }
    (output_dir / "candidate_conditioned_packet_builder_smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_gate_markdown(output_dir / "candidate_conditioned_packet_builder_smoke.md", payload)
    manifest = {
        "artifacts": [
            "candidate_conditioned_packet_builder_smoke.json",
            "candidate_conditioned_packet_builder_smoke.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "candidate_conditioned_packet_builder_smoke.json": _sha256_file(
                output_dir / "candidate_conditioned_packet_builder_smoke.json"
            ),
            "candidate_conditioned_packet_builder_smoke.md": _sha256_file(
                output_dir / "candidate_conditioned_packet_builder_smoke.md"
            ),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Candidate-Conditioned Packet Builder Smoke Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- cross-family pass: `{cross_family_pass}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8])
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("core_to_holdout", "holdout_to_core", "same_family_all"),
        default=None,
        help="Optional subset of directions to run; default runs all directions.",
    )
    parser.add_argument(
        "--family-mode",
        choices=("cross_family", "train_family_disjoint_validation"),
        default="cross_family",
        help=(
            "cross_family runs the standard final train/eval family split. "
            "train_family_disjoint_validation keeps direction names but validates on disjoint IDs from the train family."
        ),
    )
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--candidate-atom-view", default="heldout_synonym")
    parser.add_argument("--calibration-atom-view", default="synonym_stress")
    parser.add_argument("--candidate-calibration", default="all_public_eval_disjoint")
    parser.add_argument("--packet-builder-calibration", default="all_public_eval_disjoint")
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--packet-builder-examples", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--packet-builder-ridge", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--packet-min-score", type=float, default=0.0)
    parser.add_argument(
        "--packet-builder-target-mode",
        choices=("answer_candidate", "answer_minus_candidate_mean", "answer_minus_prior"),
        default="answer_candidate",
    )
    parser.add_argument(
        "--packet-builder-composition",
        choices=(
            "mapped",
            "add_source",
            "max_source",
            "centered_mapped",
            "centered_add_source",
            "centered_max_source",
            "antishuffle_innovation",
            "train_mean_antishuffle_innovation",
            "train_donor_antishuffle_innovation",
            "project_mapped",
            "project_add_source",
            "project_max_source",
            "project_centered_mapped",
            "project_centered_add_source",
            "project_centered_max_source",
        ),
        default="mapped",
    )
    parser.add_argument("--source-identity-weight", type=float, default=0.0)
    parser.add_argument("--antishuffle-train-donors", type=int, default=12)
    parser.add_argument("--antishuffle-donor-weight", type=float, default=1.0)
    parser.add_argument("--antishuffle-null-weight", type=float, default=0.50)
    parser.add_argument("--antishuffle-generic-weight", type=float, default=0.25)
    parser.add_argument("--antishuffle-carrier-mode", choices=("sum", "min", "geomean"), default="sum")
    parser.add_argument("--text-feature-mode", default="hf_last_mean")
    parser.add_argument("--adapter-target-mode", default="semantic_anchor_teacher")
    parser.add_argument("--decoder-score-mode", default="candidate_local_residual_norm")
    parser.add_argument("--min-decision-score", type=float, default=0.48)
    parser.add_argument(
        "--permuted-null-weight",
        type=float,
        default=0.75,
        help="Penalty on the deterministic permuted-teacher null score for candidate_local_permuted_null_gap_residual_norm.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-lift-over-target", type=float, default=0.15)
    parser.add_argument("--min-gap-over-control", type=float, default=0.10)
    parser.add_argument("--min-improvement-over-base", type=float, default=0.03)
    parser.add_argument("--feature-model", default="BAAI/bge-small-en")
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--feature-dtype", default="float32")
    parser.add_argument("--feature-max-length", type=int, default=128)
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_gate(
        output_dir=args.output_dir,
        budgets=args.budgets,
        directions=args.directions,
        family_mode=args.family_mode,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        seed=args.seed,
        candidate_atom_view=args.candidate_atom_view,
        calibration_atom_view=args.calibration_atom_view,
        candidate_calibration=args.candidate_calibration,
        packet_builder_calibration=args.packet_builder_calibration,
        calibration_examples=args.calibration_examples,
        packet_builder_examples=args.packet_builder_examples,
        feature_dim=args.feature_dim,
        ridge=args.ridge,
        packet_builder_ridge=args.packet_builder_ridge,
        top_k=args.top_k,
        min_score=args.min_score,
        packet_min_score=args.packet_min_score,
        packet_builder_target_mode=args.packet_builder_target_mode,
        packet_builder_composition=args.packet_builder_composition,
        source_identity_weight=args.source_identity_weight,
        antishuffle_train_donors=args.antishuffle_train_donors,
        antishuffle_donor_weight=args.antishuffle_donor_weight,
        antishuffle_null_weight=args.antishuffle_null_weight,
        antishuffle_generic_weight=args.antishuffle_generic_weight,
        antishuffle_carrier_mode=args.antishuffle_carrier_mode,
        text_feature_mode=args.text_feature_mode,
        adapter_target_mode=args.adapter_target_mode,
        decoder_score_mode=args.decoder_score_mode,
        min_decision_score=args.min_decision_score,
        permuted_null_weight=args.permuted_null_weight,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_improvement_over_base=args.min_improvement_over_base,
        feature_model=args.feature_model,
        feature_device=args.feature_device,
        feature_dtype=args.feature_dtype,
        feature_max_length=args.feature_max_length,
        local_files_only=not args.allow_downloads,
    )


if __name__ == "__main__":
    main()
