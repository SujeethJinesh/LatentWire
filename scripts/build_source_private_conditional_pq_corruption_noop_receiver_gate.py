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

from scripts.build_source_private_product_codebook_geometry_gate import _groups_for_variant  # noqa: E402
from scripts.run_source_private_candidate_embedding_receiver import _build_anchor_matrix  # noqa: E402
from scripts.run_source_private_conditional_pq_innovation_gate import (  # noqa: E402
    BASIS_VIEWS,
    CONDITIONING_MODES,
    _candidate_innovations_for_basis,
    _condition_candidates_to_public,
    _derangement,
    _dimension_utilities,
    _fit_conditional_encoder,
    _fit_innovation_codebook,
    _oracle_accuracy,
    _paired_bootstrap,
    _payload_for_condition,
    _payload_uniqueness,
    _representation_dim,
    _sha256_file,
    _unquantized_accuracy,
    _write_jsonl,
)
from scripts.run_source_private_hidden_repair_packet_smoke import Example, _prior_prediction, make_benchmark  # noqa: E402
from scripts.run_source_private_masked_innovation_receiver import _SEMANTIC_CANDIDATE_MATRIX_CACHE  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import _remap_candidate_slots  # noqa: E402
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


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
    "candidate_roll",
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
    "candidate_roll",
    "opaque_slot_basis",
]
NOOP_TRAIN_CONDITIONS = [
    "target_only",
    *CONTROL_CONDITIONS,
]


def _answer_index(example: Example) -> int:
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == prior)


def _reconstruct_payload(payload: bytes | None, *, codebook: Any, representation_dim: int) -> np.ndarray:
    reconstructed = np.zeros(representation_dim, dtype=np.float32)
    if not payload:
        return reconstructed
    raw = np.frombuffer(payload[: codebook.subspaces], dtype=np.uint8)
    for subspace_index, code in enumerate(raw):
        centroids = codebook.centroids[subspace_index]
        reconstructed[codebook.groups[subspace_index]] = centroids[int(code) % centroids.shape[0]]
    return reconstructed


def _candidate_features(
    example: Example,
    payload: bytes | None,
    *,
    codebook: Any,
    representation_dim: int,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
    conditioning_mode: str,
    candidate_permutation: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    reconstructed = _reconstruct_payload(payload, codebook=codebook, representation_dim=representation_dim)
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
    if candidate_permutation is not None:
        candidates = candidates[candidate_permutation]
    if codebook.rotation is not None:
        candidates = (candidates @ codebook.rotation).astype(np.float32)
    distances = np.sum((candidates - reconstructed[None, :]) ** 2, axis=1)
    similarities = candidates @ reconstructed
    prior_idx = _prior_index(example)
    prior_distance = float(distances[prior_idx])
    prior_similarity = float(similarities[prior_idx])
    rows: list[list[float]] = []
    for candidate_index, candidate in enumerate(example.candidates):
        rows.append(
            [
                1.0,
                float(similarities[candidate_index]),
                -float(distances[candidate_index]),
                float(candidate.prior_score),
                1.0 if candidate_index == prior_idx else 0.0,
                prior_distance - float(distances[candidate_index]),
                float(similarities[candidate_index]) - prior_similarity,
                1.0 if payload else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float64), {
        "min_l2": float(np.min(distances)),
        "max_similarity": float(np.max(similarities)),
        "candidate_permutation": None
        if candidate_permutation is None
        else [int(value) for value in candidate_permutation.tolist()],
    }


def _payload_and_permutation_for_condition(
    *,
    condition: str,
    example: Example,
    rows: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: Any,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
    seed: int,
    rng: random.Random,
) -> tuple[bytes | None, np.ndarray | None]:
    payload_condition = "source" if condition == "candidate_roll" else condition
    payload = _payload_for_condition(
        condition=payload_condition,
        example=example,
        eval_rows=rows,
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
    if condition == "candidate_roll":
        return payload, np.roll(np.arange(len(example.candidates)), 1).astype(np.int64)
    if condition == "deranged_public_basis":
        return payload, _derangement(len(example.candidates), seed=seed * 1009 + index)
    return payload, None


def _fit_receiver(
    train_rows: list[Example],
    *,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: Any,
    representation_dim: int,
    feature_dim: int,
    basis_view: str,
    anchor_matrix: np.ndarray | None,
    target_topk: int,
    source_topk: int,
    conditioning_mode: str,
    seed: int,
    ridge: float,
    noop_weight: float,
) -> dict[str, Any]:
    rng = random.Random(seed * 7919 + 17)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    counts: dict[str, int] = {}
    diagnostics_rows: list[tuple[str, np.ndarray, int, int, int]] = []
    for row_index, example in enumerate(train_rows):
        for condition in ["source", *NOOP_TRAIN_CONDITIONS]:
            payload, permutation = _payload_and_permutation_for_condition(
                condition=condition,
                example=example,
                rows=train_rows,
                index=row_index,
                encoder=encoder,
                label_shuffle_encoder=label_shuffle_encoder,
                codebook=codebook,
                feature_dim=feature_dim,
                basis_view=basis_view,
                anchor_matrix=anchor_matrix,
                source_topk=source_topk,
                target_topk=target_topk,
                conditioning_mode=conditioning_mode,
                seed=seed,
                rng=rng,
            )
            features, _ = _candidate_features(
                example,
                payload,
                codebook=codebook,
                representation_dim=representation_dim,
                feature_dim=feature_dim,
                basis_view="slot" if condition == "opaque_slot_basis" else basis_view,
                anchor_matrix=anchor_matrix,
                target_topk=target_topk,
                conditioning_mode=conditioning_mode,
                candidate_permutation=permutation,
            )
            positive_index = _answer_index(example) if condition == "source" else _prior_index(example)
            weight = 1.0 if condition == "source" else noop_weight
            root_weight = float(np.sqrt(weight))
            xs.append(features * root_weight)
            ys.extend(
                [
                    root_weight if candidate_index == positive_index else 0.0
                    for candidate_index in range(len(example.candidates))
                ]
            )
            counts[condition] = counts.get(condition, 0) + 1
            diagnostics_rows.append((condition, features, positive_index, _prior_index(example), _answer_index(example)))
    x = np.concatenate(xs, axis=0)
    y = np.asarray(ys, dtype=np.float64)
    xtx = x.T @ x
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= ridge
    weights = np.linalg.solve(xtx, x.T @ y).astype(np.float64)
    diagnostics: dict[str, dict[str, Any]] = {}
    for condition, features, positive_index, prior_index, answer_index in diagnostics_rows:
        scores = features @ weights
        best = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best, rtol=1e-6, atol=1e-8))
        chosen_index = prior_index if any(int(idx) == prior_index for idx in tied) else int(tied[0])
        row = diagnostics.setdefault(
            condition,
            {
                "rows": 0,
                "target_correct": 0,
                "predicted_prior": 0,
                "predicted_answer": 0,
            },
        )
        row["rows"] += 1
        row["target_correct"] += int(chosen_index == positive_index)
        row["predicted_prior"] += int(chosen_index == prior_index)
        row["predicted_answer"] += int(chosen_index == answer_index)
    for row in diagnostics.values():
        rows = max(1, int(row["rows"]))
        row["target_accuracy"] = float(row["target_correct"] / rows)
        row["predicted_prior_rate"] = float(row["predicted_prior"] / rows)
        row["predicted_answer_rate"] = float(row["predicted_answer"] / rows)
    return {
        "weights": weights,
        "feature_names": [
            "intercept",
            "packet_candidate_similarity",
            "negative_packet_candidate_l2",
            "target_prior_score",
            "is_target_prior",
            "prior_l2_minus_candidate_l2",
            "candidate_similarity_minus_prior_similarity",
            "has_payload",
        ],
        "condition_counts": counts,
        "ridge": ridge,
        "noop_weight": noop_weight,
        "training_diagnostics": diagnostics,
    }


def _predict_condition(
    *,
    condition: str,
    example: Example,
    rows: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: Any,
    receiver: dict[str, Any],
    representation_dim: int,
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
    payload, permutation = _payload_and_permutation_for_condition(
        condition=condition,
        example=example,
        rows=rows,
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
        seed=seed,
        rng=rng,
    )
    features, metadata = _candidate_features(
        example,
        payload,
        codebook=codebook,
        representation_dim=representation_dim,
        feature_dim=feature_dim,
        basis_view="slot" if condition == "opaque_slot_basis" else basis_view,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        candidate_permutation=permutation,
    )
    scores = features @ receiver["weights"]
    best = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, best, rtol=1e-6, atol=1e-8))
    prior_idx = _prior_index(example)
    chosen_idx = prior_idx if any(int(idx) == prior_idx for idx in tied) else int(tied[0])
    prediction = example.candidates[chosen_idx].label
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
        "metadata": metadata
        | {
            "decoder": "corruption_noop_public_candidate_receiver",
            "chosen_index": int(chosen_idx),
            "chosen_score": float(scores[chosen_idx]),
            "prior_score": float(scores[prior_idx]),
        },
    }


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


def _summarize(
    predictions: list[dict[str, Any]],
    *,
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
        condition: [row["example_id"] for row in predictions if row["condition"] == condition]
        for condition in CONDITIONS
    }
    metrics = {
        condition: _metric([row for row in predictions if row["condition"] == condition])
        for condition in CONDITIONS
    }
    target = metrics["target_only"]["accuracy"]
    source = metrics["source"]["accuracy"]
    best_control_condition = max(CONTROL_CONDITIONS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_condition]["accuracy"]
    paired_vs_best_control = _paired_bootstrap(
        predictions,
        condition_a="source",
        condition_b=best_control_condition,
        samples=bootstrap_samples,
        seed=seed + 31,
    )
    paired_vs_target = _paired_bootstrap(
        predictions,
        condition_a="source",
        condition_b="target_only",
        samples=bootstrap_samples,
        seed=seed + 17,
    )
    target_ids = set(metrics["target_only"]["correct_ids"])
    source_ids = set(metrics["source"]["correct_ids"])
    train_eval_overlap = sorted(train_ids.intersection(eval_ids))
    return {
        "n": len(eval_rows),
        "conditions": CONDITIONS,
        "exact_id_parity": all(ids == eval_ids for ids in condition_ids.values()),
        "train_eval_id_intersection_count": len(train_eval_overlap),
        "eval_id_sha256": hashlib.sha256("\n".join(eval_ids).encode("utf-8")).hexdigest(),
        "target_only_accuracy": target,
        "source_accuracy": source,
        "source_minus_target": source - target,
        "best_control_condition": best_control_condition,
        "best_control_accuracy": best_control,
        "source_minus_best_control": source - best_control,
        "helps": len(source_ids - target_ids),
        "harms": len(target_ids - source_ids),
        "unquantized_predicted_accuracy": unquantized["accuracy"],
        "target_innovation_oracle_accuracy": oracle["accuracy"],
        "paired_bootstrap": {
            "source_vs_target": paired_vs_target,
            "source_vs_best_control": paired_vs_best_control,
        },
        "payload_uniqueness": _payload_uniqueness(predictions),
        "metrics": metrics,
        "pass_gate": (
            not train_eval_overlap
            and source >= target + 0.05
            and source >= best_control + 0.10
            and paired_vs_best_control["ci95_low"] > 0.0
        ),
        "pass_rule": (
            "Pass requires disjoint train/eval IDs, source >= target+0.05, "
            "source >= best destructive/shortcut control+0.10, and paired CI95 low versus best control > 0."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Conditional PQ Corruption-to-Noop Receiver Gate",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- examples: `{summary['n']}`",
        f"- train/eval ID overlap: `{summary['train_eval_id_intersection_count']}`",
        f"- train/eval families: `{payload['train_family_set']}->{payload['eval_family_set']}`",
        f"- basis view: `{payload['basis_view']}`",
        f"- conditioning mode: `{payload['conditioning_mode']}`",
        f"- budget bytes: `{payload['budget_bytes']}`",
        f"- source accuracy: `{summary['source_accuracy']:.3f}`",
        f"- target accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
        f"- source minus best control: `{summary['source_minus_best_control']:.3f}`",
        f"- CI95 low vs best control: `{summary['paired_bootstrap']['source_vs_best_control']['ci95_low']:.3f}`",
        f"- helps/harms: `{summary['helps']}/{summary['harms']}`",
        f"- unquantized predicted accuracy: `{summary['unquantized_predicted_accuracy']:.3f}`",
        f"- target innovation oracle accuracy: `{summary['target_innovation_oracle_accuracy']:.3f}`",
        "",
        "| Condition | Accuracy | Mean bytes | p50 ms |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metric = summary["metrics"][condition]
        lines.append(
            f"| {condition} | {metric['accuracy']:.3f} | {metric['mean_payload_bytes']:.2f} | "
            f"{metric['p50_latency_ms']:.4f} |"
        )
    lines.extend(["", f"Payload uniqueness: `{summary['payload_uniqueness']}`", "", f"Pass rule: {summary['pass_rule']}", ""])
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
    conditioning_mode: str,
    budget_bytes: int,
    variant: str,
    remap_slot_seed: int | None,
    encoder_ridge: float,
    receiver_ridge: float,
    receiver_noop_weight: float,
    fit_intercept: bool,
    mask_repeats: int,
    codebook_iterations: int,
    seed: int,
    bootstrap_samples: int,
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
    anchor_matrix = (
        _build_anchor_matrix(train_rows, feature_dim=feature_dim, anchor_count=anchor_count)
        if basis_view == "anchor_relative"
        else None
    )
    representation_dim = _representation_dim(feature_dim=feature_dim, anchor_matrix=anchor_matrix)
    encoder = _fit_conditional_encoder(
        train_rows,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        ridge=encoder_ridge,
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
        ridge=encoder_ridge,
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
    receiver = _fit_receiver(
        train_rows,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        representation_dim=representation_dim,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
        source_topk=source_topk,
        conditioning_mode=conditioning_mode,
        seed=seed,
        ridge=receiver_ridge,
        noop_weight=receiver_noop_weight,
    )
    rng = random.Random(seed * 4001 + train_seed * 37 + eval_seed)
    predictions: list[dict[str, Any]] = []
    for row_index, example in enumerate(eval_rows):
        for condition in CONDITIONS:
            predictions.append(
                _predict_condition(
                    condition=condition,
                    example=example,
                    rows=eval_rows,
                    index=row_index,
                    encoder=encoder,
                    label_shuffle_encoder=label_shuffle_encoder,
                    codebook=codebook,
                    receiver=receiver,
                    representation_dim=representation_dim,
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
        predictions,
        train_rows=train_rows,
        eval_rows=eval_rows,
        unquantized=unquantized,
        oracle=oracle,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    payload = {
        "gate": "source_private_conditional_pq_corruption_noop_receiver",
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
        "encoder_ridge": encoder_ridge,
        "receiver_ridge": receiver_ridge,
        "receiver_noop_weight": receiver_noop_weight,
        "fit_intercept": fit_intercept,
        "mask_repeats": mask_repeats,
        "codebook_iterations": codebook_iterations,
        "conditions": CONDITIONS,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "label_shuffle_encoder_sha256": hashlib.sha256(label_shuffle_encoder.tobytes()).hexdigest(),
        "codebook_sha256": hashlib.sha256(b"".join(centroid.tobytes() for centroid in codebook.centroids)).hexdigest(),
        "receiver_sha256": hashlib.sha256(receiver["weights"].tobytes()).hexdigest(),
        "receiver": {
            "feature_names": receiver["feature_names"],
            "weights": [float(value) for value in receiver["weights"].tolist()],
            "condition_counts": receiver["condition_counts"],
            "ridge": receiver["ridge"],
            "noop_weight": receiver["noop_weight"],
            "training_diagnostics": receiver["training_diagnostics"],
        },
        "systems_accounting": {
            "payload_bytes": budget_bytes,
            "framed_packet_bytes_estimate": budget_bytes + 3,
            "dense_kv_floor_bytes_reference": 21504,
            "dense_kv_floor_note": (
                "Local comparator inherited from the conditional-PQ status artifact; "
                "not a native GPU throughput or energy measurement."
            ),
            "payload_to_dense_kv_floor_ratio": float(budget_bytes / 21504.0),
        },
        "summary": summary,
        "pass_gate": summary["pass_gate"],
        "interpretation": (
            "This gate keeps the conditional-PQ source packet fixed, but trains a public candidate receiver "
            "to choose the answer for matched packets and decode corrupted packets to the target prior."
        ),
    }
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
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
                "# Conditional PQ Corruption-to-Noop Receiver Manifest",
                "",
                f"- pass gate: `{summary['pass_gate']}`",
                f"- train/eval families: `{train_family_set}->{eval_family_set}`",
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
        default=pathlib.Path("results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504"),
    )
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=29)
    parser.add_argument("--train-start-index", type=int, default=10000)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="core")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="holdout")
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="plausible_decoys")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--basis-view", choices=BASIS_VIEWS, default="semantic")
    parser.add_argument("--source-topk", type=int, default=64)
    parser.add_argument("--target-topk", type=int, default=32)
    parser.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default="public_zscore")
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument(
        "--variant",
        choices=["canonical", "utility_balanced", "protected_hadamard", "utility_protected_hadamard"],
        default="utility_protected_hadamard",
    )
    parser.add_argument("--remap-slot-seed", type=int, default=101)
    parser.add_argument("--encoder-ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-noop-weight", type=float, default=0.1)
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--mask-repeats", type=int, default=1)
    parser.add_argument("--codebook-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=output_dir,
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
        encoder_ridge=args.encoder_ridge,
        receiver_ridge=args.receiver_ridge,
        receiver_noop_weight=args.receiver_noop_weight,
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
                "best_control_condition": payload["summary"]["best_control_condition"],
                "best_control_accuracy": payload["summary"]["best_control_accuracy"],
                "ci95_low_vs_best_control": payload["summary"]["paired_bootstrap"]["source_vs_best_control"]["ci95_low"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
