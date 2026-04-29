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

from scripts.run_source_private_candidate_embedding_receiver import (  # noqa: E402
    _build_anchor_matrix,
    _candidate_matrix_mode,
    _packet_bits,
    _packet_from_vector,
    _sha256_file,
    _source_vector_mode,
)
from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _deterministic_nonself_index,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_learned_syndrome import (  # noqa: E402
    _augment,
    _normalize_rows,
    _token_count,
)


CONDITIONS = [
    "target_only",
    "matched_masked_innovation_receiver",
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "public_only_innovation",
    "shuffled_mask_or_atoms",
    "random_same_byte",
    "target_derived_sidecar",
    "answer_only",
    "structured_text_matched",
    "wrong_projection_source",
    "full_diag_oracle",
]


def _relative_candidates(example: Example, *, feature_dim: int, anchor_matrix: np.ndarray) -> np.ndarray:
    candidates = _candidate_matrix_mode(example, feature_dim)
    return _normalize_rows(candidates @ anchor_matrix.T).astype(np.float32)


def _answer_index(example: Example) -> int:
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)


def _target_innovation(example: Example, *, feature_dim: int, anchor_matrix: np.ndarray, topk: int) -> np.ndarray:
    rel = _relative_candidates(example, feature_dim=feature_dim, anchor_matrix=anchor_matrix)
    delta = rel[_answer_index(example)] - rel[_prior_index(example)]
    if topk > 0 and topk < delta.size:
        keep = np.argpartition(np.abs(delta), -topk)[-topk:]
        sparse = np.zeros_like(delta)
        sparse[keep] = delta[keep]
        delta = sparse
    norm = float(np.linalg.norm(delta))
    if norm > 1e-8:
        delta = delta / norm
    return delta.astype(np.float32)


def _candidate_innovations(example: Example, *, feature_dim: int, anchor_matrix: np.ndarray, topk: int) -> np.ndarray:
    rel = _relative_candidates(example, feature_dim=feature_dim, anchor_matrix=anchor_matrix)
    prior = rel[_prior_index(example)]
    deltas = rel - prior[None, :]
    if topk > 0 and topk < deltas.shape[1]:
        sparse = np.zeros_like(deltas)
        for row_index, delta in enumerate(deltas):
            keep = np.argpartition(np.abs(delta), -topk)[-topk:]
            sparse[row_index, keep] = delta[keep]
        deltas = sparse
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    return np.divide(deltas, np.maximum(norms, 1e-8)).astype(np.float32)


def _source_innovation(example: Example, *, feature_dim: int, source_topk: int, mode: str) -> np.ndarray:
    matched = _source_vector_mode(example, feature_dim, mode="matched")
    masked = _source_vector_mode(example, feature_dim, mode="answer_masked")
    if mode == "matched":
        delta = matched - masked
    elif mode in {"answer_masked", "public_only", "zero"}:
        delta = np.zeros_like(matched)
    else:
        raise ValueError(f"unknown source innovation mode {mode!r}")
    if source_topk > 0 and source_topk < delta.size:
        keep = np.argpartition(np.abs(delta), -source_topk)[-source_topk:]
        sparse = np.zeros_like(delta)
        sparse[keep] = delta[keep]
        delta = sparse
    norm = float(np.linalg.norm(delta))
    if norm > 1e-8:
        delta = delta / norm
    return delta.astype(np.float32)


def _fit_innovation_encoder(
    train_examples: list[Example],
    *,
    feature_dim: int,
    anchor_matrix: np.ndarray,
    source_topk: int,
    target_topk: int,
    ridge: float,
    mask_repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for example in train_examples:
        base_x = _source_innovation(example, feature_dim=feature_dim, source_topk=source_topk, mode="matched")
        base_y = _target_innovation(example, feature_dim=feature_dim, anchor_matrix=anchor_matrix, topk=target_topk)
        xs.append(base_x)
        ys.append(base_y)
        for _ in range(mask_repeats):
            x_mask = (rng.random(base_x.shape) >= 0.20).astype(np.float32)
            y_mask = (rng.random(base_y.shape) >= 0.10).astype(np.float32)
            xs.append(base_x * x_mask)
            ys.append(base_y * y_mask)
    x_arr = np.stack(xs, axis=0).astype(np.float64)
    y_arr = np.stack(ys, axis=0).astype(np.float64)
    x_aug = np.concatenate([x_arr, np.ones((x_arr.shape[0], 1), dtype=np.float64)], axis=1)
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[-1, -1] -= ridge
    return np.linalg.solve(xtx, x_aug.T @ y_arr).astype(np.float32)


def _predict_innovation(
    example: Example,
    *,
    encoder: np.ndarray,
    feature_dim: int,
    source_topk: int,
    mode: str,
) -> np.ndarray:
    pred = _augment(_source_innovation(example, feature_dim=feature_dim, source_topk=source_topk, mode=mode)) @ encoder
    norm = float(np.linalg.norm(pred))
    if norm > 1e-8:
        pred = pred / norm
    return pred.astype(np.float32)


def _condition_vector(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    feature_dim: int,
    source_topk: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
    random_encoder: np.ndarray,
    atom_permutation: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {}
    if condition == "matched_masked_innovation_receiver":
        return (
            _predict_innovation(example, encoder=encoder, feature_dim=feature_dim, source_topk=source_topk, mode="matched"),
            {"source": example.example_id},
        )
    if condition == "shuffled_source":
        other = eval_examples[_deterministic_nonself_index(index, len(eval_examples))]
        return (
            _predict_innovation(other, encoder=encoder, feature_dim=feature_dim, source_topk=source_topk, mode="matched"),
            {"source": other.example_id},
        )
    if condition in {"answer_masked_source", "public_only_innovation"}:
        return (
            _predict_innovation(example, encoder=encoder, feature_dim=feature_dim, source_topk=source_topk, mode="answer_masked"),
            {"source": condition},
        )
    if condition == "shuffled_mask_or_atoms":
        vec = _predict_innovation(example, encoder=encoder, feature_dim=feature_dim, source_topk=source_topk, mode="matched")
        return vec[atom_permutation], {"source": "atom_permuted"}
    if condition == "target_derived_sidecar":
        return np.zeros(anchor_matrix.shape[0], dtype=np.float32), {"source": "target_prior_zero_innovation"}
    if condition == "wrong_projection_source":
        pred = _augment(_source_innovation(example, feature_dim=feature_dim, source_topk=source_topk, mode="matched")) @ random_encoder
        norm = float(np.linalg.norm(pred))
        if norm > 1e-8:
            pred = pred / norm
        return pred.astype(np.float32), {"source": "wrong_encoder"}
    if condition == "full_diag_oracle":
        return _target_innovation(example, feature_dim=feature_dim, anchor_matrix=anchor_matrix, topk=target_topk), {"source": "target_innovation_oracle"}
    raise ValueError(f"condition {condition!r} does not use an innovation vector")


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    source_topk: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
    random_encoder: np.ndarray,
    atom_permutation: np.ndarray,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"answer_only", "structured_text_matched"}:
        if condition == "answer_only":
            return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_hidden_log"}
    if condition == "random_same_byte":
        return rng.randbytes(budget_bytes), {"source": "random"}
    vec, meta = _condition_vector(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        encoder=encoder,
        feature_dim=feature_dim,
        source_topk=source_topk,
        target_topk=target_topk,
        anchor_matrix=anchor_matrix,
        random_encoder=random_encoder,
        atom_permutation=atom_permutation,
    )
    if vec is None:
        return None, meta
    return _packet_from_vector(vec, code_projection, budget_bytes), meta


def _candidate_codes(
    example: Example,
    *,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
) -> np.ndarray:
    innovations = _candidate_innovations(example, feature_dim=feature_dim, anchor_matrix=anchor_matrix, topk=target_topk)
    logits = innovations @ code_projection[: budget_bytes * 8].T
    return (logits >= 0).astype(np.uint8)


def _score_candidates(
    example: Example,
    payload: bytes,
    *,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
) -> np.ndarray:
    bits = _packet_bits(payload, budget_bytes * 8)
    codes = _candidate_codes(
        example,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        target_topk=target_topk,
        anchor_matrix=anchor_matrix,
    ).astype(np.float32)
    signed_packet = bits * 2.0 - 1.0
    signed_codes = codes * 2.0 - 1.0
    return (signed_codes * signed_packet[None, :]).mean(axis=1)


def _predict_with_receiver(
    example: Example,
    payload: bytes | None,
    *,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
    margin_threshold: float,
) -> tuple[str, dict[str, Any]]:
    prior = _prior_prediction(example)
    if payload is None:
        return prior, {"decoder": "prior"}
    scores = _score_candidates(
        example,
        payload,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        target_topk=target_topk,
        anchor_matrix=anchor_matrix,
    )
    labels = [candidate.label for candidate in example.candidates]
    prior_index = labels.index(prior)
    best_score = float(np.max(scores))
    tied = [idx for idx, score in enumerate(scores) if abs(float(score) - best_score) <= 1e-8]
    margin_vs_prior = best_score - float(scores[prior_index])
    if labels[tied[0]] != prior and margin_vs_prior < margin_threshold:
        return prior, {
            "decoder": "masked_innovation_target_preserve",
            "scores": [float(score) for score in scores],
            "ties": tied,
            "margin_vs_prior": margin_vs_prior,
            "margin_threshold": margin_threshold,
            "preserved_prior": True,
        }
    if any(labels[idx] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[tied[0]]
    return prediction, {
        "decoder": "masked_innovation_target_preserve",
        "scores": [float(score) for score in scores],
        "ties": tied,
        "margin_vs_prior": margin_vs_prior,
        "margin_threshold": margin_threshold,
        "preserved_prior": False,
    }


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    source_topk: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
    margin_threshold: float,
    random_encoder: np.ndarray,
    atom_permutation: np.ndarray,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, metadata = _payload_for_condition(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        encoder=encoder,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        source_topk=source_topk,
        target_topk=target_topk,
        anchor_matrix=anchor_matrix,
        random_encoder=random_encoder,
        atom_permutation=atom_permutation,
        rng=rng,
    )
    prediction, decode_meta = _predict_with_receiver(
        example,
        payload,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        target_topk=target_topk,
        anchor_matrix=anchor_matrix,
        margin_threshold=0.0 if condition == "full_diag_oracle" else margin_threshold,
    )
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "payload_hex": payload_hex,
        "metadata": {**metadata, **decode_meta},
    }


def _summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in predictions]
    correct = sum(1 for row in predictions if row["correct"])
    return {
        "n": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in predictions),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in predictions),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _evaluate_threshold(
    examples: list[Example],
    *,
    conditions: list[str],
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    source_topk: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
    random_encoder: np.ndarray,
    atom_permutation: np.ndarray,
    margin_threshold: float,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    by_condition: dict[str, list[bool]] = {condition: [] for condition in conditions}
    for row_index, example in enumerate(examples):
        for condition in conditions:
            row = _predict_condition(
                condition=condition,
                example=example,
                eval_examples=examples,
                index=row_index,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                source_topk=source_topk,
                target_topk=target_topk,
                anchor_matrix=anchor_matrix,
                margin_threshold=margin_threshold,
                random_encoder=random_encoder,
                atom_permutation=atom_permutation,
                rng=rng,
            )
            by_condition[condition].append(row["correct"])
    return {condition: sum(values) / len(values) for condition, values in by_condition.items()}


def _calibrate_margin_threshold(
    train_examples: list[Example],
    *,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    source_topk: int,
    target_topk: int,
    anchor_matrix: np.ndarray,
    random_encoder: np.ndarray,
    atom_permutation: np.ndarray,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    controls = [
        "zero_source",
        "shuffled_source",
        "answer_masked_source",
        "public_only_innovation",
        "shuffled_mask_or_atoms",
        "random_same_byte",
        "target_derived_sidecar",
        "answer_only",
        "structured_text_matched",
        "wrong_projection_source",
    ]
    margins = [0.0]
    for example in train_examples:
        payload = _packet_from_vector(
            _predict_innovation(example, encoder=encoder, feature_dim=feature_dim, source_topk=source_topk, mode="matched"),
            code_projection,
            budget_bytes,
        )
        scores = _score_candidates(
            example,
            payload,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            target_topk=target_topk,
            anchor_matrix=anchor_matrix,
        )
        labels = [candidate.label for candidate in example.candidates]
        prior_index = labels.index(_prior_prediction(example))
        margins.append(max(0.0, float(np.max(scores) - scores[prior_index])))
    ordered = sorted(margins)
    thresholds = sorted(
        {
            round(ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))], 6)
            for q in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
        }
        | {0.0}
    )
    best_threshold = 0.0
    best_record: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = _evaluate_threshold(
            train_examples,
            conditions=["target_only", "matched_masked_innovation_receiver", *controls],
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            source_topk=source_topk,
            target_topk=target_topk,
            anchor_matrix=anchor_matrix,
            random_encoder=random_encoder,
            atom_permutation=atom_permutation,
            margin_threshold=threshold,
            seed=seed,
        )
        target = metrics["target_only"]
        best_control = max(metrics[name] for name in controls)
        control_ok = all(metrics[name] <= target + 0.05 for name in controls)
        record = {
            "threshold": threshold,
            "matched_accuracy": metrics["matched_masked_innovation_receiver"],
            "target_accuracy": target,
            "best_control_accuracy": best_control,
            "control_ok": control_ok,
        }
        if best_record is None:
            best_record = record
            best_threshold = threshold
        elif control_ok and (
            not best_record["control_ok"]
            or record["matched_accuracy"] > best_record["matched_accuracy"]
            or (
                record["matched_accuracy"] == best_record["matched_accuracy"]
                and record["best_control_accuracy"] < best_record["best_control_accuracy"]
            )
        ):
            best_record = record
            best_threshold = threshold
        elif not best_record["control_ok"] and record["best_control_accuracy"] < best_record["best_control_accuracy"]:
            best_record = record
            best_threshold = threshold
    assert best_record is not None
    return best_threshold, best_record


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    anchor_count: int,
    source_topk: int,
    target_topk: int,
    budgets: list[int],
    train_seed: int,
    eval_seed: int,
    ridge: float,
    mask_repeats: int,
    calibration_examples: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(examples=train_examples, candidates=candidates, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=candidates, seed=eval_seed, family_set=eval_family_set)
    anchor_matrix = _build_anchor_matrix(train_rows, feature_dim=feature_dim, anchor_count=anchor_count)
    rng_np = np.random.default_rng(train_seed * 1009 + eval_seed)
    encoder = _fit_innovation_encoder(
        train_rows,
        feature_dim=feature_dim,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        ridge=ridge,
        mask_repeats=mask_repeats,
        rng=rng_np,
    )
    code_projection = _normalize_rows(rng_np.normal(size=(max(budgets) * 8, anchor_count))).astype(np.float32)
    random_encoder = rng_np.normal(0.0, 1.0 / np.sqrt(anchor_count), size=encoder.shape).astype(np.float32)
    atom_permutation = rng_np.permutation(anchor_count)
    rng = random.Random(train_seed * 2003 + eval_seed)
    prediction_files: dict[str, str] = {}
    budget_summaries: list[dict[str, Any]] = []
    calibration_rows = train_rows[: min(len(train_rows), calibration_examples)]
    for budget in budgets:
        margin_threshold, calibration = _calibrate_margin_threshold(
            calibration_rows,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget,
            source_topk=source_topk,
            target_topk=target_topk,
            anchor_matrix=anchor_matrix,
            random_encoder=random_encoder,
            atom_permutation=atom_permutation,
            seed=train_seed * 3011 + eval_seed + budget,
        )
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
        for row_index, example in enumerate(eval_rows):
            for condition in CONDITIONS:
                by_condition[condition].append(
                    _predict_condition(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        encoder=encoder,
                        code_projection=code_projection,
                        feature_dim=feature_dim,
                        budget_bytes=budget,
                        source_topk=source_topk,
                        target_topk=target_topk,
                        anchor_matrix=anchor_matrix,
                        margin_threshold=margin_threshold,
                        random_encoder=random_encoder,
                        atom_permutation=atom_permutation,
                        rng=rng,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        metrics = {condition: _summarize(rows) for condition, rows in by_condition.items()}
        controls = [
            "zero_source",
            "shuffled_source",
            "answer_masked_source",
            "public_only_innovation",
            "shuffled_mask_or_atoms",
            "random_same_byte",
            "target_derived_sidecar",
            "answer_only",
            "structured_text_matched",
            "wrong_projection_source",
        ]
        best_no_source = max(metrics[name]["accuracy"] for name in ["target_only", *controls])
        best_destructive = max(metrics[name]["accuracy"] for name in controls)
        matched = metrics["matched_masked_innovation_receiver"]["accuracy"]
        target = metrics["target_only"]["accuracy"]
        pass_gate = (
            matched >= target + 0.15
            and matched >= best_destructive + 0.15
            and all(metrics[name]["accuracy"] <= target + 0.05 for name in controls)
            and metrics["full_diag_oracle"]["accuracy"] >= 0.95
        )
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for condition in CONDITIONS:
                for row in by_condition[condition]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_files[str(budget)] = predictions_path.name
        budget_summaries.append(
            {
                "budget_bytes": budget,
                "pass_gate": pass_gate,
                "matched_accuracy": matched,
                "target_only_accuracy": target,
                "best_no_source_accuracy": best_no_source,
                "best_destructive_control_accuracy": best_destructive,
                "matched_minus_target": matched - target,
                "matched_minus_best_destructive_control": matched - best_destructive,
                "full_diag_oracle_accuracy": metrics["full_diag_oracle"]["accuracy"],
                "margin_threshold": margin_threshold,
                "margin_calibration": calibration,
                "metrics": metrics,
            }
        )
    exact_ids = [row.example_id for row in eval_rows]
    payload = {
        "gate": "source_private_masked_innovation_receiver",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "anchor_count": anchor_count,
        "source_topk": source_topk,
        "target_topk": target_topk,
        "budgets": budgets,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "ridge": ridge,
        "mask_repeats": mask_repeats,
        "calibration_examples": len(calibration_rows),
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
        "prediction_files": prediction_files,
        "pass_rule": (
            "Matched masked innovation receiver must beat target by >=0.15, beat every destructive control "
            "by >=0.15, keep all destructive controls within target+0.05, and keep full diagnostic oracle >=0.95."
        ),
        "interpretation": (
            "This gate transmits a sparse source-private innovation: hashed matched-source minus answer-masked-source "
            "features are mapped to anchor-relative target innovation from target prior candidate to answer candidate. "
            "It tests whether private source evidence survives without raw candidate-coordinate regression."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", *prediction_files.values(), "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md", *prediction_files.values()]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Masked Innovation Receiver Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- train family set: `{train_family_set}`",
                f"- eval family set: `{eval_family_set}`",
                f"- budgets: `{budgets}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Masked Innovation Receiver",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{payload['train_family_set']}:{payload['train_examples']}` / `{payload['eval_family_set']}:{payload['eval_examples']}`",
        f"- anchor count: `{payload['anchor_count']}`",
        f"- source top-k: `{payload['source_topk']}`",
        f"- target top-k: `{payload['target_topk']}`",
        f"- mask repeats: `{payload['mask_repeats']}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_destructive_control_accuracy']:.3f} | "
            f"{row['matched_minus_target']:.3f} | {row['matched_minus_best_destructive_control']:.3f} | "
            f"{row['full_diag_oracle_accuracy']:.3f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_masked_innovation_receiver_20260429"))
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="core")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="holdout")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--source-topk", type=int, default=64)
    parser.add_argument("--target-topk", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--mask-repeats", type=int, default=2)
    parser.add_argument("--calibration-examples", type=int, default=128)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        anchor_count=args.anchor_count,
        source_topk=args.source_topk,
        target_topk=args.target_topk,
        budgets=args.budgets,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        ridge=args.ridge,
        mask_repeats=args.mask_repeats,
        calibration_examples=args.calibration_examples,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
