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
from dataclasses import replace
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _deterministic_nonself_index,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_learned_syndrome import (  # noqa: E402
    _augment,
    _candidate_matrix,
    _decode_packet,
    _fit_ridge_encoder,
    _hashed_text_features,
    _normalize_rows,
    _packet_from_vector,
    _source_packet,
    _source_vector,
    _token_count,
)


def _candidate_texts_for_view(example: Example, *, candidate_view: str) -> list[str]:
    rows: list[str] = []
    for candidate in example.candidates:
        if candidate_view == "full":
            rows.append(
                "\n".join(
                    [
                        f"patch={candidate.patch_name}",
                        f"intent={candidate.patch_intent}",
                        f"handles_repair_diag={candidate.handles_diagnostic}",
                        f"public_issue={example.public_issue}",
                    ]
                )
            )
        elif candidate_view == "no_diag":
            rows.append(
                "\n".join(
                    [
                        f"patch={candidate.patch_name}",
                        f"intent={candidate.patch_intent}",
                        f"handles_repair_diag=<MASKED>",
                        f"public_issue={example.public_issue}",
                    ]
                )
            )
        elif candidate_view == "semantic":
            rows.append(
                "\n".join(
                    [
                        f"intent={candidate.patch_intent}",
                        f"public_issue={example.public_issue}",
                    ]
                )
            )
        elif candidate_view == "slot":
            rows.append(f"candidate_slot={len(rows)}")
        else:
            raise ValueError(f"unknown candidate view {candidate_view!r}")
    return rows


def _remap_candidate_slots(examples: list[Example], *, remap_seed: int | None) -> list[Example]:
    if remap_seed is None:
        return examples
    remapped: list[Example] = []
    for index, example in enumerate(examples):
        rng = random.Random(remap_seed * 1000003 + index)
        order = list(range(len(example.candidates)))
        rng.shuffle(order)
        if order == list(range(len(example.candidates))):
            order = order[1:] + order[:1]
        remapped.append(replace(example, candidates=tuple(example.candidates[i] for i in order)))
    return remapped


def _candidate_matrix_for_view(example: Example, feature_dim: int, *, candidate_view: str) -> np.ndarray:
    if candidate_view == "full":
        return _candidate_matrix(example, feature_dim)
    return _normalize_rows(
        np.stack(
            [
                _hashed_text_features(text, feature_dim, namespace=f"candidate:{candidate_view}")
                for text in _candidate_texts_for_view(example, candidate_view=candidate_view)
            ]
        )
    ).astype(np.float32)


def _fit_ridge_encoder_for_view(
    train_examples: list[Example],
    *,
    feature_dim: int,
    ridge: float,
    candidate_view: str,
    fit_intercept: bool,
    label_shuffle_seed: int | None = None,
) -> np.ndarray:
    x = np.stack([_source_vector(example, feature_dim, mode="matched") for example in train_examples], axis=0).astype(np.float64)
    y = []
    label_indices = list(range(len(train_examples)))
    if label_shuffle_seed is not None:
        rng = random.Random(label_shuffle_seed)
        rng.shuffle(label_indices)
    for example in train_examples:
        label_example = train_examples[label_indices[len(y)]]
        candidates = _candidate_matrix_for_view(label_example, feature_dim, candidate_view=candidate_view).astype(np.float64)
        answer_index = next(idx for idx, candidate in enumerate(label_example.candidates) if candidate.label == label_example.answer_label)
        y.append(candidates[answer_index])
    y_arr = np.stack(y, axis=0).astype(np.float64)
    if fit_intercept:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    else:
        x_aug = x
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    if fit_intercept:
        xtx[-1, -1] -= ridge
    return np.linalg.solve(xtx, x_aug.T @ y_arr).astype(np.float32)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_scalar_calibration(
    train_examples: list[Example],
    *,
    encoder: np.ndarray,
    scalar_projection: np.ndarray,
    feature_dim: int,
    candidate_view: str,
) -> tuple[np.ndarray, np.ndarray]:
    projected: list[np.ndarray] = []
    for example in train_examples:
        predicted = _project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched")
        projected.append(scalar_projection @ predicted)
        projected.extend(_candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view) @ scalar_projection.T)
    values = np.stack(projected, axis=0).astype(np.float32)
    lo = np.quantile(values, 0.01, axis=0).astype(np.float32)
    hi = np.quantile(values, 0.99, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 1e-4)
    return lo, hi


def _project_source(example: Example, *, encoder: np.ndarray, feature_dim: int, mode: str) -> np.ndarray:
    source = _source_vector(example, feature_dim, mode=mode)
    if encoder.shape[0] == feature_dim + 1:
        return _augment(source) @ encoder
    if encoder.shape[0] == feature_dim:
        return source @ encoder
    raise ValueError(f"encoder shape {encoder.shape} is incompatible with feature_dim={feature_dim}")


def _quantize_scalar(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bytes:
    scaled = np.clip((values - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8).tobytes()


def _dequantize_scalar(payload: bytes, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    ints = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
    return lo[: ints.shape[0]] + (ints / 255.0) * (hi[: ints.shape[0]] - lo[: ints.shape[0]])


def _scalar_packet(
    example: Example,
    *,
    encoder: np.ndarray,
    scalar_projection: np.ndarray,
    feature_dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    mode: str,
) -> bytes:
    predicted = _project_source(example, encoder=encoder, feature_dim=feature_dim, mode=mode)
    return _quantize_scalar(scalar_projection @ predicted, lo, hi)


def _orthogonal_residual_projection(sign_projection: np.ndarray, coarse_projection: np.ndarray) -> np.ndarray:
    if coarse_projection.size == 0:
        return _normalize_rows(sign_projection).astype(np.float32)
    q, _ = np.linalg.qr(coarse_projection.astype(np.float64).T)
    residual = sign_projection.astype(np.float64) - (sign_projection.astype(np.float64) @ q) @ q.T
    return _normalize_rows(residual).astype(np.float32)


def _qjl_layout(budget_bytes: int) -> tuple[int, int]:
    if budget_bytes < 2:
        return budget_bytes, 0
    scalar_bytes = max(1, budget_bytes // 2)
    return scalar_bytes, budget_bytes - scalar_bytes


def _qjl_residual_packet(
    example: Example,
    *,
    encoder: np.ndarray,
    scalar_projection: np.ndarray,
    residual_projection: np.ndarray,
    feature_dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    budget_bytes: int,
    mode: str,
) -> bytes:
    scalar_bytes, sign_bytes = _qjl_layout(budget_bytes)
    predicted = _project_source(example, encoder=encoder, feature_dim=feature_dim, mode=mode)
    coarse = _quantize_scalar(scalar_projection[:scalar_bytes] @ predicted, lo[:scalar_bytes], hi[:scalar_bytes])
    if sign_bytes == 0:
        return coarse
    signs = _packet_from_vector(predicted, residual_projection, sign_bytes)
    return coarse + signs


def _decode_scalar_packet(
    example: Example,
    payload: bytes | None,
    *,
    scalar_projection: np.ndarray,
    feature_dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    candidate_view: str,
) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    decoded = _dequantize_scalar(payload, lo, hi)
    candidate_values = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view) @ scalar_projection[: len(payload)].T
    distances = np.sum((candidate_values - decoded[None, :]) ** 2, axis=1)
    min_distance = float(np.min(distances))
    tied = np.flatnonzero(np.isclose(distances, min_distance, rtol=1e-6, atol=1e-8))
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[int(idx)] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[int(tied[0])]
    return prediction, {"decoder": "scalar_quantized_l2", "min_l2": min_distance, "ties": [int(i) for i in tied.tolist()]}


def _decode_qjl_residual_packet(
    example: Example,
    payload: bytes | None,
    *,
    scalar_projection: np.ndarray,
    residual_projection: np.ndarray,
    feature_dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    budget_bytes: int,
    candidate_view: str,
) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    scalar_bytes, sign_bytes = _qjl_layout(min(budget_bytes, len(payload)))
    scalar_payload = payload[:scalar_bytes]
    sign_payload = payload[scalar_bytes : scalar_bytes + sign_bytes]
    decoded = _dequantize_scalar(scalar_payload, lo[:scalar_bytes], hi[:scalar_bytes])
    candidates = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
    candidate_values = candidates @ scalar_projection[:scalar_bytes].T
    scalar_distances = np.sum((candidate_values - decoded[None, :]) ** 2, axis=1)
    scalar_scale = max(float(np.max(scalar_distances)), 1e-8)
    scores = scalar_distances / scalar_scale
    sign_distance = None
    if sign_bytes:
        bit_count = sign_bytes * 8
        packet_bits = np.unpackbits(np.frombuffer(sign_payload, dtype=np.uint8), bitorder="big")[:bit_count].astype(np.uint8)
        logits = candidates @ residual_projection[:bit_count].T
        candidate_bits = (logits >= 0).astype(np.uint8)
        sign_distances = np.sum(candidate_bits != packet_bits[None, :], axis=1).astype(np.float32)
        scores = scores + sign_distances / max(float(bit_count), 1.0)
        sign_distance = int(np.min(sign_distances))
    min_score = float(np.min(scores))
    tied = np.flatnonzero(np.isclose(scores, min_score, rtol=1e-6, atol=1e-8))
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[int(idx)] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[int(tied[0])]
    return prediction, {
        "decoder": "qjl_residual",
        "scalar_bytes": scalar_bytes,
        "sign_bytes": sign_bytes,
        "min_score": min_score,
        "min_sign_hamming": sign_distance,
        "ties": [int(i) for i in tied.tolist()],
    }


def _raw_source_sign_packet(example: Example, code_projection: np.ndarray, feature_dim: int, budget_bytes: int) -> bytes:
    return _packet_from_vector(_source_vector(example, feature_dim, mode="matched"), code_projection, budget_bytes)


def _answer_slot(example: Example) -> int:
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _constrained_nonself_index(index: int, examples: list[Example]) -> int:
    current = examples[index]
    current_slot = _answer_slot(current)
    n = len(examples)
    for offset in range(1, n):
        candidate_index = (index * 17 + 11 + offset) % n
        candidate = examples[candidate_index]
        if candidate_index != index and candidate.family_name != current.family_name and _answer_slot(candidate) != current_slot:
            return candidate_index
    for offset in range(1, n):
        candidate_index = (index + offset) % n
        if examples[candidate_index].family_name != current.family_name:
            return candidate_index
    return _deterministic_nonself_index(index, n)


def _payload_and_decode(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    code_projection: np.ndarray,
    scalar_projection: np.ndarray,
    residual_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    lo: np.ndarray,
    hi: np.ndarray,
    candidate_view: str,
    rng: random.Random,
) -> tuple[str, bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        prediction, meta = _decode_packet(example, None, code_projection, feature_dim, budget_bytes)
        return prediction, None, meta
    if condition == "matched_learned_syndrome":
        if encoder.shape[0] == feature_dim + 1:
            payload = _source_packet(example, encoder, code_projection, feature_dim, budget_bytes, mode="matched")
        else:
            payload = _packet_from_vector(_project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched"), code_projection, budget_bytes)
        prediction, meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
        return prediction, payload, meta | {"packet_family": "learned_syndrome"}
    if condition == "scalar_quantized_source":
        payload = _scalar_packet(
            example,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="matched",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "candidate_view": candidate_view}
    if condition == "scalar_label_shuffled_ridge":
        payload = _scalar_packet(
            example,
            encoder=label_shuffle_encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="matched",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "source": "label_shuffled_ridge", "candidate_view": candidate_view}
    if condition == "qjl_residual_source":
        payload = _qjl_residual_packet(
            example,
            encoder=encoder,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        prediction, meta = _decode_qjl_residual_packet(
            example,
            payload,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "qjl_residual", "candidate_view": candidate_view}
    if condition == "qjl_label_shuffled_ridge":
        payload = _qjl_residual_packet(
            example,
            encoder=label_shuffle_encoder,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        prediction, meta = _decode_qjl_residual_packet(
            example,
            payload,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "qjl_residual", "source": "label_shuffled_ridge", "candidate_view": candidate_view}
    if condition == "scalar_shuffled_source":
        other = eval_examples[_deterministic_nonself_index(index, len(eval_examples))]
        payload = _scalar_packet(
            other,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="matched",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "source": other.example_id, "candidate_view": candidate_view}
    if condition == "scalar_constrained_shuffled_source":
        other = eval_examples[_constrained_nonself_index(index, eval_examples)]
        payload = _scalar_packet(
            other,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="matched",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "source": other.example_id, "shuffle": "cross_family_slot", "candidate_view": candidate_view}
    if condition == "qjl_constrained_shuffled_source":
        other = eval_examples[_constrained_nonself_index(index, eval_examples)]
        payload = _qjl_residual_packet(
            other,
            encoder=encoder,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        prediction, meta = _decode_qjl_residual_packet(
            example,
            payload,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "qjl_residual", "source": other.example_id, "shuffle": "cross_family_slot", "candidate_view": candidate_view}
    if condition == "scalar_answer_masked_source":
        payload = _scalar_packet(
            example,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="answer_masked",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "source": "answer_masked", "candidate_view": candidate_view}
    if condition == "qjl_answer_masked_source":
        payload = _qjl_residual_packet(
            example,
            encoder=encoder,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            mode="answer_masked",
        )
        prediction, meta = _decode_qjl_residual_packet(
            example,
            payload,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "qjl_residual", "source": "answer_masked", "candidate_view": candidate_view}
    if condition == "qjl_random_same_byte":
        payload = rng.randbytes(budget_bytes)
        prediction, meta = _decode_qjl_residual_packet(
            example,
            payload,
            scalar_projection=scalar_projection,
            residual_projection=residual_projection,
            feature_dim=feature_dim,
            lo=lo,
            hi=hi,
            budget_bytes=budget_bytes,
            candidate_view=candidate_view,
        )
        return prediction, payload, meta | {"packet_family": "qjl_residual", "source": "random", "candidate_view": candidate_view}
    if condition == "raw_source_sign_sketch":
        payload = _raw_source_sign_packet(example, code_projection, feature_dim, budget_bytes)
        prediction, meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
        return prediction, payload, meta | {"packet_family": "raw_source_sign_sketch"}
    if condition == "random_same_byte":
        payload = rng.randbytes(budget_bytes)
        prediction, meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
        return prediction, payload, meta | {"packet_family": "random"}
    raise ValueError(f"unknown condition {condition!r}")


def _predict(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    code_projection: np.ndarray,
    scalar_projection: np.ndarray,
    residual_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    lo: np.ndarray,
    hi: np.ndarray,
    candidate_view: str,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    prediction, payload, metadata = _payload_and_decode(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        code_projection=code_projection,
        scalar_projection=scalar_projection,
        residual_projection=residual_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        lo=lo,
        hi=hi,
        candidate_view=candidate_view,
        rng=rng,
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
        "metadata": metadata,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    budgets: list[int],
    train_seed: int,
    eval_seed: int,
    ridge: float,
    candidate_view: str = "full",
    fit_intercept: bool = True,
    label_shuffle_seed: int | None = None,
    remap_slot_seed: int | None = None,
    packet_variants: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(examples=train_examples, candidates=candidates, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=candidates, seed=eval_seed, family_set=eval_family_set)
    train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_slot_seed)
    eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_slot_seed)
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
        label_shuffle_seed=label_shuffle_seed if label_shuffle_seed is not None else train_seed * 5003 + eval_seed,
    )
    rng_np = np.random.default_rng(train_seed * 3001 + eval_seed)
    max_budget = max(budgets)
    code_projection = _normalize_rows(rng_np.normal(size=(max_budget * 8, feature_dim))).astype(np.float32)
    scalar_projection = _normalize_rows(rng_np.normal(size=(max_budget, feature_dim))).astype(np.float32)
    qjl_sign_projection = _normalize_rows(rng_np.normal(size=(max_budget * 8, feature_dim))).astype(np.float32)
    residual_projection = _orthogonal_residual_projection(qjl_sign_projection, scalar_projection)
    lo, hi = _fit_scalar_calibration(
        train_rows,
        encoder=encoder,
        scalar_projection=scalar_projection,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
    )
    rng = random.Random(train_seed * 4001 + eval_seed)
    packet_variants = list(packet_variants or [])
    conditions = [
        "target_only",
        "matched_learned_syndrome",
        "scalar_quantized_source",
        "scalar_label_shuffled_ridge",
        "scalar_shuffled_source",
        "scalar_constrained_shuffled_source",
        "scalar_answer_masked_source",
        "raw_source_sign_sketch",
        "zero_source",
        "random_same_byte",
    ]
    if "qjl_residual" in packet_variants:
        conditions.extend(
            [
                "qjl_residual_source",
                "qjl_label_shuffled_ridge",
                "qjl_constrained_shuffled_source",
                "qjl_answer_masked_source",
                "qjl_random_same_byte",
            ]
        )
    budget_summaries: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}
    for budget in budgets:
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
        for row_index, example in enumerate(eval_rows):
            for condition in conditions:
                by_condition[condition].append(
                    _predict(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        encoder=encoder,
                        label_shuffle_encoder=label_shuffle_encoder,
                        code_projection=code_projection,
                        scalar_projection=scalar_projection,
                        residual_projection=residual_projection,
                        feature_dim=feature_dim,
                        budget_bytes=budget,
                        lo=lo,
                        hi=hi,
                        candidate_view=candidate_view,
                        rng=rng,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        metrics = {condition: _summarize(rows) for condition, rows in by_condition.items()}
        no_source = max(metrics[name]["accuracy"] for name in ["target_only", "zero_source", "random_same_byte"])
        compression = max(
            metrics[name]["accuracy"]
            for name in ["scalar_quantized_source", "raw_source_sign_sketch"]
        )
        matched = metrics["matched_learned_syndrome"]["accuracy"]
        learned_vs_compression_pass = (
            matched >= no_source + 0.15
            and matched >= compression + 0.02
            and metrics["scalar_shuffled_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["scalar_answer_masked_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
        )
        scalar = metrics["scalar_quantized_source"]["accuracy"]
        qjl = metrics["qjl_residual_source"]["accuracy"] if "qjl_residual_source" in metrics else None
        scalar_controls_ok = (
            metrics["scalar_constrained_shuffled_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["scalar_answer_masked_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["scalar_label_shuffled_ridge"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
        )
        qjl_controls_ok = (
            qjl is not None
            and metrics["qjl_constrained_shuffled_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["qjl_answer_masked_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["qjl_label_shuffled_ridge"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["qjl_random_same_byte"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
        )
        scalar_source_packet_pass = scalar >= no_source + 0.15 and scalar_controls_ok
        qjl_source_packet_pass = qjl is not None and qjl >= no_source + 0.15 and qjl_controls_ok
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for condition in conditions:
                for row in by_condition[condition]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_files[str(budget)] = predictions_path.name
        budget_summaries.append(
            {
                "budget_bytes": budget,
                "pass_gate": learned_vs_compression_pass,
                "learned_vs_compression_pass": learned_vs_compression_pass,
                "scalar_source_packet_pass": scalar_source_packet_pass,
                "matched_accuracy": matched,
                "scalar_quantized_source_accuracy": scalar,
                "qjl_residual_source_accuracy": qjl,
                "target_only_accuracy": metrics["target_only"]["accuracy"],
                "best_no_source_accuracy": no_source,
                "best_compression_baseline_accuracy": compression,
                "matched_minus_best_no_source": matched - no_source,
                "matched_minus_best_compression": matched - compression,
                "scalar_minus_best_no_source": scalar - no_source,
                "qjl_minus_best_no_source": None if qjl is None else qjl - no_source,
                "qjl_minus_scalar": None if qjl is None else qjl - scalar,
                "scalar_controls_ok": scalar_controls_ok,
                "qjl_controls_ok": qjl_controls_ok,
                "qjl_source_packet_pass": qjl_source_packet_pass,
                "metrics": metrics,
            }
        )
    payload = {
        "gate": "source_private_tool_trace_compression_baselines",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "budgets": budgets,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "ridge": ridge,
        "candidate_view": candidate_view,
        "fit_intercept": fit_intercept,
        "packet_variants": packet_variants,
        "label_shuffle_seed": label_shuffle_seed if label_shuffle_seed is not None else train_seed * 5003 + eval_seed,
        "remap_slot_seed": remap_slot_seed,
        "exact_id_parity": len({row.example_id for row in eval_rows}) == len(eval_rows),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "label_shuffle_encoder_sha256": hashlib.sha256(label_shuffle_encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "scalar_projection_sha256": hashlib.sha256(scalar_projection.tobytes()).hexdigest(),
        "qjl_residual_projection_sha256": hashlib.sha256(residual_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["scalar_source_packet_pass"] for row in budget_summaries),
        "pass_rule": "learned syndrome pass: beats target/no-source by >=0.15 and beats best matched-byte compression baseline by >=0.02. Scalar packet pass: scalar quantized source packet beats no-source by >=0.15 and scalar source-destroying controls stay within target_only +0.05. Optional packet variants are reported as comparator rows and do not change the historical pass gate.",
        "prediction_files": prediction_files,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Source-Private Tool-Trace Compression Baselines",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{train_family_set}:{train_examples}` / `{eval_family_set}:{eval_examples}`",
        f"- candidate view: `{candidate_view}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget bytes | Learned > compression | Scalar pass | QJL pass | Syndrome | Scalar | QJL | Target | Best no-source | QJL - scalar |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_summaries:
        qjl_acc = "n/a" if row["qjl_residual_source_accuracy"] is None else f"{row['qjl_residual_source_accuracy']:.3f}"
        qjl_delta = "n/a" if row["qjl_minus_scalar"] is None else f"{row['qjl_minus_scalar']:.3f}"
        lines.append(
            f"| {row['budget_bytes']} | `{row['learned_vs_compression_pass']}` | "
            f"`{row['scalar_source_packet_pass']}` | `{row['qjl_source_packet_pass']}` | "
            f"{row['matched_accuracy']:.3f} | {row['scalar_quantized_source_accuracy']:.3f} | "
            f"{qjl_acc} | {row['target_only_accuracy']:.3f} | "
            f"{row['best_no_source_accuracy']:.3f} | {qjl_delta} |"
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
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
                "# Source-Private Tool-Trace Compression Baselines Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- budgets: `{budgets}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_tool_trace_compression_baselines_20260429"))
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budgets", type=int, nargs="+", default=[6])
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "slot"], default="full")
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=None)
    parser.add_argument("--remap-slot-seed", type=int, default=None)
    parser.add_argument("--packet-variants", choices=["qjl_residual"], nargs="*", default=[])
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
        budgets=args.budgets,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        ridge=args.ridge,
        candidate_view=args.candidate_view,
        fit_intercept=not args.no_intercept,
        label_shuffle_seed=args.label_shuffle_seed,
        remap_slot_seed=args.remap_slot_seed,
        packet_variants=args.packet_variants,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
