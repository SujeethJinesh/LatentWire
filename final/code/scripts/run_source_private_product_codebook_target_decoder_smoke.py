from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import re
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
    ProductCodebook,
    _candidate_matrix_for_view,
    _constrained_nonself_index,
    _fit_product_codebook,
    _fit_ridge_encoder_for_view,
    _product_codebook_packet,
    _project_source,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402
from scripts.run_source_private_tool_trace_target_decoder_smoke import (  # noqa: E402
    _format_prompt,
    _load_model,
)


@dataclass(frozen=True)
class ProductCodebookReceiverState:
    train_rows: list[Example]
    eval_rows: list[Example]
    encoder: np.ndarray
    label_shuffle_encoder: np.ndarray
    codebook: ProductCodebook
    wrong_codebook: ProductCodebook
    feature_dim: int
    budget_bytes: int
    candidate_view: str
    remap_slot_seed: int | None


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remap_candidate_slots(examples: list[Example], *, remap_seed: int | None) -> list[Example]:
    from scripts.run_source_private_tool_trace_compression_baselines import _remap_candidate_slots as remap

    return remap(examples, remap_seed=remap_seed)


def build_receiver_state(
    *,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    budget_bytes: int,
    ridge: float,
    candidate_view: str,
    fit_intercept: bool,
    remap_slot_seed: int | None,
    train_start_index: int = 0,
    eval_start_index: int = 0,
    label_shuffle_seed: int | None = None,
) -> ProductCodebookReceiverState:
    train_rows = make_benchmark(
        examples=train_examples,
        candidates=candidates,
        seed=train_seed,
        family_set=train_family_set,
        start_index=train_start_index,
    )
    eval_rows = make_benchmark(
        examples=eval_examples,
        candidates=candidates,
        seed=eval_seed,
        family_set=eval_family_set,
        start_index=eval_start_index,
    )
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
    codebook = _fit_product_codebook(
        train_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        seed=train_seed * 9001 + eval_seed * 17 + budget_bytes,
    )
    wrong_codebook = _fit_product_codebook(
        train_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        seed=train_seed * 9001 + eval_seed * 17 + budget_bytes + 7919,
    )
    return ProductCodebookReceiverState(
        train_rows=train_rows,
        eval_rows=eval_rows,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        wrong_codebook=wrong_codebook,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        candidate_view=candidate_view,
        remap_slot_seed=remap_slot_seed,
    )


def _candidate_code_signature(example: Example, state: ProductCodebookReceiverState) -> list[tuple[int, ...]]:
    candidates = _candidate_matrix_for_view(example, state.feature_dim, candidate_view=state.candidate_view)
    signatures: list[tuple[int, ...]] = []
    for candidate in candidates:
        codes: list[int] = []
        for sub_centroids, dim_slice in zip(state.codebook.centroids, state.codebook.slices, strict=True):
            part = candidate[dim_slice]
            distances = np.sum((sub_centroids - part[None, :]) ** 2, axis=1)
            codes.append(int(np.argmin(distances)))
        signatures.append(tuple(codes))
    return signatures


def _packet_to_reconstructed_vector(payload: bytes, state: ProductCodebookReceiverState) -> np.ndarray:
    reconstructed = np.zeros(state.feature_dim, dtype=np.float32)
    used_subspaces = min(len(payload), state.codebook.subspaces)
    raw_codes = np.frombuffer(payload[:used_subspaces], dtype=np.uint8)
    for subspace_index, code in enumerate(raw_codes):
        sub_centroids = state.codebook.centroids[subspace_index]
        centroid_index = int(code) % sub_centroids.shape[0]
        reconstructed[state.codebook.slices[subspace_index]] = sub_centroids[centroid_index]
    return reconstructed


def _candidate_distances(example: Example, payload: bytes | None, state: ProductCodebookReceiverState) -> list[float | None]:
    if not payload:
        return [None for _ in example.candidates]
    reconstructed = _packet_to_reconstructed_vector(payload, state)
    candidates = _candidate_matrix_for_view(example, state.feature_dim, candidate_view=state.candidate_view)
    return [float(value) for value in np.sum((candidates - reconstructed[None, :]) ** 2, axis=1)]


def _permute_payload_codes(payload: bytes) -> bytes:
    if len(payload) <= 1:
        return payload
    return payload[1:] + payload[:1]


def _target_derived_payload(example: Example, state: ProductCodebookReceiverState) -> bytes:
    material = f"{example.example_id}|{_prior_prediction(example)}|target-derived-product-codebook".encode("utf-8")
    return hashlib.sha256(material).digest()[: state.codebook.subspaces]


def _prompt_candidate_order(example: Example) -> list[int]:
    digest = hashlib.sha256(f"{example.example_id}|product-codebook-target-decoder-order".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(len(example.candidates)))
    rng.shuffle(order)
    return order


def _choice_labels(example: Example) -> list[str]:
    if len(example.candidates) > 26:
        raise ValueError("candidate choice rendering supports at most 26 candidates")
    return [chr(ord("A") + index) for index in range(len(example.candidates))]


def _choice_for_candidate_label(example: Example, label: str) -> str:
    choices = _choice_labels(example)
    for display_index, candidate_index in enumerate(_prompt_candidate_order(example)):
        if example.candidates[candidate_index].label == label:
            return choices[display_index]
    raise ValueError(f"candidate label {label!r} not found")


def _candidate_label_for_choice(example: Example, choice: str) -> str:
    choices = _choice_labels(example)
    if choice not in choices:
        return ""
    candidate_index = _prompt_candidate_order(example)[choices.index(choice)]
    return example.candidates[candidate_index].label


def _condition_payload(
    *,
    condition: str,
    example: Example,
    state: ProductCodebookReceiverState,
    index: int,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {"packet_kind": "none"}
    if condition == "matched_product_codebook":
        payload = _product_codebook_packet(
            example,
            encoder=state.encoder,
            codebook=state.codebook,
            feature_dim=state.feature_dim,
            mode="matched",
        )
        return payload, {"packet_kind": "product_codebook", "source": "matched"}
    if condition == "label_shuffled_ridge":
        payload = _product_codebook_packet(
            example,
            encoder=state.label_shuffle_encoder,
            codebook=state.codebook,
            feature_dim=state.feature_dim,
            mode="matched",
        )
        return payload, {"packet_kind": "product_codebook", "source": "label_shuffled_ridge"}
    if condition == "constrained_shuffled_source":
        other = state.eval_rows[_constrained_nonself_index(index, state.eval_rows)]
        payload = _product_codebook_packet(
            other,
            encoder=state.encoder,
            codebook=state.codebook,
            feature_dim=state.feature_dim,
            mode="matched",
        )
        return payload, {"packet_kind": "product_codebook", "source": other.example_id, "shuffle": "cross_family_slot"}
    if condition == "answer_masked_source":
        payload = _product_codebook_packet(
            example,
            encoder=state.encoder,
            codebook=state.codebook,
            feature_dim=state.feature_dim,
            mode="answer_masked",
        )
        return payload, {"packet_kind": "product_codebook", "source": "answer_masked"}
    if condition == "permuted_codes":
        payload = _product_codebook_packet(
            example,
            encoder=state.encoder,
            codebook=state.codebook,
            feature_dim=state.feature_dim,
            mode="matched",
        )
        return _permute_payload_codes(payload), {"packet_kind": "product_codebook", "source": "permuted_codes"}
    if condition == "wrong_codebook_packet":
        payload = _product_codebook_packet(
            example,
            encoder=state.encoder,
            codebook=state.wrong_codebook,
            feature_dim=state.feature_dim,
            mode="matched",
        )
        return payload, {"packet_kind": "product_codebook", "source": "wrong_codebook"}
    if condition == "random_same_byte":
        return rng.randbytes(state.codebook.subspaces), {"packet_kind": "random_product_codebook"}
    if condition == "structured_json_same_byte":
        payload = json.dumps({"pq_code": list(_condition_payload(
            condition="matched_product_codebook",
            example=example,
            state=state,
            index=index,
            rng=rng,
        )[0] or b"")}, sort_keys=True).encode("utf-8")[: state.codebook.subspaces]
        return payload, {"packet_kind": "truncated_json_product_codebook"}
    if condition == "structured_free_text_same_byte":
        matched = _condition_payload(
            condition="matched_product_codebook",
            example=example,
            state=state,
            index=index,
            rng=rng,
        )[0] or b""
        payload = f"product codebook packet is {matched.hex()}".encode("utf-8")[: state.codebook.subspaces]
        return payload, {"packet_kind": "truncated_free_text_product_codebook"}
    if condition == "target_derived_sidecar":
        return _target_derived_payload(example, state), {"packet_kind": "target_derived_sidecar"}
    raise ValueError(f"unknown condition {condition!r}")


def _conditions() -> list[str]:
    return [
        "target_only",
        "zero_source",
        "matched_product_codebook",
        "label_shuffled_ridge",
        "constrained_shuffled_source",
        "answer_masked_source",
        "permuted_codes",
        "wrong_codebook_packet",
        "random_same_byte",
        "structured_json_same_byte",
        "structured_free_text_same_byte",
        "target_derived_sidecar",
    ]


def _validate_conditions(conditions: list[str] | None) -> list[str]:
    available = _conditions()
    if not conditions:
        return available
    unknown = sorted(set(conditions) - set(available))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    return list(conditions)


def _format_payload(payload: bytes | None) -> str:
    if not payload:
        return "<NO_SOURCE_PACKET>"
    return "[" + ", ".join(str(int(value)) for value in payload) + f"] (hex={payload.hex()})"


def _prompt_for_product_codebook_decoder(
    example: Example,
    *,
    payload: bytes | None,
    state: ProductCodebookReceiverState,
    candidate_metadata_mode: str,
) -> str:
    signatures = _candidate_code_signature(example, state)
    distances = _candidate_distances(example, payload, state)
    order = _prompt_candidate_order(example)
    choices = _choice_labels(example)
    candidate_lines: list[str] = []
    for display_index, candidate_index in enumerate(order):
        candidate = example.candidates[candidate_index]
        signature = signatures[candidate_index]
        distance = distances[candidate_index]
        fields = [
            f"- {choices[display_index]}",
            f"prior_score={candidate.prior_score:.6f}",
            "pq_signature=[" + ", ".join(str(value) for value in signature) + "]",
        ]
        if candidate_metadata_mode == "distance":
            fields.append("distance_to_packet=<NO_PACKET>" if distance is None else f"distance_to_packet={distance:.8f}")
        candidate_lines.append("; ".join(fields))
    if candidate_metadata_mode == "signature":
        rule = (
            "Choose the candidate whose pq_signature has the most exact byte matches with the source packet. "
            "If no packet is present, or if the best match is tied, choose the candidate with the highest prior_score. "
            "Do not use prior_score when a unique packet match exists."
        )
    elif candidate_metadata_mode == "distance":
        rule = (
            "Choose the candidate with the smallest distance_to_packet value. "
            "If no packet is present, or if the smallest distance is tied, choose the candidate with the highest prior_score. "
            "Do not use prior_score when a unique packet distance winner exists."
        )
    else:
        raise ValueError(f"unknown candidate metadata mode {candidate_metadata_mode!r}")
    return (
        "You are the target-side decoder in a source-private product-codebook handoff.\n"
        "The public question gives blinded candidate choices and public candidate-side PQ signatures.\n"
        "The source packet is a rate-capped list of product-codebook byte indices derived from private evidence.\n"
        f"{rule}\n"
        f"Return only one candidate choice ({', '.join(choices)}) and no explanation.\n\n"
        f"Source packet bytes: {_format_payload(payload)}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\nCandidate choice:"
    )


def _prompt_for_product_codebook_binary_match(
    example: Example,
    *,
    payload: bytes | None,
    state: ProductCodebookReceiverState,
    candidate_metadata_mode: str,
    focus_choice: str,
) -> str:
    if focus_choice not in _choice_labels(example):
        raise ValueError(f"unknown focus choice {focus_choice!r}")
    base_prompt = _prompt_for_product_codebook_decoder(
        example,
        payload=payload,
        state=state,
        candidate_metadata_mode=candidate_metadata_mode,
    ).removesuffix("Candidate choice:")
    if candidate_metadata_mode == "signature":
        rule = "Answer yes only if the focus candidate has the unique strongest exact byte match to the packet."
    elif candidate_metadata_mode == "distance":
        rule = "Answer yes only if the focus candidate has the unique smallest distance_to_packet value."
    else:
        raise ValueError(f"unknown candidate metadata mode {candidate_metadata_mode!r}")
    return (
        base_prompt
        + f"Focus candidate: {focus_choice}\n"
        + f"{rule} If no packet is present or the best packet match is tied, answer yes only for the target-prior candidate.\n"
        + "Return only yes or no.\n\nDoes the focus candidate match the source packet best? Answer:"
    )


def _parse_candidate_choice(generated: str, example: Example) -> str:
    stripped = generated.strip()
    choices = _choice_labels(example)
    for choice in choices:
        if stripped == choice:
            return choice
    for choice in choices:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(choice)}(?![A-Za-z0-9])", stripped):
            return choice
    return ""


def _choice_prediction_from_scores(example: Example, choice_scores: dict[str, float]) -> tuple[str, str]:
    best_score = max(choice_scores.values())
    tied = [
        choice
        for choice, score in choice_scores.items()
        if math.isclose(score, best_score, rel_tol=1e-6, abs_tol=1e-8)
    ]
    prior_choice = _choice_for_candidate_label(example, _prior_prediction(example))
    display_prediction = prior_choice if prior_choice in tied else sorted(tied)[0]
    return display_prediction, _candidate_label_for_choice(example, display_prediction)


def _token_surface_score(tokenizer: Any, logits: Any, surfaces: tuple[str, ...]) -> tuple[float, str]:
    log_probs = logits.log_softmax(dim=-1)
    scored: list[tuple[float, str]] = []
    for surface in surfaces:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            scored.append((float(log_probs[int(ids[0])].item()), surface))
    if not scored:
        raise ValueError(f"could not find a single-token encoding for surfaces {surfaces!r}")
    return max(scored, key=lambda item: item[0])


def _yes_no_token_scores(tokenizer: Any, logits: Any) -> dict[str, Any]:
    yes_score, yes_surface = _token_surface_score(tokenizer, logits, (" yes", " Yes", "yes", "Yes"))
    no_score, no_surface = _token_surface_score(tokenizer, logits, (" no", " No", "no", "No"))
    return {
        "yes_logprob": yes_score,
        "no_logprob": no_score,
        "yes_minus_no": yes_score - no_score,
        "yes_surface": yes_surface,
        "no_surface": no_surface,
    }


def _choice_token_scores(tokenizer: Any, logits: Any, choices: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    surfaces: dict[str, str] = {}
    for choice in choices:
        score, surface = _token_surface_score(tokenizer, logits, (choice, f" {choice}"))
        scores[choice] = score
        surfaces[choice] = surface
    return scores, surfaces


def _binary_prediction_from_scores(
    example: Example,
    binary_scores: list[dict[str, Any]],
    *,
    threshold: float,
) -> tuple[str, str, bool]:
    best_score = max(row["yes_minus_no"] for row in binary_scores)
    prior_choice = _choice_for_candidate_label(example, _prior_prediction(example))
    if best_score <= threshold:
        return prior_choice, _prior_prediction(example), True
    tied = [
        row
        for row in binary_scores
        if math.isclose(row["yes_minus_no"], best_score, rel_tol=1e-6, abs_tol=1e-8)
    ]
    display_prediction = (
        prior_choice
        if any(row["display_choice"] == prior_choice for row in tied)
        else sorted(row["display_choice"] for row in tied)[0]
    )
    return display_prediction, _candidate_label_for_choice(example, display_prediction), False


def _generate_target_predictions(
    state: ProductCodebookReceiverState,
    *,
    model_name: str,
    device: str,
    dtype: str,
    seed: int,
    max_new_tokens: int,
    enable_thinking: bool | None,
    candidate_metadata_mode: str,
    conditions: list[str] | None = None,
    progress_jsonl: pathlib.Path | None = None,
    partial_predictions_jsonl: pathlib.Path | None = None,
    progress_every: int = 16,
    decode_mode: str = "generate",
    binary_fallback_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    import torch

    active_conditions = _validate_conditions(conditions)
    tokenizer, model = _load_model(model_name, device=device, dtype=dtype)
    torch.manual_seed(seed)
    rng = random.Random(seed + 20260430)
    rows: list[dict[str, Any]] = []
    progress_handle = progress_jsonl.open("a", encoding="utf-8") if progress_jsonl is not None else None
    partial_handle = partial_predictions_jsonl.open("a", encoding="utf-8") if partial_predictions_jsonl is not None else None
    for index, example in enumerate(state.eval_rows):
        for condition in active_conditions:
            payload, metadata = _condition_payload(
                condition=condition,
                example=example,
                state=state,
                index=index,
                rng=rng,
            )
            prompt = _prompt_for_product_codebook_decoder(
                example,
                payload=payload,
                state=state,
                candidate_metadata_mode=candidate_metadata_mode,
            )
            text_prompt = _format_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
            inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
            start = time.perf_counter()
            choice_scores: dict[str, float] | None = None
            choice_surfaces: dict[str, str] | None = None
            binary_scores: list[dict[str, Any]] | None = None
            binary_fallback_to_prior = False
            if decode_mode == "candidate_binary_logprob" and payload is None:
                display_prediction = _choice_for_candidate_label(example, _prior_prediction(example))
                prediction = _prior_prediction(example)
                generated = display_prediction
                generated_tokens = 0
                latency_ms = (time.perf_counter() - start) * 1000.0
                binary_scores = []
                binary_fallback_to_prior = True
            elif decode_mode == "candidate_binary_logprob":
                binary_scores = []
                for focus_choice in _choice_labels(example):
                    binary_prompt = _prompt_for_product_codebook_binary_match(
                        example,
                        payload=payload,
                        state=state,
                        candidate_metadata_mode=candidate_metadata_mode,
                        focus_choice=focus_choice,
                    )
                    text_binary_prompt = _format_prompt(tokenizer, binary_prompt, enable_thinking=enable_thinking)
                    binary_inputs = tokenizer(text_binary_prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        binary_output = model(**binary_inputs)
                    binary_scores.append(
                        {
                            "display_choice": focus_choice,
                            "candidate_label": _candidate_label_for_choice(example, focus_choice),
                            **_yes_no_token_scores(tokenizer, binary_output.logits[0, -1]),
                        }
                    )
                latency_ms = (time.perf_counter() - start) * 1000.0
                display_prediction, prediction, binary_fallback_to_prior = _binary_prediction_from_scores(
                    example,
                    binary_scores,
                    threshold=binary_fallback_threshold,
                )
                generated = display_prediction
                generated_tokens = len(binary_scores)
            else:
                with torch.no_grad():
                    if decode_mode == "choice_logprob":
                        output = model(**inputs)
                    else:
                        output = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                latency_ms = (time.perf_counter() - start) * 1000.0
                if decode_mode == "choice_logprob":
                    choice_scores, choice_surfaces = _choice_token_scores(
                        tokenizer,
                        output.logits[0, -1],
                        _choice_labels(example),
                    )
                    display_prediction, prediction = _choice_prediction_from_scores(example, choice_scores)
                    generated = display_prediction
                    generated_tokens = 1
                else:
                    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
                    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    display_prediction = _parse_candidate_choice(generated, example)
                    prediction = _candidate_label_for_choice(example, display_prediction) if display_prediction else ""
                    generated_tokens = len(new_tokens)
            payload_hex = (payload or b"").hex()
            row = {
                "example_id": example.example_id,
                "family_name": example.family_name,
                "condition": condition,
                "answer_label": example.answer_label,
                "target_prior_label": _prior_prediction(example),
                "payload_hex": payload_hex,
                "payload_bytes": len(payload or b""),
                "payload_tokens": _token_count(payload_hex),
                "candidate_metadata_mode": candidate_metadata_mode,
                "decode_mode": decode_mode,
                "binary_fallback_threshold": binary_fallback_threshold
                if decode_mode == "candidate_binary_logprob"
                else None,
                "generated_text": generated,
                "display_prediction": display_prediction,
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "valid_prediction": bool(prediction),
                "latency_ms": latency_ms,
                "generated_tokens": generated_tokens,
                **metadata,
            }
            if choice_scores is not None:
                row["choice_logprobs"] = choice_scores
                row["choice_surfaces"] = choice_surfaces
            if binary_scores is not None:
                row["candidate_binary_logprobs"] = binary_scores
                row["binary_fallback_to_prior"] = binary_fallback_to_prior
            rows.append(row)
            if partial_handle is not None:
                partial_handle.write(json.dumps(row, sort_keys=True) + "\n")
                partial_handle.flush()
        if progress_handle is not None and ((index + 1) % max(progress_every, 1) == 0 or index + 1 == len(state.eval_rows)):
            progress_handle.write(
                json.dumps(
                    {
                        "completed_examples": index + 1,
                        "total_examples": len(state.eval_rows),
                        "rows": len(rows),
                        "conditions": active_conditions,
                        "last_example_id": example.example_id,
                        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            progress_handle.flush()
    if progress_handle is not None:
        progress_handle.close()
    if partial_handle is not None:
        partial_handle.close()
    return rows


def _summarize(rows: list[dict[str, Any]], *, conditions: list[str] | None = None) -> dict[str, Any]:
    conditions = _validate_conditions(conditions)
    example_ids = sorted({row["example_id"] for row in rows})
    metrics: dict[str, Any] = {}
    condition_id_parity: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            raise ValueError(f"missing rows for condition {condition!r}")
        condition_ids = [row["example_id"] for row in condition_rows]
        duplicate_ids = sorted({example_id for example_id in condition_ids if condition_ids.count(example_id) > 1})
        missing_ids = sorted(set(example_ids) - set(condition_ids))
        extra_ids = sorted(set(condition_ids) - set(example_ids))
        condition_id_parity[condition] = {
            "rows": len(condition_rows),
            "unique_ids": len(set(condition_ids)),
            "duplicate_ids": duplicate_ids,
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "passes": len(condition_rows) == len(example_ids)
            and len(set(condition_ids)) == len(example_ids)
            and not duplicate_ids
            and not missing_ids
            and not extra_ids,
        }
        correct = [row["example_id"] for row in condition_rows if row["correct"]]
        metrics[condition] = {
            "correct": len(correct),
            "accuracy": len(correct) / len(condition_rows),
            "correct_ids": correct,
            "valid_prediction_rate": statistics.fmean(float(row["valid_prediction"]) for row in condition_rows),
            "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in condition_rows),
            "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in condition_rows),
            "mean_generated_tokens": statistics.fmean(row["generated_tokens"] for row in condition_rows),
            "p50_latency_ms": statistics.median(row["latency_ms"] for row in condition_rows),
        }
    target = metrics["target_only"]["accuracy"]
    matched = metrics["matched_product_codebook"]["accuracy"]
    control_names = [
        "zero_source",
        "label_shuffled_ridge",
        "constrained_shuffled_source",
        "answer_masked_source",
        "permuted_codes",
        "wrong_codebook_packet",
        "random_same_byte",
        "structured_json_same_byte",
        "structured_free_text_same_byte",
        "target_derived_sidecar",
    ]
    controls = [name for name in control_names if name in metrics]
    best_control_condition = max(controls, key=lambda name: metrics[name]["accuracy"]) if controls else "target_only"
    best_control = metrics[best_control_condition]["accuracy"] if controls else target
    max_valid_gap = max((1.0 - metrics[name]["valid_prediction_rate"] for name in metrics), default=0.0)
    exact_id_parity = len(example_ids) * len(conditions) == len(rows) and all(
        row["passes"] for row in condition_id_parity.values()
    )
    paired = _paired_bootstrap(rows, example_ids=example_ids, best_control_condition=best_control_condition)
    threshold_pass = exact_id_parity and matched - target >= 0.15 and best_control <= target + 0.05 and max_valid_gap <= 0.05
    return {
        "n": len(example_ids),
        "exact_id_count": len(example_ids),
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "conditions": conditions,
        "exact_id_parity": exact_id_parity,
        "condition_id_parity": condition_id_parity,
        "target_only_accuracy": target,
        "matched_accuracy": matched,
        "best_control_condition": best_control_condition,
        "best_control_accuracy": best_control,
        "matched_minus_target": matched - target,
        "matched_minus_best_control": matched - best_control,
        "paired_bootstrap": paired,
        "pass_gate": threshold_pass,
        "strict_ci_pass_gate": threshold_pass
        and paired["matched_vs_target"]["ci95_low"] >= 0.10
        and paired["matched_vs_best_control"]["ci95_low"] >= 0.10,
        "pass_rule": (
            "matched product-codebook target decoder must beat target-only by >=0.15, every source-destroying "
            "or same-byte text/target-derived control must stay within target+0.05, exact-ID parity must hold, "
            "and every condition must keep valid prediction rate >=0.95. strict_ci_pass_gate additionally requires "
            "paired bootstrap CI95 lower bounds >=+0.10 for matched-vs-target and matched-vs-best-control."
        ),
        "metrics": metrics,
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    example_ids: list[str],
    best_control_condition: str,
    samples: int = 2000,
    seed: int = 20260430,
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], {})[row["example_id"]] = bool(row["correct"])

    def delta(condition_a: str, condition_b: str) -> dict[str, float]:
        diffs = np.array(
            [
                float(by_condition.get(condition_a, {}).get(example_id, False))
                - float(by_condition.get(condition_b, {}).get(example_id, False))
                for example_id in example_ids
            ],
            dtype=np.float32,
        )
        point = float(np.mean(diffs)) if len(diffs) else 0.0
        if len(diffs) <= 1:
            return {"point": point, "ci95_low": point, "ci95_high": point}
        rng = np.random.default_rng(seed + sum(ord(ch) for ch in condition_a + condition_b))
        boot = np.empty(samples, dtype=np.float32)
        for sample_index in range(samples):
            indices = rng.integers(0, len(diffs), size=len(diffs))
            boot[sample_index] = float(np.mean(diffs[indices]))
        return {
            "point": point,
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
        }

    return {
        "samples": samples,
        "seed": seed,
        "best_control_condition": best_control_condition,
        "matched_vs_target": delta("matched_product_codebook", "target_only"),
        "matched_vs_best_control": delta("matched_product_codebook", best_control_condition),
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_partial_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Product-Codebook Target-Decoder Smoke",
        "",
        f"- examples: `{summary['n']}`",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- strict CI pass gate: `{summary['strict_ci_pass_gate']}`",
        f"- matched minus target: `{summary['matched_minus_target']:.3f}`",
        f"- matched minus best control: `{summary['matched_minus_best_control']:.3f}`",
        f"- best control condition: `{summary['best_control_condition']}`",
        f"- matched vs target CI95: `[{summary['paired_bootstrap']['matched_vs_target']['ci95_low']:.3f}, "
        f"{summary['paired_bootstrap']['matched_vs_target']['ci95_high']:.3f}]`",
        f"- matched vs best control CI95: `[{summary['paired_bootstrap']['matched_vs_best_control']['ci95_low']:.3f}, "
        f"{summary['paired_bootstrap']['matched_vs_best_control']['ci95_high']:.3f}]`",
        "",
        "| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["metrics"].items():
        lines.append(
            "| "
            f"{condition} | {metrics['correct']}/{summary['n']} | "
            f"{metrics['accuracy']:.3f} | {metrics['valid_prediction_rate']:.3f} | "
            f"{metrics['mean_payload_bytes']:.2f} | {metrics['mean_generated_tokens']:.2f} | "
            f"{metrics['p50_latency_ms']:.2f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Source-Private Product-Codebook Target-Decoder Smoke Manifest",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Outcome",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- strict CI pass gate: `{summary['strict_ci_pass_gate']}`",
        f"- examples: `{summary['n']}`",
        f"- matched accuracy: `{summary['matched_accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best control accuracy: `{summary['best_control_accuracy']:.3f}`",
        f"- train/eval ID overlap count: `{manifest.get('train_eval_id_overlap_count', 'n/a')}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest["artifacts"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--train-start-index", type=int, default=0)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--train-family-set", default="all")
    parser.add_argument("--eval-family-set", default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", default="slot")
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remap-slot-seed", type=int, default=101)
    parser.add_argument("--label-shuffle-seed", type=int, default=None)
    parser.add_argument("--candidate-metadata-mode", choices=["signature", "distance"], default="signature")
    parser.add_argument("--decode-mode", choices=["generate", "choice_logprob", "candidate_binary_logprob"], default="generate")
    parser.add_argument("--binary-fallback-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--conditions", choices=_conditions(), nargs="*", default=None)
    parser.add_argument("--progress-jsonl", type=pathlib.Path, default=None)
    parser.add_argument("--partial-predictions-jsonl", type=pathlib.Path, default=None)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = args.progress_jsonl if args.progress_jsonl is None or args.progress_jsonl.is_absolute() else ROOT / args.progress_jsonl
    if progress_jsonl is not None:
        progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    partial_predictions_jsonl = (
        args.partial_predictions_jsonl
        if args.partial_predictions_jsonl is None or args.partial_predictions_jsonl.is_absolute()
        else ROOT / args.partial_predictions_jsonl
    )
    if partial_predictions_jsonl is not None:
        partial_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    conditions = _validate_conditions(args.conditions)
    state = build_receiver_state(
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        budget_bytes=args.budget_bytes,
        ridge=args.ridge,
        candidate_view=args.candidate_view,
        fit_intercept=args.fit_intercept,
        remap_slot_seed=args.remap_slot_seed,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    rows = _generate_target_predictions(
        state,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        candidate_metadata_mode=args.candidate_metadata_mode,
        conditions=conditions,
        progress_jsonl=progress_jsonl,
        partial_predictions_jsonl=partial_predictions_jsonl,
        progress_every=args.progress_every,
        decode_mode=args.decode_mode,
        binary_fallback_threshold=args.binary_fallback_threshold,
    )
    if partial_predictions_jsonl is not None:
        partial_rows = _read_partial_jsonl(partial_predictions_jsonl)
        if len(partial_rows) < len(rows):
            raise RuntimeError(
                f"partial prediction log {partial_predictions_jsonl} has {len(partial_rows)} rows but final run produced {len(rows)}"
            )
    summary = _summarize(rows, conditions=conditions)
    _write_jsonl(output_dir / "target_predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    artifacts = ["target_predictions.jsonl", "summary.json", "summary.md", "manifest.json", "manifest.md"]
    if partial_predictions_jsonl is not None and partial_predictions_jsonl.parent.resolve() == output_dir.resolve():
        artifacts.insert(1, partial_predictions_jsonl.name)
    train_eval_id_overlap = sorted({row.example_id for row in state.train_rows} & {row.example_id for row in state.eval_rows})
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_product_codebook_target_decoder_smoke.py",
                f"--output-dir {args.output_dir}",
                f"--model {args.model}",
                f"--device {args.device}",
                f"--dtype {args.dtype}",
                f"--train-examples {args.train_examples}",
                f"--eval-examples {args.eval_examples}",
                f"--train-seed {args.train_seed}",
                f"--eval-seed {args.eval_seed}",
                f"--train-start-index {args.train_start_index}",
                f"--eval-start-index {args.eval_start_index}",
                f"--train-family-set {args.train_family_set}",
                f"--eval-family-set {args.eval_family_set}",
                f"--candidates {args.candidates}",
                f"--feature-dim {args.feature_dim}",
                f"--budget-bytes {args.budget_bytes}",
                f"--ridge {args.ridge}",
                f"--candidate-view {args.candidate_view}",
                "--fit-intercept" if args.fit_intercept else "--no-fit-intercept",
                f"--remap-slot-seed {args.remap_slot_seed}",
                "" if args.label_shuffle_seed is None else f"--label-shuffle-seed {args.label_shuffle_seed}",
                f"--candidate-metadata-mode {args.candidate_metadata_mode}",
                f"--decode-mode {args.decode_mode}",
                f"--binary-fallback-threshold {args.binary_fallback_threshold}",
                f"--seed {args.seed}",
                f"--max-new-tokens {args.max_new_tokens}",
                "--no-enable-thinking" if args.enable_thinking is False else "--enable-thinking",
                "" if args.conditions is None else "--conditions " + " ".join(args.conditions),
                "" if args.progress_jsonl is None else f"--progress-jsonl {args.progress_jsonl}",
                ""
                if args.partial_predictions_jsonl is None
                else f"--partial-predictions-jsonl {args.partial_predictions_jsonl}",
                f"--progress-every {args.progress_every}",
                "--require-pass" if args.require_pass else "--no-require-pass",
            ]
        ),
        "args": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "progress_jsonl": None if args.progress_jsonl is None else str(args.progress_jsonl),
            "partial_predictions_jsonl": None if partial_predictions_jsonl is None else str(partial_predictions_jsonl),
            "do_sample": False,
        },
        "train_eval_id_overlap": train_eval_id_overlap,
        "train_eval_id_overlap_count": len(train_eval_id_overlap),
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "python": sys.version,
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if args.require_pass and not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
