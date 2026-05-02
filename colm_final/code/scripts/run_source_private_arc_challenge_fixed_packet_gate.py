from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import re
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_bridge_contract as arc_contract  # noqa: E402
from scripts import run_source_private_learned_synonym_dictionary_packet_gate as syn  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_fixed_packet_gate_20260501")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)

MATCHED_CONDITION = "matched_source_private_packet"
STRICT_DESTRUCTIVE_CONTROLS = (
    "zero_source",
    "shuffled_source_packet",
    "random_same_byte_packet",
    "target_derived_sidecar",
    "candidate_derangement",
)
REPORT_CONDITIONS = (
    "target_only",
    MATCHED_CONDITION,
    *STRICT_DESTRUCTIVE_CONTROLS,
    "label_permutation",
    "same_byte_structured_text",
    "answer_only_text_forbidden_oracle",
)
FORBIDDEN_SOURCE_KEYS = ("answer", "answerKey", "answer_index", "answer_label", "gold")


@dataclasses.dataclass(frozen=True)
class ArcRow:
    row_id: str
    content_id: str
    question: str
    choices: tuple[str, ...]
    choice_labels: tuple[str, ...]
    answer_index: int
    answer_label: str
    source_name: str = ""


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _peak_rss_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _rounded_cache_bytes(record_bytes: int, *, granularity: int, batch_size: int = 1) -> float:
    total = int(record_bytes) * int(batch_size)
    rounded = int(math.ceil(total / granularity) * granularity)
    return float(rounded / batch_size)


def _load_rows(path: pathlib.Path, *, limit: int | None = None) -> list[ArcRow]:
    rows: list[ArcRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if {"content_id", "choice_labels", "answer_index", "answer_label"} <= set(raw):
                canonical = raw
            else:
                canonical = arc_contract.canonical_arc_row(raw, source_name=_display_path(path), row_index=row_index)
            rows.append(
                ArcRow(
                    row_id=str(canonical.get("id") or canonical["content_id"][:16]),
                    content_id=str(canonical["content_id"]),
                    question=str(canonical["question"]),
                    choices=tuple(str(choice) for choice in canonical["choices"]),
                    choice_labels=tuple(str(label) for label in canonical["choice_labels"]),
                    answer_index=int(canonical["answer_index"]),
                    answer_label=str(canonical["answer_label"]),
                    source_name=str(canonical.get("source_name", "")),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path} contained no ARC rows")
    return rows


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _stable_index(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def _hashed_features(texts: list[str], *, dim: int) -> np.ndarray:
    rows = np.zeros((len(texts), dim), dtype=np.float64)
    for row_index, text in enumerate(texts):
        lowered = text.lower()
        words = _tokens(lowered)
        for token in words:
            rows[row_index, _stable_index(f"w:{token}", dim)] += 1.0
        for left, right in zip(words, words[1:]):
            rows[row_index, _stable_index(f"b:{left}_{right}", dim)] += 0.75
        compact = re.sub(r"\s+", " ", lowered)
        for ngram in (3, 4, 5):
            if len(compact) < ngram:
                continue
            for offset in range(len(compact) - ngram + 1):
                rows[row_index, _stable_index(f"c{ngram}:{compact[offset:offset + ngram]}", dim)] += 0.15
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, np.maximum(norms, 1e-12), out=np.zeros_like(rows), where=norms > 0)


def _select_anchor_texts(anchor_texts: list[str], *, dim: int) -> list[str]:
    unique = sorted(set(anchor_texts), key=lambda text: _sha256_text(f"anchor:{text}"))
    if not unique:
        raise ValueError("anchor-relative features require at least one anchor text")
    return unique[: min(dim, len(unique))]


def _anchor_relative_hashed_features(
    texts: list[str],
    *,
    anchor_texts: list[str],
    dim: int,
) -> np.ndarray:
    anchors = _select_anchor_texts(anchor_texts, dim=dim)
    return _anchor_relative_hashed_features_from_anchors(texts, anchors=anchors, dim=dim)


def _anchor_relative_hashed_features_from_anchors(
    texts: list[str],
    *,
    anchors: list[str],
    dim: int,
) -> np.ndarray:
    if not anchors:
        raise ValueError("anchor-relative features require at least one anchor text")
    anchors = anchors[:dim]
    base_dim = max(512, dim * 2)
    text_features = _hashed_features(texts, dim=base_dim)
    anchor_features = _hashed_features(anchors, dim=base_dim)
    similarities = text_features @ anchor_features.T
    if len(anchors) < dim:
        similarities = np.pad(similarities, ((0, 0), (0, dim - len(anchors))), mode="constant")
    similarities = similarities[:, :dim].astype(np.float64, copy=False)
    means = similarities.mean(axis=1, keepdims=True)
    centered = similarities - means
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, np.maximum(norms, 1e-12), out=np.zeros_like(centered), where=norms > 0)


def _features(
    texts: list[str],
    *,
    dim: int,
    feature_mode: str,
    feature_model: str,
    feature_device: str,
    feature_dtype: str,
    feature_max_length: int,
    local_files_only: bool,
    anchor_texts: list[str] | None = None,
) -> np.ndarray:
    if feature_mode == "hashed":
        return _hashed_features(texts, dim=dim)
    if feature_mode == "anchor_relative_hashed":
        if anchor_texts is None:
            raise ValueError("anchor_relative_hashed requires anchor_texts")
        return _anchor_relative_hashed_features(texts, anchor_texts=anchor_texts, dim=dim)
    if feature_mode not in {"hf_last_mean", "hf_mid_last_mean"}:
        raise ValueError(f"unknown feature mode {feature_mode!r}")
    syn._HF_FEATURE_MODEL = feature_model
    syn._HF_FEATURE_DEVICE = feature_device
    syn._HF_FEATURE_DTYPE = feature_dtype
    syn._HF_FEATURE_MAX_LENGTH = feature_max_length
    syn._HF_FEATURE_LOCAL_FILES_ONLY = local_files_only
    return syn._hf_text_features(texts, dim=dim, text_feature_mode=feature_mode)


def _pair_text(row: ArcRow, choice: str) -> str:
    return f"Question: {row.question}\nCandidate answer: {choice}"


def _choice_pair_texts(rows: list[ArcRow]) -> list[str]:
    return [_pair_text(row, choice) for row in rows for choice in row.choices]


def _row_offsets(rows: list[ArcRow]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for row in rows:
        end = start + len(row.choices)
        offsets.append((start, end))
        start = end
    return offsets


def _fit_ridge_pair_scorer(
    train_rows: list[ArcRow],
    pair_features: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    labels: list[float] = []
    for row in train_rows:
        labels.extend(1.0 if index == row.answer_index else -1.0 for index, _ in enumerate(row.choices))
    x = np.concatenate([np.ones((pair_features.shape[0], 1), dtype=np.float64), pair_features], axis=1)
    y = np.asarray(labels, dtype=np.float64)
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    xtx[0, 0] -= float(ridge)
    weights = np.linalg.solve(xtx, x.T @ y)
    train_scores = x @ weights
    return {
        "weights": weights,
        "ridge": float(ridge),
        "train_pair_rows": int(pair_features.shape[0]),
        "train_positive_rows": int(sum(value > 0 for value in labels)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
    }


def _score_rows(
    rows: list[ArcRow],
    pair_features: np.ndarray,
    scorer: dict[str, Any],
) -> tuple[list[list[float]], list[int]]:
    weights = np.asarray(scorer["weights"], dtype=np.float64)
    x = np.concatenate([np.ones((pair_features.shape[0], 1), dtype=np.float64), pair_features], axis=1)
    flat_scores = x @ weights
    scores_by_row: list[list[float]] = []
    predictions: list[int] = []
    for start, end in _row_offsets(rows):
        scores = [float(value) for value in flat_scores[start:end]]
        best = max(range(len(scores)), key=lambda index: (scores[index], -index))
        scores_by_row.append(scores)
        predictions.append(int(best))
    return scores_by_row, predictions


def _torch_dtype(dtype: str) -> Any:
    import torch

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unknown torch dtype {dtype!r}")


def _lm_choice_prompt(row: ArcRow, *, prompt_mode: str = "qa") -> str:
    if prompt_mode == "qa":
        return f"Answer the science question with the best answer.\nQuestion: {row.question}\nAnswer:"
    if prompt_mode == "continuation":
        return f"Choose the most plausible continuation.\nContext: {row.question}\nContinuation:"
    if prompt_mode == "generic_mcq":
        return f"Choose the best option.\nQuestion: {row.question}\nAnswer:"
    raise ValueError(f"unknown LM choice prompt mode {prompt_mode!r}")


def _lm_choice_loglikelihood_scores(
    rows: list[ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    local_files_only: bool,
    normalization: str,
    prompt_mode: str = "qa",
    attn_implementation: str | None = None,
    choice_batch_size: int | None = None,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if normalization not in {"mean", "sum"}:
        raise ValueError(f"unknown LM score normalization {normalization!r}")
    if choice_batch_size is not None and choice_batch_size <= 0:
        raise ValueError("choice_batch_size must be positive when provided")
    resolved_device = "cpu" if device == "auto_cpu" else syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if (
        isinstance(getattr(config, "rope_scaling", None), dict)
        and config.rope_scaling.get("rope_type") == "default"
        and "type" not in config.rope_scaling
    ):
        # Older remote model code, notably cached Phi-3, expects no RoPE scaling
        # for default RoPE while newer Transformers normalizes it to a dict.
        config.rope_scaling = None
    model_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(dtype),
    }
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(resolved_device)
    model.eval()

    def _score_text_batch(texts: list[str], *, prompt_len: int) -> list[float]:
        padding_mode: bool | str = True
        tokenizer_max_length = max_length
        if str(resolved_device).startswith("mps"):
            raw_lengths = tokenizer(texts, padding=False, truncation=True, max_length=max_length)["input_ids"]
            row_max_length = max(len(input_ids) for input_ids in raw_lengths)
            tokenizer_max_length = min(max_length, int(math.ceil(row_max_length / 32.0) * 32))
            padding_mode = "max_length"
        encoded = tokenizer(
            texts,
            padding=padding_mode,
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=False)
        logits = output.logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        attention = encoded["attention_mask"][:, 1:].bool()
        batch_scores: list[float] = []
        for batch_index in range(len(texts)):
            valid = attention[batch_index].clone()
            valid[: max(0, prompt_len - 1)] = False
            values = token_logp[batch_index][valid]
            if values.numel() == 0:
                batch_scores.append(float("-inf"))
            elif normalization == "sum":
                batch_scores.append(float(values.sum().detach().cpu()))
            else:
                batch_scores.append(float(values.mean().detach().cpu()))
        return batch_scores

    scores_by_row: list[list[float]] = []
    predictions: list[int] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            prompt = _lm_choice_prompt(row, prompt_mode=prompt_mode)
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            texts = [prompt + " " + choice for choice in row.choices]
            if choice_batch_size is None or choice_batch_size >= len(texts):
                row_scores = _score_text_batch(texts, prompt_len=prompt_len)
            else:
                row_scores = []
                for batch_start in range(0, len(texts), choice_batch_size):
                    row_scores.extend(
                        _score_text_batch(
                            texts[batch_start : batch_start + choice_batch_size],
                            prompt_len=prompt_len,
                        )
                    )
            scores_by_row.append(row_scores)
            predictions.append(int(max(range(len(row_scores)), key=lambda index: (row_scores[index], -index))))
    return scores_by_row, predictions, {
        "kind": "local_causal_lm_choice_loglikelihood",
        "model_path": model_path,
        "device": resolved_device,
        "dtype": dtype,
        "max_length": max_length,
        "normalization": normalization,
        "prompt_mode": prompt_mode,
        "attn_implementation": attn_implementation or "auto",
        "choice_batch_size": choice_batch_size,
        "latency_s": float(time.perf_counter() - start),
    }


def _candidate_residuals(rows: list[ArcRow], pair_features: np.ndarray) -> list[np.ndarray]:
    residuals: list[np.ndarray] = []
    for start, end in _row_offsets(rows):
        matrix = pair_features[start:end].astype(np.float64)
        residuals.append(matrix - matrix.mean(axis=0, keepdims=True))
    return residuals


def _projection_matrix(feature_dim: int, code_dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(max(1, feature_dim)), size=(feature_dim, code_dim))
    col_norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return np.divide(matrix, np.maximum(col_norms, 1e-12), out=np.zeros_like(matrix), where=col_norms > 0)


def _encode_packet(vector: np.ndarray, projection: np.ndarray, *, budget_bytes: int) -> tuple[bytes, dict[str, Any]]:
    if budget_bytes < 2:
        raise ValueError("budget must be at least 2 bytes")
    code = vector @ projection
    if not np.any(code):
        return b"\x00" * budget_bytes, {"packet_nonzero_dims": 0, "packet_code_l2": 0.0, "packet_dims": []}
    slots = max(1, budget_bytes // 2)
    ranked = np.argsort(-np.abs(code))[:slots]
    scale = float(max(abs(float(code[index])) for index in ranked) or 1.0)
    raw = bytearray()
    dims: list[dict[str, Any]] = []
    for index in ranked:
        value = float(code[int(index)])
        encoded_index = int(index) & 0x7F
        if value < 0:
            encoded_index |= 0x80
        magnitude = int(round(min(1.0, abs(value) / scale) * 255.0))
        raw.extend([encoded_index, magnitude])
        dims.append({"index": int(index), "sign": -1 if value < 0 else 1, "magnitude": magnitude})
    while len(raw) < budget_bytes:
        raw.append(0)
    payload = bytes(raw[:budget_bytes])
    return payload, {
        "packet_nonzero_dims": len(dims),
        "packet_code_l2": float(np.linalg.norm(code)),
        "packet_scale": scale,
        "packet_dims": dims,
    }


def _decode_packet(payload: bytes | None, *, code_dim: int) -> np.ndarray:
    code = np.zeros(code_dim, dtype=np.float64)
    if not payload:
        return code
    for offset in range(0, len(payload) - 1, 2):
        raw_index = payload[offset]
        magnitude = payload[offset + 1]
        index = raw_index & 0x7F
        if index >= code_dim or magnitude <= 0:
            continue
        sign = -1.0 if raw_index & 0x80 else 1.0
        code[index] = sign * (float(magnitude) / 255.0)
    return code


def _target_prior_index(row: ArcRow, index_prior: list[float]) -> int:
    valid = list(range(len(row.choices)))
    return max(valid, key=lambda index: (index_prior[index] if index < len(index_prior) else 0.0, -index))


def _index_prior(train_rows: list[ArcRow]) -> list[float]:
    max_choices = max(len(row.choices) for row in train_rows)
    counts = [1.0 for _ in range(max_choices)]
    for row in train_rows:
        counts[row.answer_index] += 1.0
    total = sum(counts)
    return [count / total for count in counts]


def _predict_from_code(
    *,
    row: ArcRow,
    residuals: np.ndarray,
    payload: bytes | None,
    projection: np.ndarray,
    index_prior: list[float],
    derange_candidates: bool = False,
) -> tuple[int, dict[str, Any]]:
    if not payload:
        prior = _target_prior_index(row, index_prior)
        return prior, {"decoder": "index_prior", "scores": [], "payload_dims": 0}
    candidate_code = residuals @ projection
    if derange_candidates and len(row.choices) > 1:
        candidate_code = np.roll(candidate_code, shift=1, axis=0)
    payload_code = _decode_packet(payload, code_dim=projection.shape[1])
    payload_norm = float(np.linalg.norm(payload_code))
    if payload_norm <= 0.0:
        prior = _target_prior_index(row, index_prior)
        return prior, {"decoder": "index_prior_empty_packet", "scores": [], "payload_dims": 0}
    row_norms = np.linalg.norm(candidate_code, axis=1)
    safe_rows = np.divide(
        candidate_code,
        np.maximum(row_norms[:, None], 1e-12),
        out=np.zeros_like(candidate_code),
        where=row_norms[:, None] > 0,
    )
    scores = safe_rows @ (payload_code / payload_norm)
    prediction = int(max(range(len(scores)), key=lambda index: (float(scores[index]), -index)))
    return prediction, {
        "decoder": "sparse_projection_candidate_residual",
        "scores": [float(score) for score in scores],
        "best_score": float(np.max(scores)),
        "payload_dims": int(np.count_nonzero(payload_code)),
    }


def _text_payload_prediction(row: ArcRow, payload: bytes | None, index_prior: list[float]) -> tuple[int, dict[str, Any]]:
    text = (payload or b"").decode("utf-8", errors="ignore").lower()
    if not text.strip():
        return _target_prior_index(row, index_prior), {"decoder": "index_prior_empty_text", "text": text}
    scores = []
    payload_tokens = set(_tokens(text))
    for choice in row.choices:
        lowered = choice.lower()
        choice_tokens = set(_tokens(lowered))
        overlap = len(payload_tokens & choice_tokens)
        substring = 1.0 if text.strip() and text.strip() in lowered else 0.0
        prefix = 0.5 if lowered.startswith(text.strip()) else 0.0
        scores.append(float(overlap + substring + prefix))
    prediction = int(max(range(len(scores)), key=lambda index: (scores[index], -index)))
    return prediction, {"decoder": "same_byte_text_overlap", "scores": scores, "text": text}


def _answer_text_oracle_prediction(row: ArcRow, payload: bytes | None, index_prior: list[float]) -> tuple[int, dict[str, Any]]:
    text = (payload or b"").decode("utf-8", errors="ignore").strip()
    if text in row.choice_labels:
        return row.choice_labels.index(text), {"decoder": "forbidden_answer_label_text", "text": text}
    return _target_prior_index(row, index_prior), {"decoder": "answer_label_parse_failed", "text": text}


def _nonself_index(rows: list[ArcRow], index: int) -> int:
    if len(rows) <= 1:
        return index
    seed = int(hashlib.blake2b(f"{rows[index].content_id}|shuffle".encode("utf-8"), digest_size=8).hexdigest(), 16)
    offset = seed % (len(rows) - 1) + 1
    return (index + offset) % len(rows)


def _permuted_row(row: ArcRow) -> ArcRow:
    rng = random.Random(int(hashlib.sha256(f"{row.content_id}|label-permutation".encode("utf-8")).hexdigest()[:16], 16))
    labels = list(row.choice_labels)
    rng.shuffle(labels)
    if len(labels) > 1 and tuple(labels) == row.choice_labels:
        labels = labels[1:] + labels[:1]
    answer_index = row.answer_index
    return ArcRow(
        row_id=row.row_id,
        content_id=row.content_id,
        question=row.question,
        choices=row.choices,
        choice_labels=tuple(labels),
        answer_index=answer_index,
        answer_label=labels[answer_index],
        source_name=row.source_name,
    )


def _rows_for_predictions(
    *,
    eval_rows: list[ArcRow],
    residuals: list[np.ndarray],
    decode_residuals: list[np.ndarray] | None = None,
    source_predictions: list[int],
    projection: np.ndarray,
    budget_bytes: int,
    index_prior: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    receiver_residuals = residuals if decode_residuals is None else decode_residuals
    if len(receiver_residuals) != len(residuals):
        raise ValueError("decode_residuals must match residual row count")
    packets: list[tuple[bytes, dict[str, Any]]] = []
    for row, row_residuals, selected_index in zip(eval_rows, residuals, source_predictions, strict=True):
        packets.append(_encode_packet(row_residuals[selected_index], projection, budget_bytes=budget_bytes))

    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for row_index, row in enumerate(eval_rows):
        for condition in REPORT_CONDITIONS:
            start = time.perf_counter()
            source_index = int(source_predictions[row_index])
            payload: bytes | None = None
            meta: dict[str, Any] = {
                "source_visible_fields": ["question", "choices"],
                "forbidden_source_fields": list(FORBIDDEN_SOURCE_KEYS),
                "source_selected_index": source_index,
                "source_selected_label": row.choice_labels[source_index],
                "source_selected_choice_sha256": _sha256_text(row.choices[source_index]),
            }
            prediction_index: int
            decode_meta: dict[str, Any]

            if condition == "target_only":
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=None,
                    projection=projection,
                    index_prior=index_prior,
                )
            elif condition == MATCHED_CONDITION:
                payload, packet_meta = packets[row_index]
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta
            elif condition == "zero_source":
                payload, packet_meta = _encode_packet(
                    np.zeros(residuals[row_index].shape[1], dtype=np.float64),
                    projection,
                    budget_bytes=budget_bytes,
                )
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta
            elif condition == "shuffled_source_packet":
                other_index = _nonself_index(eval_rows, row_index)
                payload, packet_meta = packets[other_index]
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"shuffled_source_row_id": eval_rows[other_index].row_id}
            elif condition == "random_same_byte_packet":
                payload = bytes(rng.randrange(256) for _ in range(budget_bytes))
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
            elif condition == "target_derived_sidecar":
                prior = _target_prior_index(row, index_prior)
                payload, packet_meta = _encode_packet(
                    receiver_residuals[row_index][prior],
                    projection,
                    budget_bytes=budget_bytes,
                )
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"target_prior_index": prior}
            elif condition == "candidate_derangement":
                payload, packet_meta = packets[row_index]
                prediction_index, decode_meta = _predict_from_code(
                    row=row,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                    derange_candidates=True,
                )
                meta |= packet_meta
            elif condition == "label_permutation":
                permuted = _permuted_row(row)
                payload, packet_meta = packets[row_index]
                prediction_index, decode_meta = _predict_from_code(
                    row=permuted,
                    residuals=receiver_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"label_permutation": True}
                row_for_eval = permuted
                rows.append(
                    _prediction_row(
                        condition=condition,
                        row=row_for_eval,
                        original_row=row,
                        prediction_index=prediction_index,
                        payload=payload,
                        meta=meta,
                        decode_meta=decode_meta,
                        latency_ms=(time.perf_counter() - start) * 1000.0,
                    )
                )
                continue
            elif condition == "same_byte_structured_text":
                payload = row.choices[source_index].encode("utf-8")[:budget_bytes]
                prediction_index, decode_meta = _text_payload_prediction(row, payload, index_prior)
            elif condition == "answer_only_text_forbidden_oracle":
                payload = row.answer_label.encode("utf-8")[:budget_bytes]
                prediction_index, decode_meta = _answer_text_oracle_prediction(row, payload, index_prior)
            else:
                raise ValueError(f"unknown condition {condition!r}")

            rows.append(
                _prediction_row(
                    condition=condition,
                    row=row,
                    original_row=row,
                    prediction_index=prediction_index,
                    payload=payload,
                    meta=meta,
                    decode_meta=decode_meta,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            )
    return rows


def _prediction_row(
    *,
    condition: str,
    row: ArcRow,
    original_row: ArcRow,
    prediction_index: int,
    payload: bytes | None,
    meta: dict[str, Any],
    decode_meta: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    prediction_label = row.choice_labels[prediction_index]
    correct = prediction_index == row.answer_index
    return {
        "condition": condition,
        "row_id": original_row.row_id,
        "content_id": original_row.content_id,
        "answer_index": row.answer_index,
        "answer_label": row.answer_label,
        "prediction_index": prediction_index,
        "prediction_label": prediction_label,
        "correct": bool(correct),
        "payload_bytes": len(payload or b""),
        "payload_hex": (payload or b"").hex(),
        "latency_ms": float(latency_ms),
        "metadata": {**meta, **decode_meta},
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": 0.0,
            "mean_payload_bytes": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    return {
        "n": len(rows),
        "correct": int(sum(1 for row in rows if row["correct"])),
        "accuracy": float(sum(1 for row in rows if row["correct"]) / len(rows)),
        "mean_payload_bytes": float(statistics.fmean(float(row["payload_bytes"]) for row in rows)),
        "p50_latency_ms": float(statistics.median(latencies)),
        "p95_latency_ms": float(latencies[max(0, int(0.95 * len(latencies)) - 1)]),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    baseline: str,
    seed: int,
    samples: int,
) -> dict[str, float]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    deltas = [
        float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"])
        for _, conditions in sorted(by_id.items())
        if condition in conditions and baseline in conditions
    ]
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        condition: _summarize([row for row in rows if row["condition"] == condition])
        for condition in REPORT_CONDITIONS
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private ARC-Challenge Fixed-Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- public benchmark result ready: `{payload['public_benchmark_result_ready']}`",
        f"- train/eval rows: `{payload['train_rows']}` / `{payload['eval_rows']}`",
        f"- packet budget: `{payload['budget_bytes']}B`",
        f"- feature mode: `{payload['feature_mode']}`",
        "",
        "| Condition | Accuracy | Correct / N | Mean bytes |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in payload["condition_metrics"].items():
        lines.append(
            f"| `{condition}` | {metrics['accuracy']:.3f} | {metrics['correct']} / {metrics['n']} | "
            f"{metrics['mean_payload_bytes']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Readout",
            "",
            f"- best destructive control: `{payload['headline']['best_destructive_control']}`",
            f"- matched minus target: `{payload['headline']['matched_minus_target']:.3f}`",
            f"- matched minus best destructive control: `{payload['headline']['matched_minus_best_destructive']:.3f}`",
            f"- matched minus same-byte structured text: `{payload['headline']['matched_minus_same_byte_text']:.3f}`",
            f"- paired CI95 vs target: `{payload['headline']['paired_ci95_vs_target']}`",
            "",
            "This is a public ARC bridge gate. It is only paper-positive if the fixed packet beats target-only, "
            "the destructive controls, and the same-byte text comparator without using answer fields at eval time.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    train_limit: int | None,
    eval_limit: int | None,
    budget_bytes: int,
    feature_dim: int,
    code_dim: int,
    feature_mode: str,
    feature_model: str,
    feature_device: str,
    feature_dtype: str,
    feature_max_length: int,
    local_files_only: bool,
    source_score_mode: str,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str = "qa",
    ridge: float,
    seed: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    phase_timings_s: dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train_rows = _load_rows(train_path, limit=train_limit)
    eval_rows = _load_rows(eval_path, limit=eval_limit)
    anchor_texts = _choice_pair_texts(train_rows)
    phase_timings_s["load_rows"] = float(time.perf_counter() - start)
    overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in eval_rows})

    start = time.perf_counter()
    eval_pair_features = _features(
        _choice_pair_texts(eval_rows),
        dim=feature_dim,
        feature_mode=feature_mode,
        feature_model=feature_model,
        feature_device=feature_device,
        feature_dtype=feature_dtype,
        feature_max_length=feature_max_length,
        local_files_only=local_files_only,
        anchor_texts=anchor_texts,
    )
    phase_timings_s["eval_pair_features"] = float(time.perf_counter() - start)
    if source_score_mode == "pair_ridge":
        start = time.perf_counter()
        train_pair_features = _features(
            _choice_pair_texts(train_rows),
            dim=feature_dim,
            feature_mode=feature_mode,
            feature_model=feature_model,
            feature_device=feature_device,
            feature_dtype=feature_dtype,
            feature_max_length=feature_max_length,
            local_files_only=local_files_only,
            anchor_texts=anchor_texts,
        )
        phase_timings_s["train_pair_features"] = float(time.perf_counter() - start)
        start = time.perf_counter()
        scorer = _fit_ridge_pair_scorer(train_rows, train_pair_features, ridge=ridge)
        source_scores, source_predictions = _score_rows(eval_rows, eval_pair_features, scorer)
        phase_timings_s["source_scoring"] = float(time.perf_counter() - start)
        source_model_state = {
            "kind": "train_split_ridge_pair_scorer",
            "train_state": {key: value for key, value in scorer.items() if key != "weights"},
        }
    elif source_score_mode == "lm_choice_loglikelihood":
        phase_timings_s["train_pair_features"] = 0.0
        start = time.perf_counter()
        source_scores, source_predictions, lm_state = _lm_choice_loglikelihood_scores(
            eval_rows,
            model_path=source_lm_model,
            device=source_lm_device,
            dtype=source_lm_dtype,
            max_length=source_lm_max_length,
            local_files_only=local_files_only,
            normalization=source_lm_normalization,
            prompt_mode=source_lm_prompt_mode,
        )
        phase_timings_s["source_scoring"] = float(time.perf_counter() - start)
        source_model_state = lm_state
    else:
        raise ValueError(f"unknown source_score_mode {source_score_mode!r}")
    start = time.perf_counter()
    residuals = _candidate_residuals(eval_rows, eval_pair_features)
    phase_timings_s["candidate_residuals"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    projection = _projection_matrix(feature_dim, code_dim, seed=seed + 171)
    phase_timings_s["projection_matrix"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    priors = _index_prior(train_rows)
    phase_timings_s["index_prior"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    prediction_rows = _rows_for_predictions(
        eval_rows=eval_rows,
        residuals=residuals,
        source_predictions=source_predictions,
        projection=projection,
        budget_bytes=budget_bytes,
        index_prior=priors,
        seed=seed + 911,
    )
    phase_timings_s["packet_encode_decode_all_conditions"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    phase_timings_s["write_predictions"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    metrics = _condition_metrics(prediction_rows)
    matched = metrics[MATCHED_CONDITION]["accuracy"]
    target = metrics["target_only"]["accuracy"]
    same_byte_text = metrics["same_byte_structured_text"]["accuracy"]
    best_control_name = max(STRICT_DESTRUCTIVE_CONTROLS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    target_ci = _paired_bootstrap(
        prediction_rows,
        condition=MATCHED_CONDITION,
        baseline="target_only",
        seed=seed + 1001,
        samples=bootstrap_samples,
    )
    control_ci = _paired_bootstrap(
        prediction_rows,
        condition=MATCHED_CONDITION,
        baseline=best_control_name,
        seed=seed + 1002,
        samples=bootstrap_samples,
    )
    phase_timings_s["metrics_and_bootstrap"] = float(time.perf_counter() - start)
    source_accuracy = float(
        sum(int(prediction == row.answer_index) for prediction, row in zip(source_predictions, eval_rows, strict=True))
        / len(eval_rows)
    )
    train_candidate_pairs = sum(len(row.choices) for row in train_rows)
    eval_candidate_pairs = sum(len(row.choices) for row in eval_rows)
    record_bytes = budget_bytes + 3
    phase_timings_s["total_before_artifact_write"] = float(time.perf_counter() - total_start)
    pass_gate = (
        not overlap
        and matched >= target + min_lift_over_target
        and matched >= best_control + min_gap_over_control
        and matched >= same_byte_text + min_gap_over_text
        and target_ci["ci95_low"] > 0.0
        and metrics["candidate_derangement"]["accuracy"] <= target + 0.05
    )
    payload = {
        "gate": "source_private_arc_challenge_fixed_packet_gate",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "eval_path": _display_path(eval_path),
        "train_sha256": _sha256_file(train_path),
        "eval_sha256": _sha256_file(eval_path),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_eval_content_overlap_count": len(overlap),
        "train_eval_content_overlap_sha256": hashlib.sha256("\n".join(overlap).encode("utf-8")).hexdigest(),
        "budget_bytes": budget_bytes,
        "feature_dim": feature_dim,
        "code_dim": code_dim,
        "feature_mode": feature_mode,
        "feature_model": feature_model if feature_mode.startswith("hf_") else None,
        "feature_device": syn._resolve_torch_device(feature_device) if feature_mode.startswith("hf_") else None,
        "feature_dtype": feature_dtype if feature_mode.startswith("hf_") else None,
        "feature_max_length": feature_max_length if feature_mode.startswith("hf_") else None,
        "anchor_relative_basis": (
            {
                "anchor_source": "train split question/candidate texts",
                "anchor_count": min(feature_dim, len(set(anchor_texts))),
                "base_feature_mode": "hashed",
                "basis_claim": "public anchor-relative coordinate chart plus downstream random projection",
            }
            if feature_mode == "anchor_relative_hashed"
            else None
        ),
        "ridge": ridge,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "source_model": {
            **source_model_state,
            "source_score_mode": source_score_mode,
            "source_eval_accuracy_before_packet": source_accuracy,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(FORBIDDEN_SOURCE_KEYS),
            "source_score_digest": hashlib.sha256(
                json.dumps(source_scores, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "method_contract": {
            "fixed_packet_budget_bytes": budget_bytes,
            "packet_format": "top-k signed random-projection residual sketch; two bytes per selected dimension",
            "source_packet_inputs_at_eval": ["question", "choices"],
            "forbidden_eval_source_inputs": list(FORBIDDEN_SOURCE_KEYS),
            "target_side_information": "candidate answer texts and public projection matrix",
            "claim_boundary": (
                "Mac-local public benchmark bridge. This is not a native GPU systems result and not a "
                "full ICLR benchmark suite."
            ),
        },
        "systems_trace": {
            "measurement_scope": "Mac-local Python/NumPy/PyTorch phase trace; not a native GPU serving measurement.",
            "phase_timings_s": phase_timings_s,
            "peak_rss_mib": _peak_rss_mib(),
            "train_candidate_pairs": train_candidate_pairs,
            "eval_candidate_pairs": eval_candidate_pairs,
            "feature_cache_bytes_eval_float64": int(eval_candidate_pairs * feature_dim * 8),
            "feature_cache_bytes_eval_float32_floor": int(eval_candidate_pairs * feature_dim * 4),
            "projection_matrix_bytes_float64": int(feature_dim * code_dim * 8),
            "raw_payload_bytes_per_request": int(budget_bytes),
            "record_bytes_with_header_crc": int(record_bytes),
            "single_request_cacheline_bytes": _rounded_cache_bytes(record_bytes, granularity=64),
            "single_request_dma_bytes": _rounded_cache_bytes(record_bytes, granularity=128),
            "batch64_cacheline_bytes_per_request": _rounded_cache_bytes(record_bytes, granularity=64, batch_size=64),
            "batch64_dma_bytes_per_request": _rounded_cache_bytes(record_bytes, granularity=128, batch_size=64),
            "source_private": True,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "native_metrics_pending": [
                "vLLM TTFT",
                "vLLM TPOT",
                "SLO goodput",
                "GPU peak memory",
                "HBM read/write bytes",
                "native C2C/KVComm/TurboQuant/QJL baseline rows",
            ],
        },
        "condition_metrics": metrics,
        "headline": {
            "matched_accuracy": matched,
            "target_accuracy": target,
            "same_byte_structured_text_accuracy": same_byte_text,
            "best_destructive_control": best_control_name,
            "best_destructive_control_accuracy": best_control,
            "matched_minus_target": matched - target,
            "matched_minus_best_destructive": matched - best_control,
            "matched_minus_same_byte_text": matched - same_byte_text,
            "paired_ci95_vs_target": target_ci,
            "paired_ci95_vs_best_destructive": control_ci,
        },
        "pass_gate": bool(pass_gate),
        "public_benchmark_result_ready": bool(pass_gate),
        "pass_rule": (
            "Pass requires no train/eval content overlap; fixed packet beats target-only, the best strict "
            "destructive control, and same-byte structured text by configured margins; paired CI95 lower "
            "bound versus target is positive; candidate derangement remains within target+0.05."
        ),
    }
    (output_dir / "arc_challenge_fixed_packet_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "arc_challenge_fixed_packet_gate.md", payload)
    manifest = {
        "artifacts": [
            "arc_challenge_fixed_packet_gate.json",
            "arc_challenge_fixed_packet_gate.md",
            "predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "arc_challenge_fixed_packet_gate.json",
                "arc_challenge_fixed_packet_gate.md",
                "predictions.jsonl",
            )
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private ARC-Challenge Fixed-Packet Gate Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- public benchmark result ready: `{payload['public_benchmark_result_ready']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument(
        "--feature-mode",
        choices=("hashed", "anchor_relative_hashed", "hf_last_mean", "hf_mid_last_mean"),
        default="hashed",
    )
    parser.add_argument("--feature-model", default="BAAI/bge-small-en")
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--feature-dtype", default="float32")
    parser.add_argument("--feature-max-length", type=int, default=128)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--source-score-mode", choices=("pair_ridge", "lm_choice_loglikelihood"), default="pair_ridge")
    parser.add_argument(
        "--source-lm-model",
        default="/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    )
    parser.add_argument(
        "--source-lm-device",
        default="auto_cpu",
        help="Use auto_cpu to force CPU; Qwen attention can be unstable on some MPS stacks.",
    )
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="qa")
    parser.add_argument("--ridge", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lift-over-target", type=float, default=0.03)
    parser.add_argument("--min-gap-over-control", type=float, default=0.03)
    parser.add_argument("--min-gap-over-text", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_gate(
        output_dir=args.output_dir,
        train_path=_resolve(args.train_path),
        eval_path=_resolve(args.eval_path),
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        budget_bytes=args.budget_bytes,
        feature_dim=args.feature_dim,
        code_dim=args.code_dim,
        feature_mode=args.feature_mode,
        feature_model=args.feature_model,
        feature_device=args.feature_device,
        feature_dtype=args.feature_dtype,
        feature_max_length=args.feature_max_length,
        local_files_only=not args.allow_downloads,
        source_score_mode=args.source_score_mode,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        ridge=args.ridge,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
    )


if __name__ == "__main__":
    main()
