from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _deterministic_nonself_index,
    _mask_log_components,
    _mask_repair_diag,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_shared_sparse_crosscoder_packet_gate import (  # noqa: E402
    ATOM_ORDER,
    ATOM_TO_ID,
    ID_TO_ATOM,
    SOURCE_DESTROYING_CONTROLS,
    _answer_index,
    _candidate_atoms,
    _constrained_nonoverlap_example,
    _decode_payload_atoms,
    _encode_atoms,
    _line_value,
    _percentile,
    _random_packet,
    _source_private_atoms,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


CONDITIONS = (
    "target_only",
    "learned_synonym_dictionary_packet",
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "public_only_sidecar",
    "target_derived_sidecar",
    "random_same_byte",
    "answer_only_text",
    "structured_text_matched",
    "atom_id_derangement",
    "private_random_source_atoms",
    "permuted_teacher_receiver",
    "top_atom_knockout",
    "private_random_knockout",
    "oracle_learned_candidate_atoms",
)

STRICT_SOURCE_DESTROYING_CONTROLS = tuple(
    condition for condition in SOURCE_DESTROYING_CONTROLS if condition in CONDITIONS
) + ("private_random_source_atoms", "permuted_teacher_receiver")
JEPA_RECEIVER_MODES = {
    "jepa_query_resampler",
    "jepa_query_resampler_trainable",
    "jepa_query_resampler_control_regularized",
    "jepa_query_resampler_pool_contrastive",
}
ADAPTER_TARGET_MODES = {
    "native_atoms",
    "semantic_anchor_teacher",
    "permuted_semantic_anchor_teacher",
}
DECODER_SCORE_MODES = {
    "global_dot",
    "candidate_local_residual",
    "candidate_local_residual_norm",
    "candidate_local_innovation_residual_norm",
    "candidate_local_permuted_null_gap_residual_norm",
    "candidate_local_random_rotation_sign_residual_norm",
    "candidate_local_random_rotation_rank_sign_residual_norm",
    "relative_anchor_dot",
    "relative_anchor_residual_norm",
    "relative_anchor_innovation_residual_norm",
    "relative_anchor_rank_innovation_residual_norm",
    "procrustes_dot",
    "ridge_cca_dot",
    "ridge_cca_residual_norm",
    "lstirp_relative_dot",
    "lstirp_relative_residual_norm",
    "inverse_relative_dot",
    "inverse_relative_residual_norm",
    "sinkhorn_ot_dot",
    "sinkhorn_ot_residual_norm",
    "gromov_wasserstein_dot",
    "gromov_wasserstein_residual_norm",
    "ot_gw_dot",
    "ot_gw_residual_norm",
}

HELDOUT_SYNONYM_REPLACEMENTS = (
    ("empty-list guard", "no-element collection safeguard"),
    ("empty-list none default", "no-element collection null return"),
    ("empty-list zero default", "no-element collection baseline number"),
    ("missing-key default", "unavailable slot replacement"),
    ("nested missing-key default", "deep unavailable slot replacement"),
    ("parse-failure zero default", "failed parse returns baseline number"),
    ("half-up rounding", "tie-breaking round toward larger value"),
    ("inclusive comparison", "edge-inclusive comparator"),
    ("preserve-order unique", "remove repeats while preserving input sequence"),
    ("none-to-empty", "null value becomes blank text"),
    ("sum all values", "total every numeric entry"),
    ("case-insensitive equality", "compare text without lettercase"),
    ("clamp negative to zero", "raise subzero values to baseline"),
    ("modulo-wrapped index", "wrap position around collection length"),
    ("strictly positive filter", "keep numbers greater than baseline"),
    ("average all values", "mean over every numeric entry"),
    ("strip and lowercase", "trim whitespace then normalize casing"),
    ("strict key access", "direct required slot retrieval"),
    ("sort before reading", "order values before lookup"),
    ("last element fallback", "use terminal item as backup"),
    ("string coercion", "convert value into text"),
    ("first value fallback", "use initial item as backup"),
    ("sort before final value", "order values before terminal lookup"),
    ("float parsing", "decimal number conversion"),
    ("return raw text", "return original text unchanged"),
    ("length fallback", "collection size backup"),
    ("sum only", "total without averaging"),
    ("divide by first value", "scale by initial numeric entry"),
    ("drop last before averaging", "exclude terminal item before mean"),
    ("strip only", "trim whitespace only"),
    ("uppercase only", "capitalize all letters only"),
    ("remove spaces inside text", "remove internal whitespace"),
    ("top-level name lookup", "outer slot retrieval"),
    ("return whole user mapping", "return complete record mapping"),
    ("uppercase nested name", "capitalize inner name field"),
    ("clamp to final index", "cap lookup at terminal position"),
    ("always first value", "always pick initial entry"),
    ("sort before index", "order values before position lookup"),
    ("keep nonnegative values", "retain baseline-or-higher numbers"),
    ("keep negative values", "select values below baseline"),
    ("sort positive values", "order numbers greater than baseline"),
)


def _candidate_surface_text(candidate_intent: str, *, candidate_atom_view: str) -> str:
    if candidate_atom_view == "native":
        return candidate_intent
    if candidate_atom_view == "synonym_stress":
        from scripts.run_source_private_shared_sparse_crosscoder_packet_gate import _stress_candidate_intent

        return _stress_candidate_intent(candidate_intent, candidate_atom_view="synonym_stress")
    if candidate_atom_view != "heldout_synonym":
        raise ValueError(f"unknown candidate atom view {candidate_atom_view!r}")
    text = candidate_intent
    for old, new in HELDOUT_SYNONYM_REPLACEMENTS:
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_index(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def _stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("-", " ").replace("_", " "))


_HF_TEXT_FEATURE_CACHE: dict[tuple[str, str, str, str, int, int, str, str], np.ndarray] = {}
_HF_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, str, Any]] = {}
_HF_FEATURE_MODEL = "BAAI/bge-small-en"
_HF_FEATURE_DEVICE = "auto"
_HF_FEATURE_DTYPE = "float32"
_HF_FEATURE_MAX_LENGTH = 128
_HF_FEATURE_LOCAL_FILES_ONLY = True


def _resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_torch_dtype(dtype: str) -> Any:
    import torch

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unknown feature dtype {dtype!r}")


def _load_hf_feature_model() -> tuple[Any, Any, str, Any]:
    device = _resolve_torch_device(_HF_FEATURE_DEVICE)
    dtype = _resolve_torch_dtype(_HF_FEATURE_DTYPE)
    key = (_HF_FEATURE_MODEL, device, _HF_FEATURE_DTYPE)
    cached = _HF_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _HF_FEATURE_MODEL,
        local_files_only=_HF_FEATURE_LOCAL_FILES_ONLY,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        _HF_FEATURE_MODEL,
        local_files_only=_HF_FEATURE_LOCAL_FILES_ONLY,
        trust_remote_code=True,
    ).to(device)
    if device in {"mps", "cuda"} and dtype != _resolve_torch_dtype("float32"):
        model = model.to(dtype)
    model.eval()
    loaded = (tokenizer, model, device, dtype)
    _HF_MODEL_CACHE[key] = loaded
    return loaded


def _project_feature_dim(values: np.ndarray, *, dim: int, namespace: str) -> np.ndarray:
    if values.shape[1] == dim:
        projected = values
    else:
        rng_seed = int.from_bytes(
            hashlib.blake2b(f"{namespace}:{values.shape[1]}:{dim}".encode("utf-8"), digest_size=8).digest(),
            "little",
        )
        rng = np.random.default_rng(rng_seed)
        projection = rng.normal(0.0, 1.0 / np.sqrt(values.shape[1]), size=(values.shape[1], dim))
        projected = values @ projection
    norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    return (projected / np.maximum(norm, 1e-8)).astype(np.float64)


def _hf_text_features(texts: list[str], *, dim: int, text_feature_mode: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float64)
    device = _resolve_torch_device(_HF_FEATURE_DEVICE)
    cache_prefix = (
        _HF_FEATURE_MODEL,
        device,
        _HF_FEATURE_DTYPE,
        _HF_FEATURE_MAX_LENGTH,
        dim,
        text_feature_mode,
    )
    missing = [text for text in dict.fromkeys(texts) if (text, *cache_prefix) not in _HF_TEXT_FEATURE_CACHE]
    if missing:
        import torch

        tokenizer, model, device, _ = _load_hf_feature_model()
        encoded = tokenizer(
            missing,
            padding=True,
            truncation=True,
            max_length=_HF_FEATURE_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=text_feature_mode == "hf_mid_last_mean")
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            if text_feature_mode == "hf_mid_last_mean" and output.hidden_states is not None:
                layers = [output.hidden_states[len(output.hidden_states) // 2], output.hidden_states[-1]]
                pooled_layers = []
                for hidden in layers:
                    hidden = hidden.float()
                    pooled_layers.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
                pooled = torch.cat(pooled_layers, dim=1)
            else:
                hidden = output.last_hidden_state.float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vectors = _project_feature_dim(pooled.cpu().numpy().astype(np.float64), dim=dim, namespace=text_feature_mode)
        for text, vector in zip(missing, vectors, strict=True):
            _HF_TEXT_FEATURE_CACHE[(text, *cache_prefix)] = vector
    return np.stack([_HF_TEXT_FEATURE_CACHE[(text, *cache_prefix)] for text in texts], axis=0)


SEMANTIC_ANCHOR_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("empty", "vacant", "element", "no element", "no-element"), ("empty",)),
    (("list", "sequence", "collection"), ("list", "all_values")),
    (("guard", "contingency", "safeguard"), ("guard",)),
    (("none", "null"), ("none", "default")),
    (("zero", "neutral", "baseline"), ("zero", "default")),
    (("missing", "absent", "unavailable"), ("missing",)),
    (("key", "field", "slot"), ("key", "mapping")),
    (("default", "fallback", "substitute", "replacement", "backup", "alternative"), ("default", "fallback")),
    (("nested", "deep", "inner"), ("nested", "mapping")),
    (("parse", "parsing", "conversion", "convert", "decimal", "number"), ("parse", "integer")),
    (("failure", "failed", "miss"), ("failure",)),
    (("round", "rounding", "midpoint", "tie", "tie breaking", "tie-breaking"), ("round", "half_up")),
    (("upward", "larger"), ("half_up",)),
    (("inclusive", "boundary", "edge", "admitting"), ("inclusive", "threshold")),
    (("comparison", "comparator", "relation", "equivalence", "equality"), ("equality", "threshold")),
    (("preserve", "preserving", "retaining"), ("preserve",)),
    (("order", "chronology"), ("order",)),
    (("unique", "deduplicate", "repeats"), ("unique",)),
    (("sum", "aggregate", "total"), ("sum", "all_values")),
    (("average", "mean"), ("average", "mean", "all_values")),
    (("case", "lettercase", "casing"), ("case",)),
    (("insensitive", "agnostic", "without lettercase"), ("insensitive",)),
    (("lowercase", "downcase"), ("lowercase", "case")),
    (("clamp", "floor", "raise", "cap"), ("clamp",)),
    (("negative", "subzero"), ("negative",)),
    (("final", "last", "terminal"), ("final", "last")),
    (("index", "position", "lookup"), ("index",)),
    (("modulo", "cyclic", "wrap", "wrapped"), ("modulo", "index", "list")),
    (("strict", "direct", "required"), ("strict",)),
    (("positive", "above neutral", "greater than baseline"), ("positive",)),
    (("filter", "retain", "keep"), ("filter",)),
    (("first", "initial"), ("first",)),
    (("string", "text"), ("string",)),
)


def _semantic_anchor_terms(text: str) -> dict[str, float]:
    lowered = text.lower().replace("_", " ")
    tokens = set(_word_tokens(lowered))
    terms: dict[str, float] = {}
    for token in tokens:
        if token in ATOM_TO_ID:
            terms[f"anchor:{token}"] = max(terms.get(f"anchor:{token}", 0.0), 1.0)
    for triggers, anchors in SEMANTIC_ANCHOR_EXPANSIONS:
        if any((trigger in tokens) or (trigger in lowered) for trigger in triggers):
            for anchor in anchors:
                terms[f"anchor:{anchor}"] = max(terms.get(f"anchor:{anchor}", 0.0), 1.0)
    return terms


def _featurize_text(text: str, *, dim: int, text_feature_mode: str = "hashed") -> np.ndarray:
    if text_feature_mode not in {
        "hashed",
        "semantic_anchor",
        "hf_last_mean",
        "hf_mid_last_mean",
        "hashed_hf_last_mean",
        "hashed_hf_mid_last_mean",
    }:
        raise ValueError(f"unknown text feature mode {text_feature_mode!r}")
    if text_feature_mode in {"hf_last_mean", "hf_mid_last_mean"}:
        return _hf_text_features([text], dim=dim, text_feature_mode=text_feature_mode)[0]
    if text_feature_mode in {"hashed_hf_last_mean", "hashed_hf_mid_last_mean"}:
        hf_mode = "hf_last_mean" if text_feature_mode == "hashed_hf_last_mean" else "hf_mid_last_mean"
        hashed = _featurize_text(text, dim=dim, text_feature_mode="hashed")
        hf = _hf_text_features([text], dim=dim, text_feature_mode=hf_mode)
        combined = hashed + hf[0]
        norm = float(np.linalg.norm(combined))
        if norm > 0:
            combined = combined / norm
        return combined.astype(np.float64)
    lowered = text.lower()
    words = _word_tokens(lowered)
    features = np.zeros(dim, dtype=np.float64)
    for token in words:
        features[_stable_index(f"w:{token}", dim)] += 1.0
    for left, right in zip(words, words[1:]):
        features[_stable_index(f"b:{left}_{right}", dim)] += 1.0
    compact = re.sub(r"\s+", " ", lowered)
    for n in (3, 4, 5):
        if len(compact) >= n:
            for idx in range(len(compact) - n + 1):
                features[_stable_index(f"c{n}:{compact[idx:idx+n]}", dim)] += 0.35
    if text_feature_mode == "semantic_anchor":
        for term, weight in _semantic_anchor_terms(text).items():
            features[_stable_index(term, dim)] += 2.5 * weight
    norm = float(np.linalg.norm(features))
    if norm > 0:
        features /= norm
    return features


def _atom_vector(atoms: dict[str, float]) -> np.ndarray:
    vector = np.zeros(len(ATOM_ORDER), dtype=np.float64)
    for atom, score in atoms.items():
        if atom in ATOM_TO_ID:
            vector[ATOM_TO_ID[atom]] = max(vector[ATOM_TO_ID[atom]], float(score))
    return vector


def _semantic_anchor_atom_vector(text: str) -> np.ndarray:
    vector = np.zeros(len(ATOM_ORDER), dtype=np.float64)
    for term, score in _semantic_anchor_terms(text).items():
        if not term.startswith("anchor:"):
            continue
        atom = term.removeprefix("anchor:")
        if atom in ATOM_TO_ID:
            vector[ATOM_TO_ID[atom]] = max(vector[ATOM_TO_ID[atom]], float(score))
    return vector


def _permute_atom_vector(vector: np.ndarray, *, namespace: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(namespace.encode("utf-8"), digest_size=8).digest(), "little")
    permutation = np.random.default_rng(seed).permutation(len(ATOM_ORDER))
    return vector[permutation]


def _atoms_from_vector(vector: np.ndarray, *, top_k: int, min_score: float) -> dict[str, float]:
    if vector.size == 0:
        return {}
    ranked = np.argsort(-vector)[:top_k]
    atoms: dict[str, float] = {}
    for atom_id in ranked:
        score = float(vector[atom_id])
        if score >= min_score:
            atoms[ID_TO_ATOM[int(atom_id)]] = score
    return atoms


def _relative_anchor_matrix(anchor_rows: list[np.ndarray], *, max_anchors: int = 256) -> np.ndarray:
    unique: dict[bytes, np.ndarray] = {}
    for row in anchor_rows:
        if not np.any(row):
            continue
        normalized = row.astype(np.float64, copy=True)
        norm = float(np.linalg.norm(normalized))
        if norm <= 0:
            continue
        normalized /= norm
        key = np.round(normalized, decimals=6).tobytes()
        unique.setdefault(key, normalized)
    if not unique:
        return np.zeros((0, len(ATOM_ORDER)), dtype=np.float64)
    anchors = sorted(unique.values(), key=lambda value: hashlib.sha256(value.tobytes()).hexdigest())
    if len(anchors) <= max_anchors:
        return np.stack(anchors, axis=0)
    step = len(anchors) / max_anchors
    selected = [anchors[int(idx * step)] for idx in range(max_anchors)]
    return np.stack(selected, axis=0)


def _orthogonal_procrustes_matrix(
    source_rows: list[np.ndarray],
    target_rows: list[np.ndarray],
) -> np.ndarray:
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target rows must be aligned for Procrustes fitting")
    if not source_rows:
        return np.eye(len(ATOM_ORDER), dtype=np.float64)
    kept_source: list[np.ndarray] = []
    kept_target: list[np.ndarray] = []
    for source_row, target_row in zip(source_rows, target_rows, strict=True):
        source_norm = float(np.linalg.norm(source_row))
        target_norm = float(np.linalg.norm(target_row))
        if source_norm <= 0 or target_norm <= 0:
            continue
        kept_source.append(source_row.astype(np.float64) / source_norm)
        kept_target.append(target_row.astype(np.float64) / target_norm)
    if not kept_source:
        return np.eye(len(ATOM_ORDER), dtype=np.float64)
    source = np.stack(kept_source, axis=0)
    target = np.stack(kept_target, axis=0)
    left, _, right_t = np.linalg.svd(source.T @ target, full_matrices=False)
    return left @ right_t


def _inverse_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    safe_values = np.maximum(values, 1e-12)
    return (vectors * (1.0 / np.sqrt(safe_values))) @ vectors.T


def _ridge_cca_components(
    source_rows: list[np.ndarray],
    target_rows: list[np.ndarray],
    *,
    ridge: float,
    max_rank: int = 32,
) -> dict[str, np.ndarray | int]:
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target rows must be aligned for CCA fitting")
    dim = len(ATOM_ORDER)
    if not source_rows:
        return {
            "source_mean": np.zeros(dim, dtype=np.float64),
            "target_mean": np.zeros(dim, dtype=np.float64),
            "source_projection": np.zeros((dim, 0), dtype=np.float64),
            "target_projection": np.zeros((dim, 0), dtype=np.float64),
            "correlations": np.zeros(0, dtype=np.float64),
            "rank": 0,
        }
    source = np.stack(source_rows, axis=0).astype(np.float64)
    target = np.stack(target_rows, axis=0).astype(np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    denom = max(source.shape[0] - 1, 1)
    source_cov = (source_centered.T @ source_centered) / denom + ridge * np.eye(dim, dtype=np.float64)
    target_cov = (target_centered.T @ target_centered) / denom + ridge * np.eye(dim, dtype=np.float64)
    cross_cov = (source_centered.T @ target_centered) / denom
    source_whitener = _inverse_sqrt_psd(source_cov)
    target_whitener = _inverse_sqrt_psd(target_cov)
    left, singular_values, right_t = np.linalg.svd(source_whitener @ cross_cov @ target_whitener, full_matrices=False)
    positive = int(np.sum(singular_values > 1e-8))
    rank = min(max_rank, positive, dim)
    if rank <= 0:
        return {
            "source_mean": source_mean,
            "target_mean": target_mean,
            "source_projection": np.zeros((dim, 0), dtype=np.float64),
            "target_projection": np.zeros((dim, 0), dtype=np.float64),
            "correlations": np.zeros(0, dtype=np.float64),
            "rank": 0,
        }
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "source_projection": source_whitener @ left[:, :rank],
        "target_projection": target_whitener @ right_t.T[:, :rank],
        "correlations": singular_values[:rank],
        "rank": rank,
    }


def _normalized_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=norms > 0,
    )


def _rank_normalized_rows(matrix: np.ndarray) -> np.ndarray:
    rows = np.asarray(matrix, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("rank-normalized rows must be a 2D matrix")
    if rows.size == 0:
        return np.zeros(rows.shape, dtype=np.float64)
    cols = rows.shape[1]
    if cols <= 1:
        return np.zeros(rows.shape, dtype=np.float64)
    order = np.argsort(rows, axis=1, kind="mergesort")
    ranks = np.empty(order.shape, dtype=np.float64)
    ranks[np.arange(rows.shape[0])[:, None], order] = np.arange(cols, dtype=np.float64)
    return (2.0 * ranks / float(cols - 1)) - 1.0


def _public_orthogonal_matrix(dim: int, *, namespace: str) -> np.ndarray:
    if dim <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    rng = np.random.default_rng(_stable_seed(f"{namespace}|dim={dim}"))
    raw = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float64)


def _sign_sketch_rows(matrix: np.ndarray) -> np.ndarray:
    signs = np.sign(matrix.astype(np.float64))
    return signs.astype(np.float64)


def _candidate_local_residual_scores(
    candidate_matrix: np.ndarray,
    payload_vector: np.ndarray,
    *,
    decoder_score_mode: str,
) -> tuple[list[float], dict[str, Any]]:
    local_mean = candidate_matrix.mean(axis=0, keepdims=True)
    residual_matrix = candidate_matrix - local_mean
    row_norms = np.linalg.norm(residual_matrix, axis=1)
    payload_for_score = payload_vector
    payload_transform = "raw_payload"
    if decoder_score_mode == "candidate_local_innovation_residual_norm":
        payload_for_score = payload_vector - local_mean[0]
        payload_transform = "subtract_candidate_pool_mean"
    payload_norm = float(np.linalg.norm(payload_for_score))
    metadata: dict[str, Any] = {
        "decoder_score_mode": decoder_score_mode,
        "candidate_local_mean_l2": float(np.linalg.norm(local_mean)),
        "candidate_local_row_norms": [float(value) for value in row_norms],
        "candidate_local_payload_l2": payload_norm,
        "candidate_local_raw_payload_l2": float(np.linalg.norm(payload_vector)),
        "candidate_local_payload_transform": payload_transform,
    }
    if decoder_score_mode == "candidate_local_residual":
        scores_array = residual_matrix @ payload_vector
    else:
        safe_rows = np.divide(
            residual_matrix,
            np.maximum(row_norms[:, None], 1e-12),
            out=np.zeros_like(residual_matrix),
            where=row_norms[:, None] > 0,
        )
        safe_payload = payload_for_score / payload_norm if payload_norm > 0 else payload_for_score
        if decoder_score_mode in {
            "candidate_local_random_rotation_sign_residual_norm",
            "candidate_local_random_rotation_rank_sign_residual_norm",
        }:
            namespace = "latentwire_candidate_local_random_rotation_sign_v1"
            rotation = _public_orthogonal_matrix(residual_matrix.shape[1], namespace=namespace)
            rotated_rows = safe_rows @ rotation
            rotated_payload = safe_payload @ rotation
            quantization = "sign"
            if decoder_score_mode == "candidate_local_random_rotation_rank_sign_residual_norm":
                rotated_rows = _rank_normalized_rows(rotated_rows)
                rotated_payload = _rank_normalized_rows(rotated_payload.reshape(1, -1))[0]
                quantization = "rank_sign"
            sketch_rows = _sign_sketch_rows(rotated_rows)
            sketch_payload = _sign_sketch_rows(rotated_payload.reshape(1, -1))[0]
            sketch_dim = int(sketch_payload.shape[0])
            scores_array = (sketch_rows @ sketch_payload) / float(max(sketch_dim, 1))
            metadata.update(
                {
                    "candidate_local_transform": "public_orthogonal_sign_sketch",
                    "candidate_local_transform_namespace": namespace,
                    "candidate_local_quantization": quantization,
                    "candidate_local_sketch_bits": sketch_dim,
                    "candidate_local_sketch_payload_l1": float(np.linalg.norm(sketch_payload, ord=1)),
                    "candidate_local_sketch_row_l1": [
                        float(np.linalg.norm(row, ord=1)) for row in sketch_rows
                    ],
                }
            )
        else:
            scores_array = safe_rows @ safe_payload
    return [float(score) for score in scores_array], metadata


def _relative_coordinates(rows: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if rows.size == 0 or anchors.size == 0:
        return np.zeros((rows.shape[0], 0), dtype=np.float64)
    return _normalized_rows(rows.astype(np.float64)) @ _normalized_rows(anchors.astype(np.float64)).T


def _lstirp_components(
    source_rows: list[np.ndarray],
    target_rows: list[np.ndarray],
    *,
    ridge: float,
) -> dict[str, np.ndarray | int]:
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target rows must be aligned for LSTIRP-lite fitting")
    dim = len(ATOM_ORDER)
    if not source_rows:
        return {
            "source_anchor_vectors": np.zeros((0, dim), dtype=np.float64),
            "target_anchor_vectors": np.zeros((0, dim), dtype=np.float64),
            "source_relative_mean": np.zeros(0, dtype=np.float64),
            "target_relative_mean": np.zeros(0, dtype=np.float64),
            "translation": np.zeros((0, 0), dtype=np.float64),
            "source_anchor_count": 0,
            "target_anchor_count": 0,
        }
    source_anchor_vectors = _relative_anchor_matrix(source_rows)
    target_anchor_vectors = _relative_anchor_matrix(target_rows)
    source = np.stack(source_rows, axis=0).astype(np.float64)
    target = np.stack(target_rows, axis=0).astype(np.float64)
    source_relative = _relative_coordinates(source, source_anchor_vectors)
    target_relative = _relative_coordinates(target, target_anchor_vectors)
    if source_relative.shape[1] == 0 or target_relative.shape[1] == 0:
        return {
            "source_anchor_vectors": source_anchor_vectors,
            "target_anchor_vectors": target_anchor_vectors,
            "source_relative_mean": np.zeros(source_relative.shape[1], dtype=np.float64),
            "target_relative_mean": np.zeros(target_relative.shape[1], dtype=np.float64),
            "translation": np.zeros((source_relative.shape[1], target_relative.shape[1]), dtype=np.float64),
            "source_anchor_count": int(source_anchor_vectors.shape[0]),
            "target_anchor_count": int(target_anchor_vectors.shape[0]),
        }
    source_mean = source_relative.mean(axis=0)
    target_mean = target_relative.mean(axis=0)
    centered_source = source_relative - source_mean
    centered_target = target_relative - target_mean
    regularized = centered_source.T @ centered_source + ridge * np.eye(centered_source.shape[1], dtype=np.float64)
    translation = np.linalg.solve(regularized, centered_source.T @ centered_target)
    return {
        "source_anchor_vectors": source_anchor_vectors,
        "target_anchor_vectors": target_anchor_vectors,
        "source_relative_mean": source_mean,
        "target_relative_mean": target_mean,
        "translation": translation,
        "source_anchor_count": int(source_anchor_vectors.shape[0]),
        "target_anchor_count": int(target_anchor_vectors.shape[0]),
    }


def _select_paired_anchor_rows(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    *,
    max_anchors: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    unique: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source_row, target_row in zip(source_rows, target_rows, strict=True):
        if not np.any(source_row) or not np.any(target_row):
            continue
        key = hashlib.sha256(np.concatenate([source_row, target_row]).round(6).tobytes()).hexdigest()
        unique.setdefault(key, (source_row, target_row))
    if not unique:
        dim = len(ATOM_ORDER)
        return np.zeros((0, dim), dtype=np.float64), np.zeros((0, dim), dtype=np.float64)
    pairs = [unique[key] for key in sorted(unique)]
    if len(pairs) > max_anchors:
        step = len(pairs) / max_anchors
        pairs = [pairs[int(idx * step)] for idx in range(max_anchors)]
    source_anchor_rows, target_anchor_rows = zip(*pairs, strict=True)
    return np.stack(source_anchor_rows, axis=0), np.stack(target_anchor_rows, axis=0)


def _inverse_relative_components(
    source_rows: list[np.ndarray],
    target_rows: list[np.ndarray],
    *,
    ridge: float,
    max_anchors: int = 256,
) -> dict[str, np.ndarray | int | float]:
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target rows must be aligned for inverse-relative fitting")
    dim = len(ATOM_ORDER)
    if not source_rows:
        return {
            "source_mean": np.zeros(dim, dtype=np.float64),
            "target_mean": np.zeros(dim, dtype=np.float64),
            "source_anchors": np.zeros((0, dim), dtype=np.float64),
            "target_anchors": np.zeros((0, dim), dtype=np.float64),
            "map": np.zeros((dim, dim), dtype=np.float64),
            "anchor_count": 0,
            "condition_number": 0.0,
        }
    source = np.stack(source_rows, axis=0).astype(np.float64)
    target = np.stack(target_rows, axis=0).astype(np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = _normalized_rows(source - source_mean[None, :])
    target_centered = _normalized_rows(target - target_mean[None, :])
    source_anchors, target_anchors = _select_paired_anchor_rows(
        source_centered,
        target_centered,
        max_anchors=max_anchors,
    )
    if source_anchors.shape[0] == 0:
        return {
            "source_mean": source_mean,
            "target_mean": target_mean,
            "source_anchors": source_anchors,
            "target_anchors": target_anchors,
            "map": np.zeros((dim, dim), dtype=np.float64),
            "anchor_count": 0,
            "condition_number": 0.0,
        }
    gram = target_anchors.T @ target_anchors + ridge * np.eye(dim, dtype=np.float64)
    condition_number = float(np.linalg.cond(gram))
    inverse_relative_map = source_anchors.T @ target_anchors @ np.linalg.pinv(gram)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "source_anchors": source_anchors,
        "target_anchors": target_anchors,
        "map": inverse_relative_map,
        "anchor_count": int(source_anchors.shape[0]),
        "condition_number": condition_number,
    }


def _pairwise_column_geometry(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
    columns = rows.astype(np.float64).T
    columns = columns - columns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(columns, axis=1, keepdims=True)
    normalized = np.divide(
        columns,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(columns),
        where=norms > 0,
    )
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    scale = float(np.max(distance))
    return distance / scale if scale > 0 else distance


def _normalized_marginal(values: np.ndarray, *, smoothing: float = 1e-3) -> np.ndarray:
    marginal = np.maximum(values.astype(np.float64), 0.0) + smoothing
    total = float(marginal.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
    return marginal / total


def _entropy(values: np.ndarray) -> float:
    safe = np.maximum(values.astype(np.float64), 1e-300)
    return float(-np.sum(safe * np.log(safe)))


def _row_normalized_transport(plan: np.ndarray) -> np.ndarray:
    row_sums = plan.sum(axis=1, keepdims=True)
    return np.divide(
        plan,
        np.maximum(row_sums, 1e-12),
        out=np.zeros_like(plan),
        where=row_sums > 0,
    )


def _sinkhorn_plan(
    cost: np.ndarray,
    *,
    entropy: float,
    iterations: int,
    source_marginal: np.ndarray | None = None,
    target_marginal: np.ndarray | None = None,
) -> np.ndarray:
    dim_source, dim_target = cost.shape
    if dim_source == 0 or dim_target == 0:
        return np.zeros_like(cost, dtype=np.float64)
    shifted = cost.astype(np.float64) - float(np.min(cost))
    kernel = np.exp(-shifted / max(entropy, 1e-6)) + 1e-300
    if source_marginal is None:
        source_marginal = np.full(dim_source, 1.0 / dim_source, dtype=np.float64)
    else:
        source_marginal = _normalized_marginal(source_marginal, smoothing=1e-12)
    if target_marginal is None:
        target_marginal = np.full(dim_target, 1.0 / dim_target, dtype=np.float64)
    else:
        target_marginal = _normalized_marginal(target_marginal, smoothing=1e-12)
    u = np.ones(dim_source, dtype=np.float64)
    v = np.ones(dim_target, dtype=np.float64)
    for _ in range(iterations):
        u = source_marginal / np.maximum(kernel @ v, 1e-300)
        v = target_marginal / np.maximum(kernel.T @ u, 1e-300)
    plan = (u[:, None] * kernel) * v[None, :]
    return plan / max(float(plan.sum()), 1e-300)


def _ot_gw_components(
    source_rows: list[np.ndarray],
    target_rows: list[np.ndarray],
    *,
    entropy: float = 0.08,
    fused_weight: float = 0.35,
    iterations: int = 24,
    sinkhorn_iterations: int = 80,
) -> dict[str, np.ndarray | int | float]:
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target rows must be aligned for OT/GW fitting")
    dim = len(ATOM_ORDER)
    if not source_rows:
        return {
            "map": np.eye(dim, dtype=np.float64),
            "sinkhorn_map": np.eye(dim, dtype=np.float64),
            "iterations": 0,
            "sinkhorn_iterations": sinkhorn_iterations,
            "entropy": entropy,
            "fused_weight": fused_weight,
            "coupling_l1": 0.0,
            "objective": 0.0,
            "coupling_entropy": 0.0,
            "coupling_top1_mass": 0.0,
            "row_residual_l1": 0.0,
            "col_residual_l1": 0.0,
            "sinkhorn_coupling_l1": 0.0,
            "sinkhorn_objective": 0.0,
            "sinkhorn_coupling_entropy": 0.0,
            "sinkhorn_coupling_top1_mass": 0.0,
            "sinkhorn_row_residual_l1": 0.0,
            "sinkhorn_col_residual_l1": 0.0,
            "source_marginal_entropy": 0.0,
            "target_marginal_entropy": 0.0,
        }
    source = np.stack(source_rows, axis=0).astype(np.float64)
    target = np.stack(target_rows, axis=0).astype(np.float64)
    source_geometry = _pairwise_column_geometry(source)
    target_geometry = _pairwise_column_geometry(target)
    source_columns = _normalized_rows((source - source.mean(axis=0, keepdims=True)).T)
    target_columns = _normalized_rows((target - target.mean(axis=0, keepdims=True)).T)
    feature_cost = 1.0 - np.clip(source_columns @ target_columns.T, -1.0, 1.0)
    feature_scale = float(np.max(feature_cost))
    if feature_scale > 0:
        feature_cost = feature_cost / feature_scale
    dim_source = source_geometry.shape[0]
    dim_target = target_geometry.shape[0]
    p = _normalized_marginal(source.sum(axis=0))
    q = _normalized_marginal(target.sum(axis=0))
    sinkhorn_plan = _sinkhorn_plan(
        feature_cost,
        entropy=entropy,
        iterations=sinkhorn_iterations,
        source_marginal=p,
        target_marginal=q,
    )
    plan = sinkhorn_plan
    fused_cost = feature_cost
    gw_cost = np.zeros_like(feature_cost)
    for _ in range(iterations):
        const_source = (source_geometry**2) @ p
        const_target = (target_geometry**2) @ q
        gw_cost = const_source[:, None] + const_target[None, :] - 2.0 * source_geometry @ plan @ target_geometry.T
        gw_scale = float(np.max(gw_cost) - np.min(gw_cost))
        if gw_scale > 0:
            gw_cost = (gw_cost - float(np.min(gw_cost))) / gw_scale
        fused_cost = (1.0 - fused_weight) * gw_cost + fused_weight * feature_cost
        plan = _sinkhorn_plan(
            fused_cost,
            entropy=entropy,
            iterations=sinkhorn_iterations,
            source_marginal=p,
            target_marginal=q,
        )
    transport_map = _row_normalized_transport(plan)
    sinkhorn_transport_map = _row_normalized_transport(sinkhorn_plan)
    row_residual = float(np.sum(np.abs(plan.sum(axis=1) - p)))
    col_residual = float(np.sum(np.abs(plan.sum(axis=0) - q)))
    sinkhorn_row_residual = float(np.sum(np.abs(sinkhorn_plan.sum(axis=1) - p)))
    sinkhorn_col_residual = float(np.sum(np.abs(sinkhorn_plan.sum(axis=0) - q)))
    objective = float(np.sum(plan * fused_cost))
    sinkhorn_objective = float(np.sum(sinkhorn_plan * feature_cost))
    return {
        "map": transport_map,
        "sinkhorn_map": sinkhorn_transport_map,
        "iterations": iterations,
        "sinkhorn_iterations": sinkhorn_iterations,
        "entropy": entropy,
        "fused_weight": fused_weight,
        "coupling_l1": float(np.sum(np.abs(plan))),
        "objective": objective,
        "coupling_entropy": _entropy(plan.reshape(-1)),
        "coupling_top1_mass": float(np.max(plan)),
        "row_residual_l1": row_residual,
        "col_residual_l1": col_residual,
        "sinkhorn_coupling_l1": float(np.sum(np.abs(sinkhorn_plan))),
        "sinkhorn_objective": sinkhorn_objective,
        "sinkhorn_coupling_entropy": _entropy(sinkhorn_plan.reshape(-1)),
        "sinkhorn_coupling_top1_mass": float(np.max(sinkhorn_plan)),
        "sinkhorn_row_residual_l1": sinkhorn_row_residual,
        "sinkhorn_col_residual_l1": sinkhorn_col_residual,
        "source_marginal_entropy": _entropy(p),
        "target_marginal_entropy": _entropy(q),
    }


def _transport_scores(
    *,
    decoder_score_mode: str,
    prefix: str,
    transport_map: np.ndarray | None,
    payload_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    residual_norm: bool,
) -> tuple[list[float], dict[str, Any]]:
    if transport_map is None:
        return [0.0 for _ in candidate_matrix], {
            "decoder_score_mode": decoder_score_mode,
            f"{prefix}_payload_l2": 0.0,
            "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
        }
    mapped_payload = payload_vector @ transport_map
    if residual_norm:
        candidate_rows = candidate_matrix - candidate_matrix.mean(axis=0, keepdims=True)
    else:
        candidate_rows = candidate_matrix
    row_norms = np.linalg.norm(candidate_rows, axis=1)
    safe_candidates = np.divide(
        candidate_rows,
        np.maximum(row_norms[:, None], 1e-12),
        out=np.zeros_like(candidate_rows),
        where=row_norms[:, None] > 0,
    )
    payload_norm = float(np.linalg.norm(mapped_payload))
    safe_payload = mapped_payload / payload_norm if payload_norm > 0 else mapped_payload
    scores_array = safe_candidates @ safe_payload
    return [float(score) for score in scores_array], {
        "decoder_score_mode": decoder_score_mode,
        f"{prefix}_transport_frobenius": float(np.linalg.norm(transport_map)),
        f"{prefix}_candidate_row_norms": [float(value) for value in row_norms],
        f"{prefix}_payload_l2": payload_norm,
        "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
    }


class LearnedSynonymDictionary:
    def __init__(
        self,
        *,
        feature_dim: int,
        weights: np.ndarray,
        top_k: int,
        min_score: float,
        text_feature_mode: str,
        receiver_mode: str = "atom_ridge",
        bias: float = 0.0,
        contrastive_rank: int | None = None,
        receiver_effective_rank: int | None = None,
        left_factors: np.ndarray | None = None,
        right_factors: np.ndarray | None = None,
        atom_embeddings: np.ndarray | None = None,
        query_vectors: np.ndarray | None = None,
        candidate_projector: np.ndarray | None = None,
        resampler_query_factors: np.ndarray | None = None,
        resampler_atom_keys: np.ndarray | None = None,
        resampler_atom_values: np.ndarray | None = None,
        resampler_output: np.ndarray | None = None,
        relative_anchor_vectors: np.ndarray | None = None,
        procrustes_matrix: np.ndarray | None = None,
        cca_source_mean: np.ndarray | None = None,
        cca_target_mean: np.ndarray | None = None,
        cca_source_projection: np.ndarray | None = None,
        cca_target_projection: np.ndarray | None = None,
        cca_correlations: np.ndarray | None = None,
        cca_rank: int | None = None,
        lstirp_source_anchor_vectors: np.ndarray | None = None,
        lstirp_target_anchor_vectors: np.ndarray | None = None,
        lstirp_source_relative_mean: np.ndarray | None = None,
        lstirp_target_relative_mean: np.ndarray | None = None,
        lstirp_translation: np.ndarray | None = None,
        lstirp_source_anchor_count: int | None = None,
        lstirp_target_anchor_count: int | None = None,
        inverse_relative_source_mean: np.ndarray | None = None,
        inverse_relative_target_mean: np.ndarray | None = None,
        inverse_relative_source_anchors: np.ndarray | None = None,
        inverse_relative_target_anchors: np.ndarray | None = None,
        inverse_relative_map: np.ndarray | None = None,
        inverse_relative_anchor_count: int | None = None,
        inverse_relative_condition_number: float | None = None,
        sinkhorn_ot_transport_map: np.ndarray | None = None,
        sinkhorn_ot_entropy: float | None = None,
        sinkhorn_ot_coupling_l1: float | None = None,
        sinkhorn_ot_objective: float | None = None,
        sinkhorn_ot_coupling_entropy: float | None = None,
        sinkhorn_ot_coupling_top1_mass: float | None = None,
        sinkhorn_ot_row_residual_l1: float | None = None,
        sinkhorn_ot_col_residual_l1: float | None = None,
        ot_gw_transport_map: np.ndarray | None = None,
        ot_gw_iterations: int | None = None,
        ot_gw_sinkhorn_iterations: int | None = None,
        ot_gw_entropy: float | None = None,
        ot_gw_fused_weight: float | None = None,
        ot_gw_coupling_l1: float | None = None,
        ot_gw_objective: float | None = None,
        ot_gw_coupling_entropy: float | None = None,
        ot_gw_coupling_top1_mass: float | None = None,
        ot_gw_row_residual_l1: float | None = None,
        ot_gw_col_residual_l1: float | None = None,
        transport_source_marginal_entropy: float | None = None,
        transport_target_marginal_entropy: float | None = None,
        jepa_query_count: int | None = None,
        jepa_hidden_dim: int | None = None,
        jepa_query_entropy: float | None = None,
        jepa_context_variance: float | None = None,
        jepa_trainable_factors: bool = False,
        jepa_train_epochs: int | None = None,
        jepa_lr: float | None = None,
        jepa_weight_decay: float | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.weights = weights
        self.top_k = top_k
        self.min_score = min_score
        self.text_feature_mode = text_feature_mode
        self.receiver_mode = receiver_mode
        self.bias = bias
        self.contrastive_rank = contrastive_rank
        self.receiver_effective_rank = receiver_effective_rank
        self.left_factors = left_factors
        self.right_factors = right_factors
        self.atom_embeddings = atom_embeddings
        self.query_vectors = query_vectors
        self.candidate_projector = candidate_projector
        self.resampler_query_factors = resampler_query_factors
        self.resampler_atom_keys = resampler_atom_keys
        self.resampler_atom_values = resampler_atom_values
        self.resampler_output = resampler_output
        self.relative_anchor_vectors = relative_anchor_vectors
        self.procrustes_matrix = procrustes_matrix
        self.cca_source_mean = cca_source_mean
        self.cca_target_mean = cca_target_mean
        self.cca_source_projection = cca_source_projection
        self.cca_target_projection = cca_target_projection
        self.cca_correlations = cca_correlations
        self.cca_rank = cca_rank
        self.lstirp_source_anchor_vectors = lstirp_source_anchor_vectors
        self.lstirp_target_anchor_vectors = lstirp_target_anchor_vectors
        self.lstirp_source_relative_mean = lstirp_source_relative_mean
        self.lstirp_target_relative_mean = lstirp_target_relative_mean
        self.lstirp_translation = lstirp_translation
        self.lstirp_source_anchor_count = lstirp_source_anchor_count
        self.lstirp_target_anchor_count = lstirp_target_anchor_count
        self.inverse_relative_source_mean = inverse_relative_source_mean
        self.inverse_relative_target_mean = inverse_relative_target_mean
        self.inverse_relative_source_anchors = inverse_relative_source_anchors
        self.inverse_relative_target_anchors = inverse_relative_target_anchors
        self.inverse_relative_map = inverse_relative_map
        self.inverse_relative_anchor_count = inverse_relative_anchor_count
        self.inverse_relative_condition_number = inverse_relative_condition_number
        self.sinkhorn_ot_transport_map = sinkhorn_ot_transport_map
        self.sinkhorn_ot_entropy = sinkhorn_ot_entropy
        self.sinkhorn_ot_coupling_l1 = sinkhorn_ot_coupling_l1
        self.sinkhorn_ot_objective = sinkhorn_ot_objective
        self.sinkhorn_ot_coupling_entropy = sinkhorn_ot_coupling_entropy
        self.sinkhorn_ot_coupling_top1_mass = sinkhorn_ot_coupling_top1_mass
        self.sinkhorn_ot_row_residual_l1 = sinkhorn_ot_row_residual_l1
        self.sinkhorn_ot_col_residual_l1 = sinkhorn_ot_col_residual_l1
        self.ot_gw_transport_map = ot_gw_transport_map
        self.ot_gw_iterations = ot_gw_iterations
        self.ot_gw_sinkhorn_iterations = ot_gw_sinkhorn_iterations
        self.ot_gw_entropy = ot_gw_entropy
        self.ot_gw_fused_weight = ot_gw_fused_weight
        self.ot_gw_coupling_l1 = ot_gw_coupling_l1
        self.ot_gw_objective = ot_gw_objective
        self.ot_gw_coupling_entropy = ot_gw_coupling_entropy
        self.ot_gw_coupling_top1_mass = ot_gw_coupling_top1_mass
        self.ot_gw_row_residual_l1 = ot_gw_row_residual_l1
        self.ot_gw_col_residual_l1 = ot_gw_col_residual_l1
        self.transport_source_marginal_entropy = transport_source_marginal_entropy
        self.transport_target_marginal_entropy = transport_target_marginal_entropy
        self.jepa_query_count = jepa_query_count
        self.jepa_hidden_dim = jepa_hidden_dim
        self.jepa_query_entropy = jepa_query_entropy
        self.jepa_context_variance = jepa_context_variance
        self.jepa_trainable_factors = jepa_trainable_factors
        self.jepa_train_epochs = jepa_train_epochs
        self.jepa_lr = jepa_lr
        self.jepa_weight_decay = jepa_weight_decay

    def predict_atom_scores(self, text: str) -> np.ndarray:
        features = _featurize_text(text, dim=self.feature_dim, text_feature_mode=self.text_feature_mode)
        return np.maximum(features @ self.weights, 0.0)

    def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
        scores = self.predict_atom_scores(text)
        if not apply_top_k:
            return scores
        return _atom_vector(_atoms_from_vector(scores, top_k=self.top_k, min_score=self.min_score))

    def predict_atoms(self, text: str) -> dict[str, float]:
        scores = self.predict_atom_scores(text)
        return _atoms_from_vector(scores, top_k=self.top_k, min_score=self.min_score)

    def score_text(self, text: str, payload_atoms: dict[str, float]) -> float:
        if not payload_atoms:
            return 0.0
        if self.receiver_mode == "atom_ridge":
            learned_atoms = self.predict_atoms(text)
            return sum(payload_atoms.get(atom, 0.0) * score for atom, score in learned_atoms.items())
        if self.receiver_mode in {"contrastive_bilinear", "contrastive_low_rank_query"}:
            features = _featurize_text(text, dim=self.feature_dim, text_feature_mode=self.text_feature_mode)
            payload_vector = _atom_vector(payload_atoms)
            return float(features @ self.weights @ payload_vector + self.bias)
        if self.receiver_mode == "contrastive_low_rank_factor":
            if self.left_factors is None or self.right_factors is None:
                raise ValueError("contrastive_low_rank_factor receiver is missing explicit factors")
            features = _featurize_text(text, dim=self.feature_dim, text_feature_mode=self.text_feature_mode)
            payload_vector = _atom_vector(payload_atoms)
            return float((features @ self.left_factors) @ (payload_vector @ self.right_factors) + self.bias)
        if self.receiver_mode in JEPA_RECEIVER_MODES:
            if (
                self.resampler_query_factors is None
                or self.resampler_atom_keys is None
                or self.resampler_atom_values is None
                or self.resampler_output is None
            ):
                raise ValueError("jepa_query_resampler receiver is missing query-resampler parameters")
            features = _featurize_text(text, dim=self.feature_dim, text_feature_mode=self.text_feature_mode)
            payload_vector = _atom_vector(payload_atoms)
            context, _ = _jepa_attention_features(
                features=features,
                payload_vector=payload_vector,
                query_factors=self.resampler_query_factors,
                atom_keys=self.resampler_atom_keys,
                atom_values=self.resampler_atom_values,
            )
            return float(context @ self.resampler_output + self.bias)
        raise ValueError(f"unknown receiver mode {self.receiver_mode!r}")


def _fit_ridge_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    adapter_target_mode: str = "native_atoms",
) -> LearnedSynonymDictionary:
    if adapter_target_mode not in ADAPTER_TARGET_MODES:
        raise ValueError(f"unknown adapter target mode {adapter_target_mode!r}")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    procrustes_source_rows: list[np.ndarray] = []
    for example in examples:
        for candidate in example.candidates:
            texts = [candidate.patch_intent]
            calibrated = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view)
            if calibrated != candidate.patch_intent:
                texts.append(calibrated)
            source_y = _atom_vector(_candidate_atoms(candidate.patch_intent, candidate_atom_view="native"))
            for text in texts:
                if adapter_target_mode == "native_atoms":
                    y = _atom_vector(_candidate_atoms(candidate.patch_intent, candidate_atom_view="native"))
                else:
                    y = _semantic_anchor_atom_vector(text)
                    if adapter_target_mode == "permuted_semantic_anchor_teacher":
                        y = _permute_atom_vector(y, namespace="semantic-anchor-teacher-negative")
                if not np.any(y):
                    continue
                x_rows.append(_featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode))
                y_rows.append(y)
                procrustes_source_rows.append(source_y)
    if not x_rows:
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=np.zeros((feature_dim, len(ATOM_ORDER)), dtype=np.float64),
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode="atom_ridge",
            relative_anchor_vectors=np.zeros((0, len(ATOM_ORDER)), dtype=np.float64),
            procrustes_matrix=np.eye(len(ATOM_ORDER), dtype=np.float64),
            cca_source_mean=np.zeros(len(ATOM_ORDER), dtype=np.float64),
            cca_target_mean=np.zeros(len(ATOM_ORDER), dtype=np.float64),
            cca_source_projection=np.zeros((len(ATOM_ORDER), 0), dtype=np.float64),
            cca_target_projection=np.zeros((len(ATOM_ORDER), 0), dtype=np.float64),
            cca_correlations=np.zeros(0, dtype=np.float64),
            cca_rank=0,
            lstirp_source_anchor_vectors=np.zeros((0, len(ATOM_ORDER)), dtype=np.float64),
            lstirp_target_anchor_vectors=np.zeros((0, len(ATOM_ORDER)), dtype=np.float64),
            lstirp_source_relative_mean=np.zeros(0, dtype=np.float64),
            lstirp_target_relative_mean=np.zeros(0, dtype=np.float64),
            lstirp_translation=np.zeros((0, 0), dtype=np.float64),
            lstirp_source_anchor_count=0,
            lstirp_target_anchor_count=0,
            inverse_relative_source_mean=np.zeros(len(ATOM_ORDER), dtype=np.float64),
            inverse_relative_target_mean=np.zeros(len(ATOM_ORDER), dtype=np.float64),
            inverse_relative_source_anchors=np.zeros((0, len(ATOM_ORDER)), dtype=np.float64),
            inverse_relative_target_anchors=np.zeros((0, len(ATOM_ORDER)), dtype=np.float64),
            inverse_relative_map=np.zeros((len(ATOM_ORDER), len(ATOM_ORDER)), dtype=np.float64),
            inverse_relative_anchor_count=0,
            inverse_relative_condition_number=0.0,
            sinkhorn_ot_transport_map=np.eye(len(ATOM_ORDER), dtype=np.float64),
            sinkhorn_ot_entropy=0.0,
            sinkhorn_ot_coupling_l1=0.0,
            sinkhorn_ot_objective=0.0,
            sinkhorn_ot_coupling_entropy=0.0,
            sinkhorn_ot_coupling_top1_mass=0.0,
            sinkhorn_ot_row_residual_l1=0.0,
            sinkhorn_ot_col_residual_l1=0.0,
            ot_gw_transport_map=np.eye(len(ATOM_ORDER), dtype=np.float64),
            ot_gw_iterations=0,
            ot_gw_sinkhorn_iterations=0,
            ot_gw_entropy=0.0,
            ot_gw_fused_weight=0.0,
            ot_gw_coupling_l1=0.0,
            ot_gw_objective=0.0,
            ot_gw_coupling_entropy=0.0,
            ot_gw_coupling_top1_mass=0.0,
            ot_gw_row_residual_l1=0.0,
            ot_gw_col_residual_l1=0.0,
            transport_source_marginal_entropy=0.0,
            transport_target_marginal_entropy=0.0,
        )
    x = np.stack(x_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    xtx = x.T @ x
    regularized = xtx + ridge * np.eye(feature_dim, dtype=np.float64)
    weights = np.linalg.solve(regularized, x.T @ y)
    cca = _ridge_cca_components(procrustes_source_rows, y_rows, ridge=ridge)
    lstirp = _lstirp_components(procrustes_source_rows, y_rows, ridge=ridge)
    inverse_relative = _inverse_relative_components(procrustes_source_rows, y_rows, ridge=ridge)
    ot_gw = _ot_gw_components(procrustes_source_rows, y_rows)
    return LearnedSynonymDictionary(
        feature_dim=feature_dim,
        weights=weights,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        receiver_mode="atom_ridge",
        relative_anchor_vectors=_relative_anchor_matrix(y_rows),
        procrustes_matrix=_orthogonal_procrustes_matrix(procrustes_source_rows, y_rows),
        cca_source_mean=cca["source_mean"],
        cca_target_mean=cca["target_mean"],
        cca_source_projection=cca["source_projection"],
        cca_target_projection=cca["target_projection"],
        cca_correlations=cca["correlations"],
        cca_rank=int(cca["rank"]),
        lstirp_source_anchor_vectors=lstirp["source_anchor_vectors"],
        lstirp_target_anchor_vectors=lstirp["target_anchor_vectors"],
        lstirp_source_relative_mean=lstirp["source_relative_mean"],
        lstirp_target_relative_mean=lstirp["target_relative_mean"],
        lstirp_translation=lstirp["translation"],
        lstirp_source_anchor_count=int(lstirp["source_anchor_count"]),
        lstirp_target_anchor_count=int(lstirp["target_anchor_count"]),
        inverse_relative_source_mean=inverse_relative["source_mean"],
        inverse_relative_target_mean=inverse_relative["target_mean"],
        inverse_relative_source_anchors=inverse_relative["source_anchors"],
        inverse_relative_target_anchors=inverse_relative["target_anchors"],
        inverse_relative_map=inverse_relative["map"],
        inverse_relative_anchor_count=int(inverse_relative["anchor_count"]),
        inverse_relative_condition_number=float(inverse_relative["condition_number"]),
        sinkhorn_ot_transport_map=ot_gw["sinkhorn_map"],
        sinkhorn_ot_entropy=float(ot_gw["entropy"]),
        sinkhorn_ot_coupling_l1=float(ot_gw["sinkhorn_coupling_l1"]),
        sinkhorn_ot_objective=float(ot_gw["sinkhorn_objective"]),
        sinkhorn_ot_coupling_entropy=float(ot_gw["sinkhorn_coupling_entropy"]),
        sinkhorn_ot_coupling_top1_mass=float(ot_gw["sinkhorn_coupling_top1_mass"]),
        sinkhorn_ot_row_residual_l1=float(ot_gw["sinkhorn_row_residual_l1"]),
        sinkhorn_ot_col_residual_l1=float(ot_gw["sinkhorn_col_residual_l1"]),
        ot_gw_transport_map=ot_gw["map"],
        ot_gw_iterations=int(ot_gw["iterations"]),
        ot_gw_sinkhorn_iterations=int(ot_gw["sinkhorn_iterations"]),
        ot_gw_entropy=float(ot_gw["entropy"]),
        ot_gw_fused_weight=float(ot_gw["fused_weight"]),
        ot_gw_coupling_l1=float(ot_gw["coupling_l1"]),
        ot_gw_objective=float(ot_gw["objective"]),
        ot_gw_coupling_entropy=float(ot_gw["coupling_entropy"]),
        ot_gw_coupling_top1_mass=float(ot_gw["coupling_top1_mass"]),
        ot_gw_row_residual_l1=float(ot_gw["row_residual_l1"]),
        ot_gw_col_residual_l1=float(ot_gw["col_residual_l1"]),
        transport_source_marginal_entropy=float(ot_gw["source_marginal_entropy"]),
        transport_target_marginal_entropy=float(ot_gw["target_marginal_entropy"]),
    )


def _fit_contrastive_bilinear_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    negative_source_controls: int = 0,
    seed: int = 0,
    receiver_mode: str = "contrastive_bilinear",
) -> LearnedSynonymDictionary:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    source_vectors = [
        _atom_vector(_source_private_atoms(example.private_test_log, mode="matched")) for example in examples
    ]
    rng = random.Random(seed)

    def add_candidate_rows(example: Example, source_vector: np.ndarray, matched_source: bool) -> None:
        if not np.any(source_vector):
            return
        answer_index = _answer_index(example)
        for idx, candidate in enumerate(example.candidates):
            texts = [candidate.patch_intent]
            calibrated = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view)
            if calibrated != candidate.patch_intent:
                texts.append(calibrated)
            label = 1.0 if matched_source and idx == answer_index else 0.0
            for text in texts:
                features = _featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode)
                x_rows.append(np.outer(features, source_vector).reshape(-1))
                y_rows.append(label)

    for example_index, example in enumerate(examples):
        add_candidate_rows(example, source_vectors[example_index], matched_source=True)
        if negative_source_controls <= 0 or len(examples) < 2:
            continue
        candidate_indices = [idx for idx in range(len(examples)) if idx != example_index and np.any(source_vectors[idx])]
        rng.shuffle(candidate_indices)
        for negative_index in candidate_indices[:negative_source_controls]:
            add_candidate_rows(example, source_vectors[negative_index], matched_source=False)
    if not x_rows:
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=np.zeros((feature_dim, len(ATOM_ORDER)), dtype=np.float64),
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode=receiver_mode,
            receiver_effective_rank=0,
        )
    x = np.stack(x_rows, axis=0)
    y = np.array(y_rows, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    if x_aug.shape[1] > x_aug.shape[0]:
        k = x_aug @ x_aug.T
        k += ridge * np.eye(k.shape[0], dtype=np.float64)
        alpha = np.linalg.solve(k, y)
        solution = x_aug.T @ alpha
    else:
        xtx = x_aug.T @ x_aug
        xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
        xtx[-1, -1] -= ridge
        solution = np.linalg.solve(xtx, x_aug.T @ y)
    return LearnedSynonymDictionary(
        feature_dim=feature_dim,
        weights=solution[:-1].reshape(feature_dim, len(ATOM_ORDER)),
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        receiver_mode=receiver_mode,
        bias=float(solution[-1]),
        receiver_effective_rank=int(np.linalg.matrix_rank(solution[:-1].reshape(feature_dim, len(ATOM_ORDER)))),
    )


def _fit_contrastive_low_rank_query_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    negative_source_controls: int = 0,
    seed: int = 0,
    contrastive_rank: int = 4,
) -> LearnedSynonymDictionary:
    if contrastive_rank <= 0:
        raise ValueError("contrastive_rank must be positive for contrastive_low_rank_query")
    dictionary = _fit_contrastive_bilinear_dictionary(
        examples=examples,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        negative_source_controls=negative_source_controls,
        seed=seed,
        receiver_mode="contrastive_low_rank_query",
    )
    if dictionary.weights.size == 0:
        effective_rank = 0
    else:
        u, singular_values, vh = np.linalg.svd(dictionary.weights, full_matrices=False)
        effective_rank = min(contrastive_rank, int(np.count_nonzero(singular_values > 1e-12)))
        if effective_rank == 0:
            dictionary.weights = np.zeros_like(dictionary.weights)
        else:
            dictionary.weights = (u[:, :effective_rank] * singular_values[:effective_rank]) @ vh[:effective_rank, :]
    dictionary.contrastive_rank = contrastive_rank
    dictionary.receiver_effective_rank = effective_rank
    return dictionary


def _fit_contrastive_low_rank_factor_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    negative_source_controls: int = 0,
    seed: int = 0,
    contrastive_rank: int = 4,
    low_rank_factor_epochs: int = 250,
    low_rank_factor_lr: float = 0.05,
    low_rank_factor_loss: str = "bce",
) -> LearnedSynonymDictionary:
    if contrastive_rank <= 0:
        raise ValueError("contrastive_rank must be positive for contrastive_low_rank_factor")
    if low_rank_factor_epochs <= 0:
        raise ValueError("low_rank_factor_epochs must be positive")
    if low_rank_factor_lr <= 0:
        raise ValueError("low_rank_factor_lr must be positive")
    if low_rank_factor_loss not in {"bce", "squared"}:
        raise ValueError(f"unknown low_rank_factor_loss {low_rank_factor_loss!r}")

    x_rows: list[np.ndarray] = []
    atom_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    source_vectors = [
        _atom_vector(_source_private_atoms(example.private_test_log, mode="matched")) for example in examples
    ]
    py_rng = random.Random(seed)

    def add_candidate_rows(example: Example, source_vector: np.ndarray, matched_source: bool) -> None:
        if not np.any(source_vector):
            return
        answer_index = _answer_index(example)
        for idx, candidate in enumerate(example.candidates):
            texts = [candidate.patch_intent]
            calibrated = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view)
            if calibrated != candidate.patch_intent:
                texts.append(calibrated)
            label = 1.0 if matched_source and idx == answer_index else 0.0
            for text in texts:
                x_rows.append(_featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode))
                atom_rows.append(source_vector)
                y_rows.append(label)

    for example_index, example in enumerate(examples):
        add_candidate_rows(example, source_vectors[example_index], matched_source=True)
        if negative_source_controls <= 0 or len(examples) < 2:
            continue
        candidate_indices = [idx for idx in range(len(examples)) if idx != example_index and np.any(source_vectors[idx])]
        py_rng.shuffle(candidate_indices)
        for negative_index in candidate_indices[:negative_source_controls]:
            add_candidate_rows(example, source_vectors[negative_index], matched_source=False)

    atom_dim = len(ATOM_ORDER)
    if not x_rows:
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=np.zeros((feature_dim, atom_dim), dtype=np.float64),
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode="contrastive_low_rank_factor",
            contrastive_rank=contrastive_rank,
            receiver_effective_rank=0,
            left_factors=np.zeros((feature_dim, contrastive_rank), dtype=np.float64),
            right_factors=np.zeros((atom_dim, contrastive_rank), dtype=np.float64),
        )

    x = np.stack(x_rows, axis=0)
    atoms = np.stack(atom_rows, axis=0)
    y = np.array(y_rows, dtype=np.float64)
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(max(feature_dim, atom_dim))
    left = rng.normal(0.0, scale, size=(feature_dim, contrastive_rank))
    right = rng.normal(0.0, scale, size=(atom_dim, contrastive_rank))
    prior = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    bias = float(np.log(prior / (1.0 - prior))) if low_rank_factor_loss == "bce" else prior

    for _ in range(low_rank_factor_epochs):
        projected_x = x @ left
        projected_atoms = atoms @ right
        scores = np.sum(projected_x * projected_atoms, axis=1) + bias
        if low_rank_factor_loss == "bce":
            clipped = np.clip(scores, -40.0, 40.0)
            errors = 1.0 / (1.0 + np.exp(-clipped)) - y
        else:
            errors = scores - y
        errors /= max(1, len(y))
        grad_left = x.T @ (errors[:, None] * projected_atoms) + ridge * left
        grad_right = atoms.T @ (errors[:, None] * projected_x) + ridge * right
        grad_bias = float(errors.sum())
        left -= low_rank_factor_lr * grad_left
        right -= low_rank_factor_lr * grad_right
        bias -= low_rank_factor_lr * grad_bias

    weights = left @ right.T
    return LearnedSynonymDictionary(
        feature_dim=feature_dim,
        weights=weights,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        receiver_mode="contrastive_low_rank_factor",
        bias=bias,
        contrastive_rank=contrastive_rank,
        receiver_effective_rank=int(np.linalg.matrix_rank(weights)),
        left_factors=left,
        right_factors=right,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    denom = float(exp_values.sum())
    if denom <= 0.0:
        return np.full_like(values, 1.0 / len(values), dtype=np.float64)
    return exp_values / denom


def _jepa_query_context(
    *,
    payload_vector: np.ndarray,
    atom_embeddings: np.ndarray,
    query_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(payload_vector > 0.0)
    hidden_dim = atom_embeddings.shape[1]
    if active.size == 0:
        return np.zeros(hidden_dim, dtype=np.float64), np.zeros((query_vectors.shape[0], 0), dtype=np.float64)
    active_embeddings = atom_embeddings[active]
    active_weights = payload_vector[active]
    active_weights = active_weights / max(float(active_weights.sum()), 1e-8)
    contexts: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    scale = np.sqrt(max(1, hidden_dim))
    for query in query_vectors:
        logits = (active_embeddings @ query) / scale + np.log(np.maximum(active_weights, 1e-8))
        attention = _softmax(logits)
        attentions.append(attention)
        contexts.append(attention @ active_embeddings)
    context = np.mean(np.stack(contexts, axis=0), axis=0)
    norm = float(np.linalg.norm(context))
    if norm > 0:
        context = context / norm
    return context, np.stack(attentions, axis=0)


def _jepa_attention_features(
    *,
    features: np.ndarray,
    payload_vector: np.ndarray,
    query_factors: np.ndarray,
    atom_keys: np.ndarray,
    atom_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(payload_vector > 0.0)
    query_count = query_factors.shape[1]
    hidden_dim = query_factors.shape[2]
    if active.size == 0:
        return np.zeros(query_count * hidden_dim, dtype=np.float64), np.zeros((query_count, 0), dtype=np.float64)
    queries = np.einsum("f,fkd->kd", features, query_factors)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    keys = atom_keys[active]
    values = atom_values[active] * payload_vector[active, None]
    attentions: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    scale = np.sqrt(max(1, hidden_dim))
    for query in queries:
        attention = _softmax((keys @ query) / scale)
        attentions.append(attention)
        contexts.append(attention @ values)
    flat = np.stack(contexts, axis=0).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm > 0:
        flat = flat / norm
    return flat, np.stack(attentions, axis=0)


def _fit_jepa_query_resampler_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    negative_source_controls: int = 0,
    seed: int = 0,
    jepa_query_count: int = 8,
    jepa_hidden_dim: int = 32,
) -> LearnedSynonymDictionary:
    if jepa_query_count <= 0:
        raise ValueError("jepa_query_count must be positive")
    if jepa_hidden_dim <= 0:
        raise ValueError("jepa_hidden_dim must be positive")
    atom_predictor = _fit_ridge_dictionary(
        examples=examples,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
    )
    atom_dim = len(ATOM_ORDER)
    rng = np.random.default_rng(seed)
    query_factors = rng.normal(
        0.0,
        1.0 / np.sqrt(max(1, feature_dim)),
        size=(feature_dim, jepa_query_count, jepa_hidden_dim),
    )
    atom_keys = rng.normal(0.0, 1.0 / np.sqrt(jepa_hidden_dim), size=(atom_dim, jepa_hidden_dim))
    atom_values = rng.normal(0.0, 1.0 / np.sqrt(jepa_hidden_dim), size=(atom_dim, jepa_hidden_dim))
    atom_keys /= np.maximum(np.linalg.norm(atom_keys, axis=1, keepdims=True), 1e-8)
    atom_values /= np.maximum(np.linalg.norm(atom_values, axis=1, keepdims=True), 1e-8)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    source_vectors = [
        _atom_vector(_source_private_atoms(example.private_test_log, mode="matched")) for example in examples
    ]
    py_rng = random.Random(seed)

    def add_candidate_rows(example: Example, source_vector: np.ndarray, matched_source: bool) -> None:
        if not np.any(source_vector):
            return
        answer_index = _answer_index(example)
        for idx, candidate in enumerate(example.candidates):
            texts = [candidate.patch_intent]
            calibrated = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view)
            if calibrated != candidate.patch_intent:
                texts.append(calibrated)
            label = 1.0 if matched_source and idx == answer_index else 0.0
            for text in texts:
                features = _featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode)
                context, _ = _jepa_attention_features(
                    features=features,
                    payload_vector=source_vector,
                    query_factors=query_factors,
                    atom_keys=atom_keys,
                    atom_values=atom_values,
                )
                x_rows.append(context)
                y_rows.append(label)

    for example_index, example in enumerate(examples):
        add_candidate_rows(example, source_vectors[example_index], matched_source=True)
        if negative_source_controls <= 0 or len(examples) < 2:
            continue
        candidate_indices = [idx for idx in range(len(examples)) if idx != example_index and np.any(source_vectors[idx])]
        py_rng.shuffle(candidate_indices)
        for negative_index in candidate_indices[:negative_source_controls]:
            add_candidate_rows(example, source_vectors[negative_index], matched_source=False)

    if not x_rows:
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=atom_predictor.weights,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode="jepa_query_resampler",
            receiver_effective_rank=0,
            resampler_query_factors=query_factors,
            resampler_atom_keys=atom_keys,
            resampler_atom_values=atom_values,
            resampler_output=np.zeros(jepa_query_count * jepa_hidden_dim, dtype=np.float64),
            jepa_query_count=jepa_query_count,
            jepa_hidden_dim=jepa_hidden_dim,
            jepa_query_entropy=0.0,
            jepa_context_variance=0.0,
        )

    x = np.stack(x_rows, axis=0)
    y = np.array(y_rows, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[-1, -1] -= ridge
    solution = np.linalg.solve(xtx, x_aug.T @ y)
    resampler_output = solution[:-1]
    bias = float(solution[-1])
    receiver_effective_rank = int(np.linalg.matrix_rank(x))

    entropy_values: list[float] = []
    source_contexts: list[np.ndarray] = []
    for source_vector in source_vectors:
        context, attentions = _jepa_attention_features(
            features=np.ones(feature_dim, dtype=np.float64) / np.sqrt(feature_dim),
            payload_vector=source_vector,
            query_factors=query_factors,
            atom_keys=atom_keys,
            atom_values=atom_values,
        )
        source_contexts.append(context)
        if attentions.size:
            entropy_values.extend(
                float(-np.sum(attention * np.log(np.maximum(attention, 1e-8)))) for attention in attentions
            )
    context_variance = float(np.mean(np.var(np.stack(source_contexts, axis=0), axis=0))) if source_contexts else 0.0
    return LearnedSynonymDictionary(
        feature_dim=feature_dim,
        weights=atom_predictor.weights,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        receiver_mode="jepa_query_resampler",
        bias=bias,
        receiver_effective_rank=receiver_effective_rank,
        resampler_query_factors=query_factors,
        resampler_atom_keys=atom_keys,
        resampler_atom_values=atom_values,
        resampler_output=resampler_output,
        jepa_query_count=jepa_query_count,
        jepa_hidden_dim=jepa_hidden_dim,
        jepa_query_entropy=statistics.fmean(entropy_values) if entropy_values else 0.0,
        jepa_context_variance=context_variance,
    )


def _fit_jepa_query_resampler_trainable_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    negative_source_controls: int = 0,
    seed: int = 0,
    jepa_query_count: int = 8,
    jepa_hidden_dim: int = 32,
    jepa_train_epochs: int = 100,
    jepa_lr: float = 0.01,
    jepa_weight_decay: float = 0.001,
    control_regularized: bool = False,
    pool_contrastive: bool = False,
) -> LearnedSynonymDictionary:
    if jepa_query_count <= 0:
        raise ValueError("jepa_query_count must be positive")
    if jepa_hidden_dim <= 0:
        raise ValueError("jepa_hidden_dim must be positive")
    if jepa_train_epochs <= 0:
        raise ValueError("jepa_train_epochs must be positive")
    if jepa_lr <= 0:
        raise ValueError("jepa_lr must be positive")
    if jepa_weight_decay < 0:
        raise ValueError("jepa_weight_decay must be non-negative")

    atom_predictor = _fit_ridge_dictionary(
        examples=examples,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
    )
    atom_dim = len(ATOM_ORDER)
    rng = np.random.default_rng(seed)
    query_factors_init = rng.normal(
        0.0,
        1.0 / np.sqrt(max(1, feature_dim)),
        size=(feature_dim, jepa_query_count, jepa_hidden_dim),
    )
    atom_keys_init = rng.normal(0.0, 1.0 / np.sqrt(jepa_hidden_dim), size=(atom_dim, jepa_hidden_dim))
    atom_values_init = rng.normal(0.0, 1.0 / np.sqrt(jepa_hidden_dim), size=(atom_dim, jepa_hidden_dim))

    feature_rows: list[np.ndarray] = []
    atom_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    row_weights: list[float] = []
    source_vectors = [
        _atom_vector(_source_private_atoms(example.private_test_log, mode="matched")) for example in examples
    ]
    py_rng = random.Random(seed)

    def deranged_source_vector(source_vector: np.ndarray) -> np.ndarray:
        active = np.flatnonzero(source_vector > 0.0)
        if active.size == 0:
            return source_vector.copy()
        deranged = np.zeros_like(source_vector)
        if active.size == 1:
            dst = (int(active[0]) + 1) % len(source_vector)
            deranged[dst] = source_vector[active[0]]
            return deranged
        for src, dst in zip(active, np.roll(active, 1), strict=True):
            deranged[dst] = source_vector[src]
        return deranged

    def random_same_byte_source_vector(source_vector: np.ndarray) -> np.ndarray:
        active = np.flatnonzero(source_vector > 0.0)
        random_vector = np.zeros_like(source_vector)
        if active.size == 0:
            return random_vector
        inactive = [idx for idx in range(len(source_vector)) if idx not in set(int(atom) for atom in active)]
        pool = inactive or list(range(len(source_vector)))
        choices = py_rng.sample(pool, k=min(len(pool), len(active)))
        values = [float(source_vector[idx]) for idx in active]
        py_rng.shuffle(values)
        for idx, value in zip(choices, values, strict=True):
            random_vector[idx] = value
        return random_vector

    def receiver_mode_name() -> str:
        if pool_contrastive:
            return "jepa_query_resampler_pool_contrastive"
        if control_regularized:
            return "jepa_query_resampler_control_regularized"
        return "jepa_query_resampler_trainable"

    if pool_contrastive:
        pool_feature_rows: list[np.ndarray] = []
        pool_atom_rows: list[np.ndarray] = []
        pool_targets: list[int] = []
        pool_weights: list[float] = []

        def candidate_feature_matrix(example: Example, *, calibrated: bool) -> np.ndarray:
            texts = []
            for candidate in example.candidates:
                text = candidate.patch_intent
                if calibrated:
                    text = _candidate_surface_text(text, candidate_atom_view=calibration_atom_view)
                texts.append(text)
            return np.stack(
                [_featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode) for text in texts],
                axis=0,
            )

        def add_pool_group(
            example: Example,
            source_vector: np.ndarray,
            target_index: int,
            *,
            weight: float,
        ) -> None:
            if not np.any(source_vector):
                return
            matrices = [candidate_feature_matrix(example, calibrated=False)]
            calibrated = candidate_feature_matrix(example, calibrated=True)
            if not np.allclose(matrices[0], calibrated):
                matrices.append(calibrated)
            for matrix in matrices:
                pool_feature_rows.append(matrix)
                pool_atom_rows.append(source_vector)
                pool_targets.append(target_index)
                pool_weights.append(weight)

        for example_index, example in enumerate(examples):
            answer_index = _answer_index(example)
            prior_index = _prior_index(example)
            source_vector = source_vectors[example_index]
            add_pool_group(example, source_vector, answer_index, weight=1.0)
            if negative_source_controls > 0 and len(examples) >= 2:
                candidate_indices = [
                    idx for idx in range(len(examples)) if idx != example_index and np.any(source_vectors[idx])
                ]
                py_rng.shuffle(candidate_indices)
                for negative_index in candidate_indices[:negative_source_controls]:
                    add_pool_group(example, source_vectors[negative_index], prior_index, weight=2.0)
            add_pool_group(example, deranged_source_vector(source_vector), prior_index, weight=2.0)
            add_pool_group(example, random_same_byte_source_vector(source_vector), prior_index, weight=2.0)

        if not pool_feature_rows:
            return LearnedSynonymDictionary(
                feature_dim=feature_dim,
                weights=atom_predictor.weights,
                top_k=top_k,
                min_score=min_score,
                text_feature_mode=text_feature_mode,
                receiver_mode=receiver_mode_name(),
                receiver_effective_rank=0,
                resampler_query_factors=query_factors_init,
                resampler_atom_keys=atom_keys_init,
                resampler_atom_values=atom_values_init,
                resampler_output=np.zeros(jepa_query_count * jepa_hidden_dim, dtype=np.float64),
                jepa_query_count=jepa_query_count,
                jepa_hidden_dim=jepa_hidden_dim,
                jepa_query_entropy=0.0,
                jepa_context_variance=0.0,
            )

        import torch
        import torch.nn.functional as F

        torch.manual_seed(seed)
        features = torch.tensor(np.stack(pool_feature_rows, axis=0), dtype=torch.float32)
        atoms = torch.tensor(np.stack(pool_atom_rows, axis=0), dtype=torch.float32)
        targets = torch.tensor(np.array(pool_targets, dtype=np.int64), dtype=torch.long)
        weights = torch.tensor(np.array(pool_weights, dtype=np.float32), dtype=torch.float32)
        query_factors = torch.tensor(query_factors_init, dtype=torch.float32, requires_grad=True)
        atom_keys = torch.tensor(atom_keys_init, dtype=torch.float32, requires_grad=True)
        atom_values = torch.tensor(atom_values_init, dtype=torch.float32, requires_grad=True)
        output = torch.zeros(jepa_query_count * jepa_hidden_dim, dtype=torch.float32, requires_grad=True)
        bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.AdamW(
            [query_factors, atom_keys, atom_values, output, bias],
            lr=jepa_lr,
            weight_decay=jepa_weight_decay,
        )
        candidate_count = features.shape[1]

        def pool_forward_scores() -> tuple[Any, Any]:
            flat_features = features.reshape(-1, feature_dim)
            repeated_atoms = atoms[:, None, :].expand(-1, candidate_count, -1).reshape(-1, atom_dim)
            queries = torch.einsum("bf,fkh->bkh", flat_features, query_factors)
            queries = F.normalize(queries, p=2, dim=-1, eps=1e-8)
            keys = F.normalize(atom_keys, p=2, dim=-1, eps=1e-8)
            values = F.normalize(atom_values, p=2, dim=-1, eps=1e-8)
            logits = torch.einsum("bkh,ah->bka", queries, keys) / np.sqrt(max(1, jepa_hidden_dim))
            active = repeated_atoms > 0.0
            logits = logits.masked_fill(~active[:, None, :], -1.0e4)
            attention = torch.softmax(logits, dim=-1)
            weighted_values = values[None, :, :] * repeated_atoms[:, :, None]
            context = torch.einsum("bka,bah->bkh", attention, weighted_values).reshape(
                flat_features.shape[0], -1
            )
            context = F.normalize(context, p=2, dim=-1, eps=1e-8)
            scores = (context @ output + bias).reshape(features.shape[0], candidate_count)
            return scores, context

        for _ in range(jepa_train_epochs):
            optimizer.zero_grad(set_to_none=True)
            scores, context = pool_forward_scores()
            per_group_loss = F.cross_entropy(scores, targets, reduction="none")
            loss = (per_group_loss * weights).sum() / weights.sum().clamp_min(1.0)
            answer_scores = scores[torch.arange(scores.shape[0]), targets]
            masked_scores = scores.masked_fill(
                F.one_hot(targets, num_classes=candidate_count).to(torch.bool),
                -1.0e4,
            )
            rank_margin = torch.relu(masked_scores.max(dim=1).values - answer_scores + 0.35).mean()
            loss = loss + 0.10 * rank_margin
            if context.shape[0] > 1:
                variance_floor = torch.relu(0.02 - torch.var(context, dim=0)).mean()
                loss = loss + 0.03 * variance_floor
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, context = pool_forward_scores()
            trained_query_factors = query_factors.detach().cpu().numpy().astype(np.float64)
            trained_atom_keys = F.normalize(atom_keys, p=2, dim=-1, eps=1e-8).detach().cpu().numpy().astype(np.float64)
            trained_atom_values = F.normalize(atom_values, p=2, dim=-1, eps=1e-8).detach().cpu().numpy().astype(np.float64)
            trained_output = output.detach().cpu().numpy().astype(np.float64)
            trained_bias = float(bias.detach().cpu().item())
            receiver_effective_rank = int(np.linalg.matrix_rank(context.detach().cpu().numpy()))

        entropy_values: list[float] = []
        source_contexts: list[np.ndarray] = []
        probe_features = np.ones(feature_dim, dtype=np.float64) / np.sqrt(feature_dim)
        for source_vector in source_vectors:
            context_np, attentions = _jepa_attention_features(
                features=probe_features,
                payload_vector=source_vector,
                query_factors=trained_query_factors,
                atom_keys=trained_atom_keys,
                atom_values=trained_atom_values,
            )
            source_contexts.append(context_np)
            if attentions.size:
                entropy_values.extend(
                    float(-np.sum(attention * np.log(np.maximum(attention, 1e-8)))) for attention in attentions
                )
        context_variance = (
            float(np.mean(np.var(np.stack(source_contexts, axis=0), axis=0))) if source_contexts else 0.0
        )
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=atom_predictor.weights,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode=receiver_mode_name(),
            bias=trained_bias,
            receiver_effective_rank=receiver_effective_rank,
            resampler_query_factors=trained_query_factors,
            resampler_atom_keys=trained_atom_keys,
            resampler_atom_values=trained_atom_values,
            resampler_output=trained_output,
            jepa_query_count=jepa_query_count,
            jepa_hidden_dim=jepa_hidden_dim,
            jepa_query_entropy=statistics.fmean(entropy_values) if entropy_values else 0.0,
            jepa_context_variance=context_variance,
            jepa_trainable_factors=True,
            jepa_train_epochs=jepa_train_epochs,
            jepa_lr=jepa_lr,
            jepa_weight_decay=jepa_weight_decay,
        )

    def add_candidate_rows(
        example: Example,
        source_vector: np.ndarray,
        matched_source: bool,
        *,
        weight: float = 1.0,
    ) -> None:
        if not np.any(source_vector):
            return
        answer_index = _answer_index(example)
        for idx, candidate in enumerate(example.candidates):
            texts = [candidate.patch_intent]
            calibrated = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view)
            if calibrated != candidate.patch_intent:
                texts.append(calibrated)
            label = 1.0 if matched_source and idx == answer_index else 0.0
            for text in texts:
                feature_rows.append(_featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode))
                atom_rows.append(source_vector)
                y_rows.append(label)
                row_weights.append(weight)

    for example_index, example in enumerate(examples):
        add_candidate_rows(example, source_vectors[example_index], matched_source=True)
        if negative_source_controls <= 0 or len(examples) < 2:
            continue
        candidate_indices = [idx for idx in range(len(examples)) if idx != example_index and np.any(source_vectors[idx])]
        py_rng.shuffle(candidate_indices)
        for negative_index in candidate_indices[:negative_source_controls]:
            add_candidate_rows(
                example,
                source_vectors[negative_index],
                matched_source=False,
                weight=3.0 if control_regularized else 1.0,
            )
        if control_regularized:
            add_candidate_rows(example, deranged_source_vector(source_vectors[example_index]), matched_source=False, weight=3.0)
            add_candidate_rows(
                example,
                random_same_byte_source_vector(source_vectors[example_index]),
                matched_source=False,
                weight=3.0,
            )

    if not feature_rows:
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=atom_predictor.weights,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode=receiver_mode_name(),
            receiver_effective_rank=0,
            resampler_query_factors=query_factors_init,
            resampler_atom_keys=atom_keys_init,
            resampler_atom_values=atom_values_init,
            resampler_output=np.zeros(jepa_query_count * jepa_hidden_dim, dtype=np.float64),
            jepa_query_count=jepa_query_count,
            jepa_hidden_dim=jepa_hidden_dim,
            jepa_query_entropy=0.0,
            jepa_context_variance=0.0,
        )

    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.tensor(np.stack(feature_rows, axis=0), dtype=torch.float32)
    atoms = torch.tensor(np.stack(atom_rows, axis=0), dtype=torch.float32)
    y = torch.tensor(np.array(y_rows, dtype=np.float32), dtype=torch.float32)
    weights = torch.tensor(np.array(row_weights, dtype=np.float32), dtype=torch.float32)
    query_factors = torch.tensor(query_factors_init, dtype=torch.float32, requires_grad=True)
    atom_keys = torch.tensor(atom_keys_init, dtype=torch.float32, requires_grad=True)
    atom_values = torch.tensor(atom_values_init, dtype=torch.float32, requires_grad=True)
    output = torch.zeros(jepa_query_count * jepa_hidden_dim, dtype=torch.float32, requires_grad=True)
    prior = float(np.clip(float(np.mean(y_rows)), 1e-4, 1.0 - 1e-4))
    bias = torch.tensor(np.log(prior / (1.0 - prior)), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [query_factors, atom_keys, atom_values, output, bias],
        lr=jepa_lr,
        weight_decay=jepa_weight_decay,
    )
    positives = float(np.sum(y_rows))
    negatives = float(len(y_rows) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)

    def forward_scores() -> tuple[Any, Any]:
        queries = torch.einsum("bf,fkh->bkh", x, query_factors)
        queries = F.normalize(queries, p=2, dim=-1, eps=1e-8)
        keys = F.normalize(atom_keys, p=2, dim=-1, eps=1e-8)
        values = F.normalize(atom_values, p=2, dim=-1, eps=1e-8)
        logits = torch.einsum("bkh,ah->bka", queries, keys) / np.sqrt(max(1, jepa_hidden_dim))
        active = atoms > 0.0
        logits = logits.masked_fill(~active[:, None, :], -1.0e4)
        attention = torch.softmax(logits, dim=-1)
        weighted_values = values[None, :, :] * atoms[:, :, None]
        context = torch.einsum("bka,bah->bkh", attention, weighted_values).reshape(x.shape[0], -1)
        context = F.normalize(context, p=2, dim=-1, eps=1e-8)
        return context @ output + bias, context

    for _ in range(jepa_train_epochs):
        optimizer.zero_grad(set_to_none=True)
        scores, context = forward_scores()
        per_row_loss = F.binary_cross_entropy_with_logits(scores, y, pos_weight=pos_weight, reduction="none")
        loss = (per_row_loss * weights).sum() / weights.sum().clamp_min(1.0)
        if control_regularized:
            positive_scores = scores[y > 0.5]
            negative_scores = scores[y <= 0.5]
            if positive_scores.numel() and negative_scores.numel():
                control_margin = torch.relu(negative_scores[:, None] - positive_scores[None, :] + 0.25).mean()
                loss = loss + 0.15 * control_margin
        if context.shape[0] > 1:
            variance_floor = torch.relu(0.02 - torch.var(context, dim=0)).mean()
            loss = loss + 0.05 * variance_floor
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, context = forward_scores()
        trained_query_factors = query_factors.detach().cpu().numpy().astype(np.float64)
        trained_atom_keys = F.normalize(atom_keys, p=2, dim=-1, eps=1e-8).detach().cpu().numpy().astype(np.float64)
        trained_atom_values = F.normalize(atom_values, p=2, dim=-1, eps=1e-8).detach().cpu().numpy().astype(np.float64)
        trained_output = output.detach().cpu().numpy().astype(np.float64)
        trained_bias = float(bias.detach().cpu().item())
        receiver_effective_rank = int(np.linalg.matrix_rank(context.detach().cpu().numpy()))

    entropy_values: list[float] = []
    source_contexts: list[np.ndarray] = []
    probe_features = np.ones(feature_dim, dtype=np.float64) / np.sqrt(feature_dim)
    for source_vector in source_vectors:
        context_np, attentions = _jepa_attention_features(
            features=probe_features,
            payload_vector=source_vector,
            query_factors=trained_query_factors,
            atom_keys=trained_atom_keys,
            atom_values=trained_atom_values,
        )
        source_contexts.append(context_np)
        if attentions.size:
            entropy_values.extend(
                float(-np.sum(attention * np.log(np.maximum(attention, 1e-8)))) for attention in attentions
            )
    context_variance = float(np.mean(np.var(np.stack(source_contexts, axis=0), axis=0))) if source_contexts else 0.0
    return LearnedSynonymDictionary(
        feature_dim=feature_dim,
        weights=atom_predictor.weights,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        receiver_mode=receiver_mode_name(),
        bias=trained_bias,
        receiver_effective_rank=receiver_effective_rank,
        resampler_query_factors=trained_query_factors,
        resampler_atom_keys=trained_atom_keys,
        resampler_atom_values=trained_atom_values,
        resampler_output=trained_output,
        jepa_query_count=jepa_query_count,
        jepa_hidden_dim=jepa_hidden_dim,
        jepa_query_entropy=statistics.fmean(entropy_values) if entropy_values else 0.0,
        jepa_context_variance=context_variance,
        jepa_trainable_factors=True,
        jepa_train_epochs=jepa_train_epochs,
        jepa_lr=jepa_lr,
        jepa_weight_decay=jepa_weight_decay,
    )


def _fit_dictionary(
    *,
    examples: list[Example],
    feature_dim: int,
    ridge: float,
    calibration_atom_view: str,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    receiver_mode: str,
    adapter_target_mode: str = "native_atoms",
    contrastive_negative_sources: int,
    contrastive_rank: int,
    low_rank_factor_epochs: int = 250,
    low_rank_factor_lr: float = 0.05,
    low_rank_factor_loss: str = "bce",
    low_rank_factor_seed: int | None = None,
    jepa_query_count: int = 8,
    jepa_hidden_dim: int = 32,
    jepa_train_epochs: int = 100,
    jepa_lr: float = 0.01,
    jepa_weight_decay: float = 0.001,
    seed: int = 0,
) -> LearnedSynonymDictionary:
    if adapter_target_mode not in ADAPTER_TARGET_MODES:
        raise ValueError(f"unknown adapter target mode {adapter_target_mode!r}")
    if adapter_target_mode != "native_atoms" and receiver_mode != "atom_ridge":
        raise ValueError("adapter_target_mode is only supported with the public atom_ridge receiver")
    if receiver_mode == "atom_ridge":
        return _fit_ridge_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            adapter_target_mode=adapter_target_mode,
        )
    if receiver_mode == "contrastive_bilinear":
        return _fit_contrastive_bilinear_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            negative_source_controls=contrastive_negative_sources,
            seed=seed,
        )
    if receiver_mode == "contrastive_low_rank_query":
        return _fit_contrastive_low_rank_query_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            negative_source_controls=contrastive_negative_sources,
            seed=seed,
            contrastive_rank=contrastive_rank,
        )
    if receiver_mode == "contrastive_low_rank_factor":
        return _fit_contrastive_low_rank_factor_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            negative_source_controls=contrastive_negative_sources,
            seed=seed if low_rank_factor_seed is None else low_rank_factor_seed,
            contrastive_rank=contrastive_rank,
            low_rank_factor_epochs=low_rank_factor_epochs,
            low_rank_factor_lr=low_rank_factor_lr,
            low_rank_factor_loss=low_rank_factor_loss,
        )
    if receiver_mode == "jepa_query_resampler":
        return _fit_jepa_query_resampler_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            negative_source_controls=contrastive_negative_sources,
            seed=seed,
            jepa_query_count=jepa_query_count,
            jepa_hidden_dim=jepa_hidden_dim,
        )
    if receiver_mode in {
        "jepa_query_resampler_trainable",
        "jepa_query_resampler_control_regularized",
        "jepa_query_resampler_pool_contrastive",
    }:
        return _fit_jepa_query_resampler_trainable_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            negative_source_controls=contrastive_negative_sources,
            seed=seed,
            jepa_query_count=jepa_query_count,
            jepa_hidden_dim=jepa_hidden_dim,
            jepa_train_epochs=jepa_train_epochs,
            jepa_lr=jepa_lr,
            jepa_weight_decay=jepa_weight_decay,
            control_regularized=receiver_mode == "jepa_query_resampler_control_regularized",
            pool_contrastive=receiver_mode == "jepa_query_resampler_pool_contrastive",
        )
    raise ValueError(f"unknown receiver mode {receiver_mode!r}")


def _calibration_examples(
    *,
    mode: str,
    train_examples: list[Example],
    eval_examples: list[Example],
    calibration_count: int,
    seed: int,
) -> list[Example]:
    if mode == "train_only":
        return train_examples
    if mode == "all_public":
        return make_benchmark(examples=calibration_count, candidates=4, seed=seed, family_set="all")
    if mode == "all_public_eval_disjoint":
        excluded = {_qualified_example_id(example) for example in eval_examples}
        pool = make_benchmark(
            examples=calibration_count + max(len(eval_examples) * 4, 128),
            candidates=4,
            seed=seed,
            family_set="all",
        )
        return [example for example in pool if _qualified_example_id(example) not in excluded][:calibration_count]
    raise ValueError(f"unknown calibration mode {mode!r}")


def _qualified_example_id(example: Example) -> str:
    return f"{example.family_name}:{example.example_id}"


def _private_random_source_atoms(atoms: dict[str, float], *, rng: random.Random) -> dict[str, float]:
    if not atoms:
        return {}
    replacement_atoms = [atom for atom in ATOM_ORDER if atom not in atoms]
    rng.shuffle(replacement_atoms)
    randomized: dict[str, float] = {}
    for atom, (_, score) in zip(replacement_atoms, sorted(atoms.items(), key=lambda item: (-item[1], item[0]))):
        randomized[atom] = float(score)
    return randomized


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)


def _surface_overlap_audit(
    *,
    calibration_rows: list[Example],
    eval_rows: list[Example],
    calibration_atom_view: str,
    candidate_atom_view: str,
) -> dict[str, Any]:
    calibration_local_ids = {example.example_id for example in calibration_rows}
    eval_local_ids = [example.example_id for example in eval_rows]
    local_id_overlap = [example_id for example_id in eval_local_ids if example_id in calibration_local_ids]
    calibration_ids = {_qualified_example_id(example) for example in calibration_rows}
    eval_ids = [_qualified_example_id(example) for example in eval_rows]
    exact_id_overlap = [example_id for example_id in eval_ids if example_id in calibration_ids]
    calibration_surfaces = {
        _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view).lower()
        for example in calibration_rows
        for candidate in example.candidates
    }
    eval_surface_pairs = [
        (
            candidate.patch_intent.lower(),
            _candidate_surface_text(candidate.patch_intent, candidate_atom_view=candidate_atom_view).lower(),
        )
        for example in eval_rows
        for candidate in example.candidates
    ]
    eval_surfaces = [surface for _, surface in eval_surface_pairs]
    transformed_eval_surfaces = [
        surface for native_surface, surface in eval_surface_pairs if surface != native_surface
    ]
    calibration_tokens = {
        token
        for surface in calibration_surfaces
        for token in _word_tokens(surface)
    }
    eval_tokens = {token for surface in eval_surfaces for token in _word_tokens(surface)}
    exact_overlap = [surface for surface in eval_surfaces if surface in calibration_surfaces]
    transformed_exact_overlap = [
        surface for surface in transformed_eval_surfaces if surface in calibration_surfaces
    ]
    shared_tokens = sorted(calibration_tokens & eval_tokens)
    return {
        "calibration_atom_view": calibration_atom_view,
        "candidate_atom_view": candidate_atom_view,
        "calibration_example_count": len(calibration_rows),
        "eval_example_count": len(eval_rows),
        "calibration_eval_exact_id_overlap_count": len(exact_id_overlap),
        "calibration_eval_exact_id_overlap_sample": exact_id_overlap[:10],
        "calibration_eval_local_id_overlap_count": len(local_id_overlap),
        "calibration_eval_local_id_overlap_sample": local_id_overlap[:10],
        "calibration_surface_count": len(calibration_surfaces),
        "eval_surface_count": len(eval_surfaces),
        "exact_eval_surface_overlap_count": len(exact_overlap),
        "exact_eval_surface_overlap_rate": len(exact_overlap) / len(eval_surfaces) if eval_surfaces else 0.0,
        "transformed_eval_surface_count": len(transformed_eval_surfaces),
        "exact_transformed_eval_surface_overlap_count": len(transformed_exact_overlap),
        "exact_transformed_eval_surface_overlap_rate": (
            len(transformed_exact_overlap) / len(transformed_eval_surfaces) if transformed_eval_surfaces else 0.0
        ),
        "exact_transformed_eval_surface_overlap_sample": transformed_exact_overlap[:10],
        "shared_token_count": len(shared_tokens),
        "shared_token_sample": shared_tokens[:25],
    }


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    budget_bytes: int,
    dictionary: LearnedSynonymDictionary,
    candidate_atom_view: str,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any], dict[str, Any]]:
    decode_kwargs: dict[str, Any] = {}
    if condition == "target_only":
        return None, {"source": "target_prior"}, decode_kwargs
    if condition == "learned_synonym_dictionary_packet":
        return _encode_atoms(_source_private_atoms(example.private_test_log, mode="matched"), budget_bytes=budget_bytes), {
            "source": example.example_id
        }, decode_kwargs
    if condition == "zero_source":
        return _encode_atoms({}, budget_bytes=budget_bytes), {"source": "zero"}, decode_kwargs
    if condition == "shuffled_source":
        other = _constrained_nonoverlap_example(example, eval_examples, index)
        return _encode_atoms(_source_private_atoms(other.private_test_log, mode="matched"), budget_bytes=budget_bytes), {
            "source": other.example_id
        }, decode_kwargs
    if condition == "answer_masked_source":
        # This gate treats answer masking as a destructive source control: no
        # private exception/value/test cues may survive, because the learned
        # synonym dictionary can otherwise decode repair families from partial
        # diagnostic traces.
        return _encode_atoms({}, budget_bytes=budget_bytes), {"source": "answer_masked_strict"}, decode_kwargs
    if condition == "public_only_sidecar":
        return _encode_atoms({}, budget_bytes=budget_bytes), {"source": "public_only"}, decode_kwargs
    if condition == "target_derived_sidecar":
        return bytes([255, 0] * max(1, budget_bytes // 2))[:budget_bytes], {"source": "target_prompt_only"}, decode_kwargs
    if condition == "random_same_byte":
        return _random_packet(budget_bytes=budget_bytes, rng=rng), {"source": "random"}, decode_kwargs
    if condition == "answer_only_text":
        return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}, decode_kwargs
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_hidden_log"}, decode_kwargs
    if condition == "atom_id_derangement":
        return _encode_atoms(_source_private_atoms(example.private_test_log, mode="matched"), budget_bytes=budget_bytes), {
            "source": example.example_id
        }, {"derange": True}
    if condition == "private_random_source_atoms":
        matched_atoms = _source_private_atoms(example.private_test_log, mode="matched")
        randomized_atoms = _private_random_source_atoms(matched_atoms, rng=rng)
        return _encode_atoms(randomized_atoms, budget_bytes=budget_bytes), {
            "source": example.example_id,
            "control": "private_random_source_atoms",
        }, decode_kwargs
    if condition == "permuted_teacher_receiver":
        return _encode_atoms(_source_private_atoms(example.private_test_log, mode="matched"), budget_bytes=budget_bytes), {
            "source": example.example_id,
            "control": "permuted_teacher_receiver",
        }, decode_kwargs
    if condition == "top_atom_knockout":
        return _encode_atoms(_source_private_atoms(example.private_test_log, mode="matched"), budget_bytes=budget_bytes), {
            "source": example.example_id
        }, {"knockout": "top"}
    if condition == "private_random_knockout":
        return _encode_atoms(_source_private_atoms(example.private_test_log, mode="matched"), budget_bytes=budget_bytes), {
            "source": example.example_id
        }, {"knockout": "random"}
    if condition == "oracle_learned_candidate_atoms":
        answer = example.candidates[_answer_index(example)]
        learned_atoms = dictionary.predict_atoms(
            _candidate_surface_text(answer.patch_intent, candidate_atom_view=candidate_atom_view)
        )
        return _encode_atoms(learned_atoms, budget_bytes=budget_bytes), {"source": "oracle_learned_candidate_atoms"}, decode_kwargs
    raise ValueError(f"unknown condition {condition!r}")


def _score_candidates(
    *,
    example: Example,
    payload_atoms: dict[str, float],
    dictionary: LearnedSynonymDictionary,
    candidate_atom_view: str,
    decoder_score_mode: str,
) -> tuple[list[float], dict[str, Any]]:
    if decoder_score_mode not in DECODER_SCORE_MODES:
        raise ValueError(f"unknown decoder score mode {decoder_score_mode!r}")
    if decoder_score_mode == "global_dot":
        scores = []
        for candidate in example.candidates:
            text = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=candidate_atom_view)
            scores.append(dictionary.score_text(text, payload_atoms))
        return scores, {"decoder_score_mode": decoder_score_mode}

    if dictionary.receiver_mode != "atom_ridge":
        raise ValueError("candidate-local and relative-anchor scoring require an atom_ridge dictionary")
    candidate_vectors = []
    for candidate in example.candidates:
        text = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=candidate_atom_view)
        candidate_vectors.append(dictionary.predict_vector(text, apply_top_k=True))
    candidate_matrix = np.stack(candidate_vectors, axis=0)
    payload_vector = _atom_vector(payload_atoms)
    if decoder_score_mode == "procrustes_dot":
        procrustes_matrix = dictionary.procrustes_matrix
        if procrustes_matrix is None:
            procrustes_matrix = np.eye(len(ATOM_ORDER), dtype=np.float64)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        safe_candidates = np.divide(
            candidate_matrix,
            np.maximum(candidate_norms, 1e-12),
            out=np.zeros_like(candidate_matrix),
            where=candidate_norms > 0,
        )
        mapped_payload = payload_vector @ procrustes_matrix
        payload_norm = float(np.linalg.norm(mapped_payload))
        safe_payload = mapped_payload / payload_norm if payload_norm > 0 else mapped_payload
        scores_array = safe_candidates @ safe_payload
        scores = [float(score) for score in scores_array]
        return scores, {
            "decoder_score_mode": decoder_score_mode,
            "procrustes_matrix_frobenius": float(np.linalg.norm(procrustes_matrix)),
            "procrustes_payload_l2": payload_norm,
            "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
        }

    if decoder_score_mode in {"ridge_cca_dot", "ridge_cca_residual_norm"}:
        rank = int(dictionary.cca_rank or 0)
        source_projection = dictionary.cca_source_projection
        target_projection = dictionary.cca_target_projection
        correlations = dictionary.cca_correlations
        source_mean = dictionary.cca_source_mean
        target_mean = dictionary.cca_target_mean
        if (
            rank <= 0
            or source_projection is None
            or target_projection is None
            or correlations is None
            or source_mean is None
            or target_mean is None
        ):
            return [0.0 for _ in example.candidates], {
                "decoder_score_mode": decoder_score_mode,
                "cca_rank": 0,
                "cca_payload_l2": 0.0,
                "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
            }
        source_shared = (payload_vector - source_mean) @ source_projection
        target_shared = (candidate_matrix - target_mean[None, :]) @ target_projection
        weights = np.sqrt(np.maximum(correlations[:rank], 0.0))
        source_shared = source_shared[:rank] * weights
        target_shared = target_shared[:, :rank] * weights[None, :]
        if decoder_score_mode == "ridge_cca_residual_norm":
            target_shared = target_shared - target_shared.mean(axis=0, keepdims=True)
        row_norms = np.linalg.norm(target_shared, axis=1)
        safe_candidates = np.divide(
            target_shared,
            np.maximum(row_norms[:, None], 1e-12),
            out=np.zeros_like(target_shared),
            where=row_norms[:, None] > 0,
        )
        payload_norm = float(np.linalg.norm(source_shared))
        safe_payload = source_shared / payload_norm if payload_norm > 0 else source_shared
        scores_array = safe_candidates @ safe_payload
        scores = [float(score) for score in scores_array]
        return scores, {
            "decoder_score_mode": decoder_score_mode,
            "cca_rank": rank,
            "cca_correlations": [float(value) for value in correlations[:rank]],
            "cca_candidate_row_norms": [float(value) for value in row_norms],
            "cca_payload_l2": payload_norm,
            "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
        }

    if decoder_score_mode in {"lstirp_relative_dot", "lstirp_relative_residual_norm"}:
        source_anchors = dictionary.lstirp_source_anchor_vectors
        target_anchors = dictionary.lstirp_target_anchor_vectors
        source_mean = dictionary.lstirp_source_relative_mean
        target_mean = dictionary.lstirp_target_relative_mean
        translation = dictionary.lstirp_translation
        if (
            source_anchors is None
            or target_anchors is None
            or source_mean is None
            or target_mean is None
            or translation is None
            or source_anchors.shape[0] == 0
            or target_anchors.shape[0] == 0
        ):
            return [0.0 for _ in example.candidates], {
                "decoder_score_mode": decoder_score_mode,
                "lstirp_source_anchor_count": 0,
                "lstirp_target_anchor_count": 0,
                "lstirp_payload_l2": 0.0,
                "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
            }
        source_relative = _relative_coordinates(payload_vector[None, :], source_anchors)[0]
        mapped_payload = (source_relative - source_mean) @ translation + target_mean
        candidate_relative = _relative_coordinates(candidate_matrix, target_anchors)
        if decoder_score_mode == "lstirp_relative_residual_norm":
            candidate_relative = candidate_relative - candidate_relative.mean(axis=0, keepdims=True)
        row_norms = np.linalg.norm(candidate_relative, axis=1)
        safe_candidates = np.divide(
            candidate_relative,
            np.maximum(row_norms[:, None], 1e-12),
            out=np.zeros_like(candidate_relative),
            where=row_norms[:, None] > 0,
        )
        payload_norm = float(np.linalg.norm(mapped_payload))
        safe_payload = mapped_payload / payload_norm if payload_norm > 0 else mapped_payload
        scores_array = safe_candidates @ safe_payload
        scores = [float(score) for score in scores_array]
        return scores, {
            "decoder_score_mode": decoder_score_mode,
            "lstirp_source_anchor_count": int(source_anchors.shape[0]),
            "lstirp_target_anchor_count": int(target_anchors.shape[0]),
            "lstirp_translation_frobenius": float(np.linalg.norm(translation)),
            "lstirp_candidate_row_norms": [float(value) for value in row_norms],
            "lstirp_payload_l2": payload_norm,
            "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
        }

    if decoder_score_mode in {"inverse_relative_dot", "inverse_relative_residual_norm"}:
        source_mean = dictionary.inverse_relative_source_mean
        target_mean = dictionary.inverse_relative_target_mean
        inverse_relative_map = dictionary.inverse_relative_map
        anchor_count = int(dictionary.inverse_relative_anchor_count or 0)
        if source_mean is None or target_mean is None or inverse_relative_map is None or anchor_count <= 0:
            return [0.0 for _ in example.candidates], {
                "decoder_score_mode": decoder_score_mode,
                "inverse_relative_anchor_count": 0,
                "inverse_relative_payload_l2": 0.0,
                "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
            }
        centered_payload = payload_vector - source_mean
        payload_centered_norm = float(np.linalg.norm(centered_payload))
        safe_payload_source = centered_payload / payload_centered_norm if payload_centered_norm > 0 else centered_payload
        mapped_payload = safe_payload_source @ inverse_relative_map
        payload_norm = float(np.linalg.norm(mapped_payload))
        safe_payload = mapped_payload / payload_norm if payload_norm > 0 else mapped_payload
        centered_candidates = candidate_matrix - target_mean[None, :]
        candidate_norms = np.linalg.norm(centered_candidates, axis=1, keepdims=True)
        candidate_rows = np.divide(
            centered_candidates,
            np.maximum(candidate_norms, 1e-12),
            out=np.zeros_like(centered_candidates),
            where=candidate_norms > 0,
        )
        if decoder_score_mode == "inverse_relative_residual_norm":
            candidate_rows = candidate_rows - candidate_rows.mean(axis=0, keepdims=True)
        row_norms = np.linalg.norm(candidate_rows, axis=1)
        safe_candidates = np.divide(
            candidate_rows,
            np.maximum(row_norms[:, None], 1e-12),
            out=np.zeros_like(candidate_rows),
            where=row_norms[:, None] > 0,
        )
        scores_array = safe_candidates @ safe_payload
        scores = [float(score) for score in scores_array]
        return scores, {
            "decoder_score_mode": decoder_score_mode,
            "inverse_relative_anchor_count": anchor_count,
            "inverse_relative_condition_number": float(dictionary.inverse_relative_condition_number or 0.0),
            "inverse_relative_map_frobenius": float(np.linalg.norm(inverse_relative_map)),
            "inverse_relative_candidate_row_norms": [float(value) for value in row_norms],
            "inverse_relative_payload_l2": payload_norm,
            "candidate_local_payload_l2": float(np.linalg.norm(payload_vector)),
        }

    if decoder_score_mode in {"sinkhorn_ot_dot", "sinkhorn_ot_residual_norm"}:
        scores, metadata = _transport_scores(
            decoder_score_mode=decoder_score_mode,
            prefix="sinkhorn_ot",
            transport_map=dictionary.sinkhorn_ot_transport_map,
            payload_vector=payload_vector,
            candidate_matrix=candidate_matrix,
            residual_norm=decoder_score_mode == "sinkhorn_ot_residual_norm",
        )
        metadata.update(
            {
                "transport_kind": "sinkhorn_ot",
                "transport_cost_mode": "calibration_column_cosine",
                "transport_marginal_mode": "smoothed_empirical_atom_mass",
                "sinkhorn_ot_entropy": float(getattr(dictionary, "sinkhorn_ot_entropy", 0.0) or 0.0),
                "sinkhorn_ot_coupling_l1": float(getattr(dictionary, "sinkhorn_ot_coupling_l1", 0.0) or 0.0),
                "sinkhorn_ot_objective": float(getattr(dictionary, "sinkhorn_ot_objective", 0.0) or 0.0),
                "sinkhorn_ot_coupling_entropy": float(
                    getattr(dictionary, "sinkhorn_ot_coupling_entropy", 0.0) or 0.0
                ),
                "sinkhorn_ot_coupling_top1_mass": float(
                    getattr(dictionary, "sinkhorn_ot_coupling_top1_mass", 0.0) or 0.0
                ),
                "sinkhorn_ot_row_residual_l1": float(
                    getattr(dictionary, "sinkhorn_ot_row_residual_l1", 0.0) or 0.0
                ),
                "sinkhorn_ot_col_residual_l1": float(
                    getattr(dictionary, "sinkhorn_ot_col_residual_l1", 0.0) or 0.0
                ),
                "transport_source_marginal_entropy": float(
                    getattr(dictionary, "transport_source_marginal_entropy", 0.0) or 0.0
                ),
                "transport_target_marginal_entropy": float(
                    getattr(dictionary, "transport_target_marginal_entropy", 0.0) or 0.0
                ),
            }
        )
        return scores, metadata

    if decoder_score_mode in {
        "gromov_wasserstein_dot",
        "gromov_wasserstein_residual_norm",
        "ot_gw_dot",
        "ot_gw_residual_norm",
    }:
        residual_norm = decoder_score_mode in {"gromov_wasserstein_residual_norm", "ot_gw_residual_norm"}
        scores, metadata = _transport_scores(
            decoder_score_mode=decoder_score_mode,
            prefix="ot_gw",
            transport_map=dictionary.ot_gw_transport_map,
            payload_vector=payload_vector,
            candidate_matrix=candidate_matrix,
            residual_norm=residual_norm,
        )
        metadata.update(
            {
                "transport_kind": "fused_gromov_wasserstein",
                "transport_cost_mode": "calibration_geometry_plus_column_cosine",
                "transport_marginal_mode": "smoothed_empirical_atom_mass",
                "ot_gw_iterations": int(getattr(dictionary, "ot_gw_iterations", 0) or 0),
                "ot_gw_sinkhorn_iterations": int(getattr(dictionary, "ot_gw_sinkhorn_iterations", 0) or 0),
                "ot_gw_entropy": float(getattr(dictionary, "ot_gw_entropy", 0.0) or 0.0),
                "ot_gw_fused_weight": float(getattr(dictionary, "ot_gw_fused_weight", 0.0) or 0.0),
                "ot_gw_coupling_l1": float(getattr(dictionary, "ot_gw_coupling_l1", 0.0) or 0.0),
                "ot_gw_objective": float(getattr(dictionary, "ot_gw_objective", 0.0) or 0.0),
                "ot_gw_coupling_entropy": float(getattr(dictionary, "ot_gw_coupling_entropy", 0.0) or 0.0),
                "ot_gw_coupling_top1_mass": float(getattr(dictionary, "ot_gw_coupling_top1_mass", 0.0) or 0.0),
                "ot_gw_row_residual_l1": float(getattr(dictionary, "ot_gw_row_residual_l1", 0.0) or 0.0),
                "ot_gw_col_residual_l1": float(getattr(dictionary, "ot_gw_col_residual_l1", 0.0) or 0.0),
                "transport_source_marginal_entropy": float(
                    getattr(dictionary, "transport_source_marginal_entropy", 0.0) or 0.0
                ),
                "transport_target_marginal_entropy": float(
                    getattr(dictionary, "transport_target_marginal_entropy", 0.0) or 0.0
                ),
            }
        )
        return scores, metadata

    if decoder_score_mode in {
        "relative_anchor_dot",
        "relative_anchor_residual_norm",
        "relative_anchor_innovation_residual_norm",
        "relative_anchor_rank_innovation_residual_norm",
    }:
        anchors = dictionary.relative_anchor_vectors
        if anchors is None:
            anchors = np.zeros((0, len(ATOM_ORDER)), dtype=np.float64)
        anchor_norms = np.linalg.norm(anchors, axis=1, keepdims=True) if anchors.size else np.zeros((0, 1))
        safe_anchors = np.divide(
            anchors,
            np.maximum(anchor_norms, 1e-12),
            out=np.zeros_like(anchors),
            where=anchor_norms > 0,
        )
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        safe_candidates = np.divide(
            candidate_matrix,
            np.maximum(candidate_norms, 1e-12),
            out=np.zeros_like(candidate_matrix),
            where=candidate_norms > 0,
        )
        payload_norm = float(np.linalg.norm(payload_vector))
        safe_payload = payload_vector / payload_norm if payload_norm > 0 else payload_vector
        relative_candidates = safe_candidates @ safe_anchors.T
        relative_payload = safe_payload @ safe_anchors.T
        rank_normalized = decoder_score_mode == "relative_anchor_rank_innovation_residual_norm"
        if rank_normalized:
            relative_candidates = _rank_normalized_rows(relative_candidates)
            relative_payload = _rank_normalized_rows(relative_payload[None, :])[0]
        if decoder_score_mode == "relative_anchor_dot":
            row_norms = np.linalg.norm(relative_candidates, axis=1)
            safe_relative_candidates = np.divide(
                relative_candidates,
                np.maximum(row_norms[:, None], 1e-12),
                out=np.zeros_like(relative_candidates),
                where=row_norms[:, None] > 0,
            )
            relative_payload_norm = float(np.linalg.norm(relative_payload))
            safe_relative_payload = (
                relative_payload / relative_payload_norm if relative_payload_norm > 0 else relative_payload
            )
            scores_array = safe_relative_candidates @ safe_relative_payload
        else:
            local_mean = relative_candidates.mean(axis=0, keepdims=True)
            residual_matrix = relative_candidates - local_mean
            row_norms = np.linalg.norm(residual_matrix, axis=1)
            safe_rows = np.divide(
                residual_matrix,
                np.maximum(row_norms[:, None], 1e-12),
                out=np.zeros_like(residual_matrix),
                where=row_norms[:, None] > 0,
            )
            relative_payload_norm = float(np.linalg.norm(relative_payload))
            if decoder_score_mode in {
                "relative_anchor_innovation_residual_norm",
                "relative_anchor_rank_innovation_residual_norm",
            }:
                relative_payload = relative_payload - local_mean[0]
                relative_payload_norm = float(np.linalg.norm(relative_payload))
            safe_relative_payload = relative_payload / relative_payload_norm if relative_payload_norm > 0 else relative_payload
            scores_array = safe_rows @ safe_relative_payload
        scores = [float(score) for score in scores_array]
        return scores, {
            "decoder_score_mode": decoder_score_mode,
            "relative_anchor_count": int(anchors.shape[0]),
            "relative_anchor_row_norms": [float(value) for value in row_norms],
            "relative_anchor_payload_l2": float(np.linalg.norm(relative_payload)),
            "relative_anchor_local_mean_l2": float(np.linalg.norm(relative_candidates.mean(axis=0))),
            "relative_anchor_rank_normalized": bool(rank_normalized),
            "candidate_local_payload_l2": payload_norm,
        }

    return _candidate_local_residual_scores(
        candidate_matrix,
        payload_vector,
        decoder_score_mode=decoder_score_mode,
    )


def _predict_from_payload(
    *,
    example: Example,
    payload: bytes | None,
    budget_bytes: int,
    dictionary: LearnedSynonymDictionary,
    candidate_atom_view: str,
    decoder_score_mode: str,
    min_decision_score: float,
    null_dictionary: LearnedSynonymDictionary | None = None,
    permuted_null_weight: float = 0.75,
    derange: bool = False,
    knockout: str | None = None,
    rng: random.Random | None = None,
) -> tuple[str, dict[str, Any]]:
    prior = _prior_prediction(example)
    payload_atoms = _decode_payload_atoms(payload, budget_bytes=budget_bytes, derange=derange, knockout=knockout, rng=rng)
    if not payload_atoms:
        return prior, {"decoder": "prior", "payload_atoms": {}, "decoder_score_mode": decoder_score_mode}
    if decoder_score_mode == "candidate_local_permuted_null_gap_residual_norm":
        if null_dictionary is None:
            raise ValueError("permuted-null gap scoring requires a null_dictionary")
        base_mode = "candidate_local_residual_norm"
        active_scores, active_meta = _score_candidates(
            example=example,
            payload_atoms=payload_atoms,
            dictionary=dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode=base_mode,
        )
        null_scores, null_meta = _score_candidates(
            example=example,
            payload_atoms=payload_atoms,
            dictionary=null_dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode=base_mode,
        )
        scores = [
            float(active_score - float(permuted_null_weight) * null_score)
            for active_score, null_score in zip(active_scores, null_scores, strict=True)
        ]
        score_meta = {
            "decoder_score_mode": decoder_score_mode,
            "decoder_score_base_mode": base_mode,
            "permuted_null_weight": float(permuted_null_weight),
            "active_scores": active_scores,
            "permuted_null_scores": null_scores,
            "active_best_score": max(active_scores),
            "permuted_null_best_score": max(null_scores),
            "active_candidate_local_payload_l2": active_meta.get("candidate_local_payload_l2", 0.0),
            "permuted_null_candidate_local_payload_l2": null_meta.get("candidate_local_payload_l2", 0.0),
            "candidate_local_payload_l2": active_meta.get("candidate_local_payload_l2", 0.0),
        }
    else:
        scores, score_meta = _score_candidates(
            example=example,
            payload_atoms=payload_atoms,
            dictionary=dictionary,
            candidate_atom_view=candidate_atom_view,
            decoder_score_mode=decoder_score_mode,
        )
    best_score = max(scores)
    sorted_scores = sorted(scores, reverse=True)
    best_margin = best_score - sorted_scores[1] if len(sorted_scores) > 1 else best_score
    if best_score < min_decision_score:
        return prior, {
            "decoder": "learned_synonym_dictionary_target_preserve",
            "payload_atoms": payload_atoms,
            "scores": scores,
            "best_score": best_score,
            "best_margin": best_margin,
            "min_decision_score": min_decision_score,
            **score_meta,
        }
    tied = [idx for idx, score in enumerate(scores) if abs(score - best_score) <= 1e-8]
    labels = [candidate.label for candidate in example.candidates]
    prediction = prior if any(labels[idx] == prior for idx in tied) else labels[tied[0]]
    return prediction, {
        "decoder": "learned_synonym_dictionary",
        "payload_atoms": payload_atoms,
        "scores": scores,
        "best_score": best_score,
        "best_margin": best_margin,
        "min_decision_score": min_decision_score,
        **score_meta,
    }


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    budget_bytes: int,
    dictionary: LearnedSynonymDictionary,
    permuted_teacher_dictionary: LearnedSynonymDictionary,
    candidate_atom_view: str,
    decoder_score_mode: str,
    min_decision_score: float,
    permuted_null_weight: float,
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
        candidate_atom_view=candidate_atom_view,
        rng=rng,
    )
    active_dictionary = permuted_teacher_dictionary if condition == "permuted_teacher_receiver" else dictionary
    null_dictionary = dictionary if condition == "permuted_teacher_receiver" else permuted_teacher_dictionary
    prediction, decode_meta = _predict_from_payload(
        example=example,
        payload=payload,
        budget_bytes=budget_bytes,
        dictionary=active_dictionary,
        null_dictionary=null_dictionary,
        candidate_atom_view=candidate_atom_view,
        decoder_score_mode=decoder_score_mode,
        min_decision_score=0.0 if condition == "oracle_learned_candidate_atoms" else min_decision_score,
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
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "payload_hex": payload_hex,
        "answer_index": _answer_index(example),
        "prior_index": _prior_index(example),
        "metadata": {**meta, **decode_meta},
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    return {
        "n": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
        "strict_accuracy": sum(1 for row in rows if row["strict_correct"]) / len(rows),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _paired_bootstrap(rows: list[dict[str, Any]], *, condition: str, baseline: str, seed: int, samples: int = 1000) -> dict[str, float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = [
        float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"])
        for _, conditions in sorted(by_example.items())
    ]
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(deltas),
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def _direction_summary(rows: list[dict[str, Any]], *, budget_bytes: int, seed: int) -> dict[str, Any]:
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    metrics = {condition: _summarize(condition_rows) for condition, condition_rows in by_condition.items()}
    target = metrics["target_only"]["accuracy"]
    matched = metrics["learned_synonym_dictionary_packet"]["accuracy"]
    best_control_name = max(STRICT_SOURCE_DESTROYING_CONTROLS, key=lambda c: metrics[c]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    top_knockout = metrics["top_atom_knockout"]["accuracy"]
    random_knockout = metrics["private_random_knockout"]["accuracy"]
    oracle = metrics["oracle_learned_candidate_atoms"]["accuracy"]
    target_ci = _paired_bootstrap(rows, condition="learned_synonym_dictionary_packet", baseline="target_only", seed=seed)
    control_ci = _paired_bootstrap(rows, condition="learned_synonym_dictionary_packet", baseline=best_control_name, seed=seed + 1)
    lift = matched - target
    knockout_reduction = 0.0 if lift <= 0 else max(0.0, matched - top_knockout) / lift
    random_knockout_reduction = 0.0 if lift <= 0 else max(0.0, matched - random_knockout) / lift
    controls_ok = all(metrics[condition]["accuracy"] <= target + 0.03 for condition in STRICT_SOURCE_DESTROYING_CONTROLS)
    pass_gate = (
        matched >= target + 0.15
        and matched >= best_control + 0.10
        and controls_ok
        and target_ci["ci95_low"] > 0.05
        and oracle >= 0.80
        and knockout_reduction >= 0.50
    )
    return {
        "budget_bytes": budget_bytes,
        "n": metrics["target_only"]["n"],
        "target_accuracy": target,
        "learned_synonym_dictionary_accuracy": matched,
        "best_control_accuracy": best_control,
        "best_control_name": best_control_name,
        "learned_minus_target": matched - target,
        "learned_minus_best_control": matched - best_control,
        "oracle_learned_candidate_atoms_accuracy": oracle,
        "top_atom_knockout_accuracy": top_knockout,
        "private_random_knockout_accuracy": random_knockout,
        "top_atom_knockout_lift_reduction": knockout_reduction,
        "private_random_knockout_lift_reduction": random_knockout_reduction,
        "paired_bootstrap_vs_target": target_ci,
        "paired_bootstrap_vs_best_control": control_ci,
        "controls_ok": controls_ok,
        "pass_gate": pass_gate,
        "metrics": metrics,
    }


def _run_direction(
    *,
    output_dir: pathlib.Path,
    direction: str,
    train_family_set: str,
    eval_family_set: str,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    budgets: list[int],
    candidate_atom_view: str,
    calibration_atom_view: str,
    candidate_calibration: str,
    calibration_examples: int,
    feature_dim: int,
    ridge: float,
    top_k: int,
    min_score: float,
    text_feature_mode: str,
    adapter_target_mode: str,
    decoder_score_mode: str,
    min_decision_score: float,
    permuted_null_weight: float,
    receiver_mode: str,
    contrastive_negative_sources: int,
    contrastive_rank: int,
    low_rank_factor_epochs: int,
    low_rank_factor_lr: float,
    low_rank_factor_loss: str,
    low_rank_factor_seed: int | None,
    jepa_query_count: int,
    jepa_hidden_dim: int,
    jepa_train_epochs: int,
    jepa_lr: float,
    jepa_weight_decay: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(examples=train_examples, candidates=4, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=4, seed=eval_seed, family_set=eval_family_set)
    calibration_rows = _calibration_examples(
        mode=candidate_calibration,
        train_examples=train_rows,
        eval_examples=eval_rows,
        calibration_count=calibration_examples,
        seed=train_seed + 101,
    )
    dictionary = _fit_dictionary(
        examples=calibration_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        adapter_target_mode=adapter_target_mode,
        receiver_mode=receiver_mode,
        contrastive_negative_sources=contrastive_negative_sources,
        contrastive_rank=contrastive_rank,
        low_rank_factor_epochs=low_rank_factor_epochs,
        low_rank_factor_lr=low_rank_factor_lr,
        low_rank_factor_loss=low_rank_factor_loss,
        low_rank_factor_seed=low_rank_factor_seed,
        jepa_query_count=jepa_query_count,
        jepa_hidden_dim=jepa_hidden_dim,
        jepa_train_epochs=jepa_train_epochs,
        jepa_lr=jepa_lr,
        jepa_weight_decay=jepa_weight_decay,
        seed=train_seed + 211,
    )
    permuted_teacher_dictionary = _fit_ridge_dictionary(
        examples=calibration_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        adapter_target_mode="permuted_semantic_anchor_teacher",
    )
    surface_overlap_audit = _surface_overlap_audit(
        calibration_rows=calibration_rows,
        eval_rows=eval_rows,
        calibration_atom_view=calibration_atom_view,
        candidate_atom_view=candidate_atom_view,
    )
    budget_summaries: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}
    for budget in budgets:
        rng = random.Random(train_seed * 1000003 + eval_seed * 1009 + budget)
        rows: list[dict[str, Any]] = []
        for row_index, example in enumerate(eval_rows):
            for condition in CONDITIONS:
                rows.append(
                    _predict_condition(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        budget_bytes=budget,
                        dictionary=dictionary,
                        permuted_teacher_dictionary=permuted_teacher_dictionary,
                        candidate_atom_view=candidate_atom_view,
                        decoder_score_mode=decoder_score_mode,
                        min_decision_score=min_decision_score,
                        permuted_null_weight=permuted_null_weight,
                        rng=rng,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        predictions_name = f"predictions_budget{budget}.jsonl"
        (output_dir / predictions_name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        prediction_files[str(budget)] = predictions_name
        budget_summaries.append(_direction_summary(rows, budget_bytes=budget, seed=train_seed + eval_seed + budget))
    exact_ids = [example.example_id for example in eval_rows]
    payload = {
        "gate": "source_private_learned_synonym_dictionary_packet_direction",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "direction": direction,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "budgets": budgets,
        "candidate_atom_view": candidate_atom_view,
        "calibration_atom_view": calibration_atom_view,
        "candidate_calibration": candidate_calibration,
        "calibration_examples": len(calibration_rows),
        "surface_overlap_audit": surface_overlap_audit,
        "feature_dim": feature_dim,
        "text_feature_mode": text_feature_mode,
        "adapter_target_mode": adapter_target_mode,
        "decoder_score_mode": decoder_score_mode,
        "permuted_null_weight": permuted_null_weight
        if decoder_score_mode == "candidate_local_permuted_null_gap_residual_norm"
        else None,
        "receiver_mode": receiver_mode,
        "contrastive_negative_sources": contrastive_negative_sources,
        "contrastive_rank": contrastive_rank if receiver_mode in {"contrastive_low_rank_query", "contrastive_low_rank_factor"} else None,
        "low_rank_factor_epochs": low_rank_factor_epochs if receiver_mode == "contrastive_low_rank_factor" else None,
        "low_rank_factor_lr": low_rank_factor_lr if receiver_mode == "contrastive_low_rank_factor" else None,
        "low_rank_factor_loss": low_rank_factor_loss if receiver_mode == "contrastive_low_rank_factor" else None,
        "low_rank_factor_seed": low_rank_factor_seed if receiver_mode == "contrastive_low_rank_factor" else None,
        "jepa_query_count": dictionary.jepa_query_count if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_hidden_dim": dictionary.jepa_hidden_dim if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_query_entropy": dictionary.jepa_query_entropy if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_context_variance": dictionary.jepa_context_variance if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_trainable_factors": dictionary.jepa_trainable_factors if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_train_epochs": dictionary.jepa_train_epochs if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_lr": dictionary.jepa_lr if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_weight_decay": dictionary.jepa_weight_decay if receiver_mode in JEPA_RECEIVER_MODES else None,
        "receiver_effective_rank": dictionary.receiver_effective_rank,
        "ridge": ridge,
        "top_k": top_k,
        "min_score": min_score,
        "min_decision_score": min_decision_score,
        "atom_dictionary": list(ATOM_ORDER),
        "conditions": list(CONDITIONS),
        "source_destroying_controls": list(STRICT_SOURCE_DESTROYING_CONTROLS),
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "exact_id_sha256": hashlib.sha256("\n".join(exact_ids).encode("utf-8")).hexdigest(),
        "budget_summaries": budget_summaries,
        "prediction_files": prediction_files,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
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
        "\n".join(["# Learned Synonym Dictionary Direction Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def run_gate(
    *,
    output_dir: pathlib.Path,
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    seed: int,
    candidate_atom_view: str,
    candidate_calibration: str,
    calibration_examples: int,
    feature_dim: int,
    ridge: float,
    top_k: int,
    min_score: float,
    calibration_atom_view: str | None = None,
    text_feature_mode: str = "hashed",
    adapter_target_mode: str = "native_atoms",
    decoder_score_mode: str = "global_dot",
    receiver_mode: str = "atom_ridge",
    contrastive_negative_sources: int = 0,
    contrastive_rank: int = 4,
    low_rank_factor_epochs: int = 250,
    low_rank_factor_lr: float = 0.05,
    low_rank_factor_loss: str = "bce",
    low_rank_factor_seed: int | None = None,
    jepa_query_count: int = 8,
    jepa_hidden_dim: int = 32,
    jepa_train_epochs: int = 100,
    jepa_lr: float = 0.01,
    jepa_weight_decay: float = 0.001,
    min_decision_score: float = 0.20,
    permuted_null_weight: float = 0.75,
    feature_model: str = "BAAI/bge-small-en",
    feature_device: str = "auto",
    feature_dtype: str = "float32",
    feature_max_length: int = 128,
    local_files_only: bool = True,
) -> dict[str, Any]:
    global _HF_FEATURE_MODEL, _HF_FEATURE_DEVICE, _HF_FEATURE_DTYPE, _HF_FEATURE_MAX_LENGTH, _HF_FEATURE_LOCAL_FILES_ONLY
    _HF_FEATURE_MODEL = feature_model
    _HF_FEATURE_DEVICE = feature_device
    _HF_FEATURE_DTYPE = feature_dtype
    _HF_FEATURE_MAX_LENGTH = feature_max_length
    _HF_FEATURE_LOCAL_FILES_ONLY = local_files_only
    output_dir.mkdir(parents=True, exist_ok=True)
    if decoder_score_mode not in DECODER_SCORE_MODES:
        raise ValueError(f"unknown decoder score mode {decoder_score_mode!r}")
    effective_calibration_atom_view = calibration_atom_view or candidate_atom_view
    specs = [
        ("core_to_holdout", "core", "holdout", seed, seed + 1),
        ("holdout_to_core", "holdout", "core", seed + 1, seed),
        ("same_family_all", "all", "all", seed, seed + 2),
    ]
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for direction, train_family, eval_family, train_seed, eval_seed in specs:
        run_dir = output_dir / direction
        result = _run_direction(
            output_dir=run_dir,
            direction=direction,
            train_family_set=train_family,
            eval_family_set=eval_family,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_seed=train_seed,
            eval_seed=eval_seed,
            budgets=budgets,
            candidate_atom_view=candidate_atom_view,
            calibration_atom_view=effective_calibration_atom_view,
            candidate_calibration=candidate_calibration,
            calibration_examples=calibration_examples,
            feature_dim=feature_dim,
            ridge=ridge,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            adapter_target_mode=adapter_target_mode,
            decoder_score_mode=decoder_score_mode,
            receiver_mode=receiver_mode,
            contrastive_negative_sources=contrastive_negative_sources,
            contrastive_rank=contrastive_rank,
            low_rank_factor_epochs=low_rank_factor_epochs,
            low_rank_factor_lr=low_rank_factor_lr,
            low_rank_factor_loss=low_rank_factor_loss,
            low_rank_factor_seed=low_rank_factor_seed,
            jepa_query_count=jepa_query_count,
            jepa_hidden_dim=jepa_hidden_dim,
            jepa_train_epochs=jepa_train_epochs,
            jepa_lr=jepa_lr,
            jepa_weight_decay=jepa_weight_decay,
            min_decision_score=min_decision_score,
            permuted_null_weight=permuted_null_weight,
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
                    "learned_synonym_dictionary_accuracy": summary["learned_synonym_dictionary_accuracy"],
                    "best_control_accuracy": summary["best_control_accuracy"],
                    "learned_minus_target": summary["learned_minus_target"],
                    "learned_minus_best_control": summary["learned_minus_best_control"],
                    "oracle_learned_candidate_atoms_accuracy": summary["oracle_learned_candidate_atoms_accuracy"],
                    "best_control_name": summary["best_control_name"],
                    "top_atom_knockout_accuracy": summary["top_atom_knockout_accuracy"],
                    "private_random_knockout_accuracy": summary["private_random_knockout_accuracy"],
                    "top_atom_knockout_lift_reduction": summary["top_atom_knockout_lift_reduction"],
                    "private_random_knockout_lift_reduction": summary["private_random_knockout_lift_reduction"],
                    "paired_ci95_low_vs_target": summary["paired_bootstrap_vs_target"]["ci95_low"],
                    "paired_ci95_high_vs_target": summary["paired_bootstrap_vs_target"]["ci95_high"],
                    "controls_ok": summary["controls_ok"],
                    "pass_gate": summary["pass_gate"],
                }
            )
    direction_pass = {
        direction: any(row["pass_gate"] for row in rows if row["direction"] == direction)
        for direction, *_ in specs
    }
    cross_family_pass = direction_pass["core_to_holdout"] and direction_pass["holdout_to_core"]
    payload = {
        "gate": "source_private_learned_synonym_dictionary_packet_gate",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "run_dirs": run_dirs,
        "budgets": budgets,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "seed": seed,
        "candidate_atom_view": candidate_atom_view,
        "calibration_atom_view": effective_calibration_atom_view,
        "candidate_calibration": candidate_calibration,
        "calibration_examples": calibration_examples,
        "feature_dim": feature_dim,
        "text_feature_mode": text_feature_mode,
        "adapter_target_mode": adapter_target_mode,
        "decoder_score_mode": decoder_score_mode,
        "permuted_null_weight": permuted_null_weight
        if decoder_score_mode == "candidate_local_permuted_null_gap_residual_norm"
        else None,
        "receiver_mode": receiver_mode,
        "contrastive_negative_sources": contrastive_negative_sources,
        "contrastive_rank": contrastive_rank if receiver_mode in {"contrastive_low_rank_query", "contrastive_low_rank_factor"} else None,
        "low_rank_factor_epochs": low_rank_factor_epochs if receiver_mode == "contrastive_low_rank_factor" else None,
        "low_rank_factor_lr": low_rank_factor_lr if receiver_mode == "contrastive_low_rank_factor" else None,
        "low_rank_factor_loss": low_rank_factor_loss if receiver_mode == "contrastive_low_rank_factor" else None,
        "low_rank_factor_seed": low_rank_factor_seed if receiver_mode == "contrastive_low_rank_factor" else None,
        "jepa_query_count": jepa_query_count if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_hidden_dim": jepa_hidden_dim if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_trainable_factors": receiver_mode in {
            "jepa_query_resampler_trainable",
            "jepa_query_resampler_control_regularized",
            "jepa_query_resampler_pool_contrastive",
        } if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_train_epochs": jepa_train_epochs if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_lr": jepa_lr if receiver_mode in JEPA_RECEIVER_MODES else None,
        "jepa_weight_decay": jepa_weight_decay if receiver_mode in JEPA_RECEIVER_MODES else None,
        "feature_model": feature_model if "hf_" in text_feature_mode else None,
        "feature_device": _resolve_torch_device(feature_device) if "hf_" in text_feature_mode else None,
        "feature_dtype": feature_dtype if "hf_" in text_feature_mode else None,
        "feature_max_length": feature_max_length if "hf_" in text_feature_mode else None,
        "ridge": ridge,
        "top_k": top_k,
        "min_score": min_score,
        "min_decision_score": min_decision_score,
        "rows": rows,
        "headline": {
            "direction_pass": direction_pass,
            "cross_family_pass": cross_family_pass,
            "pass_rows": sum(1 for row in rows if row["pass_gate"]),
            "max_learned_synonym_dictionary_accuracy": max(row["learned_synonym_dictionary_accuracy"] for row in rows),
            "max_learned_minus_target": max(row["learned_minus_target"] for row in rows),
            "min_passing_ci95_low_vs_target": min(
                [row["paired_ci95_low_vs_target"] for row in rows if row["pass_gate"]] or [0.0]
            ),
        },
        "pass_gate": cross_family_pass,
        "pass_rule": (
            "Bidirectional cross-family pass requires at least one budget per direction with learned synonym dictionary "
            "packet beating target by >=0.15, best source-destroying control by >=0.10, all source-destroying controls "
            "within target+0.03, paired CI95 lower bound >0.05, learned candidate oracle >=0.80, and top-feature "
            "knockout removing >=50% of lift. Strict controls include shuffled source packets, atom-ID derangement, "
            "private-random atom packets, and a permuted-teacher receiver. Private-random single-atom knockout is "
            "reported as a packet-fragility diagnostic but is not a hard veto because low-rate 2-4 atom packets are "
            "expected to lose lift when a real transmitted atom is removed."
        ),
    }
    (output_dir / "learned_synonym_dictionary_packet_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_gate_markdown(output_dir / "learned_synonym_dictionary_packet_gate.md", payload)
    manifest = {
        "artifacts": [
            "learned_synonym_dictionary_packet_gate.json",
            "learned_synonym_dictionary_packet_gate.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "learned_synonym_dictionary_packet_gate.json": _sha256_file(
                output_dir / "learned_synonym_dictionary_packet_gate.json"
            ),
            "learned_synonym_dictionary_packet_gate.md": _sha256_file(
                output_dir / "learned_synonym_dictionary_packet_gate.md"
            ),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Learned Synonym Dictionary Packet Gate Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- cross-family pass: `{cross_family_pass}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_direction_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Learned Synonym Dictionary Direction",
        "",
        f"- direction: `{payload['direction']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval families: `{payload['train_family_set']} -> {payload['eval_family_set']}`",
        f"- candidate atom view: `{payload['candidate_atom_view']}`",
        f"- calibration atom view: `{payload['calibration_atom_view']}`",
        f"- candidate calibration: `{payload['candidate_calibration']}`",
        f"- text feature mode: `{payload['text_feature_mode']}`",
        f"- adapter target mode: `{payload['adapter_target_mode']}`",
        f"- decoder score mode: `{payload['decoder_score_mode']}`",
        f"- receiver mode: `{payload['receiver_mode']}`",
        f"- contrastive negative sources: `{payload['contrastive_negative_sources']}`",
        f"- contrastive rank: `{payload['contrastive_rank']}`",
        f"- low-rank factor epochs: `{payload['low_rank_factor_epochs']}`",
        f"- low-rank factor lr: `{payload['low_rank_factor_lr']}`",
        f"- low-rank factor loss: `{payload['low_rank_factor_loss']}`",
        f"- low-rank factor seed: `{payload['low_rank_factor_seed']}`",
        f"- JEPA query count: `{payload['jepa_query_count']}`",
        f"- JEPA hidden dim: `{payload['jepa_hidden_dim']}`",
        f"- JEPA trainable factors: `{payload['jepa_trainable_factors']}`",
        f"- JEPA train epochs: `{payload['jepa_train_epochs']}`",
        f"- JEPA lr: `{payload['jepa_lr']}`",
        f"- JEPA weight decay: `{payload['jepa_weight_decay']}`",
        f"- JEPA query entropy: `{payload['jepa_query_entropy']}`",
        f"- JEPA context variance: `{payload['jepa_context_variance']}`",
        f"- receiver effective rank: `{payload['receiver_effective_rank']}`",
        f"- min decision score: `{payload['min_decision_score']}`",
        f"- calibration/eval exact ID overlap count: `{payload['surface_overlap_audit']['calibration_eval_exact_id_overlap_count']}`",
        f"- exact eval surface overlap count: `{payload['surface_overlap_audit']['exact_eval_surface_overlap_count']}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | "
            f"{row['learned_synonym_dictionary_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['learned_minus_target']:.3f} | "
            f"{row['paired_bootstrap_vs_target']['ci95_low']:.3f} | "
            f"{row['top_atom_knockout_lift_reduction']:.3f} | "
            f"{row['oracle_learned_candidate_atoms_accuracy']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Learned Synonym Dictionary Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- direction pass: `{h['direction_pass']}`",
        f"- cross-family pass: `{h['cross_family_pass']}`",
        f"- budgets: `{payload['budgets']}`",
        f"- candidate atom view: `{payload['candidate_atom_view']}`",
        f"- calibration atom view: `{payload['calibration_atom_view']}`",
        f"- candidate calibration: `{payload['candidate_calibration']}`",
        f"- text feature mode: `{payload['text_feature_mode']}`",
        f"- adapter target mode: `{payload['adapter_target_mode']}`",
        f"- decoder score mode: `{payload['decoder_score_mode']}`",
        f"- receiver mode: `{payload['receiver_mode']}`",
        f"- contrastive negative sources: `{payload['contrastive_negative_sources']}`",
        f"- contrastive rank: `{payload['contrastive_rank']}`",
        f"- low-rank factor epochs: `{payload['low_rank_factor_epochs']}`",
        f"- low-rank factor lr: `{payload['low_rank_factor_lr']}`",
        f"- low-rank factor loss: `{payload['low_rank_factor_loss']}`",
        f"- low-rank factor seed: `{payload['low_rank_factor_seed']}`",
        f"- JEPA query count: `{payload['jepa_query_count']}`",
        f"- JEPA hidden dim: `{payload['jepa_hidden_dim']}`",
        f"- JEPA trainable factors: `{payload['jepa_trainable_factors']}`",
        f"- JEPA train epochs: `{payload['jepa_train_epochs']}`",
        f"- JEPA lr: `{payload['jepa_lr']}`",
        f"- JEPA weight decay: `{payload['jepa_weight_decay']}`",
        f"- min decision score: `{payload['min_decision_score']}`",
        f"- max learned packet accuracy: `{h['max_learned_synonym_dictionary_accuracy']:.3f}`",
        f"- max learned-target delta: `{h['max_learned_minus_target']:.3f}`",
        "",
        "## Rows",
        "",
        "| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['direction']} | {row['budget_bytes']} | {row['n']} | `{row['pass_gate']}` | "
            f"{row['learned_synonym_dictionary_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['learned_minus_target']:.3f} | "
            f"{row['paired_ci95_low_vs_target']:.3f} | {row['top_atom_knockout_lift_reduction']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_learned_synonym_dictionary_packet_gate_20260429"),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--candidate-atom-view",
        choices=["native", "synonym_stress", "heldout_synonym"],
        default="synonym_stress",
    )
    parser.add_argument(
        "--calibration-atom-view",
        choices=["native", "synonym_stress", "heldout_synonym"],
        default=None,
        help="Surface used to calibrate the learned dictionary; defaults to --candidate-atom-view.",
    )
    parser.add_argument(
        "--candidate-calibration",
        choices=["train_only", "all_public", "all_public_eval_disjoint"],
        default="all_public",
    )
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument(
        "--text-feature-mode",
        choices=[
            "hashed",
            "semantic_anchor",
            "hf_last_mean",
            "hf_mid_last_mean",
            "hashed_hf_last_mean",
            "hashed_hf_mid_last_mean",
        ],
        default="hashed",
        help=(
            "Target-side candidate dictionary features. semantic_anchor adds public atom-anchor expansions; "
            "hf_* uses frozen Transformer text features; hashed_hf_* combines generic lexical hashing with frozen embeddings."
        ),
    )
    parser.add_argument(
        "--adapter-target-mode",
        choices=["native_atoms", "semantic_anchor_teacher", "permuted_semantic_anchor_teacher"],
        default="native_atoms",
        help=(
            "Public receiver supervision target for atom_ridge. semantic_anchor_teacher distills public "
            "semantic-anchor coordinates from calibration surfaces into the chosen features; permuted_* is "
            "a negative control that should collapse."
        ),
    )
    parser.add_argument(
        "--decoder-score-mode",
        choices=sorted(DECODER_SCORE_MODES),
        default="global_dot",
        help=(
            "Candidate scoring rule. candidate_local_residual subtracts the public candidate-pool mean before "
            "scoring, candidate_local_residual_norm also normalizes candidate residual rows and the packet, "
            "candidate_local_innovation_residual_norm also subtracts the candidate-pool mean from the packet, "
            "and candidate_local_permuted_null_gap_residual_norm subtracts a deterministic permuted-receiver null score."
        ),
    )
    parser.add_argument("--feature-model", default="BAAI/bge-small-en")
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--feature-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--feature-max-length", type=int, default=128)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--permuted-null-weight",
        type=float,
        default=0.75,
        help="Penalty on the deterministic permuted-teacher null score for candidate_local_permuted_null_gap_residual_norm.",
    )
    parser.add_argument(
        "--receiver-mode",
        choices=[
            "atom_ridge",
            "contrastive_bilinear",
            "contrastive_low_rank_query",
            "contrastive_low_rank_factor",
            "jepa_query_resampler",
            "jepa_query_resampler_trainable",
            "jepa_query_resampler_control_regularized",
            "jepa_query_resampler_pool_contrastive",
        ],
        default="atom_ridge",
    )
    parser.add_argument(
        "--contrastive-negative-sources",
        type=int,
        default=0,
        help="For contrastive_bilinear, add this many shuffled source packets per calibration example as zero-label negatives.",
    )
    parser.add_argument(
        "--contrastive-rank",
        type=int,
        default=4,
        help="Low-rank receiver rank. For contrastive_low_rank_query this truncates the bilinear map; for contrastive_low_rank_factor this is the directly trained factor rank.",
    )
    parser.add_argument(
        "--low-rank-factor-epochs",
        type=int,
        default=250,
        help="For contrastive_low_rank_factor, number of numpy gradient steps.",
    )
    parser.add_argument(
        "--low-rank-factor-lr",
        type=float,
        default=0.05,
        help="For contrastive_low_rank_factor, gradient step size.",
    )
    parser.add_argument(
        "--low-rank-factor-loss",
        choices=["bce", "squared"],
        default="bce",
        help="For contrastive_low_rank_factor, objective used for matched candidate/source-control negatives.",
    )
    parser.add_argument(
        "--low-rank-factor-seed",
        type=int,
        default=None,
        help="For contrastive_low_rank_factor, explicit factor initialization seed; defaults to the direction train seed.",
    )
    parser.add_argument(
        "--jepa-query-count",
        type=int,
        default=8,
        help="For jepa_query_resampler, number of candidate-conditioned query vectors.",
    )
    parser.add_argument(
        "--jepa-hidden-dim",
        type=int,
        default=32,
        help="For jepa_query_resampler, hidden dimension of query/key/value packet attention.",
    )
    parser.add_argument(
        "--jepa-train-epochs",
        type=int,
        default=100,
        help="For jepa_query_resampler_trainable, CPU torch optimization epochs.",
    )
    parser.add_argument(
        "--jepa-lr",
        type=float,
        default=0.01,
        help="For jepa_query_resampler_trainable, CPU torch optimizer learning rate.",
    )
    parser.add_argument(
        "--jepa-weight-decay",
        type=float,
        default=0.001,
        help="For jepa_query_resampler_trainable, AdamW weight decay on trainable query/key/value factors.",
    )
    parser.add_argument("--ridge", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument(
        "--min-decision-score",
        type=float,
        default=0.20,
        help="Preserve the target prior unless the best packet/candidate atom score reaches this threshold.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=output_dir,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        seed=args.seed,
        candidate_atom_view=args.candidate_atom_view,
        calibration_atom_view=args.calibration_atom_view,
        candidate_calibration=args.candidate_calibration,
        calibration_examples=args.calibration_examples,
        feature_dim=args.feature_dim,
        text_feature_mode=args.text_feature_mode,
        adapter_target_mode=args.adapter_target_mode,
        decoder_score_mode=args.decoder_score_mode,
        receiver_mode=args.receiver_mode,
        contrastive_negative_sources=args.contrastive_negative_sources,
        contrastive_rank=args.contrastive_rank,
        low_rank_factor_epochs=args.low_rank_factor_epochs,
        low_rank_factor_lr=args.low_rank_factor_lr,
        low_rank_factor_loss=args.low_rank_factor_loss,
        low_rank_factor_seed=args.low_rank_factor_seed,
        jepa_query_count=args.jepa_query_count,
        jepa_hidden_dim=args.jepa_hidden_dim,
        jepa_train_epochs=args.jepa_train_epochs,
        jepa_lr=args.jepa_lr,
        jepa_weight_decay=args.jepa_weight_decay,
        feature_model=args.feature_model,
        feature_device=args.feature_device,
        feature_dtype=args.feature_dtype,
        feature_max_length=args.feature_max_length,
        local_files_only=args.local_files_only,
        permuted_null_weight=args.permuted_null_weight,
        ridge=args.ridge,
        top_k=args.top_k,
        min_score=args.min_score,
        min_decision_score=args.min_decision_score,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
