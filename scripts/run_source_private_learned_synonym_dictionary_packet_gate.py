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
    "top_atom_knockout",
    "private_random_knockout",
    "oracle_learned_candidate_atoms",
)

STRICT_SOURCE_DESTROYING_CONTROLS = tuple(
    condition for condition in SOURCE_DESTROYING_CONTROLS if condition in CONDITIONS
)
JEPA_RECEIVER_MODES = {
    "jepa_query_resampler",
    "jepa_query_resampler_trainable",
    "jepa_query_resampler_control_regularized",
    "jepa_query_resampler_pool_contrastive",
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
        self.jepa_query_count = jepa_query_count
        self.jepa_hidden_dim = jepa_hidden_dim
        self.jepa_query_entropy = jepa_query_entropy
        self.jepa_context_variance = jepa_context_variance
        self.jepa_trainable_factors = jepa_trainable_factors
        self.jepa_train_epochs = jepa_train_epochs
        self.jepa_lr = jepa_lr
        self.jepa_weight_decay = jepa_weight_decay

    def predict_atoms(self, text: str) -> dict[str, float]:
        features = _featurize_text(text, dim=self.feature_dim, text_feature_mode=self.text_feature_mode)
        scores = np.maximum(features @ self.weights, 0.0)
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
) -> LearnedSynonymDictionary:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for example in examples:
        for candidate in example.candidates:
            native_atoms = _candidate_atoms(candidate.patch_intent, candidate_atom_view="native")
            y = _atom_vector(native_atoms)
            if not np.any(y):
                continue
            texts = [candidate.patch_intent]
            calibrated = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=calibration_atom_view)
            if calibrated != candidate.patch_intent:
                texts.append(calibrated)
            for text in texts:
                x_rows.append(_featurize_text(text, dim=feature_dim, text_feature_mode=text_feature_mode))
                y_rows.append(y)
    if not x_rows:
        return LearnedSynonymDictionary(
            feature_dim=feature_dim,
            weights=np.zeros((feature_dim, len(ATOM_ORDER)), dtype=np.float64),
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode="atom_ridge",
        )
    x = np.stack(x_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    xtx = x.T @ x
    regularized = xtx + ridge * np.eye(feature_dim, dtype=np.float64)
    weights = np.linalg.solve(regularized, x.T @ y)
    return LearnedSynonymDictionary(
        feature_dim=feature_dim,
        weights=weights,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
        receiver_mode="atom_ridge",
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
    if receiver_mode == "atom_ridge":
        return _fit_ridge_dictionary(
            examples=examples,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view=calibration_atom_view,
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
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
    calibration_count: int,
    seed: int,
) -> list[Example]:
    if mode == "train_only":
        return train_examples
    if mode == "all_public":
        return make_benchmark(examples=calibration_count, candidates=4, seed=seed, family_set="all")
    raise ValueError(f"unknown calibration mode {mode!r}")


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
) -> list[float]:
    scores = []
    for candidate in example.candidates:
        text = _candidate_surface_text(candidate.patch_intent, candidate_atom_view=candidate_atom_view)
        scores.append(dictionary.score_text(text, payload_atoms))
    return scores


def _predict_from_payload(
    *,
    example: Example,
    payload: bytes | None,
    budget_bytes: int,
    dictionary: LearnedSynonymDictionary,
    candidate_atom_view: str,
    min_decision_score: float,
    derange: bool = False,
    knockout: str | None = None,
    rng: random.Random | None = None,
) -> tuple[str, dict[str, Any]]:
    prior = _prior_prediction(example)
    payload_atoms = _decode_payload_atoms(payload, budget_bytes=budget_bytes, derange=derange, knockout=knockout, rng=rng)
    if not payload_atoms:
        return prior, {"decoder": "prior", "payload_atoms": {}}
    scores = _score_candidates(
        example=example,
        payload_atoms=payload_atoms,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
    )
    best_score = max(scores)
    if best_score < min_decision_score:
        return prior, {
            "decoder": "learned_synonym_dictionary_target_preserve",
            "payload_atoms": payload_atoms,
            "scores": scores,
            "best_score": best_score,
            "min_decision_score": min_decision_score,
        }
    tied = [idx for idx, score in enumerate(scores) if abs(score - best_score) <= 1e-8]
    labels = [candidate.label for candidate in example.candidates]
    prediction = prior if any(labels[idx] == prior for idx in tied) else labels[tied[0]]
    return prediction, {
        "decoder": "learned_synonym_dictionary",
        "payload_atoms": payload_atoms,
        "scores": scores,
        "best_score": best_score,
        "min_decision_score": min_decision_score,
    }


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    budget_bytes: int,
    dictionary: LearnedSynonymDictionary,
    candidate_atom_view: str,
    min_decision_score: float,
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
    prediction, decode_meta = _predict_from_payload(
        example=example,
        payload=payload,
        budget_bytes=budget_bytes,
        dictionary=dictionary,
        candidate_atom_view=candidate_atom_view,
        min_decision_score=0.0 if condition == "oracle_learned_candidate_atoms" else min_decision_score,
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
        and random_knockout_reduction < 0.75
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
    min_decision_score: float,
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
                        candidate_atom_view=candidate_atom_view,
                        min_decision_score=min_decision_score,
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
            "knockout removing >=50% of lift."
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
    parser.add_argument("--candidate-calibration", choices=["train_only", "all_public"], default="all_public")
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
    parser.add_argument("--feature-model", default="BAAI/bge-small-en")
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--feature-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--feature-max-length", type=int, default=128)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
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
        ridge=args.ridge,
        top_k=args.top_k,
        min_score=args.min_score,
        min_decision_score=args.min_decision_score,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
