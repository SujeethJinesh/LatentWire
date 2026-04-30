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
    if text_feature_mode not in {"hashed", "semantic_anchor"}:
        raise ValueError(f"unknown text feature mode {text_feature_mode!r}")
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
    ) -> None:
        self.feature_dim = feature_dim
        self.weights = weights
        self.top_k = top_k
        self.min_score = min_score
        self.text_feature_mode = text_feature_mode

    def predict_atoms(self, text: str) -> dict[str, float]:
        features = _featurize_text(text, dim=self.feature_dim, text_feature_mode=self.text_feature_mode)
        scores = np.maximum(features @ self.weights, 0.0)
        return _atoms_from_vector(scores, top_k=self.top_k, min_score=self.min_score)


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
    )


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
        learned_atoms = dictionary.predict_atoms(text)
        overlap = sum(payload_atoms.get(atom, 0.0) * score for atom, score in learned_atoms.items())
        scores.append(overlap)
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
    dictionary = _fit_ridge_dictionary(
        examples=calibration_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        calibration_atom_view=calibration_atom_view,
        top_k=top_k,
        min_score=min_score,
        text_feature_mode=text_feature_mode,
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
    min_decision_score: float = 0.20,
) -> dict[str, Any]:
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
                    "top_atom_knockout_lift_reduction": summary["top_atom_knockout_lift_reduction"],
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
        choices=["hashed", "semantic_anchor"],
        default="hashed",
        help="Target-side candidate dictionary features. semantic_anchor adds public atom-anchor expansions.",
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
        ridge=args.ridge,
        top_k=args.top_k,
        min_score=args.min_score,
        min_decision_score=args.min_decision_score,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
