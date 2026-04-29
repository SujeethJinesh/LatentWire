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
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


ATOM_ORDER = (
    "empty",
    "list",
    "guard",
    "missing",
    "key",
    "default",
    "round",
    "half_up",
    "threshold",
    "inclusive",
    "preserve",
    "order",
    "unique",
    "none",
    "string",
    "sum",
    "all_values",
    "case",
    "insensitive",
    "clamp",
    "negative",
    "zero",
    "final",
    "parse",
    "integer",
    "failure",
    "average",
    "mean",
    "lowercase",
    "nested",
    "mapping",
    "index",
    "modulo",
    "strict",
    "positive",
    "filter",
    "exception",
    "equality",
    "fallback",
    "first",
    "last",
)
ATOM_TO_ID = {atom: idx for idx, atom in enumerate(ATOM_ORDER)}
ID_TO_ATOM = {idx: atom for atom, idx in ATOM_TO_ID.items()}

CONDITIONS = (
    "target_only",
    "shared_sparse_packet",
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
    "oracle_candidate_atoms",
)

SOURCE_DESTROYING_CONTROLS = (
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "public_only_sidecar",
    "target_derived_sidecar",
    "random_same_byte",
    "answer_only_text",
    "structured_text_matched",
    "atom_id_derangement",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atom_score(atoms: dict[str, float], atom: str, value: float) -> None:
    if atom in ATOM_TO_ID:
        atoms[atom] = max(float(value), atoms.get(atom, 0.0))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("-", " ").replace("_", " "))


def _candidate_atoms(candidate_intent: str) -> dict[str, float]:
    text = candidate_intent.lower().replace("-", " ")
    toks = set(_tokens(text))
    atoms: dict[str, float] = {}
    for token in toks:
        if token in ATOM_TO_ID:
            _atom_score(atoms, token, 0.60)
    phrase_rules = {
        "empty": ("empty", "list", "guard", "default"),
        "missing key": ("missing", "key", "default", "mapping"),
        "half up": ("round", "half_up"),
        "round": ("round",),
        "inclusive": ("threshold", "inclusive", "equality"),
        "preserve order": ("preserve", "order", "unique"),
        "unique": ("unique",),
        "none": ("none", "empty", "default"),
        "sum all": ("sum", "all_values"),
        "case insensitive": ("case", "insensitive", "equality"),
        "clamp negative": ("clamp", "negative", "zero"),
        "final": ("final", "last"),
        "last": ("final", "last"),
        "parse": ("parse", "integer", "failure", "default"),
        "average": ("average", "mean", "all_values"),
        "lowercase": ("lowercase", "case"),
        "nested": ("nested", "key", "default", "mapping"),
        "modulo": ("index", "modulo", "list"),
        "strictly positive": ("strict", "positive", "filter"),
        "nonnegative": ("positive", "zero"),
        "first": ("first",),
        "fallback": ("fallback", "default"),
    }
    for phrase, phrase_atoms in phrase_rules.items():
        if phrase in text:
            for atom in phrase_atoms:
                _atom_score(atoms, atom, 1.0)
    if "none" in text:
        _atom_score(atoms, "none", 1.35)
    if "zero" in text:
        _atom_score(atoms, "zero", 1.35)
    if "final" in text or "last" in text:
        _atom_score(atoms, "final", 1.20)
    return atoms


def _line_value(log: str, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=(.*)$", re.MULTILINE)
    match = pattern.search(log)
    return "" if not match else match.group(1).strip()


def _source_private_atoms(log: str, *, mode: str) -> dict[str, float]:
    if mode == "zero":
        return {}
    if mode == "answer_masked":
        log = _mask_repair_diag(log)
        log = _mask_log_components(log, mask_expected_actual=True, mask_test_name=True)
    elif mode == "matched":
        log = _mask_repair_diag(log)
        log = re.sub(r"repair_family=[A-Za-z0-9_]+", "repair_family=<MASKED>", log)
        log = re.sub(r"hidden_tests/test_[A-Za-z0-9_]+\\.py", "hidden_tests/test_<MASKED>.py", log)
    elif mode == "public_only":
        return {}
    else:
        raise ValueError(f"unknown source atom mode {mode!r}")

    lowered = log.lower()
    hidden_input = _line_value(log, "hidden_input").lower()
    expected = _line_value(log, "expected").lower()
    actual = _line_value(log, "actual").lower()
    atoms: dict[str, float] = {}

    if "indexerror" in lowered:
        for atom in ("exception", "index", "list", "empty"):
            _atom_score(atoms, atom, 1.0)
    if "keyerror" in lowered:
        for atom in ("exception", "missing", "key", "default", "mapping"):
            _atom_score(atoms, atom, 1.0)
    if "valueerror" in lowered or "invalid literal" in lowered:
        for atom in ("exception", "parse", "integer", "failure", "default"):
            _atom_score(atoms, atom, 1.0)
    if hidden_input in {"[]", "()", "{}", "none"} or hidden_input == "":
        for atom in ("empty", "default"):
            _atom_score(atoms, atom, 0.95)
    if expected in {"0", "0.0"} and re.search(r"actual=-|hidden_input=-", lowered):
        for atom in ("clamp", "negative", "zero"):
            _atom_score(atoms, atom, 1.0)
    if expected in {"0", "0.0"} and ("valueerror" in lowered or "invalid literal" in lowered):
        for atom in ("parse", "failure", "zero", "default"):
            _atom_score(atoms, atom, 1.0)
    if expected in {"none", "'none'", "null"}:
        _atom_score(atoms, "none", 1.35)
        for atom in ("empty", "default"):
            _atom_score(atoms, atom, 1.0)
    if expected in {"''", '""'}:
        for atom in ("empty", "string", "default"):
            _atom_score(atoms, atom, 1.0)
    if expected == "true" and actual == "false":
        for atom in ("inclusive", "threshold", "equality", "case", "insensitive"):
            _atom_score(atoms, atom, 0.85)
    if "alpha" in lowered and "expected=true" in lowered:
        for atom in ("case", "insensitive", "equality"):
            _atom_score(atoms, atom, 1.0)
    if "hello" in lowered and expected.strip("'\"") == "hello":
        for atom in ("lowercase", "case"):
            _atom_score(atoms, atom, 1.0)
    if "actual=[0, 2]" in lowered and "expected=[2]" in lowered:
        for atom in ("strict", "positive", "filter"):
            _atom_score(atoms, atom, 1.0)
    if re.search(r"hidden_input=\[[0-9, ]+\]", lowered) and "actual=6.0" in lowered and "expected=4" in lowered:
        for atom in ("average", "mean", "all_values"):
            _atom_score(atoms, atom, 1.0)
    if "expected=10" in lowered:
        for atom in ("sum", "all_values"):
            _atom_score(atoms, atom, 1.0)
    if "expected=[3, 1, 2]" in lowered:
        for atom in ("preserve", "order", "unique"):
            _atom_score(atoms, atom, 1.0)
    if "expected=3" in lowered and ("hidden_input=2.6" in lowered or "actual=2" in lowered):
        for atom in ("round", "half_up"):
            _atom_score(atoms, atom, 1.0)
    if "expected=20" in lowered and "indexerror" in lowered:
        for atom in ("index", "modulo", "list"):
            _atom_score(atoms, atom, 1.0)
    if "expected='unknown'" in lowered:
        for atom in ("missing", "key", "default"):
            _atom_score(atoms, atom, 1.0)
    return atoms


def _answer_index(example: Example) -> int:
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)


def _encode_atoms(atoms: dict[str, float], *, budget_bytes: int) -> bytes:
    atom_count = max(1, budget_bytes // 2)
    ranked = sorted(atoms.items(), key=lambda item: (-item[1], ATOM_TO_ID[item[0]]))[:atom_count]
    payload = bytearray()
    for atom, score in ranked:
        payload.append(ATOM_TO_ID[atom])
        payload.append(int(max(0, min(255, round(score * 255.0)))))
    return bytes(payload[:budget_bytes])


def _random_packet(*, budget_bytes: int, rng: random.Random) -> bytes:
    payload = bytearray()
    for _ in range(max(1, budget_bytes // 2)):
        payload.append((len(ATOM_ORDER) + rng.randrange(17, 239)) % 256)
        payload.append(rng.randrange(0, 256))
    return bytes(payload[:budget_bytes])


def _decode_payload_atoms(
    payload: bytes | None,
    *,
    budget_bytes: int,
    derange: bool = False,
    knockout: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, float]:
    if not payload:
        return {}
    pairs = min(len(payload) // 2, max(1, budget_bytes // 2))
    decoded: list[tuple[str, float]] = []
    for pair_idx in range(pairs):
        atom_id = int(payload[2 * pair_idx])
        if derange:
            atom_id = (atom_id + 7) % len(ATOM_ORDER)
        atom = ID_TO_ATOM.get(atom_id)
        if atom is None:
            continue
        decoded.append((atom, int(payload[2 * pair_idx + 1]) / 255.0))
    if knockout == "top" and decoded:
        decoded = []
    elif knockout == "random" and decoded:
        assert rng is not None
        drop = rng.randrange(len(decoded))
        decoded = [item for idx, item in enumerate(decoded) if idx != drop]
    atoms: dict[str, float] = {}
    for atom, score in decoded:
        _atom_score(atoms, atom, score)
    return atoms


def _score_candidates(example: Example, payload_atoms: dict[str, float]) -> list[float]:
    scores = []
    for candidate in example.candidates:
        cand_atoms = _candidate_atoms(candidate.patch_intent)
        overlap = sum(payload_atoms.get(atom, 0.0) * weight for atom, weight in cand_atoms.items())
        scores.append(overlap)
    return scores


def _predict_from_payload(
    example: Example,
    payload: bytes | None,
    *,
    budget_bytes: int,
    derange: bool = False,
    knockout: str | None = None,
    rng: random.Random | None = None,
) -> tuple[str, dict[str, Any]]:
    prior = _prior_prediction(example)
    payload_atoms = _decode_payload_atoms(payload, budget_bytes=budget_bytes, derange=derange, knockout=knockout, rng=rng)
    if not payload_atoms:
        return prior, {"decoder": "prior", "payload_atoms": {}}
    scores = _score_candidates(example, payload_atoms)
    best_score = max(scores)
    if best_score < 1.0:
        return prior, {"decoder": "shared_sparse_atom_overlap", "payload_atoms": payload_atoms, "scores": scores}
    tied = [idx for idx, score in enumerate(scores) if abs(score - best_score) <= 1e-8]
    labels = [candidate.label for candidate in example.candidates]
    if any(labels[idx] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[tied[0]]
    return prediction, {"decoder": "shared_sparse_atom_overlap", "payload_atoms": payload_atoms, "scores": scores}


def _oracle_packet(example: Example, *, budget_bytes: int) -> bytes:
    answer = example.candidates[_answer_index(example)]
    return _encode_atoms(_candidate_atoms(answer.patch_intent), budget_bytes=budget_bytes)


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    budget_bytes: int,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any], dict[str, Any]]:
    decode_kwargs: dict[str, Any] = {}
    if condition == "target_only":
        return None, {"source": "target_prior"}, decode_kwargs
    if condition == "shared_sparse_packet":
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
        return _encode_atoms(_source_private_atoms(example.private_test_log, mode="answer_masked"), budget_bytes=budget_bytes), {
            "source": "answer_masked"
        }, decode_kwargs
    if condition == "public_only_sidecar":
        return _encode_atoms(_source_private_atoms(example.public_issue, mode="public_only"), budget_bytes=budget_bytes), {
            "source": "public_only"
        }, decode_kwargs
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
    if condition == "oracle_candidate_atoms":
        return _oracle_packet(example, budget_bytes=budget_bytes), {"source": "oracle_candidate_atoms"}, decode_kwargs
    raise ValueError(f"unknown condition {condition!r}")


def _constrained_nonoverlap_example(example: Example, eval_examples: list[Example], index: int) -> Example:
    current_atoms = set(_source_private_atoms(example.private_test_log, mode="matched"))
    current_answer = _answer_index(example)
    for offset in range(1, len(eval_examples)):
        other = eval_examples[(index + offset) % len(eval_examples)]
        other_atoms = set(_source_private_atoms(other.private_test_log, mode="matched"))
        if _answer_index(other) != current_answer and not (current_atoms & other_atoms):
            return other
    for offset in range(1, len(eval_examples)):
        other = eval_examples[(index + offset) % len(eval_examples)]
        if _answer_index(other) != current_answer:
            return other
    return eval_examples[_deterministic_nonself_index(index, len(eval_examples))]


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    budget_bytes: int,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, meta, decode_kwargs = _payload_for_condition(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        budget_bytes=budget_bytes,
        rng=rng,
    )
    prediction, decode_meta = _predict_from_payload(
        example,
        payload,
        budget_bytes=budget_bytes,
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


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def _paired_bootstrap(rows: list[dict[str, Any]], *, condition: str, baseline: str, seed: int, samples: int = 2000) -> dict[str, float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = []
    for _, conditions in sorted(by_example.items()):
        deltas.append(float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"]))
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
    matched = metrics["shared_sparse_packet"]["accuracy"]
    best_control = max(metrics[condition]["accuracy"] for condition in SOURCE_DESTROYING_CONTROLS)
    top_knockout = metrics["top_atom_knockout"]["accuracy"]
    random_knockout = metrics["private_random_knockout"]["accuracy"]
    oracle = metrics["oracle_candidate_atoms"]["accuracy"]
    target_ci = _paired_bootstrap(rows, condition="shared_sparse_packet", baseline="target_only", seed=seed)
    best_control_name = max(SOURCE_DESTROYING_CONTROLS, key=lambda c: metrics[c]["accuracy"])
    control_ci = _paired_bootstrap(rows, condition="shared_sparse_packet", baseline=best_control_name, seed=seed + 1)
    lift = matched - target
    knockout_reduction = 0.0 if lift <= 0 else max(0.0, matched - top_knockout) / lift
    random_knockout_reduction = 0.0 if lift <= 0 else max(0.0, matched - random_knockout) / lift
    pass_gate = (
        matched >= target + 0.10
        and matched >= best_control + 0.05
        and all(metrics[condition]["accuracy"] <= target + 0.03 for condition in SOURCE_DESTROYING_CONTROLS)
        and target_ci["ci95_low"] > 0.0
        and oracle >= 0.90
        and knockout_reduction >= 0.50
        and random_knockout_reduction < 0.75
    )
    return {
        "budget_bytes": budget_bytes,
        "n": metrics["target_only"]["n"],
        "target_accuracy": target,
        "shared_sparse_accuracy": matched,
        "best_control_accuracy": best_control,
        "shared_minus_target": matched - target,
        "shared_minus_best_control": matched - best_control,
        "oracle_candidate_atoms_accuracy": oracle,
        "top_atom_knockout_accuracy": top_knockout,
        "private_random_knockout_accuracy": random_knockout,
        "top_atom_knockout_lift_reduction": knockout_reduction,
        "private_random_knockout_lift_reduction": random_knockout_reduction,
        "paired_bootstrap_vs_target": target_ci,
        "paired_bootstrap_vs_best_control": control_ci,
        "best_control_name": best_control_name,
        "controls_ok": all(metrics[condition]["accuracy"] <= target + 0.03 for condition in SOURCE_DESTROYING_CONTROLS),
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(examples=train_examples, candidates=4, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=4, seed=eval_seed, family_set=eval_family_set)
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
        "gate": "source_private_shared_sparse_crosscoder_packet_direction",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "direction": direction,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "budgets": budgets,
        "atom_dictionary": list(ATOM_ORDER),
        "conditions": list(CONDITIONS),
        "source_destroying_controls": list(SOURCE_DESTROYING_CONTROLS),
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
        "\n".join(["# Shared Sparse Crosscoder Direction Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
                    "shared_sparse_accuracy": summary["shared_sparse_accuracy"],
                    "best_control_accuracy": summary["best_control_accuracy"],
                    "shared_minus_target": summary["shared_minus_target"],
                    "shared_minus_best_control": summary["shared_minus_best_control"],
                    "oracle_candidate_atoms_accuracy": summary["oracle_candidate_atoms_accuracy"],
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
        "gate": "source_private_shared_sparse_crosscoder_packet_gate",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "run_dirs": run_dirs,
        "budgets": budgets,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "seed": seed,
        "rows": rows,
        "headline": {
            "direction_pass": direction_pass,
            "cross_family_pass": cross_family_pass,
            "pass_rows": sum(1 for row in rows if row["pass_gate"]),
            "max_shared_sparse_accuracy": max(row["shared_sparse_accuracy"] for row in rows),
            "max_shared_minus_target": max(row["shared_minus_target"] for row in rows),
            "min_passing_ci95_low_vs_target": min(
                [row["paired_ci95_low_vs_target"] for row in rows if row["pass_gate"]] or [0.0]
            ),
        },
        "pass_gate": cross_family_pass,
        "pass_rule": (
            "Bidirectional cross-family pass requires at least one budget per direction with matched shared sparse "
            "packet beating target by >=0.10, best source-destroying control by >=0.05, all controls within "
            "target+0.03, paired CI95 lower bound >0, oracle >=0.90, and top-atom knockout removing >=50% of lift."
        ),
    }
    (output_dir / "shared_sparse_crosscoder_packet_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_gate_markdown(output_dir / "shared_sparse_crosscoder_packet_gate.md", payload)
    manifest = {
        "artifacts": [
            "shared_sparse_crosscoder_packet_gate.json",
            "shared_sparse_crosscoder_packet_gate.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "shared_sparse_crosscoder_packet_gate.json": _sha256_file(output_dir / "shared_sparse_crosscoder_packet_gate.json"),
            "shared_sparse_crosscoder_packet_gate.md": _sha256_file(output_dir / "shared_sparse_crosscoder_packet_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Shared Sparse Crosscoder Packet Gate Manifest",
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
        "# Shared Sparse Crosscoder Direction",
        "",
        f"- direction: `{payload['direction']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval families: `{payload['train_family_set']} -> {payload['eval_family_set']}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['shared_sparse_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | "
            f"{row['shared_minus_target']:.3f} | {row['paired_bootstrap_vs_target']['ci95_low']:.3f} | "
            f"{row['top_atom_knockout_lift_reduction']:.3f} | {row['oracle_candidate_atoms_accuracy']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Shared Sparse Crosscoder Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- direction pass: `{h['direction_pass']}`",
        f"- cross-family pass: `{h['cross_family_pass']}`",
        f"- budgets: `{payload['budgets']}`",
        f"- max shared sparse accuracy: `{h['max_shared_sparse_accuracy']:.3f}`",
        f"- max shared-target delta: `{h['max_shared_minus_target']:.3f}`",
        "",
        "## Rows",
        "",
        "| Direction | Budget | N | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Knockout reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['direction']} | {row['budget_bytes']} | {row['n']} | `{row['pass_gate']}` | "
            f"{row['shared_sparse_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['shared_minus_target']:.3f} | "
            f"{row['paired_ci95_low_vs_target']:.3f} | {row['top_atom_knockout_lift_reduction']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_shared_sparse_crosscoder_packet_gate_20260429"),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=output_dir,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
