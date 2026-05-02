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
from dataclasses import dataclass
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAG_LETTERS = "ABCDEFGHJKLMNPQRSTUVWYZ"


@dataclass(frozen=True)
class RepairFamily:
    name: str
    public_issue: str
    buggy_code: str
    hidden_input: Any
    expected: Any
    diagnostic_code: str
    patch_intents: tuple[str, str, str, str]
    answer_index: int


@dataclass(frozen=True)
class Candidate:
    label: str
    patch_name: str
    patch_intent: str
    handles_diagnostic: str
    prior_score: float


@dataclass(frozen=True)
class Example:
    example_id: str
    family_name: str
    public_issue: str
    target_prompt: str
    source_prompt: str
    private_test_log: str
    candidates: tuple[Candidate, ...]
    answer_label: str
    diagnostic_code: str
    hidden_input_repr: str
    expected_repr: str
    actual_repr: str


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _core_families() -> tuple[RepairFamily, ...]:
    return (
        RepairFamily(
            name="empty_list_first",
            public_issue="Function should return the first value or 0 when the input list is empty.",
            buggy_code="def solve(values):\n    return values[0]\n",
            hidden_input=[],
            expected=0,
            diagnostic_code="E0",
            patch_intents=("empty-list guard", "sort before reading", "last element fallback", "string coercion"),
            answer_index=0,
        ),
        RepairFamily(
            name="missing_key_default",
            public_issue="Function should read key 'status' from a mapping and default to 'unknown' when absent.",
            buggy_code="def solve(row):\n    return row['status']\n",
            hidden_input={},
            expected="unknown",
            diagnostic_code="K0",
            patch_intents=("strict key access", "missing-key default", "numeric fallback", "uppercase status"),
            answer_index=1,
        ),
        RepairFamily(
            name="round_half_up",
            public_issue="Function should round a price to the nearest integer using half-up behavior.",
            buggy_code="def solve(price):\n    return int(price)\n",
            hidden_input=2.6,
            expected=3,
            diagnostic_code="R1",
            patch_intents=("floor conversion", "ceiling always", "half-up rounding", "string formatting"),
            answer_index=2,
        ),
        RepairFamily(
            name="inclusive_threshold",
            public_issue="Function should approve values greater than or equal to the threshold.",
            buggy_code="def solve(pair):\n    value, threshold = pair\n    return value > threshold\n",
            hidden_input=(5, 5),
            expected=True,
            diagnostic_code="C1",
            patch_intents=("strict greater-than", "threshold ignored", "boolean inversion", "inclusive comparison"),
            answer_index=3,
        ),
        RepairFamily(
            name="preserve_order_unique",
            public_issue="Function should remove duplicates while preserving first-seen order.",
            buggy_code="def solve(values):\n    return sorted(set(values))\n",
            hidden_input=[3, 1, 3, 2],
            expected=[3, 1, 2],
            diagnostic_code="O2",
            patch_intents=("preserve-order unique", "sort unique values", "keep duplicates", "reverse values"),
            answer_index=0,
        ),
        RepairFamily(
            name="none_to_empty",
            public_issue="Function should normalize None to an empty string before returning text.",
            buggy_code="def solve(text):\n    return text.strip()\n",
            hidden_input=None,
            expected="",
            diagnostic_code="N0",
            patch_intents=("strip only", "none-to-empty", "none-to-zero", "uppercase text"),
            answer_index=1,
        ),
        RepairFamily(
            name="sum_all_values",
            public_issue="Function should sum all numeric values in a list.",
            buggy_code="def solve(values):\n    total = 0\n    for value in values[:-1]:\n        total += value\n    return total\n",
            hidden_input=[2, 3, 5],
            expected=10,
            diagnostic_code="S1",
            patch_intents=("drop last item", "multiply values", "sum all values", "return max value"),
            answer_index=2,
        ),
        RepairFamily(
            name="case_insensitive_match",
            public_issue="Function should compare two strings case-insensitively.",
            buggy_code="def solve(pair):\n    left, right = pair\n    return left == right\n",
            hidden_input=("Alpha", "alpha"),
            expected=True,
            diagnostic_code="T1",
            patch_intents=("case-sensitive equality", "always false", "prefix match", "case-insensitive equality"),
            answer_index=3,
        ),
    )


def _holdout_families() -> tuple[RepairFamily, ...]:
    return (
        RepairFamily(
            name="clamp_negative_to_zero",
            public_issue="Function should clamp negative integers to zero and return nonnegative values unchanged.",
            buggy_code="def solve(value):\n    return value\n",
            hidden_input=-3,
            expected=0,
            diagnostic_code="L0",
            patch_intents=("clamp negative to zero", "absolute value", "always zero", "string conversion"),
            answer_index=0,
        ),
        RepairFamily(
            name="last_value_default",
            public_issue="Function should return the final list value, or None when the list is empty.",
            buggy_code="def solve(values):\n    return values[-1]\n",
            hidden_input=[],
            expected=None,
            diagnostic_code="D0",
            patch_intents=("first value fallback", "empty-list none default", "empty-list zero default", "sort before final value"),
            answer_index=1,
        ),
        RepairFamily(
            name="parse_int_default",
            public_issue="Function should parse an integer string and return 0 when parsing fails.",
            buggy_code="def solve(text):\n    return int(text)\n",
            hidden_input="n/a",
            expected=0,
            diagnostic_code="P0",
            patch_intents=("float parsing", "return raw text", "parse-failure zero default", "length fallback"),
            answer_index=2,
        ),
        RepairFamily(
            name="average_all_values",
            public_issue="Function should return the arithmetic mean over all numeric list values.",
            buggy_code="def solve(values):\n    return sum(values) / (len(values) - 1)\n",
            hidden_input=[2, 4, 6],
            expected=4,
            diagnostic_code="A1",
            patch_intents=("sum only", "divide by first value", "drop last before averaging", "average all values"),
            answer_index=3,
        ),
        RepairFamily(
            name="strip_and_lower",
            public_issue="Function should strip surrounding whitespace and lowercase the text.",
            buggy_code="def solve(text):\n    return text.strip()\n",
            hidden_input=" HELLO ",
            expected="hello",
            diagnostic_code="B1",
            patch_intents=("strip and lowercase", "strip only", "uppercase only", "remove spaces inside text"),
            answer_index=0,
        ),
        RepairFamily(
            name="nested_key_default",
            public_issue="Function should read user.name from a nested mapping and default to an empty string when missing.",
            buggy_code="def solve(row):\n    return row['user']['name']\n",
            hidden_input={"user": {}},
            expected="",
            diagnostic_code="M0",
            patch_intents=("top-level name lookup", "nested missing-key default", "return whole user mapping", "uppercase nested name"),
            answer_index=1,
        ),
        RepairFamily(
            name="wrapped_index_lookup",
            public_issue="Function should use modulo wrapping when indexing into a list.",
            buggy_code="def solve(payload):\n    values, index = payload\n    return values[index]\n",
            hidden_input=([10, 20, 30], 4),
            expected=20,
            diagnostic_code="W1",
            patch_intents=("clamp to final index", "always first value", "modulo-wrapped index", "sort before index"),
            answer_index=2,
        ),
        RepairFamily(
            name="strict_positive_filter",
            public_issue="Function should keep only strictly positive values from a list.",
            buggy_code="def solve(values):\n    return [value for value in values if value >= 0]\n",
            hidden_input=[-1, 0, 2],
            expected=[2],
            diagnostic_code="F1",
            patch_intents=("keep nonnegative values", "keep negative values", "sort positive values", "strictly positive filter"),
            answer_index=3,
        ),
    )


def _families(family_set: str) -> tuple[RepairFamily, ...]:
    if family_set == "core":
        return _core_families()
    if family_set == "holdout":
        return _holdout_families()
    if family_set == "all":
        return _core_families() + _holdout_families()
    raise ValueError(f"unknown family set {family_set!r}")


def _diagnostic_code(example_index: int, seed: int) -> str:
    return f"{DIAG_LETTERS[(example_index + seed) % len(DIAG_LETTERS)]}{example_index % 10}"


def _plausible_decoy_diagnostic_code(
    *,
    example_index: int,
    seed: int,
    candidate_index: int,
    answer_index: int,
    candidates: int,
    answer_code: str,
) -> str:
    offset = (candidate_index - answer_index) % candidates
    for attempt in range(1, candidates + 3):
        code_index = example_index + (offset + attempt) * 997 + seed * 13
        code_seed = seed + (offset + attempt) * 37
        code = _diagnostic_code(code_index, code_seed)
        if code != answer_code:
            return code
    raise RuntimeError("could not construct non-answer diagnostic decoy")


def _run_buggy_code(family: RepairFamily) -> tuple[Any, str]:
    namespace: dict[str, Any] = {}
    exec(family.buggy_code, namespace)
    solve: Callable[[Any], Any] = namespace["solve"]
    try:
        actual = solve(family.hidden_input)
        return actual, ""
    except Exception as exc:  # noqa: BLE001 - intentional private execution log
        return f"{exc.__class__.__name__}: {exc}", exc.__class__.__name__


def _candidate_label(example_index: int, candidate_index: int) -> str:
    return f"candidate_{example_index:04d}_patch_{candidate_index}_repair_record"


def _make_private_log(*, family: RepairFamily, example_index: int, actual: Any, exception_name: str) -> str:
    status = "EXCEPTION" if exception_name else "ASSERTION"
    return "\n".join(
        [
            "pytest session starts",
            f"collected 1 hidden item for repair_family={family.name}",
            f"FAILED hidden_tests/test_{family.name}.py::test_private_regression_{example_index:04d}",
            f"hidden_input={family.hidden_input!r}",
            f"expected={family.expected!r}",
            f"actual={actual!r}",
            f"failure_status={status}",
            f"private_tool_trace: REPAIR_DIAG={_diagnostic_code(example_index, 28)}",
            "hint: emit only the REPAIR_DIAG code; target maps it to public candidate repair metadata",
        ]
    )


def make_benchmark(
    *,
    examples: int,
    candidates: int,
    seed: int,
    family_set: str = "core",
    start_index: int = 0,
    diagnostic_table_mode: str = "legacy",
) -> list[Example]:
    if examples <= 0:
        raise ValueError("--examples must be positive")
    if candidates != 4:
        raise ValueError("hidden repair smoke currently expects exactly 4 candidates")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if diagnostic_table_mode not in {"legacy", "plausible_decoys"}:
        raise ValueError(f"unknown diagnostic_table_mode {diagnostic_table_mode!r}")
    rng = random.Random(seed)
    families = _families(family_set)
    rows: list[Example] = []
    for local_index in range(examples):
        example_index = start_index + local_index
        family = families[example_index % len(families)]
        prior_index = family.answer_index if example_index % candidates == 0 else (family.answer_index + 1) % candidates
        answer_index = family.answer_index
        diagnostic_code = _diagnostic_code(example_index, seed)
        actual, exception_name = _run_buggy_code(family)
        private_log = _make_private_log(family=family, example_index=example_index, actual=actual, exception_name=exception_name).replace(
            f"REPAIR_DIAG={_diagnostic_code(example_index, 28)}",
            f"REPAIR_DIAG={diagnostic_code}",
        )
        candidate_rows: list[Candidate] = []
        for candidate_index, intent in enumerate(family.patch_intents):
            if candidate_index == answer_index:
                diagnostic = diagnostic_code
            elif diagnostic_table_mode == "legacy":
                diagnostic = f"X{candidate_index}"
            else:
                diagnostic = _plausible_decoy_diagnostic_code(
                    example_index=example_index,
                    seed=seed,
                    candidate_index=candidate_index,
                    answer_index=answer_index,
                    candidates=candidates,
                    answer_code=diagnostic_code,
                )
            candidate_rows.append(
                Candidate(
                    label=_candidate_label(example_index, candidate_index),
                    patch_name=f"{family.name}_patch_{candidate_index}",
                    patch_intent=intent,
                    handles_diagnostic=diagnostic,
                    prior_score=1.0 if candidate_index == prior_index else rng.uniform(0.05, 0.65),
                )
            )
        answer = candidate_rows[answer_index]
        candidates_text = "\n".join(
            "- "
            f"{row.label}: patch={row.patch_name}; intent={row.patch_intent}; "
            f"handles_repair_diag={row.handles_diagnostic}"
            for row in candidate_rows
        )
        public_issue = f"{family.public_issue} Hidden tests are private to the source agent."
        target_prompt = (
            f"{public_issue}\n"
            "Choose the candidate patch. Return only the candidate label.\n"
            f"Buggy implementation:\n{family.buggy_code}\n"
            f"Candidates:\n{candidates_text}"
        )
        trace_line = next(line for line in private_log.splitlines() if "REPAIR_DIAG=" in line)
        source_prompt = (
            "You are the source agent. You may inspect the private hidden-test log. "
            "Send only the compact repair diagnostic packet.\n"
            f"Public issue: {public_issue}\n"
            f"Private hidden-test log:\n{private_log}\n"
            f"Private REPAIR_DIAG line copied from the log: {trace_line}\n"
            "Packet:"
        )
        rows.append(
            Example(
                example_id=f"sphr_{example_index:04d}",
                family_name=family.name,
                public_issue=public_issue,
                target_prompt=target_prompt,
                source_prompt=source_prompt,
                private_test_log=private_log,
                candidates=tuple(candidate_rows),
                answer_label=answer.label,
                diagnostic_code=diagnostic_code,
                hidden_input_repr=repr(family.hidden_input),
                expected_repr=repr(family.expected),
                actual_repr=repr(actual),
            )
        )
    return rows


def _prior_prediction(example: Example) -> str:
    return max(example.candidates, key=lambda row: row.prior_score).label


def _extract_diag(payload: str | None) -> str | None:
    if not payload:
        return None
    explicit = re.search(r"REPAIR_DIAG=([A-Z][0-9])", payload)
    if explicit:
        return explicit.group(1)
    jsonish = re.search(r'"repair_diag"\s*:\s*"([A-Z][0-9])"', payload)
    if jsonish:
        return jsonish.group(1)
    prose = re.search(r"\brepair diag(?:nostic)?\s+(?:is\s+)?([A-Z][0-9])\b", payload, flags=re.IGNORECASE)
    if prose:
        return prose.group(1).upper()
    exact = re.fullmatch(r"[A-Z][0-9]", payload.strip())
    if exact:
        return exact.group(0)
    return None


def _decode_packet(example: Example, payload: str | None) -> tuple[str, dict[str, Any]]:
    diag = _extract_diag(payload)
    if not diag:
        return _prior_prediction(example), {"parsed_repair_diag": False}
    matches = [candidate for candidate in example.candidates if candidate.handles_diagnostic == diag]
    if not matches:
        return _prior_prediction(example), {"parsed_repair_diag": True, "diag_not_in_pool": True}
    prediction = max(matches, key=lambda row: row.prior_score).label
    return prediction, {
        "parsed_repair_diag": True,
        "matched_candidates": [candidate.label for candidate in matches],
        "ambiguous": len(matches) > 1,
    }


def _deterministic_nonself_index(index: int, n: int) -> int:
    candidate = (index * 17 + 11) % n
    return candidate if candidate != index else (index + 1) % n


def _mask_log_components(private_log: str, *, mask_expected_actual: bool = False, mask_test_name: bool = False) -> str:
    lines: list[str] = []
    for line in private_log.splitlines():
        if mask_expected_actual and line.startswith("expected="):
            lines.append("expected=<MASKED>")
        elif mask_expected_actual and line.startswith("actual="):
            lines.append("actual=<MASKED>")
        elif mask_test_name and line.startswith("FAILED hidden_tests/"):
            lines.append("FAILED hidden_tests/test_MASKED.py::test_private_regression_MASKED")
        else:
            lines.append(line)
    return "\n".join(lines)


def _mask_repair_diag(private_log: str) -> str:
    return re.sub(r"REPAIR_DIAG=[A-Z][0-9]", "REPAIR_DIAG=??", private_log)


def _condition_payload(
    *,
    condition: str,
    example: Example,
    examples: list[Example],
    index: int,
    budget_bytes: int,
    rng: random.Random,
) -> tuple[str | None, int, int, dict[str, Any]]:
    if condition in {"target_only", "target_wrapper", "zero_source", "answer_masked"}:
        return None, 0, 0, {}
    if condition == "matched_repair_packet":
        payload = example.diagnostic_code[:budget_bytes]
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "repair_diag"}
    if condition == "shuffled_source":
        other = examples[_deterministic_nonself_index(index, len(examples))]
        payload = other.diagnostic_code[:budget_bytes]
        return payload, len(payload.encode("utf-8")), _token_count(payload), {
            "packet_kind": "repair_diag",
            "source_example_id": other.example_id,
        }
    if condition == "random_same_byte":
        payload = f"{rng.choice('EKRCONST')}{rng.randrange(0, 10)}"
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "random_diag"}
    if condition == "answer_only":
        payload = example.answer_label.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "answer_label"}
    if condition == "target_derived_sidecar":
        prior = _prior_prediction(example)
        prior_diag = next(candidate.handles_diagnostic for candidate in example.candidates if candidate.label == prior)
        return prior_diag, len(prior_diag.encode("utf-8")), _token_count(prior_diag), {"packet_kind": "target_prior_diag"}
    if condition == "structured_text_matched":
        payload = example.private_test_log.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "truncated_hidden_log"}
    if condition == "structured_json_matched":
        text = json.dumps({"repair_diag": example.diagnostic_code}, sort_keys=True)
        payload = text.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "truncated_json_diag"}
    if condition == "structured_free_text_matched":
        text = f"repair diag is {example.diagnostic_code}"
        payload = text.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "truncated_free_text_diag"}
    if condition == "helper_only_no_log":
        payload = "Private REPAIR_DIAG line copied from the log: <withheld>"
        payload = payload.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "helper_template_no_log"}
    if condition == "full_hidden_log":
        payload = example.private_test_log
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "full_hidden_log"}
    if condition == "expected_actual_masked_full_log":
        payload = _mask_log_components(example.private_test_log, mask_expected_actual=True)
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "expected_actual_masked_log"}
    if condition == "test_name_masked_full_log":
        payload = _mask_log_components(example.private_test_log, mask_test_name=True)
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "test_name_masked_log"}
    if condition == "diag_masked_full_log":
        payload = _mask_repair_diag(example.private_test_log)
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "diag_masked_log"}
    if condition == "full_diag_text":
        payload = f"REPAIR_DIAG={example.diagnostic_code}"
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "full_diag_text"}
    raise ValueError(f"unknown condition {condition!r}")


def _conditions() -> list[str]:
    return [
        "target_only",
        "target_wrapper",
        "matched_repair_packet",
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_derived_sidecar",
        "structured_text_matched",
        "structured_json_matched",
        "structured_free_text_matched",
        "helper_only_no_log",
        "full_hidden_log",
        "expected_actual_masked_full_log",
        "test_name_masked_full_log",
        "diag_masked_full_log",
        "full_diag_text",
    ]


def _predict_condition(
    *,
    condition: str,
    example: Example,
    examples: list[Example],
    index: int,
    budget_bytes: int,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, payload_bytes, payload_tokens, metadata = _condition_payload(
        condition=condition,
        example=example,
        examples=examples,
        index=index,
        budget_bytes=budget_bytes,
        rng=rng,
    )
    if condition in {
        "matched_repair_packet",
        "shuffled_source",
        "random_same_byte",
        "target_derived_sidecar",
        "structured_text_matched",
        "structured_json_matched",
        "structured_free_text_matched",
        "helper_only_no_log",
        "full_hidden_log",
        "expected_actual_masked_full_log",
        "test_name_masked_full_log",
        "diag_masked_full_log",
        "full_diag_text",
    }:
        prediction, decode_metadata = _decode_packet(example, payload)
    elif condition == "answer_only":
        prediction = payload if payload == example.answer_label else _prior_prediction(example)
        decode_metadata = {"answer_exact_match": payload == example.answer_label}
    else:
        prediction = _prior_prediction(example)
        decode_metadata = {}
    return {
        "prediction": prediction,
        "correct": prediction == example.answer_label,
        "payload": payload,
        "payload_bytes": payload_bytes,
        "payload_tokens": payload_tokens,
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        **metadata,
        **decode_metadata,
    }


def _ids(rows: list[dict[str, Any]], condition: str) -> list[str]:
    return [row["example_id"] for row in rows if row["conditions"][condition]["correct"]]


def summarize_budget(rows: list[dict[str, Any]], *, budget_bytes: int) -> dict[str, Any]:
    exact_ids = [row["example_id"] for row in rows]
    metrics: dict[str, Any] = {}
    for condition in _conditions():
        payloads = [row["conditions"][condition]["payload_bytes"] for row in rows]
        tokens = [row["conditions"][condition]["payload_tokens"] for row in rows]
        latencies = [row["conditions"][condition]["latency_ms"] for row in rows]
        correct_ids = _ids(rows, condition)
        metrics[condition] = {
            "correct": len(correct_ids),
            "accuracy": len(correct_ids) / len(rows),
            "correct_ids": correct_ids,
            "mean_payload_bytes": statistics.fmean(payloads),
            "max_payload_bytes": max(payloads),
            "mean_payload_tokens": statistics.fmean(tokens),
            "p50_latency_ms": statistics.median(latencies),
        }
    no_source = ["target_only", "target_wrapper"]
    controls = ["zero_source", "shuffled_source", "random_same_byte", "answer_only", "answer_masked", "target_derived_sidecar"]
    reviewer_negative_controls = [
        "structured_text_matched",
        "structured_json_matched",
        "structured_free_text_matched",
        "helper_only_no_log",
        "diag_masked_full_log",
    ]
    reviewer_positive_oracles = [
        "full_hidden_log",
        "expected_actual_masked_full_log",
        "test_name_masked_full_log",
        "full_diag_text",
    ]
    best_no_source = max(metrics[name]["accuracy"] for name in no_source)
    best_control = max(metrics[name]["accuracy"] for name in controls)
    matched = metrics["matched_repair_packet"]["accuracy"]
    matched_text = metrics["structured_text_matched"]["accuracy"]
    best_reviewer_negative = max(metrics[name]["accuracy"] for name in reviewer_negative_controls)
    min_reviewer_positive = min(metrics[name]["accuracy"] for name in reviewer_positive_oracles)
    pass_gate = (
        matched - best_no_source >= 0.15
        and best_control <= best_no_source + 0.02
        and matched_text <= best_no_source + 0.02
        and best_reviewer_negative <= best_no_source + 0.02
        and min_reviewer_positive >= matched
    )
    return {
        "budget_bytes": budget_bytes,
        "n": len(rows),
        "exact_id_count": len(set(exact_ids)),
        "exact_id_sha256": _sha256_text(exact_ids),
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "candidate_pool_recall": 1.0,
        "candidate_pool_gold_count": len(rows),
        "matched_selector_accuracy": matched,
        "best_no_source_accuracy": best_no_source,
        "best_source_destroying_control_accuracy": best_control,
        "best_reviewer_negative_control_accuracy": best_reviewer_negative,
        "min_reviewer_positive_oracle_accuracy": min_reviewer_positive,
        "matched_minus_best_no_source": matched - best_no_source,
        "matched_minus_best_control": matched - best_control,
        "matched_minus_matched_text_relay": matched - matched_text,
        "pass_gate": pass_gate,
        "metrics": metrics,
    }


def run_budget(*, examples: list[Example], seed: int, budget_bytes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed + budget_bytes * 1009)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        condition_rows = {
            condition: _predict_condition(
                condition=condition,
                example=example,
                examples=examples,
                index=index,
                budget_bytes=budget_bytes,
                rng=rng,
            )
            for condition in _conditions()
        }
        rows.append(
            {
                "example_id": example.example_id,
                "family_name": example.family_name,
                "answer_label": example.answer_label,
                "diagnostic_code": example.diagnostic_code,
                "candidate_labels": [candidate.label for candidate in example.candidates],
                "candidate_diags": [candidate.handles_diagnostic for candidate in example.candidates],
                "candidate_pool_contains_gold": any(candidate.label == example.answer_label for candidate in example.candidates),
                "hidden_input_repr": example.hidden_input_repr,
                "expected_repr": example.expected_repr,
                "actual_repr": example.actual_repr,
                "target_prompt_bytes": len(example.target_prompt.encode("utf-8")),
                "source_prompt_bytes": len(example.source_prompt.encode("utf-8")),
                "private_test_log_bytes": len(example.private_test_log.encode("utf-8")),
                "conditions": condition_rows,
            }
        )
    return rows, summarize_budget(rows, budget_bytes=budget_bytes)


def summarize_sweep(budget_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [summary for summary in budget_summaries if summary["pass_gate"]]
    best = min(passing, key=lambda row: row["budget_bytes"]) if passing else max(budget_summaries, key=lambda row: row["matched_minus_best_control"])
    return {
        "budgets": [summary["budget_bytes"] for summary in budget_summaries],
        "passing_budgets": [summary["budget_bytes"] for summary in passing],
        "best_budget_bytes": best["budget_bytes"],
        "strict_smoke_pass": bool(passing),
        "pass_rule": "At least one budget must have matched_repair_packet - best no-source >= 0.15, source-destroying controls within +0.02, reviewer negative controls within +0.02, and reviewer positive oracles at or above matched.",
        "budget_summaries": budget_summaries,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_benchmark(path: pathlib.Path, examples: list[Example]) -> None:
    rows = [
        {
            "example_id": example.example_id,
            "family_name": example.family_name,
            "public_issue": example.public_issue,
            "target_prompt": example.target_prompt,
            "source_prompt": example.source_prompt,
            "private_test_log": example.private_test_log,
            "answer_label": example.answer_label,
            "diagnostic_code": example.diagnostic_code,
            "hidden_input_repr": example.hidden_input_repr,
            "expected_repr": example.expected_repr,
            "actual_repr": example.actual_repr,
            "candidates": [
                {
                    "label": candidate.label,
                    "patch_name": candidate.patch_name,
                    "patch_intent": candidate.patch_intent,
                    "handles_diagnostic": candidate.handles_diagnostic,
                    "prior_score": candidate.prior_score,
                }
                for candidate in example.candidates
            ],
        }
        for example in examples
    ]
    _write_jsonl(path, rows)


def _write_markdown(path: pathlib.Path, sweep: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Hidden-Repair Packet Smoke",
        "",
        f"- strict smoke pass: `{sweep['strict_smoke_pass']}`",
        f"- passing budgets: `{sweep['passing_budgets']}`",
        f"- best budget bytes: `{sweep['best_budget_bytes']}`",
        "",
        "| Budget bytes | Pass | Matched | Best no-source | Best control | Best reviewer negative | Min reviewer oracle | Matched text | JSON | Free text | Helper/no-log | Diag masked | Full log | Full diag |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sweep["budget_summaries"]:
        metrics = summary["metrics"]
        lines.append(
            "| "
            f"{summary['budget_bytes']} | `{summary['pass_gate']}` | "
            f"{metrics['matched_repair_packet']['accuracy']:.3f} | "
            f"{summary['best_no_source_accuracy']:.3f} | "
            f"{summary['best_source_destroying_control_accuracy']:.3f} | "
            f"{summary['best_reviewer_negative_control_accuracy']:.3f} | "
            f"{summary['min_reviewer_positive_oracle_accuracy']:.3f} | "
            f"{metrics['structured_text_matched']['accuracy']:.3f} | "
            f"{metrics['structured_json_matched']['accuracy']:.3f} | "
            f"{metrics['structured_free_text_matched']['accuracy']:.3f} | "
            f"{metrics['helper_only_no_log']['accuracy']:.3f} | "
            f"{metrics['diag_masked_full_log']['accuracy']:.3f} | "
            f"{metrics['full_hidden_log']['accuracy']:.3f} | "
            f"{metrics['full_diag_text']['accuracy']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {sweep['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(path: pathlib.Path, *, manifest: dict[str, Any]) -> None:
    sweep = manifest["sweep"]
    lines = [
        "# Source-Private Hidden-Repair Packet Smoke Manifest",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Outcome",
        "",
        f"- strict smoke pass: `{sweep['strict_smoke_pass']}`",
        f"- passing budgets: `{sweep['passing_budgets']}`",
        f"- best budget bytes: `{sweep['best_budget_bytes']}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest["artifacts"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--budgets", type=str, default="2,4,8,16,32")
    parser.add_argument("--family-set", choices=["core", "holdout", "all"], default="core")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="legacy")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    budgets = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = make_benchmark(
        examples=args.examples,
        candidates=args.candidates,
        seed=args.seed,
        family_set=args.family_set,
        start_index=args.start_index,
        diagnostic_table_mode=args.diagnostic_table_mode,
    )
    _write_benchmark(output_dir / "benchmark.jsonl", examples)

    artifacts = ["benchmark.jsonl", "sweep_summary.json", "sweep_summary.md", "manifest.json", "manifest.md"]
    summaries: list[dict[str, Any]] = []
    for budget in budgets:
        rows, summary = run_budget(examples=examples, seed=args.seed, budget_bytes=budget)
        predictions_name = f"predictions_budget{budget}.jsonl"
        summary_name = f"summary_budget{budget}.json"
        _write_jsonl(output_dir / predictions_name, rows)
        (output_dir / summary_name).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.extend([predictions_name, summary_name])
        summaries.append(summary)
    sweep = summarize_sweep(summaries)
    (output_dir / "sweep_summary.json").write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "sweep_summary.md", sweep)
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_hidden_repair_packet_smoke.py",
                f"--examples {args.examples}",
                f"--candidates {args.candidates}",
                f"--seed {args.seed}",
                f"--budgets {args.budgets}",
                f"--family-set {args.family_set}",
                f"--start-index {args.start_index}",
                f"--diagnostic-table-mode {args.diagnostic_table_mode}",
                f"--output-dir {args.output_dir}",
            ]
        ),
        "args": vars(args) | {"output_dir": str(args.output_dir), "budgets": budgets},
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "python": sys.version,
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "sweep": sweep,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest(output_dir / "manifest.md", manifest=manifest)
    if not sweep["strict_smoke_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
