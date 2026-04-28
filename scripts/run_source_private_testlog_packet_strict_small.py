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
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]


MODULES = (
    "billing.rate_limit",
    "auth.session_refresh",
    "inventory.reorder",
    "search.index_delta",
    "notifications.digest",
    "reports.csv_export",
    "scheduler.retry_queue",
    "payments.webhook",
)

FAILURE_KINDS = (
    "assert_status",
    "assert_count",
    "assert_order",
    "assert_timeout",
    "assert_schema",
    "assert_rounding",
    "assert_idempotency",
    "assert_permission",
)

PUBLIC_CUES = (
    "recently touched",
    "low-risk patch",
    "small diff",
    "owner approved",
    "fast rollback",
    "needs migration",
    "high test coverage",
    "legacy path",
)

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass(frozen=True)
class Candidate:
    label: str
    patch_name: str
    public_cue: str
    failure_signature: str
    prior_score: float


@dataclass(frozen=True)
class Example:
    example_id: str
    module: str
    failure_kind: str
    public_issue: str
    target_prompt: str
    source_prompt: str
    private_test_log: str
    candidates: tuple[Candidate, ...]
    answer_label: str
    answer_signature: str


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


def _signature_from_int(value: int, *, length: int = 2) -> str:
    base = len(CODE_ALPHABET)
    chars: list[str] = []
    current = value
    for _ in range(length):
        chars.append(CODE_ALPHABET[current % base])
        current //= base
    return "".join(reversed(chars))


def _candidate_label(example_index: int, candidate_index: int) -> str:
    return f"candidate_{example_index:04d}_patch_{candidate_index}_public_fix_record"


def _patch_name(module: str, example_index: int, candidate_index: int) -> str:
    module_slug = module.replace(".", "_")
    return f"{module_slug}_fix_{example_index:04d}_{candidate_index}"


def _make_test_log(
    *,
    module: str,
    failure_kind: str,
    signature: str,
    seed: int,
    example_index: int,
) -> str:
    return "\n".join(
        [
            "pytest session starts",
            f"platform=darwin seed={seed} worker=gw{example_index % 4}",
            f"collected {37 + example_index % 11} items",
            f"FAILED tests/{module.replace('.', '/')}/test_regression.py::test_{failure_kind}",
            f"private_tool_trace: TRACE_SIG={signature}",
            f"private_tool_trace: module={module}",
            f"private_tool_trace: failure={failure_kind}",
            "hint: choose the candidate whose public handled_signature matches TRACE_SIG",
        ]
    )


def make_benchmark(*, examples: int, candidates: int, seed: int) -> list[Example]:
    if examples <= 0:
        raise ValueError("--examples must be positive")
    if candidates < 2:
        raise ValueError("--candidates must be at least 2")
    rng = random.Random(seed)
    rows: list[Example] = []
    for example_index in range(examples):
        module = MODULES[example_index % len(MODULES)]
        failure_kind = FAILURE_KINDS[(example_index // len(MODULES)) % len(FAILURE_KINDS)]
        prior_index = example_index % candidates
        answer_index = prior_index if example_index % candidates == 0 else (prior_index + 1) % candidates
        signature_offset = example_index * candidates * 13 + seed * 7
        candidate_rows: list[Candidate] = []
        for candidate_index in range(candidates):
            signature = _signature_from_int(signature_offset + candidate_index * 19)
            candidate_rows.append(
                Candidate(
                    label=_candidate_label(example_index, candidate_index),
                    patch_name=_patch_name(module, example_index, candidate_index),
                    public_cue=PUBLIC_CUES[(example_index + candidate_index) % len(PUBLIC_CUES)],
                    failure_signature=signature,
                    prior_score=1.0 if candidate_index == prior_index else rng.uniform(0.05, 0.65),
                )
            )
        answer = candidate_rows[answer_index]
        public_issue = (
            f"Production regression in `{module}`. The public issue says the failing test is "
            f"in the `{failure_kind}` family, but the private tool log is required to identify "
            "which candidate patch handles the exact trace signature."
        )
        candidates_text = "\n".join(
            "- "
            f"{row.label}: patch={row.patch_name}; public cue={row.public_cue}; "
            f"handles_trace_signature={row.failure_signature}"
            for row in candidate_rows
        )
        target_prompt = (
            f"{public_issue}\n"
            "Choose the candidate patch. Return only the candidate label.\n"
            f"Candidates:\n{candidates_text}"
        )
        private_log = _make_test_log(
            module=module,
            failure_kind=failure_kind,
            signature=answer.failure_signature,
            seed=seed,
            example_index=example_index,
        )
        source_prompt = (
            "You are the source agent. Read the private test log and send only the "
            "rate-capped failure-signature packet requested by the protocol.\n"
            f"Public issue: {public_issue}\n"
            f"Private test log:\n{private_log}"
        )
        rows.append(
            Example(
                example_id=f"sptl_{example_index:04d}",
                module=module,
                failure_kind=failure_kind,
                public_issue=public_issue,
                target_prompt=target_prompt,
                source_prompt=source_prompt,
                private_test_log=private_log,
                candidates=tuple(candidate_rows),
                answer_label=answer.label,
                answer_signature=answer.failure_signature,
            )
        )
    return rows


def _prior_prediction(example: Example) -> str:
    return max(example.candidates, key=lambda row: row.prior_score).label


def _extract_trace_signature(payload: str | None) -> str | None:
    if not payload:
        return None
    explicit = re.search(r"TRACE_SIG=([A-Z2-9]{2})", payload)
    if explicit:
        return explicit.group(1)
    exact = re.fullmatch(r"[A-Z2-9]{2}", payload.strip())
    if exact:
        return exact.group(0)
    return None


def _decode_signature_packet(example: Example, payload: str | None) -> tuple[str, dict[str, Any]]:
    signature = _extract_trace_signature(payload)
    if not signature:
        return _prior_prediction(example), {"parsed_trace_signature": False}
    matches = [candidate for candidate in example.candidates if candidate.failure_signature == signature]
    if not matches:
        return _prior_prediction(example), {"parsed_trace_signature": True, "signature_not_in_pool": True}
    prediction = max(matches, key=lambda row: row.prior_score).label
    return prediction, {
        "parsed_trace_signature": True,
        "matched_candidates": [candidate.label for candidate in matches],
        "ambiguous": len(matches) > 1,
    }


def _deterministic_nonself_index(index: int, n: int) -> int:
    if n < 2:
        raise ValueError("Need at least two examples for shuffled-source controls")
    candidate = (index * 17 + 11) % n
    return candidate if candidate != index else (index + 1) % n


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
    if condition == "matched_testlog_packet":
        payload = example.answer_signature[:budget_bytes]
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "trace_signature"}
    if condition == "shuffled_source":
        other = examples[_deterministic_nonself_index(index, len(examples))]
        payload = other.answer_signature[:budget_bytes]
        return payload, len(payload.encode("utf-8")), _token_count(payload), {
            "packet_kind": "trace_signature",
            "source_example_id": other.example_id,
        }
    if condition == "random_same_byte":
        payload = "".join(rng.choice(CODE_ALPHABET) for _ in range(min(2, budget_bytes)))
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "random_signature"}
    if condition == "answer_only":
        payload = example.answer_label.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "answer_label"}
    if condition == "target_derived_sidecar":
        prior = _prior_prediction(example)
        prior_signature = next(candidate.failure_signature for candidate in example.candidates if candidate.label == prior)
        payload = prior_signature[:budget_bytes]
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "target_prior_signature"}
    if condition == "structured_text_matched":
        payload = example.private_test_log.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "truncated_test_log"}
    if condition == "full_structured_log":
        payload = example.private_test_log
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "full_test_log"}
    if condition == "full_signature_text":
        payload = f"TRACE_SIG={example.answer_signature}"
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "full_signature_text"}
    raise ValueError(f"unknown condition {condition!r}")


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
        "matched_testlog_packet",
        "shuffled_source",
        "random_same_byte",
        "target_derived_sidecar",
        "structured_text_matched",
        "full_structured_log",
        "full_signature_text",
    }:
        prediction, decode_metadata = _decode_signature_packet(example, payload)
    elif condition == "answer_only":
        prediction = payload if payload == example.answer_label else _prior_prediction(example)
        decode_metadata = {"answer_exact_match": payload == example.answer_label}
    else:
        prediction = _prior_prediction(example)
        decode_metadata = {}
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "prediction": prediction,
        "correct": prediction == example.answer_label,
        "payload": payload,
        "payload_bytes": payload_bytes,
        "payload_tokens": payload_tokens,
        "latency_ms": latency_ms,
        **metadata,
        **decode_metadata,
    }


def _conditions() -> list[str]:
    return [
        "target_only",
        "target_wrapper",
        "matched_testlog_packet",
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_derived_sidecar",
        "structured_text_matched",
        "full_structured_log",
        "full_signature_text",
    ]


def _ids(rows: Iterable[dict[str, Any]], condition: str) -> list[str]:
    return [row["example_id"] for row in rows if row["conditions"][condition]["correct"]]


def summarize_budget(rows: list[dict[str, Any]], *, budget_bytes: int) -> dict[str, Any]:
    exact_ids = [row["example_id"] for row in rows]
    if len(exact_ids) != len(set(exact_ids)):
        raise ValueError("Duplicate example IDs in strict-small predictions")
    if any(not row["candidate_pool_contains_gold"] for row in rows):
        raise ValueError("Candidate pool recall is expected to be 1.0 in this gate")
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
            "p50_payload_bytes": statistics.median(payloads),
            "max_payload_bytes": max(payloads),
            "mean_payload_tokens": statistics.fmean(tokens),
            "p50_payload_tokens": statistics.median(tokens),
            "max_payload_tokens": max(tokens),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        }
    no_source_conditions = ["target_only", "target_wrapper"]
    source_destroying = [
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_derived_sidecar",
    ]
    best_no_source = max(metrics[name]["accuracy"] for name in no_source_conditions)
    best_control = max(metrics[name]["accuracy"] for name in source_destroying)
    matched = metrics["matched_testlog_packet"]["accuracy"]
    matched_text = metrics["structured_text_matched"]["accuracy"]
    pass_gate = (
        matched - best_no_source >= 0.15
        and best_control <= best_no_source + 0.02
        and matched_text <= best_no_source + 0.02
    )
    return {
        "budget_bytes": budget_bytes,
        "n": len(rows),
        "exact_id_count": len(set(exact_ids)),
        "exact_id_sha256": _sha256_text(exact_ids),
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "candidate_pool_recall": 1.0,
        "best_no_source_accuracy": best_no_source,
        "best_source_destroying_control_accuracy": best_control,
        "matched_minus_best_no_source": matched - best_no_source,
        "matched_minus_best_control": matched - best_control,
        "matched_minus_matched_text_relay": matched - matched_text,
        "pass_gate": pass_gate,
        "metrics": metrics,
    }


def run_budget(
    *,
    examples: list[Example],
    seed: int,
    budget_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                "module": example.module,
                "failure_kind": example.failure_kind,
                "answer_label": example.answer_label,
                "answer_signature": example.answer_signature,
                "candidate_labels": [candidate.label for candidate in example.candidates],
                "candidate_signatures": [candidate.failure_signature for candidate in example.candidates],
                "candidate_pool_contains_gold": any(
                    candidate.label == example.answer_label for candidate in example.candidates
                ),
                "target_prompt_bytes": len(example.target_prompt.encode("utf-8")),
                "target_prompt_tokens": _token_count(example.target_prompt),
                "source_prompt_bytes": len(example.source_prompt.encode("utf-8")),
                "source_prompt_tokens": _token_count(example.source_prompt),
                "private_test_log_bytes": len(example.private_test_log.encode("utf-8")),
                "private_test_log_tokens": _token_count(example.private_test_log),
                "conditions": condition_rows,
            }
        )
    return rows, summarize_budget(rows, budget_bytes=budget_bytes)


def summarize_sweep(budget_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [summary for summary in budget_summaries if summary["pass_gate"]]
    best = max(
        budget_summaries,
        key=lambda row: (
            row["pass_gate"],
            row["matched_minus_best_control"],
            -row["budget_bytes"],
        ),
    )
    return {
        "budgets": [summary["budget_bytes"] for summary in budget_summaries],
        "passing_budgets": [summary["budget_bytes"] for summary in passing],
        "best_budget_bytes": best["budget_bytes"],
        "best_budget_pass_gate": best["pass_gate"],
        "strict_small_pass": bool(passing),
        "pass_rule": (
            "At least one budget must have matched_testlog_packet - best no-source >= 0.15, "
            "all source-destroying controls within +0.02 of no-source, matched-byte "
            "structured text within +0.02 of no-source, exact ID parity, and candidate "
            "pool recall 1.0."
        ),
        "budget_summaries": budget_summaries,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_benchmark(path: pathlib.Path, examples: list[Example]) -> None:
    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "example_id": example.example_id,
                "module": example.module,
                "failure_kind": example.failure_kind,
                "public_issue": example.public_issue,
                "target_prompt": example.target_prompt,
                "source_prompt": example.source_prompt,
                "private_test_log": example.private_test_log,
                "answer_label": example.answer_label,
                "answer_signature": example.answer_signature,
                "candidates": [
                    {
                        "label": candidate.label,
                        "patch_name": candidate.patch_name,
                        "public_cue": candidate.public_cue,
                        "failure_signature": candidate.failure_signature,
                        "prior_score": candidate.prior_score,
                    }
                    for candidate in example.candidates
                ],
            }
        )
    _write_jsonl(path, rows)


def _write_markdown(path: pathlib.Path, sweep: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Test-Log Packet Strict-Small Gate",
        "",
        f"- strict-small pass: `{sweep['strict_small_pass']}`",
        f"- passing budgets: `{sweep['passing_budgets']}`",
        f"- best budget bytes: `{sweep['best_budget_bytes']}`",
        "",
        "| Budget bytes | Pass | Matched | Best no-source | Best control | Matched text | Full log | Full signature |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sweep["budget_summaries"]:
        metrics = summary["metrics"]
        lines.append(
            "| "
            f"{summary['budget_bytes']} | `{summary['pass_gate']}` | "
            f"{metrics['matched_testlog_packet']['accuracy']:.3f} | "
            f"{summary['best_no_source_accuracy']:.3f} | "
            f"{summary['best_source_destroying_control_accuracy']:.3f} | "
            f"{metrics['structured_text_matched']['accuracy']:.3f} | "
            f"{metrics['full_structured_log']['accuracy']:.3f} | "
            f"{metrics['full_signature_text']['accuracy']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {sweep['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _leakage_audit(examples: list[Example], budget_predictions: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    public_target_answer_label_candidate_pool_hits = 0
    public_target_private_log_hits = 0
    target_prompt_trace_sig_hits = 0
    for example in examples:
        public_target_answer_label_candidate_pool_hits += int(example.answer_label in example.target_prompt)
        public_target_private_log_hits += int(example.private_test_log in example.target_prompt)
        target_prompt_trace_sig_hits += int(f"TRACE_SIG={example.answer_signature}" in example.target_prompt)

    packet_copy_counts: dict[str, dict[str, int]] = {}
    over_budget_counts: dict[str, dict[str, int]] = {}
    parse_failure_counts: dict[str, dict[str, int]] = {}
    for budget, rows in budget_predictions.items():
        packet_copy_counts[str(budget)] = {}
        over_budget_counts[str(budget)] = {}
        parse_failure_counts[str(budget)] = {}
        for condition in _conditions():
            copied_answer_label = 0
            copied_patch_name = 0
            copied_private_log = 0
            over_budget = 0
            parse_failure = 0
            for row in rows:
                payload = row["conditions"][condition]["payload"] or ""
                over_budget += int(row["conditions"][condition]["payload_bytes"] > budget)
                parse_failure += int(
                    condition
                    in {
                        "matched_testlog_packet",
                        "shuffled_source",
                        "random_same_byte",
                        "target_derived_sidecar",
                        "structured_text_matched",
                        "full_structured_log",
                        "full_signature_text",
                    }
                    and not row["conditions"][condition].get("parsed_trace_signature", False)
                )
                copied_answer_label += int(row["answer_label"] in payload)
                for label, signature in zip(row["candidate_labels"], row["candidate_signatures"], strict=False):
                    copied_patch_name += int(label in payload)
                    copied_private_log += int(f"TRACE_SIG={signature}" in payload and condition != "full_structured_log")
            packet_copy_counts[str(budget)][condition] = {
                "payload_contains_answer_label": copied_answer_label,
                "payload_contains_candidate_label": copied_patch_name,
                "payload_contains_trace_sig_field_outside_full_log": copied_private_log,
            }
            over_budget_counts[str(budget)][condition] = over_budget
            parse_failure_counts[str(budget)][condition] = parse_failure

    return {
        "n": len(examples),
        "exact_id_sha256": _sha256_text([example.example_id for example in examples]),
        "public_target_answer_label_candidate_pool_hits": public_target_answer_label_candidate_pool_hits,
        "public_target_private_log_hits": public_target_private_log_hits,
        "target_prompt_trace_sig_hits": target_prompt_trace_sig_hits,
        "packet_copy_counts": packet_copy_counts,
        "over_budget_counts": over_budget_counts,
        "parse_failure_counts": parse_failure_counts,
        "interpretation": (
            "Matched test-log packets intentionally transmit the two-byte trace signature. "
            "Leakage counts focus on answer/candidate-label copies and accidental TRACE_SIG "
            "field leakage outside the full-log oracle."
        ),
    }


def _write_leakage_markdown(path: pathlib.Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Test-Log Packet Leakage Audit",
        "",
        f"- examples: `{audit['n']}`",
        f"- exact ID SHA256: `{audit['exact_id_sha256']}`",
        "- public target answer-label candidate-pool hits: "
        f"`{audit['public_target_answer_label_candidate_pool_hits']}`",
        f"- public target private-log hits: `{audit['public_target_private_log_hits']}`",
        f"- public target TRACE_SIG hits: `{audit['target_prompt_trace_sig_hits']}`",
        "",
        "## Over-Budget Counts",
        "",
        "| Budget | Condition | Over budget | Parse failures | Answer-label copies | Candidate-label copies | TRACE_SIG field copies |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for budget, condition_counts in audit["over_budget_counts"].items():
        for condition, over_budget in condition_counts.items():
            copies = audit["packet_copy_counts"][budget][condition]
            lines.append(
                "| "
                f"{budget} | {condition} | {over_budget} | "
                f"{audit['parse_failure_counts'][budget][condition]} | "
                f"{copies['payload_contains_answer_label']} | "
                f"{copies['payload_contains_candidate_label']} | "
                f"{copies['payload_contains_trace_sig_field_outside_full_log']} |"
            )
    lines.extend(["", audit["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(path: pathlib.Path, *, manifest: dict[str, Any]) -> None:
    sweep = manifest["sweep"]
    lines = [
        "# Source-Private Test-Log Packet Strict-Small Manifest",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Outcome",
        "",
        f"- strict-small pass: `{sweep['strict_small_pass']}`",
        f"- passing budgets: `{sweep['passing_budgets']}`",
        f"- best budget bytes: `{sweep['best_budget_bytes']}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest["artifacts"])
    lines.extend(["", "## Artifact Hashes", ""])
    for artifact, digest in sorted(manifest["artifact_sha256"].items()):
        lines.append(f"- `{artifact}`: `{digest}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=160)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--budgets", type=str, default="2,4,8,16,32")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    budgets = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]
    if not budgets:
        raise ValueError("--budgets must include at least one integer")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--budgets must be positive integers")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = make_benchmark(examples=args.examples, candidates=args.candidates, seed=args.seed)
    _write_benchmark(output_dir / "benchmark.jsonl", examples)

    budget_summaries: list[dict[str, Any]] = []
    artifacts = [
        "benchmark.jsonl",
        "sweep_summary.json",
        "sweep_summary.md",
        "leakage_audit.json",
        "leakage_audit.md",
        "manifest.json",
        "manifest.md",
    ]
    budget_predictions: dict[int, list[dict[str, Any]]] = {}
    for budget in budgets:
        rows, summary = run_budget(examples=examples, seed=args.seed, budget_bytes=budget)
        budget_predictions[budget] = rows
        predictions_name = f"predictions_budget{budget}.jsonl"
        summary_name = f"summary_budget{budget}.json"
        _write_jsonl(output_dir / predictions_name, rows)
        (output_dir / summary_name).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.extend([predictions_name, summary_name])
        budget_summaries.append(summary)

    sweep = summarize_sweep(budget_summaries)
    (output_dir / "sweep_summary.json").write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "sweep_summary.md", sweep)
    leakage_audit = _leakage_audit(examples, budget_predictions)
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_leakage_markdown(output_dir / "leakage_audit.md", leakage_audit)
    command = " ".join(
        [
            "./venv_arm64/bin/python",
            "scripts/run_source_private_testlog_packet_strict_small.py",
            f"--examples {args.examples}",
            f"--candidates {args.candidates}",
            f"--seed {args.seed}",
            f"--budgets {args.budgets}",
            f"--output-dir {args.output_dir}",
        ]
    )
    manifest = {
        "command": command,
        "args": {
            "examples": args.examples,
            "candidates": args.candidates,
            "seed": args.seed,
            "budgets": budgets,
            "output_dir": str(args.output_dir),
        },
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
    if not sweep["strict_small_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
