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


DOMAINS = (
    "clinic intake",
    "field survey",
    "warehouse audit",
    "library archive",
    "lab sample routing",
    "incident triage",
    "travel desk",
    "school roster",
)

PUBLIC_ATTRIBUTES = (
    "blue badge",
    "green folder",
    "north desk",
    "silver tag",
    "morning slot",
    "east locker",
    "paper form",
    "quiet room",
)


@dataclass(frozen=True)
class Candidate:
    label: str
    display_name: str
    public_attribute: str
    hidden_key: str
    commitment_hex: str
    prior_score: float


@dataclass(frozen=True)
class Example:
    example_id: str
    domain: str
    template_id: int
    public_question: str
    target_prompt: str
    source_prompt: str
    private_evidence: str
    candidates: tuple[Candidate, ...]
    answer_label: str
    answer_hidden_key: str


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _digest_hex(value: str, *, salt: str, digest_bytes: int = 16) -> str:
    return hashlib.blake2s(f"{salt}:{value}".encode("utf-8"), digest_size=digest_bytes).hexdigest()


def _packet_hex(value: str, *, salt: str, budget_bytes: int) -> str:
    return hashlib.blake2s(f"{salt}:{value}".encode("utf-8"), digest_size=budget_bytes).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _long_key(rng: random.Random) -> str:
    pieces = [f"{rng.randrange(0, 16**8):08x}" for _ in range(6)]
    return "-".join(pieces)


def _candidate_label(example_index: int, candidate_index: int) -> str:
    return f"candidate_{example_index:04d}_option_{candidate_index}_private_commitment_record"


def _make_display_name(domain: str, example_index: int, candidate_index: int) -> str:
    short_domain = domain.replace(" ", "_")
    return f"{short_domain}_record_{example_index:04d}_{candidate_index}"


def make_benchmark(*, examples: int, candidates: int, seed: int) -> list[Example]:
    if examples <= 0:
        raise ValueError("--examples must be positive")
    if candidates < 2:
        raise ValueError("--candidates must be at least 2")
    rng = random.Random(seed)
    rows: list[Example] = []
    commitment_salt = f"source-private-commitment:{seed}"
    for example_index in range(examples):
        domain = DOMAINS[example_index % len(DOMAINS)]
        template_id = example_index % 5
        prior_index = example_index % candidates
        answer_index = prior_index if example_index % candidates == 0 else (prior_index + 1) % candidates
        candidate_rows: list[Candidate] = []
        used_keys: set[str] = set()
        for candidate_index in range(candidates):
            key = _long_key(rng)
            while key in used_keys:
                key = _long_key(rng)
            used_keys.add(key)
            label = _candidate_label(example_index, candidate_index)
            display_name = _make_display_name(domain, example_index, candidate_index)
            public_attribute = PUBLIC_ATTRIBUTES[(example_index + candidate_index) % len(PUBLIC_ATTRIBUTES)]
            candidate_rows.append(
                Candidate(
                    label=label,
                    display_name=display_name,
                    public_attribute=public_attribute,
                    hidden_key=key,
                    commitment_hex=_digest_hex(key, salt=commitment_salt, digest_bytes=16),
                    prior_score=1.0 if candidate_index == prior_index else rng.uniform(0.05, 0.65),
                )
            )
        answer = candidate_rows[answer_index]
        candidates_text = "\n".join(
            f"- {row.label}: {row.display_name}; public cue={row.public_attribute}; commitment={row.commitment_hex}"
            for row in candidate_rows
        )
        question = (
            f"In the {domain} task, select the candidate whose private witness key matches "
            "the source packet. Return only the candidate label."
        )
        target_prompt = f"{question}\nCandidates:\n{candidates_text}"
        private_evidence = (
            f"Private evidence for {domain}: the witness key is {answer.hidden_key}. "
            f"The matching candidate is {answer.display_name}."
        )
        source_prompt = (
            "You are the source agent. Read the private evidence and send only the "
            "rate-capped packet requested by the protocol.\n"
            f"Question: {question}\nPrivate evidence: {private_evidence}"
        )
        rows.append(
            Example(
                example_id=f"spqa_{example_index:04d}",
                domain=domain,
                template_id=template_id,
                public_question=question,
                target_prompt=target_prompt,
                source_prompt=source_prompt,
                private_evidence=private_evidence,
                candidates=tuple(candidate_rows),
                answer_label=answer.label,
                answer_hidden_key=answer.hidden_key,
            )
        )
    return rows


def _prior_prediction(example: Example) -> str:
    return max(example.candidates, key=lambda row: row.prior_score).label


def _decode_syndrome(example: Example, packet: str | None, *, packet_salt: str, budget_bytes: int) -> tuple[str, dict[str, Any]]:
    if not packet:
        return _prior_prediction(example), {"matched_candidates": [], "ambiguous": False}
    matches = [
        candidate
        for candidate in example.candidates
        if _packet_hex(candidate.hidden_key, salt=packet_salt, budget_bytes=budget_bytes) == packet
    ]
    if not matches:
        return _prior_prediction(example), {"matched_candidates": [], "ambiguous": False}
    prediction = max(matches, key=lambda row: row.prior_score).label
    return prediction, {
        "matched_candidates": [candidate.label for candidate in matches],
        "ambiguous": len(matches) > 1,
    }


def _decode_full_key(example: Example, payload: str) -> tuple[str, dict[str, Any]]:
    key_match = re.search(r"([0-9a-f]{8}(?:-[0-9a-f]{8}){5})", payload)
    if not key_match:
        return _prior_prediction(example), {"parsed_full_key": False}
    key = key_match.group(1)
    for candidate in example.candidates:
        if candidate.hidden_key == key:
            return candidate.label, {"parsed_full_key": True}
    return _prior_prediction(example), {"parsed_full_key": True, "key_not_in_pool": True}


def _deterministic_nonself_index(index: int, n: int) -> int:
    if n < 2:
        raise ValueError("Need at least two examples for shuffled-source controls")
    return (index * 17 + 11) % n if (index * 17 + 11) % n != index else (index + 1) % n


def _condition_payload(
    *,
    condition: str,
    example: Example,
    examples: list[Example],
    index: int,
    packet_salt: str,
    budget_bytes: int,
    rng: random.Random,
) -> tuple[str | None, int, int, dict[str, Any]]:
    if condition in {"target_only", "target_wrapper", "zero_source", "answer_masked"}:
        return None, 0, 0, {}
    if condition == "matched_syndrome":
        packet = _packet_hex(example.answer_hidden_key, salt=packet_salt, budget_bytes=budget_bytes)
        return packet, budget_bytes, 1, {"packet_kind": "syndrome"}
    if condition == "shuffled_source":
        other_index = _deterministic_nonself_index(index, len(examples))
        other = examples[other_index]
        packet = _packet_hex(other.answer_hidden_key, salt=packet_salt, budget_bytes=budget_bytes)
        return packet, budget_bytes, 1, {"packet_kind": "syndrome", "source_example_id": other.example_id}
    if condition == "random_same_byte":
        packet = rng.randbytes(budget_bytes).hex()
        return packet, budget_bytes, 1, {"packet_kind": "random"}
    if condition == "answer_only":
        payload = example.answer_label.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "answer_label"}
    if condition == "target_derived_sidecar":
        prior_label = _prior_prediction(example)
        payload = _packet_hex(prior_label, salt=packet_salt, budget_bytes=budget_bytes)
        return payload, budget_bytes, 1, {"packet_kind": "target_derived"}
    if condition == "wrong_salt_same_source":
        payload = _packet_hex(example.answer_hidden_key, salt=f"wrong:{packet_salt}", budget_bytes=budget_bytes)
        return payload, budget_bytes, 1, {"packet_kind": "wrong_salt_syndrome"}
    if condition == "structured_text_matched":
        payload = f"witness_key={example.answer_hidden_key}".encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "truncated_text"}
    if condition == "full_structured_text":
        payload = f"witness_key={example.answer_hidden_key}"
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "full_text"}
    if condition == "full_evidence_oracle":
        payload = example.private_evidence
        return payload, len(payload.encode("utf-8")), _token_count(payload), {"packet_kind": "full_evidence"}
    raise ValueError(f"unknown condition {condition!r}")


def _predict_condition(
    *,
    condition: str,
    example: Example,
    examples: list[Example],
    index: int,
    packet_salt: str,
    budget_bytes: int,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, payload_bytes, payload_tokens, metadata = _condition_payload(
        condition=condition,
        example=example,
        examples=examples,
        index=index,
        packet_salt=packet_salt,
        budget_bytes=budget_bytes,
        rng=rng,
    )
    if condition in {
        "matched_syndrome",
        "shuffled_source",
        "random_same_byte",
        "target_derived_sidecar",
        "wrong_salt_same_source",
    }:
        prediction, decode_metadata = _decode_syndrome(
            example, payload, packet_salt=packet_salt, budget_bytes=budget_bytes
        )
    elif condition in {"structured_text_matched", "full_structured_text", "full_evidence_oracle"}:
        prediction, decode_metadata = _decode_full_key(example, payload or "")
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
        "matched_syndrome",
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_derived_sidecar",
        "wrong_salt_same_source",
        "structured_text_matched",
        "full_structured_text",
        "full_evidence_oracle",
    ]


def _accuracy(rows: list[dict[str, Any]], condition: str) -> float:
    return sum(int(row["conditions"][condition]["correct"]) for row in rows) / len(rows)


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
        "wrong_salt_same_source",
    ]
    best_no_source = max(metrics[name]["accuracy"] for name in no_source_conditions)
    best_control = max(metrics[name]["accuracy"] for name in source_destroying)
    matched = metrics["matched_syndrome"]["accuracy"]
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
    rng = random.Random(seed + budget_bytes * 997)
    packet_salt = f"source-private-packet:{seed}:budget:{budget_bytes}"
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        condition_rows = {
            condition: _predict_condition(
                condition=condition,
                example=example,
                examples=examples,
                index=index,
                packet_salt=packet_salt,
                budget_bytes=budget_bytes,
                rng=rng,
            )
            for condition in _conditions()
        }
        rows.append(
            {
                "example_id": example.example_id,
                "domain": example.domain,
                "template_id": example.template_id,
                "answer_label": example.answer_label,
                "candidate_labels": [candidate.label for candidate in example.candidates],
                "candidate_pool_contains_gold": any(
                    candidate.label == example.answer_label for candidate in example.candidates
                ),
                "target_prompt_bytes": len(example.target_prompt.encode("utf-8")),
                "target_prompt_tokens": _token_count(example.target_prompt),
                "source_prompt_bytes": len(example.source_prompt.encode("utf-8")),
                "source_prompt_tokens": _token_count(example.source_prompt),
                "private_evidence_bytes": len(example.private_evidence.encode("utf-8")),
                "private_evidence_tokens": _token_count(example.private_evidence),
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
            "At least one budget must have matched_syndrome - best no-source >= 0.15, "
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
                "domain": example.domain,
                "template_id": example.template_id,
                "public_question": example.public_question,
                "target_prompt": example.target_prompt,
                "source_prompt": example.source_prompt,
                "private_evidence": example.private_evidence,
                "answer_label": example.answer_label,
                "candidates": [
                    {
                        "label": candidate.label,
                        "display_name": candidate.display_name,
                        "public_attribute": candidate.public_attribute,
                        "commitment_hex": candidate.commitment_hex,
                        "prior_score": candidate.prior_score,
                    }
                    for candidate in example.candidates
                ],
            }
        )
    _write_jsonl(path, rows)


def _write_markdown(path: pathlib.Path, sweep: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Evidence Packet Strict-Small Gate",
        "",
        f"- strict-small pass: `{sweep['strict_small_pass']}`",
        f"- passing budgets: `{sweep['passing_budgets']}`",
        f"- best budget bytes: `{sweep['best_budget_bytes']}`",
        "",
        "| Budget bytes | Pass | Matched | Best no-source | Best control | Matched text | Full text | Full evidence |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sweep["budget_summaries"]:
        metrics = summary["metrics"]
        lines.append(
            "| "
            f"{summary['budget_bytes']} | `{summary['pass_gate']}` | "
            f"{metrics['matched_syndrome']['accuracy']:.3f} | "
            f"{summary['best_no_source_accuracy']:.3f} | "
            f"{summary['best_source_destroying_control_accuracy']:.3f} | "
            f"{metrics['structured_text_matched']['accuracy']:.3f} | "
            f"{metrics['full_structured_text']['accuracy']:.3f} | "
            f"{metrics['full_evidence_oracle']['accuracy']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {sweep['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(path: pathlib.Path, *, manifest: dict[str, Any]) -> None:
    sweep = manifest["sweep"]
    lines = [
        "# Source-Private Evidence Packet Strict-Small Manifest",
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
    artifacts = ["benchmark.jsonl", "sweep_summary.json", "sweep_summary.md", "manifest.json", "manifest.md"]
    for budget in budgets:
        rows, summary = run_budget(examples=examples, seed=args.seed, budget_bytes=budget)
        predictions_name = f"predictions_budget{budget}.jsonl"
        summary_name = f"summary_budget{budget}.json"
        _write_jsonl(output_dir / predictions_name, rows)
        (output_dir / summary_name).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.extend([predictions_name, summary_name])
        budget_summaries.append(summary)

    sweep = summarize_sweep(budget_summaries)
    (output_dir / "sweep_summary.json").write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "sweep_summary.md", sweep)
    command = " ".join(
        [
            "./venv_arm64/bin/python",
            "scripts/run_source_private_evidence_packet_strict_small.py",
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
