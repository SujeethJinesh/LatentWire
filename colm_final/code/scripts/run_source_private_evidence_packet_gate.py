from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    key: str
    prior_score: float


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    candidates: tuple[Candidate, ...]
    answer_name: str
    private_key: str
    private_evidence: str


def _digest(value: str, *, salt: str, bytes_: int) -> bytes:
    return hashlib.blake2s(f"{salt}:{value}".encode("utf-8"), digest_size=bytes_).digest()


def _hex_digest(value: str, *, salt: str, bytes_: int) -> str:
    return _digest(value, salt=salt, bytes_=bytes_).hex()


def _make_examples(*, examples: int, candidates: int, seed: int) -> list[Example]:
    if candidates < 2:
        raise ValueError("--candidates must be at least 2")
    rng = random.Random(seed)
    rows: list[Example] = []
    for idx in range(examples):
        answer_idx = idx % candidates
        # The public prior is intentionally weak but not impossible: it is
        # correct for one candidate slot and wrong for the rest.
        prior_idx = 0
        candidate_rows: list[Candidate] = []
        for j in range(candidates):
            key = f"{rng.randrange(10_000, 99_999)}-{rng.randrange(100, 999)}"
            candidate_rows.append(
                Candidate(
                    name=f"agent_{idx:04d}_{j}",
                    key=key,
                    prior_score=1.0 if j == prior_idx else rng.uniform(0.0, 0.4),
                )
            )
        answer = candidate_rows[answer_idx]
        question = "Which candidate matches the private witness key?"
        evidence = f"private witness key is {answer.key}"
        rows.append(
            Example(
                example_id=f"spv_{idx:04d}",
                question=question,
                candidates=tuple(candidate_rows),
                answer_name=answer.name,
                private_key=answer.key,
                private_evidence=evidence,
            )
        )
    return rows


def _prior_candidate(example: Example) -> Candidate:
    return max(example.candidates, key=lambda row: row.prior_score)


def _decode_digest(example: Example, digest_hex: str | None, *, salt: str, bytes_: int) -> str:
    if not digest_hex:
        return _prior_candidate(example).name
    matches = [
        candidate
        for candidate in example.candidates
        if _hex_digest(candidate.key, salt=salt, bytes_=bytes_) == digest_hex
    ]
    if not matches:
        return _prior_candidate(example).name
    return max(matches, key=lambda row: row.prior_score).name


def _structured_text_decode(example: Example, payload: str) -> str:
    prefix = "key="
    if not payload.startswith(prefix):
        return _prior_candidate(example).name
    key = payload[len(prefix) :]
    for candidate in example.candidates:
        if candidate.key == key:
            return candidate.name
    return _prior_candidate(example).name


def _condition_prediction(
    *,
    condition: str,
    example: Example,
    examples: list[Example],
    index: int,
    salt: str,
    syndrome_bytes: int,
    rng: random.Random,
) -> tuple[str, int, float]:
    start = time.perf_counter()
    if condition in {"target_only", "target_wrapper", "zero_source", "answer_masked"}:
        prediction = _prior_candidate(example).name
        payload_bytes = 0
    elif condition == "matched_syndrome":
        payload = _hex_digest(example.private_key, salt=salt, bytes_=syndrome_bytes)
        prediction = _decode_digest(example, payload, salt=salt, bytes_=syndrome_bytes)
        payload_bytes = syndrome_bytes
    elif condition == "shuffled_source":
        other = examples[(index + 1) % len(examples)]
        payload = _hex_digest(other.private_key, salt=salt, bytes_=syndrome_bytes)
        prediction = _decode_digest(example, payload, salt=salt, bytes_=syndrome_bytes)
        payload_bytes = syndrome_bytes
    elif condition == "random_same_byte":
        payload = rng.randbytes(syndrome_bytes).hex()
        prediction = _decode_digest(example, payload, salt=salt, bytes_=syndrome_bytes)
        payload_bytes = syndrome_bytes
    elif condition == "answer_only":
        payload = _hex_digest(example.answer_name, salt=salt, bytes_=syndrome_bytes)
        prediction = _decode_digest(example, payload, salt=salt, bytes_=syndrome_bytes)
        payload_bytes = syndrome_bytes
    elif condition == "target_only_sidecar":
        payload = _hex_digest(_prior_candidate(example).key, salt=salt, bytes_=syndrome_bytes)
        prediction = _decode_digest(example, payload, salt=salt, bytes_=syndrome_bytes)
        payload_bytes = syndrome_bytes
    elif condition == "structured_text_matched":
        full_payload = f"key={example.private_key}"
        payload = full_payload.encode("utf-8")[:syndrome_bytes].decode("utf-8", errors="ignore")
        prediction = _structured_text_decode(example, payload)
        payload_bytes = len(payload.encode("utf-8"))
    elif condition == "structured_text_full":
        payload = f"key={example.private_key}"
        prediction = _structured_text_decode(example, payload)
        payload_bytes = len(payload.encode("utf-8"))
    else:
        raise ValueError(f"unknown condition {condition!r}")
    latency_ms = (time.perf_counter() - start) * 1000.0
    return prediction, payload_bytes, latency_ms


def _accuracy(rows: list[dict[str, Any]], condition: str) -> float:
    if not rows:
        return 0.0
    correct = sum(int(row["conditions"][condition]["correct"]) for row in rows)
    return correct / len(rows)


def _summarize(rows: list[dict[str, Any]], *, syndrome_bytes: int) -> dict[str, Any]:
    conditions = list(rows[0]["conditions"]) if rows else []
    metrics: dict[str, Any] = {}
    for condition in conditions:
        payloads = [row["conditions"][condition]["payload_bytes"] for row in rows]
        latencies = [row["conditions"][condition]["latency_ms"] for row in rows]
        metrics[condition] = {
            "accuracy": _accuracy(rows, condition),
            "correct": sum(int(row["conditions"][condition]["correct"]) for row in rows),
            "n": len(rows),
            "mean_payload_bytes": statistics.fmean(payloads) if payloads else 0.0,
            "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        }
    no_source_conditions = ["target_only", "target_wrapper"]
    control_conditions = [
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_only_sidecar",
    ]
    best_no_source = max(metrics[name]["accuracy"] for name in no_source_conditions)
    best_control = max(metrics[name]["accuracy"] for name in control_conditions)
    matched = metrics["matched_syndrome"]["accuracy"]
    matched_text = metrics["structured_text_matched"]["accuracy"]
    pass_gate = (
        matched - best_no_source >= 0.15
        and best_control <= best_no_source + 0.02
        and matched_text <= best_no_source + 0.02
    )
    return {
        "n": len(rows),
        "syndrome_bytes": syndrome_bytes,
        "metrics": metrics,
        "best_no_source_accuracy": best_no_source,
        "best_source_destroying_control_accuracy": best_control,
        "matched_minus_best_no_source": matched - best_no_source,
        "matched_minus_best_control": matched - best_control,
        "matched_minus_matched_text_relay": matched - matched_text,
        "pass_gate": pass_gate,
        "pass_rule": (
            "matched_syndrome beats best no-source by >=0.15, source-destroying "
            "controls stay within +0.02 of no-source, and matched-byte structured "
            "text relay stays within +0.02 of no-source"
        ),
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Evidence Packet Gate",
        "",
        f"- examples: `{summary['n']}`",
        f"- syndrome bytes/example: `{summary['syndrome_bytes']}`",
        f"- best no-source accuracy: `{summary['best_no_source_accuracy']:.3f}`",
        f"- best source-destroying control accuracy: `{summary['best_source_destroying_control_accuracy']:.3f}`",
        f"- matched minus best no-source: `{summary['matched_minus_best_no_source']:.3f}`",
        f"- matched minus best control: `{summary['matched_minus_best_control']:.3f}`",
        f"- matched minus matched-byte text relay: `{summary['matched_minus_matched_text_relay']:.3f}`",
        f"- pass gate: `{summary['pass_gate']}`",
        "",
        "| Condition | Correct | Accuracy | Mean bytes | p50 latency ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["metrics"].items():
        lines.append(
            "| "
            f"{condition} | {metrics['correct']}/{metrics['n']} | "
            f"{metrics['accuracy']:.3f} | {metrics['mean_payload_bytes']:.2f} | "
            f"{metrics['p50_latency_ms']:.4f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Source-Private Evidence Packet Gate Manifest",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest["artifacts"])
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- examples: `{summary['n']}`",
            f"- syndrome bytes/example: `{summary['syndrome_bytes']}`",
            f"- matched syndrome accuracy: `{summary['metrics']['matched_syndrome']['accuracy']:.3f}`",
            f"- best no-source accuracy: `{summary['best_no_source_accuracy']:.3f}`",
            f"- best source-destroying control accuracy: `{summary['best_source_destroying_control_accuracy']:.3f}`",
            f"- matched minus best no-source: `{summary['matched_minus_best_no_source']:.3f}`",
            f"- pass gate: `{summary['pass_gate']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(*, examples: int, candidates: int, seed: int, syndrome_bytes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark = _make_examples(examples=examples, candidates=candidates, seed=seed)
    rng = random.Random(seed + 97)
    conditions = [
        "target_only",
        "target_wrapper",
        "matched_syndrome",
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_only_sidecar",
        "structured_text_matched",
        "structured_text_full",
    ]
    rows: list[dict[str, Any]] = []
    salt = f"source-private-evidence-v1:{seed}"
    for index, example in enumerate(benchmark):
        condition_rows: dict[str, Any] = {}
        for condition in conditions:
            prediction, payload_bytes, latency_ms = _condition_prediction(
                condition=condition,
                example=example,
                examples=benchmark,
                index=index,
                salt=salt,
                syndrome_bytes=syndrome_bytes,
                rng=rng,
            )
            condition_rows[condition] = {
                "prediction": prediction,
                "correct": prediction == example.answer_name,
                "payload_bytes": payload_bytes,
                "latency_ms": latency_ms,
            }
        rows.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "answer_name": example.answer_name,
                "candidate_pool_contains_gold": True,
                "candidate_count": len(example.candidates),
                "private_evidence_bytes": len(example.private_evidence.encode("utf-8")),
                "conditions": condition_rows,
            }
        )
    return rows, _summarize(rows, syndrome_bytes=syndrome_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--syndrome-bytes", type=int, default=2)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_gate(
        examples=args.examples,
        candidates=args.candidates,
        seed=args.seed,
        syndrome_bytes=args.syndrome_bytes,
    )
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_evidence_packet_gate.py",
                f"--examples {args.examples}",
                f"--candidates {args.candidates}",
                f"--seed {args.seed}",
                f"--syndrome-bytes {args.syndrome_bytes}",
                f"--output-dir {args.output_dir}",
            ]
        ),
        "artifacts": ["predictions.jsonl", "summary.json", "summary.md", "manifest.json"],
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
