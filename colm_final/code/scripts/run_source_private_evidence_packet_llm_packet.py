from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

KEY_RE = re.compile(r"([0-9a-f]{8}(?:-[0-9a-f]{8}){5})")
DISPLAY_RE = re.compile(r"The matching candidate is ([A-Za-z0-9_]+)")
HEX_RE = re.compile(r"[0-9a-fA-F]+")


@dataclass(frozen=True)
class LoadedExample:
    example_id: str
    answer_label: str
    private_evidence: str
    source_prompt: str
    candidates: tuple[dict[str, Any], ...]


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_examples(path: pathlib.Path, *, limit: int | None) -> list[LoadedExample]:
    rows = _load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    examples: list[LoadedExample] = []
    for row in rows:
        examples.append(
            LoadedExample(
                example_id=row["example_id"],
                answer_label=row["answer_label"],
                private_evidence=row["private_evidence"],
                source_prompt=row["source_prompt"],
                candidates=tuple(row["candidates"]),
            )
        )
    return examples


def _packet_hex(value: str, *, salt: str, budget_bytes: int) -> str:
    return hashlib.blake2s(f"{salt}:{value}".encode("utf-8"), digest_size=budget_bytes).hexdigest()


def _key_from_evidence(text: str) -> str | None:
    match = KEY_RE.search(text)
    return match.group(1) if match else None


def _display_from_evidence(text: str) -> str | None:
    match = DISPLAY_RE.search(text)
    return match.group(1) if match else None


def _prior_prediction(example: LoadedExample) -> str:
    return max(example.candidates, key=lambda row: float(row["prior_score"]))["label"]


def _candidate_by_display(example: LoadedExample, display_name: str | None) -> str | None:
    if not display_name:
        return None
    for candidate in example.candidates:
        if candidate["display_name"] == display_name:
            return candidate["label"]
    return None


def _extract_hex_packet(text: str, *, budget_bytes: int) -> str:
    needed = budget_bytes * 2
    for match in HEX_RE.finditer(text):
        token = match.group(0).lower()
        if len(token) >= needed:
            return token[:needed]
    return ""


def _prompt_for_digest(example: LoadedExample, *, packet_salt: str, budget_bytes: int) -> str:
    needed = budget_bytes * 2
    return (
        "You are the source agent in a communication experiment.\n"
        "Read the private evidence. Do not output the candidate label, record name, "
        "or witness key.\n"
        f"Packet rule: output exactly the first {needed} lowercase hexadecimal characters "
        "of this digest: BLAKE2s(packet_salt + ':' + witness_key), where "
        f"packet_salt is {packet_salt!r} and digest_size is {budget_bytes} bytes.\n"
        "Return only the hex packet.\n\n"
        f"{example.source_prompt}"
    )


def _format_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool | None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    return prompt


def _load_model(model_name: str, *, device: str, dtype: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, local_files_only=True)
    model.to(device).eval()
    return tokenizer, model


def _generate_packets(
    examples: list[LoadedExample],
    *,
    model_name: str,
    device: str,
    dtype: str,
    budget_bytes: int,
    seed: int,
    max_new_tokens: int,
    enable_thinking: bool | None,
) -> list[dict[str, Any]]:
    import torch

    tokenizer, model = _load_model(model_name, device=device, dtype=dtype)
    packet_salt = f"source-private-packet:28:budget:{budget_bytes}"
    rows: list[dict[str, Any]] = []
    torch.manual_seed(seed)
    for example in examples:
        prompt = _prompt_for_digest(example, packet_salt=packet_salt, budget_bytes=budget_bytes)
        text_prompt = _format_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        packet = _extract_hex_packet(generated, budget_bytes=budget_bytes)
        rows.append(
            {
                "example_id": example.example_id,
                "generated_text": generated,
                "packet": packet,
                "packet_bytes": len(bytes.fromhex(packet)) if len(packet) % 2 == 0 and packet else 0,
                "packet_tokens": len(new_tokens),
                "latency_ms": latency_ms,
            }
        )
    return rows


def _deterministic_nonself_index(index: int, n: int) -> int:
    return (index * 17 + 11) % n if (index * 17 + 11) % n != index else (index + 1) % n


def _decode_digest_packet(example: LoadedExample, packet: str, *, packet_salt: str, budget_bytes: int) -> str:
    key = _key_from_evidence(example.private_evidence)
    if not key:
        return _prior_prediction(example)
    if packet == _packet_hex(key, salt=packet_salt, budget_bytes=budget_bytes):
        return example.answer_label
    return _prior_prediction(example)


def _evaluate(
    examples: list[LoadedExample],
    generated_packets: list[dict[str, Any]],
    *,
    budget_bytes: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_salt = f"source-private-packet:28:budget:{budget_bytes}"
    by_id = {row["example_id"]: row for row in generated_packets}
    rng = random.Random(seed + budget_bytes * 1009)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        matched_packet = by_id[example.example_id]["packet"]
        shuffled_packet = by_id[examples[_deterministic_nonself_index(index, len(examples))].example_id]["packet"]
        random_packet = rng.randbytes(budget_bytes).hex()
        source_final_display = _display_from_evidence(example.private_evidence)
        source_final_label = _candidate_by_display(example, source_final_display) or _prior_prediction(example)
        answer_only = example.answer_label.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")
        conditions: dict[str, dict[str, Any]] = {}
        condition_predictions = {
            "target_only": _prior_prediction(example),
            "matched_model_packet": _decode_digest_packet(
                example, matched_packet, packet_salt=packet_salt, budget_bytes=budget_bytes
            ),
            "zero_source": _prior_prediction(example),
            "shuffled_model_packet": _decode_digest_packet(
                example, shuffled_packet, packet_salt=packet_salt, budget_bytes=budget_bytes
            ),
            "random_same_byte": _decode_digest_packet(
                example, random_packet, packet_salt=packet_salt, budget_bytes=budget_bytes
            ),
            "answer_only": answer_only if answer_only == example.answer_label else _prior_prediction(example),
            "answer_masked": _prior_prediction(example),
            "source_final_only": source_final_label,
        }
        for condition, prediction in condition_predictions.items():
            if condition == "matched_model_packet":
                payload_bytes = by_id[example.example_id]["packet_bytes"]
                payload_tokens = by_id[example.example_id]["packet_tokens"]
                payload = matched_packet
                latency_ms = by_id[example.example_id]["latency_ms"]
            elif condition == "shuffled_model_packet":
                payload_bytes = len(bytes.fromhex(shuffled_packet)) if shuffled_packet else 0
                payload_tokens = 1 if shuffled_packet else 0
                payload = shuffled_packet
                latency_ms = 0.0
            elif condition == "random_same_byte":
                payload_bytes = budget_bytes
                payload_tokens = 1
                payload = random_packet
                latency_ms = 0.0
            elif condition == "source_final_only":
                payload = source_final_display or ""
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1
                latency_ms = 0.0
            elif condition == "answer_only":
                payload = answer_only
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1 if payload else 0
                latency_ms = 0.0
            else:
                payload = ""
                payload_bytes = 0
                payload_tokens = 0
                latency_ms = 0.0
            conditions[condition] = {
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "payload": payload,
                "payload_bytes": payload_bytes,
                "payload_tokens": payload_tokens,
                "latency_ms": latency_ms,
            }
        rows.append(
            {
                "example_id": example.example_id,
                "answer_label": example.answer_label,
                "conditions": conditions,
                "generated_text": by_id[example.example_id]["generated_text"],
            }
        )
    summary = _summarize(rows, budget_bytes=budget_bytes, model_packet_rows=generated_packets)
    return rows, summary


def _summarize(
    rows: list[dict[str, Any]],
    *,
    budget_bytes: int,
    model_packet_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_ids = [row["example_id"] for row in rows]
    conditions = list(rows[0]["conditions"]) if rows else []
    metrics: dict[str, Any] = {}
    for condition in conditions:
        correct_ids = [row["example_id"] for row in rows if row["conditions"][condition]["correct"]]
        payloads = [row["conditions"][condition]["payload_bytes"] for row in rows]
        latencies = [row["conditions"][condition]["latency_ms"] for row in rows]
        metrics[condition] = {
            "correct": len(correct_ids),
            "accuracy": len(correct_ids) / len(rows),
            "correct_ids": correct_ids,
            "mean_payload_bytes": statistics.fmean(payloads),
            "max_payload_bytes": max(payloads),
            "p50_latency_ms": statistics.median(latencies),
        }
    best_no_source = metrics["target_only"]["accuracy"]
    control_names = ["zero_source", "shuffled_model_packet", "random_same_byte", "answer_only", "answer_masked"]
    best_control = max(metrics[name]["accuracy"] for name in control_names)
    matched = metrics["matched_model_packet"]["accuracy"]
    source_final = metrics["source_final_only"]["accuracy"]
    pass_gate = (
        matched - best_no_source >= 0.15
        and best_control <= best_no_source + 0.02
        and source_final <= best_no_source + 0.02
    )
    packets_nonempty = sum(int(bool(row["packet"])) for row in model_packet_rows)
    return {
        "budget_bytes": budget_bytes,
        "n": len(rows),
        "exact_id_count": len(set(exact_ids)),
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "packet_nonempty_rate": packets_nonempty / len(model_packet_rows) if model_packet_rows else 0.0,
        "best_no_source_accuracy": best_no_source,
        "best_source_destroying_control_accuracy": best_control,
        "matched_minus_best_no_source": matched - best_no_source,
        "matched_minus_best_control": matched - best_control,
        "source_final_minus_best_no_source": source_final - best_no_source,
        "pass_gate": pass_gate,
        "pass_rule": (
            "matched model-produced digest packet must beat target-only by >=0.15, "
            "source-destroying controls must remain within +0.02, and source-final-only "
            "must not explain the gain."
        ),
        "metrics": metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Evidence Packet Model-Packet Gate",
        "",
        f"- examples: `{summary['n']}`",
        f"- budget bytes: `{summary['budget_bytes']}`",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- packet nonempty rate: `{summary['packet_nonempty_rate']:.3f}`",
        f"- matched minus best no-source: `{summary['matched_minus_best_no_source']:.3f}`",
        f"- matched minus best control: `{summary['matched_minus_best_control']:.3f}`",
        f"- source-final minus best no-source: `{summary['source_final_minus_best_no_source']:.3f}`",
        "",
        "| Condition | Correct | Accuracy | Mean bytes | Max bytes | p50 latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["metrics"].items():
        lines.append(
            "| "
            f"{condition} | {metrics['correct']}/{summary['n']} | "
            f"{metrics['accuracy']:.3f} | {metrics['mean_payload_bytes']:.2f} | "
            f"{metrics['max_payload_bytes']} | {metrics['p50_latency_ms']:.2f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Source-Private Evidence Packet Model-Packet Manifest",
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
        f"- examples: `{summary['n']}`",
        f"- budget bytes: `{summary['budget_bytes']}`",
        f"- matched model packet accuracy: `{summary['metrics']['matched_model_packet']['accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['metrics']['target_only']['accuracy']:.3f}`",
        f"- source-final-only accuracy: `{summary['metrics']['source_final_only']['accuracy']:.3f}`",
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
    parser.add_argument("--benchmark-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--budget-bytes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    benchmark_path = args.benchmark_jsonl if args.benchmark_jsonl.is_absolute() else ROOT / args.benchmark_jsonl
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_examples(benchmark_path, limit=args.limit)
    packets = _generate_packets(
        examples,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        budget_bytes=args.budget_bytes,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
    )
    rows, summary = _evaluate(examples, packets, budget_bytes=args.budget_bytes, seed=args.seed)
    _write_jsonl(output_dir / "model_packets.jsonl", packets)
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    artifacts = ["model_packets.jsonl", "predictions.jsonl", "summary.json", "summary.md", "manifest.json", "manifest.md"]
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_evidence_packet_llm_packet.py",
                f"--benchmark-jsonl {args.benchmark_jsonl}",
                f"--output-dir {args.output_dir}",
                f"--model {args.model}",
                f"--device {args.device}",
                f"--dtype {args.dtype}",
                f"--limit {args.limit}",
                f"--budget-bytes {args.budget_bytes}",
                f"--seed {args.seed}",
                f"--max-new-tokens {args.max_new_tokens}",
            ]
        ),
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
