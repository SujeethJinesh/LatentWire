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
EXPLICIT_SIG_RE = re.compile(r"TRACE_SIG\s*=\s*([A-Z2-9]{2})")
STANDALONE_SIG_RE = re.compile(r"(?<![A-Z2-9])([A-Z2-9]{2})(?![A-Z2-9])")


@dataclass(frozen=True)
class LoadedExample:
    example_id: str
    answer_label: str
    answer_signature: str
    private_test_log: str
    source_prompt: str
    candidates: tuple[dict[str, Any], ...]


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_examples(path: pathlib.Path, *, limit: int | None) -> list[LoadedExample]:
    rows = _load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [
        LoadedExample(
            example_id=row["example_id"],
            answer_label=row["answer_label"],
            answer_signature=row["answer_signature"],
            private_test_log=row["private_test_log"],
            source_prompt=row["source_prompt"],
            candidates=tuple(row["candidates"]),
        )
        for row in rows
    ]


def _prior_prediction(example: LoadedExample) -> str:
    return max(example.candidates, key=lambda row: float(row["prior_score"]))["label"]


def _extract_signature(text: str) -> str:
    stripped = text.strip()
    explicit = EXPLICIT_SIG_RE.search(stripped)
    if explicit:
        return explicit.group(1)
    if "TRACE_SIG" in stripped:
        return ""
    exact = re.fullmatch(r"[A-Z2-9]{2}", stripped)
    if exact:
        return exact.group(0)
    for match in STANDALONE_SIG_RE.finditer(stripped):
        return match.group(1)
    return ""


def _decode_signature_packet(example: LoadedExample, packet: str) -> str:
    if not packet:
        return _prior_prediction(example)
    matches = [candidate for candidate in example.candidates if candidate["failure_signature"] == packet]
    if not matches:
        return _prior_prediction(example)
    return max(matches, key=lambda row: float(row["prior_score"]))["label"]


def _deterministic_nonself_index(index: int, n: int) -> int:
    return (index * 17 + 11) % n if (index * 17 + 11) % n != index else (index + 1) % n


def _prompt_for_signature(example: LoadedExample) -> str:
    trace_line = next(
        (line for line in example.private_test_log.splitlines() if "TRACE_SIG=" in line),
        "",
    )
    return (
        "You are the source agent in a private tool-log handoff experiment.\n"
        "Copy exactly the two characters after the equals sign in the private TRACE_SIG line.\n"
        "Do not output the candidate label, patch name, explanation, punctuation, or code fence.\n"
        "Return only the two-character packet.\n\n"
        f"Private test log:\n{example.private_test_log}\n\n"
        f"Private TRACE_SIG line copied from the log: {trace_line}\n"
        "Packet:"
    )


def _format_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool | None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
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
    seed: int,
    max_new_tokens: int,
    enable_thinking: bool | None,
) -> list[dict[str, Any]]:
    import torch

    tokenizer, model = _load_model(model_name, device=device, dtype=dtype)
    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = _prompt_for_signature(example)
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
        packet = _extract_signature(generated)
        rows.append(
            {
                "example_id": example.example_id,
                "generated_text": generated,
                "packet": packet,
                "packet_bytes": len(packet.encode("utf-8")),
                "packet_tokens": len(new_tokens),
                "latency_ms": latency_ms,
                "valid_packet": bool(re.fullmatch(r"[A-Z2-9]{2}", packet)),
            }
        )
    return rows


def _evaluate(
    examples: list[LoadedExample],
    generated_packets: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["example_id"]: row for row in generated_packets}
    rng = random.Random(seed + 20260428)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        matched_packet = by_id[example.example_id]["packet"]
        shuffled_packet = by_id[examples[_deterministic_nonself_index(index, len(examples))].example_id]["packet"]
        random_packet = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(2))
        prior_label = _prior_prediction(example)
        prior_signature = next(candidate["failure_signature"] for candidate in example.candidates if candidate["label"] == prior_label)
        answer_only = example.answer_label.encode("utf-8")[:2].decode("utf-8", errors="ignore")
        condition_predictions = {
            "target_only": prior_label,
            "matched_model_packet": _decode_signature_packet(example, matched_packet),
            "zero_source": prior_label,
            "shuffled_model_packet": _decode_signature_packet(example, shuffled_packet),
            "random_same_byte": _decode_signature_packet(example, random_packet),
            "answer_only": answer_only if answer_only == example.answer_label else prior_label,
            "answer_masked": prior_label,
            "target_derived_sidecar": _decode_signature_packet(example, prior_signature),
            "full_signature_oracle": _decode_signature_packet(example, example.answer_signature),
        }
        conditions: dict[str, dict[str, Any]] = {}
        for condition, prediction in condition_predictions.items():
            if condition == "matched_model_packet":
                packet_row = by_id[example.example_id]
                payload = matched_packet
                payload_bytes = packet_row["packet_bytes"]
                payload_tokens = packet_row["packet_tokens"]
                latency_ms = packet_row["latency_ms"]
            elif condition == "shuffled_model_packet":
                payload = shuffled_packet
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1 if payload else 0
                latency_ms = 0.0
            elif condition == "random_same_byte":
                payload = random_packet
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1
                latency_ms = 0.0
            elif condition == "target_derived_sidecar":
                payload = prior_signature
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1
                latency_ms = 0.0
            elif condition == "full_signature_oracle":
                payload = example.answer_signature
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
                "answer_signature": example.answer_signature,
                "generated_text": by_id[example.example_id]["generated_text"],
                "conditions": conditions,
            }
        )
    return rows, _summarize(rows, model_packet_rows=generated_packets)


def _summarize(rows: list[dict[str, Any]], *, model_packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_ids = [row["example_id"] for row in rows]
    conditions = list(rows[0]["conditions"]) if rows else []
    metrics: dict[str, Any] = {}
    for condition in conditions:
        correct_ids = [row["example_id"] for row in rows if row["conditions"][condition]["correct"]]
        payloads = [row["conditions"][condition]["payload_bytes"] for row in rows]
        tokens = [row["conditions"][condition]["payload_tokens"] for row in rows]
        latencies = [row["conditions"][condition]["latency_ms"] for row in rows]
        metrics[condition] = {
            "correct": len(correct_ids),
            "accuracy": len(correct_ids) / len(rows),
            "correct_ids": correct_ids,
            "mean_payload_bytes": statistics.fmean(payloads),
            "max_payload_bytes": max(payloads),
            "mean_payload_tokens": statistics.fmean(tokens),
            "p50_latency_ms": statistics.median(latencies),
        }
    best_no_source = metrics["target_only"]["accuracy"]
    controls = [
        "zero_source",
        "shuffled_model_packet",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_derived_sidecar",
    ]
    best_control = max(metrics[name]["accuracy"] for name in controls)
    matched = metrics["matched_model_packet"]["accuracy"]
    valid_packets = sum(int(row["valid_packet"]) for row in model_packet_rows)
    pass_gate = matched - best_no_source >= 0.15 and best_control <= best_no_source + 0.02
    return {
        "n": len(rows),
        "exact_id_count": len(set(exact_ids)),
        "exact_id_sha256": hashlib.sha256("\n".join(exact_ids).encode("utf-8")).hexdigest(),
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "packet_valid_rate": valid_packets / len(model_packet_rows) if model_packet_rows else 0.0,
        "best_no_source_accuracy": best_no_source,
        "best_source_destroying_control_accuracy": best_control,
        "matched_minus_best_no_source": matched - best_no_source,
        "matched_minus_best_control": matched - best_control,
        "pass_gate": pass_gate,
        "pass_rule": (
            "matched model-extracted test-log packet must beat target-only by >=0.15 "
            "and all source-destroying controls must remain within +0.02 of no-source."
        ),
        "metrics": metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Test-Log Model-Packet Gate",
        "",
        f"- examples: `{summary['n']}`",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- packet valid rate: `{summary['packet_valid_rate']:.3f}`",
        f"- matched minus best no-source: `{summary['matched_minus_best_no_source']:.3f}`",
        f"- matched minus best control: `{summary['matched_minus_best_control']:.3f}`",
        "",
        "| Condition | Correct | Accuracy | Mean bytes | Max bytes | Mean tokens | p50 latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["metrics"].items():
        lines.append(
            "| "
            f"{condition} | {metrics['correct']}/{summary['n']} | "
            f"{metrics['accuracy']:.3f} | {metrics['mean_payload_bytes']:.2f} | "
            f"{metrics['max_payload_bytes']} | {metrics['mean_payload_tokens']:.2f} | "
            f"{metrics['p50_latency_ms']:.2f} |"
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
        "# Source-Private Test-Log Model-Packet Manifest",
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
        f"- packet valid rate: `{summary['packet_valid_rate']:.3f}`",
        f"- matched model packet accuracy: `{summary['metrics']['matched_model_packet']['accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['metrics']['target_only']['accuracy']:.3f}`",
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
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--max-new-tokens", type=int, default=8)
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
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
    )
    rows, summary = _evaluate(examples, packets, seed=args.seed)
    _write_jsonl(output_dir / "model_packets.jsonl", packets)
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    artifacts = ["model_packets.jsonl", "predictions.jsonl", "summary.json", "summary.md", "manifest.json", "manifest.md"]
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_testlog_packet_llm_packet.py",
                f"--benchmark-jsonl {args.benchmark_jsonl}",
                f"--output-dir {args.output_dir}",
                f"--model {args.model}",
                f"--device {args.device}",
                f"--dtype {args.dtype}",
                f"--limit {args.limit}",
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
