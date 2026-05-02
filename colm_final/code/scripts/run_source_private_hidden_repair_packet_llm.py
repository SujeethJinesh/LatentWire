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
EXPLICIT_DIAG_RE = re.compile(r"REPAIR_DIAG\s*=\s*([A-Z][0-9])")
STANDALONE_DIAG_RE = re.compile(r"(?<![A-Z0-9])([A-Z][0-9])(?![A-Z0-9])")
PROMPT_MODES = ("copied_helper", "log_only", "trace_no_hint", "raw_log_no_trace")


@dataclass(frozen=True)
class LoadedExample:
    example_id: str
    answer_label: str
    diagnostic_code: str
    private_test_log: str
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
            diagnostic_code=row["diagnostic_code"],
            private_test_log=row["private_test_log"],
            candidates=tuple(row["candidates"]),
        )
        for row in rows
    ]


def _prior_prediction(example: LoadedExample) -> str:
    return max(example.candidates, key=lambda row: float(row["prior_score"]))["label"]


def _extract_diag(text: str) -> str:
    stripped = text.strip()
    explicit = EXPLICIT_DIAG_RE.search(stripped)
    if explicit:
        return explicit.group(1)
    if "REPAIR_DIAG" in stripped:
        return ""
    exact = re.fullmatch(r"[A-Z][0-9]", stripped)
    if exact:
        return exact.group(0)
    for match in STANDALONE_DIAG_RE.finditer(stripped):
        return match.group(1)
    return ""


def _decode_packet(example: LoadedExample, packet: str) -> str:
    if not packet:
        return _prior_prediction(example)
    matches = [candidate for candidate in example.candidates if candidate["handles_diagnostic"] == packet]
    if not matches:
        return _prior_prediction(example)
    return max(matches, key=lambda row: float(row["prior_score"]))["label"]


def _deterministic_nonself_index(index: int, n: int) -> int:
    candidate = (index * 17 + 11) % n
    return candidate if candidate != index else (index + 1) % n


def _private_log_for_prompt(example: LoadedExample, *, prompt_mode: str) -> str:
    if prompt_mode == "raw_log_no_trace":
        banned = ("private_tool_trace:", "hint:")
    elif prompt_mode == "trace_no_hint":
        banned = ("hint:",)
    else:
        banned = ()
    lines = [
        line
        for line in example.private_test_log.splitlines()
        if not any(line.startswith(prefix) for prefix in banned)
    ]
    return "\n".join(lines)


def _prompt_for_diag(example: LoadedExample, *, prompt_mode: str) -> str:
    private_log = _private_log_for_prompt(example, prompt_mode=prompt_mode)
    diag_line = next((line for line in private_log.splitlines() if "REPAIR_DIAG=" in line), "")
    if prompt_mode == "raw_log_no_trace":
        instruction = (
            "Infer the most compact repair packet from the private hidden-test log.\n"
            "Do not output the candidate label, patch name, explanation, punctuation, or code fence.\n"
            "Return only a two-character repair packet if one is recoverable."
        )
    else:
        instruction = (
            "Copy exactly the two-character code after the equals sign in the private REPAIR_DIAG line.\n"
            "Do not output the candidate label, patch name, explanation, punctuation, or code fence.\n"
            "Return only the two-character repair packet."
        )
    return (
        "You are the source agent in a private code-repair handoff experiment.\n"
        f"{instruction}\n\n"
        f"Private hidden-test log:\n{private_log}\n\n"
        + (f"Private REPAIR_DIAG line copied from the log: {diag_line}\n" if prompt_mode == "copied_helper" else "")
        + "Packet:"
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

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
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
    prompt_mode: str,
) -> list[dict[str, Any]]:
    import torch

    tokenizer, model = _load_model(model_name, device=device, dtype=dtype)
    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = _prompt_for_diag(example, prompt_mode=prompt_mode)
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
        packet = _extract_diag(generated)
        rows.append(
            {
                "example_id": example.example_id,
                "generated_text": generated,
                "packet": packet,
                "packet_bytes": len(packet.encode("utf-8")),
                "packet_tokens": len(new_tokens),
                "latency_ms": latency_ms,
                "valid_packet": bool(re.fullmatch(r"[A-Z][0-9]", packet)),
                "prompt_mode": prompt_mode,
            }
        )
    return rows


def _evaluate(examples: list[LoadedExample], packets: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["example_id"]: row for row in packets}
    rng = random.Random(seed + 20260428)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        shuffled_source = examples[_deterministic_nonself_index(index, len(examples))]
        matched_packet = by_id[example.example_id]["packet"]
        shuffled_packet = by_id[shuffled_source.example_id]["packet"]
        random_packet = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWYZ')}{rng.randrange(0, 10)}"
        prior_label = _prior_prediction(example)
        prior_diag = next(candidate["handles_diagnostic"] for candidate in example.candidates if candidate["label"] == prior_label)
        answer_only = example.answer_label.encode("utf-8")[:2].decode("utf-8", errors="ignore")
        predictions = {
            "target_only": prior_label,
            "matched_model_packet": _decode_packet(example, matched_packet),
            "zero_source": prior_label,
            "shuffled_model_packet": _decode_packet(example, shuffled_packet),
            "random_same_byte": _decode_packet(example, random_packet),
            "answer_only": answer_only if answer_only == example.answer_label else prior_label,
            "answer_masked": prior_label,
            "target_derived_sidecar": _decode_packet(example, prior_diag),
            "full_diag_oracle": _decode_packet(example, example.diagnostic_code),
        }
        conditions: dict[str, dict[str, Any]] = {}
        for condition, prediction in predictions.items():
            if condition == "matched_model_packet":
                packet_row = by_id[example.example_id]
                payload = matched_packet
                payload_bytes = packet_row["packet_bytes"]
                payload_tokens = packet_row["packet_tokens"]
                latency_ms = packet_row["latency_ms"]
                source_example_id = example.example_id
            elif condition == "shuffled_model_packet":
                payload = shuffled_packet
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1 if payload else 0
                latency_ms = 0.0
                source_example_id = shuffled_source.example_id
            elif condition == "random_same_byte":
                payload = random_packet
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1
                latency_ms = 0.0
                source_example_id = example.example_id
            elif condition == "target_derived_sidecar":
                payload = prior_diag
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1
                latency_ms = 0.0
                source_example_id = example.example_id
            elif condition == "full_diag_oracle":
                payload = example.diagnostic_code
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1
                latency_ms = 0.0
                source_example_id = example.example_id
            elif condition == "answer_only":
                payload = answer_only
                payload_bytes = len(payload.encode("utf-8"))
                payload_tokens = 1 if payload else 0
                latency_ms = 0.0
                source_example_id = example.example_id
            else:
                payload = ""
                payload_bytes = 0
                payload_tokens = 0
                latency_ms = 0.0
                source_example_id = example.example_id
            conditions[condition] = {
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "payload": payload,
                "payload_bytes": payload_bytes,
                "payload_tokens": payload_tokens,
                "latency_ms": latency_ms,
                "source_example_id": source_example_id,
            }
        rows.append(
            {
                "example_id": example.example_id,
                "answer_label": example.answer_label,
                "diagnostic_code": example.diagnostic_code,
                "generated_text": by_id[example.example_id]["generated_text"],
                "conditions": conditions,
            }
        )
    return rows, _summarize(rows, model_packet_rows=packets)


def _summarize(rows: list[dict[str, Any]], *, model_packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_ids = [row["example_id"] for row in rows]
    conditions = list(rows[0]["conditions"]) if rows else []
    metrics: dict[str, Any] = {}
    for condition in conditions:
        correct_ids = [row["example_id"] for row in rows if row["conditions"][condition]["correct"]]
        payloads = [row["conditions"][condition]["payload_bytes"] for row in rows]
        latencies = [row["conditions"][condition]["latency_ms"] for row in rows]
        tokens = [row["conditions"][condition]["payload_tokens"] for row in rows]
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
    controls = ["zero_source", "shuffled_model_packet", "random_same_byte", "answer_only", "answer_masked", "target_derived_sidecar"]
    best_control = max(metrics[name]["accuracy"] for name in controls)
    matched = metrics["matched_model_packet"]["accuracy"]
    valid_packets = sum(int(row["valid_packet"]) for row in model_packet_rows)
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
        "pass_gate": matched - best_no_source >= 0.15 and best_control <= best_no_source + 0.02,
        "pass_rule": "matched model-extracted hidden-repair packet must beat no-source by >=0.15 and controls stay within +0.02.",
        "metrics": metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Hidden-Repair Model-Packet Gate",
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
        "# Source-Private Hidden-Repair Model-Packet Manifest",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="copied_helper")
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
        prompt_mode=args.prompt_mode,
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
                "scripts/run_source_private_hidden_repair_packet_llm.py",
                f"--benchmark-jsonl {args.benchmark_jsonl}",
                f"--output-dir {args.output_dir}",
                f"--model {args.model}",
                f"--device {args.device}",
                f"--dtype {args.dtype}",
                f"--limit {args.limit}",
                f"--seed {args.seed}",
                f"--max-new-tokens {args.max_new_tokens}",
                f"--prompt-mode {args.prompt_mode}",
                "--no-enable-thinking" if args.enable_thinking is False else "--enable-thinking",
            ]
        ),
        "args": vars(args) | {"benchmark_jsonl": str(args.benchmark_jsonl), "output_dir": str(args.output_dir), "do_sample": False},
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "benchmark_sha256": _sha256_file(benchmark_path),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
