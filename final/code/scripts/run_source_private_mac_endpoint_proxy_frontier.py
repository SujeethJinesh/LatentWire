from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


CONDITIONS = (
    "target_only",
    "matched_packet",
    "matched_byte_text_2",
    "random_same_byte_packet",
    "deranged_candidate_diag_table",
    "query_aware_diag_span",
    "structured_json_diag",
    "structured_free_text_diag",
    "full_hidden_log",
)

PROMPT_STYLES = ("canonical", "terse", "audit", "label_strict")


@dataclass(frozen=True)
class LoadedExample:
    example_id: str
    answer_label: str
    diagnostic_code: str
    private_test_log: str
    candidates: tuple[dict[str, Any], ...]


def _load_examples(path: pathlib.Path, *, limit: int | None) -> list[LoadedExample]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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


def _random_same_byte_packet(example: LoadedExample) -> str:
    diagnostics = [str(candidate["handles_diagnostic"]) for candidate in example.candidates]
    alternatives = [diagnostic for diagnostic in diagnostics if diagnostic != example.diagnostic_code]
    if alternatives:
        return sorted(alternatives)[0]
    return "Z9" if example.diagnostic_code != "Z9" else "Z8"


def _payload_for_condition(example: LoadedExample, *, condition: str) -> str:
    if condition == "target_only":
        return ""
    if condition == "matched_packet":
        return example.diagnostic_code
    if condition == "random_same_byte_packet":
        return _random_same_byte_packet(example)
    if condition == "deranged_candidate_diag_table":
        return example.diagnostic_code
    if condition == "matched_byte_text_2":
        return f"REPAIR_DIAG={example.diagnostic_code}".encode("utf-8")[:2].decode("utf-8", errors="ignore")
    if condition == "query_aware_diag_span":
        return f"REPAIR_DIAG={example.diagnostic_code}"
    if condition == "structured_json_diag":
        return json.dumps({"repair_diag": example.diagnostic_code}, sort_keys=True)
    if condition == "structured_free_text_diag":
        return f"repair diag is {example.diagnostic_code}"
    if condition == "full_hidden_log":
        return example.private_test_log
    raise ValueError(f"unknown condition {condition!r}")


def _candidate_table_for_condition(example: LoadedExample, *, condition: str) -> tuple[dict[str, Any], ...]:
    candidates = [dict(candidate) for candidate in example.candidates]
    if condition != "deranged_candidate_diag_table" or len(candidates) < 2:
        return tuple(candidates)
    diagnostics = [candidate["handles_diagnostic"] for candidate in candidates]
    shifted = diagnostics[1:] + diagnostics[:1]
    for candidate, diagnostic in zip(candidates, shifted):
        candidate["handles_diagnostic"] = diagnostic
    return tuple(candidates)


def _prompt(example: LoadedExample, *, payload: str, prompt_style: str, condition: str = "matched_packet") -> str:
    prompt_candidates = _candidate_table_for_condition(example, condition=condition)
    candidate_rows = "\n".join(
        f"- {candidate['label']}: handles_repair_diag={candidate['handles_diagnostic']}"
        for candidate in prompt_candidates
    )
    if prompt_style == "canonical":
        return (
            "You are the target-side decoder in a source-private repair handoff.\n"
            "Use the source payload only to identify a repair diagnostic code.\n"
            "A valid code is either a two-character packet like G0 or a text span like REPAIR_DIAG=G0.\n"
            "If the code matches exactly one candidate's handles_repair_diag value, return that candidate label.\n"
            "If no valid code is present, return the target-prior label.\n"
            "Return only one candidate label and no explanation.\n\n"
            f"Target-prior label: {_prior_prediction(example)}\n"
            f"Source payload:\n{payload if payload else '<NO_SOURCE_PAYLOAD>'}\n\n"
            "Candidates:\n"
            f"{candidate_rows}\n\n"
            "Candidate label:"
        )
    if prompt_style == "terse":
        return (
            "Choose one candidate.\n"
            "The payload may contain a short repair key such as G0 or the same key in a longer string.\n"
            "Match that key to the candidate field handles_repair_diag. With no key, use the prior.\n"
            "Output only the candidate label.\n\n"
            f"Prior: {_prior_prediction(example)}\n"
            f"Payload: {payload if payload else '<NONE>'}\n\n"
            f"{candidate_rows}\n\n"
            "Answer:"
        )
    if prompt_style == "audit":
        return (
            "Selection audit task. The target prior is a fallback, not private evidence.\n"
            "Read the side message. If it contains exactly one two-character repair key, pick the row whose "
            "handles_repair_diag value is that key. Ignore formatting words around the key.\n"
            "If the side message has no usable key, emit the fallback candidate. Emit only the label.\n\n"
            f"Fallback candidate: {_prior_prediction(example)}\n"
            f"Side message:\n{payload if payload else '<EMPTY>'}\n\n"
            "Candidate rows:\n"
            f"{candidate_rows}\n\n"
            "Selected label:"
        )
    if prompt_style == "label_strict":
        return (
            "Exact-label decoder task. Your answer is valid only if it is copied verbatim from one candidate label.\n"
            "Do not output repair keys such as G0, JSON, explanations, or partial labels.\n"
            "If the side message contains exactly one two-character repair key, choose the candidate whose "
            "handles_repair_diag value exactly equals that key. If no transmitted repair key is present, choose "
            "the fallback candidate. Output one full candidate label, copied exactly.\n\n"
            f"Fallback candidate: {_prior_prediction(example)}\n"
            f"Side message:\n{payload if payload else '<EMPTY>'}\n\n"
            "Allowed candidate labels and keys:\n"
            f"{candidate_rows}\n\n"
            "Copy exactly one allowed candidate label:"
        )
    raise ValueError(f"unknown prompt style {prompt_style!r}")


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


def _parse_strict_candidate_label(generated: str, candidates: tuple[dict[str, Any], ...]) -> str:
    stripped = generated.strip()
    for candidate in candidates:
        label = candidate["label"]
        if stripped == label:
            return label
    for candidate in candidates:
        label = candidate["label"]
        if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", stripped):
            return label
    return ""


def _payload_diagnostic_codes(payload: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][0-9]\b", payload))


def _parse_candidate_label(generated: str, candidates: tuple[dict[str, Any], ...], *, payload: str = "") -> str:
    strict = _parse_strict_candidate_label(generated, candidates)
    if strict:
        return strict
    stripped = generated.strip()
    diag_match = re.search(r"\b(?:REPAIR_DIAG\s*=\s*)?([A-Z][0-9])\b", stripped)
    if diag_match:
        diag = diag_match.group(1)
        if diag not in _payload_diagnostic_codes(payload):
            return ""
        matches = [candidate for candidate in candidates if candidate["handles_diagnostic"] == diag]
        if len(matches) == 1:
            return matches[0]["label"]
    return ""


def _generate_one(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, int, float, float]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        start = time.perf_counter()
        first = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        ttft_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        e2e_ms = (time.perf_counter() - start) * 1000.0
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return generated, int(len(new_tokens)), ttft_ms, e2e_ms


def run_frontier(
    *,
    benchmark_jsonl: pathlib.Path,
    output_dir: pathlib.Path,
    model_name: str,
    device: str,
    dtype: str,
    limit: int,
    max_new_tokens: int,
    enable_thinking: bool | None,
    conditions: list[str],
    prompt_style: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(benchmark_jsonl, limit=limit)
    tokenizer, model = _load_model(model_name, device=device, dtype=dtype)
    rows: list[dict[str, Any]] = []
    for example in examples:
        for condition in conditions:
            payload = _payload_for_condition(example, condition=condition)
            prompt_candidates = _candidate_table_for_condition(example, condition=condition)
            raw_prompt = _prompt(example, payload=payload, prompt_style=prompt_style, condition=condition)
            text_prompt = _format_prompt(tokenizer, raw_prompt, enable_thinking=enable_thinking)
            prompt_tokens = len(tokenizer(text_prompt)["input_ids"])
            generated, generated_tokens, ttft_ms, e2e_ms = _generate_one(
                tokenizer=tokenizer,
                model=model,
                device=device,
                prompt=text_prompt,
                max_new_tokens=max_new_tokens,
            )
            strict_prediction = _parse_strict_candidate_label(generated, prompt_candidates)
            prediction = _parse_candidate_label(generated, prompt_candidates, payload=payload)
            rows.append(
                {
                    "example_id": example.example_id,
                    "condition": condition,
                    "prompt_style": prompt_style,
                    "answer_label": example.answer_label,
                    "target_prior_label": _prior_prediction(example),
                    "payload": payload,
                    "payload_bytes": len(payload.encode("utf-8")),
                    "payload_tokens_proxy": len(re.findall(r"\S+", payload)),
                    "prompt_bytes": len(text_prompt.encode("utf-8")),
                    "prompt_tokens": prompt_tokens,
                    "generated_text": generated,
                    "generated_tokens": generated_tokens,
                    "strict_prediction": strict_prediction,
                    "prediction": prediction,
                    "correct": prediction == example.answer_label,
                    "strict_correct": strict_prediction == example.answer_label,
                    "valid_prediction": bool(prediction),
                    "strict_valid_prediction": bool(strict_prediction),
                    "ttft_ms": ttft_ms,
                    "e2e_ms": e2e_ms,
                }
            )
    summary = summarize(rows, conditions=conditions, prompt_style=prompt_style)
    _write_jsonl(output_dir / "endpoint_proxy_rows.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    manifest = {
        "artifacts": ["endpoint_proxy_rows.jsonl", "summary.json", "summary.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["endpoint_proxy_rows.jsonl", "summary.json", "summary.md"]
        },
        "benchmark_sha256": _sha256_file(benchmark_jsonl),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "prompt_style": prompt_style,
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Mac Endpoint-Proxy Frontier Manifest", "", f"- pass gate: `{summary['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return summary


def summarize(rows: list[dict[str, Any]], *, conditions: list[str], prompt_style: str = "canonical") -> dict[str, Any]:
    example_ids = sorted({row["example_id"] for row in rows})
    metrics: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            raise ValueError(f"missing rows for {condition}")
        metrics[condition] = {
            "correct": sum(1 for row in condition_rows if row["correct"]),
            "accuracy": sum(1 for row in condition_rows if row["correct"]) / len(condition_rows),
            "strict_correct": sum(1 for row in condition_rows if row.get("strict_correct", row["correct"])),
            "strict_accuracy": sum(1 for row in condition_rows if row.get("strict_correct", row["correct"])) / len(condition_rows),
            "valid_prediction_rate": statistics.fmean(float(row["valid_prediction"]) for row in condition_rows),
            "strict_valid_prediction_rate": statistics.fmean(float(row.get("strict_valid_prediction", row["valid_prediction"])) for row in condition_rows),
            "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in condition_rows),
            "mean_payload_tokens_proxy": statistics.fmean(row["payload_tokens_proxy"] for row in condition_rows),
            "mean_prompt_bytes": statistics.fmean(row["prompt_bytes"] for row in condition_rows),
            "mean_prompt_tokens": statistics.fmean(row["prompt_tokens"] for row in condition_rows),
            "mean_generated_tokens": statistics.fmean(row["generated_tokens"] for row in condition_rows),
            "p50_ttft_ms": statistics.median(row["ttft_ms"] for row in condition_rows),
            "p95_ttft_ms": sorted(row["ttft_ms"] for row in condition_rows)[int(0.95 * (len(condition_rows) - 1))],
            "p50_e2e_ms": statistics.median(row["e2e_ms"] for row in condition_rows),
            "p95_e2e_ms": sorted(row["e2e_ms"] for row in condition_rows)[int(0.95 * (len(condition_rows) - 1))],
        }
    packet = metrics["matched_packet"]
    target = metrics["target_only"]
    full_log = metrics["full_hidden_log"]
    query = metrics["query_aware_diag_span"]
    text_at_packet = metrics["matched_byte_text_2"]
    source_destroying_controls = [
        name
        for name in ("matched_byte_text_2", "random_same_byte_packet", "deranged_candidate_diag_table")
        if name in metrics
    ]
    best_source_destroying_control_accuracy = max(metrics[name]["accuracy"] for name in source_destroying_controls)
    return {
        "n": len(example_ids),
        "conditions": conditions,
        "prompt_style": prompt_style,
        "exact_id_parity": len(rows) == len(example_ids) * len(conditions),
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "packet_minus_target_accuracy": packet["accuracy"] - target["accuracy"],
        "packet_minus_matched_byte_text_accuracy": packet["accuracy"] - text_at_packet["accuracy"],
        "packet_strict_accuracy": packet["strict_accuracy"],
        "packet_strict_minus_target_accuracy": packet["strict_accuracy"] - target["strict_accuracy"],
        "best_source_destroying_control_accuracy": best_source_destroying_control_accuracy,
        "packet_minus_best_source_destroying_control_accuracy": packet["accuracy"] - best_source_destroying_control_accuracy,
        "packet_vs_query_payload_compression": query["mean_payload_bytes"] / max(packet["mean_payload_bytes"], 1e-9),
        "packet_vs_full_log_payload_compression": full_log["mean_payload_bytes"] / max(packet["mean_payload_bytes"], 1e-9),
        "packet_prompt_token_delta_vs_target": packet["mean_prompt_tokens"] - target["mean_prompt_tokens"],
        "full_log_prompt_token_delta_vs_packet": full_log["mean_prompt_tokens"] - packet["mean_prompt_tokens"],
        "full_log_ttft_delta_vs_packet_ms": full_log["p50_ttft_ms"] - packet["p50_ttft_ms"],
        "full_log_e2e_delta_vs_packet_ms": full_log["p50_e2e_ms"] - packet["p50_e2e_ms"],
        "pass_gate": (
            len(rows) == len(example_ids) * len(conditions)
            and packet["accuracy"] >= target["accuracy"] + 0.15
            and best_source_destroying_control_accuracy <= target["accuracy"] + 0.05
            and packet["valid_prediction_rate"] >= 0.95
            and query["mean_payload_bytes"] > packet["mean_payload_bytes"]
            and full_log["mean_payload_bytes"] > packet["mean_payload_bytes"]
        ),
        "pass_rule": (
            "Packet must beat target by >=0.15, all included source-destroying controls must stay within "
            "target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and "
            "query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated."
        ),
        "metrics": metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Mac Endpoint-Proxy Frontier",
        "",
        f"- examples: `{summary['n']}`",
        f"- prompt style: `{summary['prompt_style']}`",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- packet minus target accuracy: `{summary['packet_minus_target_accuracy']:.3f}`",
        f"- packet strict-label accuracy: `{summary['packet_strict_accuracy']:.3f}`",
        f"- best source-destroying control accuracy: `{summary['best_source_destroying_control_accuracy']:.3f}`",
        f"- packet vs query-aware payload compression: `{summary['packet_vs_query_payload_compression']:.1f}x`",
        f"- packet vs full-log payload compression: `{summary['packet_vs_full_log_payload_compression']:.1f}x`",
        f"- full-log p50 TTFT delta vs packet: `{summary['full_log_ttft_delta_vs_packet_ms']:.2f} ms`",
        f"- full-log p50 E2E delta vs packet: `{summary['full_log_e2e_delta_vs_packet_ms']:.2f} ms`",
        "",
        "| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metric in summary["metrics"].items():
        lines.append(
            f"| {condition} | {metric['accuracy']:.3f} | {metric['strict_accuracy']:.3f} | "
            f"{metric['valid_prediction_rate']:.3f} | "
            f"{metric['mean_payload_bytes']:.1f} | {metric['mean_prompt_tokens']:.1f} | "
            f"{metric['p50_ttft_ms']:.2f} | {metric['p50_e2e_ms']:.2f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="*", default=list(CONDITIONS))
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="canonical")
    args = parser.parse_args()
    benchmark_path = args.benchmark_jsonl if args.benchmark_jsonl.is_absolute() else ROOT / args.benchmark_jsonl
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    summary = run_frontier(
        benchmark_jsonl=benchmark_path,
        output_dir=output_dir,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        conditions=list(args.conditions),
        prompt_style=args.prompt_style,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": summary["pass_gate"]}, indent=2, sort_keys=True))
    if not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
