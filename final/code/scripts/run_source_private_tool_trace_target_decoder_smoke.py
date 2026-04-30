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
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LoadedExample:
    example_id: str
    answer_label: str
    diagnostic_code: str
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
            candidates=tuple(row["candidates"]),
        )
        for row in rows
    ]


def _prior_prediction(example: LoadedExample) -> str:
    return max(example.candidates, key=lambda row: float(row["prior_score"]))["label"]


def _deterministic_nonself_index(index: int, n: int) -> int:
    candidate = (index * 17 + 11) % n
    return candidate if candidate != index else (index + 1) % n


def _condition_payload(
    *,
    condition: str,
    example: LoadedExample,
    examples: list[LoadedExample],
    index: int,
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    if condition == "target_only":
        return "", {"packet_kind": "none"}
    if condition == "matched_packet":
        return example.diagnostic_code, {"packet_kind": "repair_diag"}
    if condition == "shuffled_packet":
        other = examples[_deterministic_nonself_index(index, len(examples))]
        return other.diagnostic_code, {"packet_kind": "repair_diag", "source_example_id": other.example_id}
    if condition == "random_same_byte":
        return f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWYZ')}{rng.randrange(0, 10)}", {"packet_kind": "random_diag"}
    if condition == "structured_json_2byte":
        payload = json.dumps({"repair_diag": example.diagnostic_code}, sort_keys=True).encode("utf-8")[:2].decode(
            "utf-8", errors="ignore"
        )
        return payload, {"packet_kind": "truncated_json_diag"}
    if condition == "structured_free_text_2byte":
        payload = f"repair diag is {example.diagnostic_code}".encode("utf-8")[:2].decode("utf-8", errors="ignore")
        return payload, {"packet_kind": "truncated_free_text_diag"}
    raise ValueError(f"unknown condition {condition!r}")


def _conditions() -> list[str]:
    return [
        "target_only",
        "matched_packet",
        "shuffled_packet",
        "random_same_byte",
        "structured_json_2byte",
        "structured_free_text_2byte",
    ]


def _validate_conditions(conditions: list[str] | None) -> list[str]:
    available = _conditions()
    if not conditions:
        return available
    unknown = sorted(set(conditions) - set(available))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    return list(conditions)


def _prompt_for_target_decoder(example: LoadedExample, *, payload: str) -> str:
    prior = _prior_prediction(example)
    candidate_rows = "\n".join(
        f"- {candidate['label']}: handles_repair_diag={candidate['handles_diagnostic']}"
        for candidate in example.candidates
    )
    packet = payload if payload else "<NO_SOURCE_PACKET>"
    return (
        "You are the target-side decoder in a source-private repair handoff.\n"
        "Use only the source packet and candidate handles_repair_diag metadata.\n"
        "If the source packet is a two-character code matching exactly one candidate's "
        "handles_repair_diag value, return that candidate label.\n"
        "If no valid packet is present or no candidate matches, return the target-prior label.\n"
        "Return only one candidate label and no explanation.\n\n"
        f"Target-prior label: {prior}\n"
        f"Source packet: {packet}\n"
        "Candidates:\n"
        f"{candidate_rows}\n\n"
        "Candidate label:"
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


def _parse_candidate_label(generated: str, example: LoadedExample) -> str:
    stripped = generated.strip()
    for candidate in example.candidates:
        label = candidate["label"]
        if stripped == label:
            return label
    for candidate in example.candidates:
        label = candidate["label"]
        if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", stripped):
            return label
    return ""


def _generate_target_predictions(
    examples: list[LoadedExample],
    *,
    model_name: str,
    device: str,
    dtype: str,
    seed: int,
    max_new_tokens: int,
    enable_thinking: bool | None,
    conditions: list[str] | None = None,
    progress_jsonl: pathlib.Path | None = None,
    partial_predictions_jsonl: pathlib.Path | None = None,
    progress_every: int = 16,
) -> list[dict[str, Any]]:
    import torch

    active_conditions = _validate_conditions(conditions)
    tokenizer, model = _load_model(model_name, device=device, dtype=dtype)
    torch.manual_seed(seed)
    rng = random.Random(seed + 20260429)
    rows: list[dict[str, Any]] = []
    progress_handle = progress_jsonl.open("a", encoding="utf-8") if progress_jsonl is not None else None
    partial_handle = partial_predictions_jsonl.open("a", encoding="utf-8") if partial_predictions_jsonl is not None else None
    for index, example in enumerate(examples):
        for condition in active_conditions:
            payload, metadata = _condition_payload(
                condition=condition,
                example=example,
                examples=examples,
                index=index,
                rng=rng,
            )
            prompt = _prompt_for_target_decoder(example, payload=payload)
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
            prediction = _parse_candidate_label(generated, example)
            row = {
                "example_id": example.example_id,
                "condition": condition,
                "answer_label": example.answer_label,
                "target_prior_label": _prior_prediction(example),
                "payload": payload,
                "payload_bytes": len(payload.encode("utf-8")),
                "payload_tokens": len(re.findall(r"\S+", payload)),
                "generated_text": generated,
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "valid_prediction": bool(prediction),
                "latency_ms": latency_ms,
                "generated_tokens": len(new_tokens),
                **metadata,
            }
            rows.append(row)
            if partial_handle is not None:
                partial_handle.write(json.dumps(row, sort_keys=True) + "\n")
                partial_handle.flush()
        if progress_handle is not None and ((index + 1) % max(progress_every, 1) == 0 or index + 1 == len(examples)):
            progress_handle.write(
                json.dumps(
                    {
                        "completed_examples": index + 1,
                        "total_examples": len(examples),
                        "rows": len(rows),
                        "conditions": active_conditions,
                        "last_example_id": example.example_id,
                        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            progress_handle.flush()
    if progress_handle is not None:
        progress_handle.close()
    if partial_handle is not None:
        partial_handle.close()
    return rows


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(rows: list[dict[str, Any]], *, conditions: list[str] | None = None) -> dict[str, Any]:
    conditions = _validate_conditions(conditions)
    example_ids = sorted({row["example_id"] for row in rows})
    metrics: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            raise ValueError(f"missing rows for condition {condition!r}")
        correct = [row["example_id"] for row in condition_rows if row["correct"]]
        metrics[condition] = {
            "correct": len(correct),
            "accuracy": len(correct) / len(condition_rows),
            "correct_ids": correct,
            "valid_prediction_rate": statistics.fmean(float(row["valid_prediction"]) for row in condition_rows),
            "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in condition_rows),
            "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in condition_rows),
            "mean_generated_tokens": statistics.fmean(row["generated_tokens"] for row in condition_rows),
            "p50_latency_ms": statistics.median(row["latency_ms"] for row in condition_rows),
        }
    target = metrics["target_only"]["accuracy"]
    controls = [
        "shuffled_packet",
        "random_same_byte",
        "structured_json_2byte",
        "structured_free_text_2byte",
    ]
    controls = [name for name in controls if name in metrics]
    best_control = max((metrics[name]["accuracy"] for name in controls), default=target)
    matched = metrics["matched_packet"]["accuracy"]
    return {
        "n": len(example_ids),
        "exact_id_count": len(example_ids),
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "conditions": conditions,
        "exact_id_parity": len(example_ids) * len(conditions) == len(rows),
        "target_only_accuracy": target,
        "matched_accuracy": matched,
        "best_control_accuracy": best_control,
        "matched_minus_target": matched - target,
        "matched_minus_best_control": matched - best_control,
        "pass_gate": matched - target >= 0.15 and best_control <= target + 0.05,
        "pass_rule": "matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.",
        "metrics": metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_partial_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return _load_jsonl(path) if path.exists() else []


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Tool-Trace Target-Decoder Smoke",
        "",
        f"- examples: `{summary['n']}`",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- matched minus target: `{summary['matched_minus_target']:.3f}`",
        f"- matched minus best control: `{summary['matched_minus_best_control']:.3f}`",
        "",
        "| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["metrics"].items():
        lines.append(
            "| "
            f"{condition} | {metrics['correct']}/{summary['n']} | "
            f"{metrics['accuracy']:.3f} | {metrics['valid_prediction_rate']:.3f} | "
            f"{metrics['mean_payload_bytes']:.2f} | {metrics['mean_generated_tokens']:.2f} | "
            f"{metrics['p50_latency_ms']:.2f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Source-Private Tool-Trace Target-Decoder Smoke Manifest",
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
        f"- matched accuracy: `{summary['matched_accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best control accuracy: `{summary['best_control_accuracy']:.3f}`",
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
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--conditions", choices=_conditions(), nargs="*", default=None)
    parser.add_argument("--progress-jsonl", type=pathlib.Path, default=None)
    parser.add_argument("--partial-predictions-jsonl", type=pathlib.Path, default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    args = parser.parse_args()

    benchmark_path = args.benchmark_jsonl if args.benchmark_jsonl.is_absolute() else ROOT / args.benchmark_jsonl
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = _validate_conditions(args.conditions)
    progress_jsonl = None
    if args.progress_jsonl is not None:
        progress_jsonl = args.progress_jsonl if args.progress_jsonl.is_absolute() else ROOT / args.progress_jsonl
        progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    partial_predictions_jsonl = None
    if args.partial_predictions_jsonl is not None:
        partial_predictions_jsonl = (
            args.partial_predictions_jsonl
            if args.partial_predictions_jsonl.is_absolute()
            else ROOT / args.partial_predictions_jsonl
        )
        partial_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(benchmark_path, limit=args.limit)
    rows = _generate_target_predictions(
        examples,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        conditions=conditions,
        progress_jsonl=progress_jsonl,
        partial_predictions_jsonl=partial_predictions_jsonl,
        progress_every=args.progress_every,
    )
    if partial_predictions_jsonl is not None:
        partial_rows = _read_partial_jsonl(partial_predictions_jsonl)
        if len(partial_rows) < len(rows):
            raise RuntimeError(
                f"partial prediction log {partial_predictions_jsonl} has {len(partial_rows)} rows but final run produced {len(rows)}"
            )
    summary = _summarize(rows, conditions=conditions)
    _write_jsonl(output_dir / "target_predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    artifacts = ["target_predictions.jsonl", "summary.json", "summary.md", "manifest.json", "manifest.md"]
    if partial_predictions_jsonl is not None and partial_predictions_jsonl.parent.resolve() == output_dir.resolve():
        artifacts.insert(1, partial_predictions_jsonl.name)
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_tool_trace_target_decoder_smoke.py",
                f"--benchmark-jsonl {args.benchmark_jsonl}",
                f"--output-dir {args.output_dir}",
                f"--model {args.model}",
                f"--device {args.device}",
                f"--dtype {args.dtype}",
                f"--limit {args.limit}",
                f"--seed {args.seed}",
                f"--max-new-tokens {args.max_new_tokens}",
                "--no-enable-thinking" if args.enable_thinking is False else "--enable-thinking",
                "" if args.conditions is None else "--conditions " + " ".join(args.conditions),
                "" if args.progress_jsonl is None else f"--progress-jsonl {args.progress_jsonl}",
                ""
                if args.partial_predictions_jsonl is None
                else f"--partial-predictions-jsonl {args.partial_predictions_jsonl}",
                f"--progress-every {args.progress_every}",
            ]
        ),
        "args": vars(args)
        | {
            "benchmark_jsonl": str(args.benchmark_jsonl),
            "output_dir": str(args.output_dir),
            "progress_jsonl": None if args.progress_jsonl is None else str(args.progress_jsonl),
            "partial_predictions_jsonl": None if partial_predictions_jsonl is None else str(partial_predictions_jsonl),
            "do_sample": False,
        },
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "benchmark_sha256": _sha256_file(benchmark_path),
        "python": sys.version,
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
