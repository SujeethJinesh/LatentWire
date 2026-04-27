#!/usr/bin/env python3
"""Sample target-only generation candidates for an exact-ID eval slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge import evaluate


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def _optional_bool(value: str) -> bool | None:
    return evaluate._optional_bool_from_arg(value)


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(dict(row))
        by_id[str(row["example_id"])].append(dict(row))
    method_summaries = {
        method: {
            "n": len(items),
            "correct": int(sum(bool(row.get("correct")) for row in items)),
            "numeric_coverage": int(sum(bool(row.get("normalized_prediction")) for row in items)),
        }
        for method, items in sorted(by_method.items())
    }
    oracle_ids = sorted(
        example_id
        for example_id, items in by_id.items()
        if any(bool(row.get("correct")) for row in items)
    )
    return {
        "methods": method_summaries,
        "example_n": len(by_id),
        "sample_n": len(by_method),
        "candidate_oracle_correct": len(oracle_ids),
        "candidate_oracle_ids": oracle_ids,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate Surface Sampling",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- model: `{payload['model']}`",
        f"- eval file: `{payload['eval_file']}`",
        f"- samples per example: `{payload['config']['samples']}`",
        f"- candidate oracle: `{payload['summary']['candidate_oracle_correct']}/{payload['summary']['example_n']}`",
        "",
        "| Method | Correct | Numeric Coverage |",
        "|---|---:|---:|",
    ]
    for method, row in payload["summary"]["methods"].items():
        lines.append(f"| `{method}` | {row['correct']}/{row['n']} | {row['numeric_coverage']}/{row['n']} |")
    lines.extend(
        [
            "",
            "## Oracle IDs",
            "",
            ", ".join(f"`{example_id}`" for example_id in payload["summary"]["candidate_oracle_ids"]) or "none",
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def sample_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_file = _resolve(args.eval_file)
    examples = evaluate.load_generation(str(eval_file))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    rows: list[dict[str, Any]] = []
    for sample_index in range(int(args.samples)):
        method = f"{args.method_prefix}_s{sample_index}"
        for index, example in enumerate(examples):
            seed = int(args.seed) + sample_index * 1009 + index
            torch.manual_seed(seed)
            raw_prompt = (
                evaluate._source_reasoning_prompt(example.prompt, args.source_reasoning_mode)
                if args.prompt_mode == "source_reasoning"
                else example.prompt
            )
            prompt = evaluate._format_prompt_for_tokenizer(
                tokenizer,
                raw_prompt,
                use_chat_template=bool(args.use_chat_template),
                enable_thinking=args.enable_thinking,
            )
            started = time.perf_counter()
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            output = model.generate(
                **encoded,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - started
            generated = output[0, encoded.input_ids.shape[1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            rows.append(
                {
                    "answer": example.answers,
                    "correct": bool(evaluate._generation_match(text, example.answers)),
                    "elapsed_sec": float(elapsed),
                    "example_id": evaluate._generation_example_id(example),
                    "generated_tokens": int(generated.numel()),
                    "index": int(index),
                    "method": method,
                    "normalized_prediction": evaluate._extract_prediction_numeric_answer(text),
                    "prediction": text,
                    "sample_index": int(sample_index),
                    "sample_seed": int(seed),
                    "prompt_mode": args.prompt_mode,
                    "source_reasoning_mode": args.source_reasoning_mode,
                    **evaluate._prompt_token_telemetry(
                        raw_prompt,
                        target_tokenizer=tokenizer,
                        target_use_chat_template=bool(args.use_chat_template),
                        target_enable_thinking=args.enable_thinking,
                    ),
                }
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--method-prefix", default="target_sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-mode", choices=["direct", "source_reasoning"], default="direct")
    parser.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="brief_analysis",
    )
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", type=_optional_bool, default=False)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/sample_target_candidate_surface.py", *argv]
    output_jsonl = _resolve(args.output_jsonl)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    rows = sample_records(args)
    _write_jsonl(output_jsonl, rows)
    payload = {
        "date": str(date.today()),
        "status": "candidate_surface_sampled",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "eval_file": _display_path(_resolve(args.eval_file)),
        "eval_file_sha256": _sha256_file(_resolve(args.eval_file)),
        "model": args.model,
        "config": {
            "samples": int(args.samples),
            "method_prefix": args.method_prefix,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "seed": int(args.seed),
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": int(args.max_new_tokens),
            "prompt_mode": args.prompt_mode,
            "source_reasoning_mode": args.source_reasoning_mode,
            "use_chat_template": bool(args.use_chat_template),
            "enable_thinking": args.enable_thinking,
        },
        "output_jsonl": _display_path(output_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "summary": summarize(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "oracle": payload["summary"]["candidate_oracle_correct"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
