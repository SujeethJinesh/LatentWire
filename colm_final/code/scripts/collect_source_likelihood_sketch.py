#!/usr/bin/env python3
"""Collect source-model likelihood sketches over a candidate answer pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_bridge import evaluate
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def _optional_bool(value: str) -> bool | None:
    return evaluate._optional_bool_from_arg(value)


def _by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _candidate_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None and field != "prediction":
        value = row.get("prediction")
    if value is None:
        value = row.get("normalized_prediction")
    if value is None:
        value = ""
    return str(value).strip()


def _format_continuation(text: str, template: str) -> str:
    return template.format(text=text).strip()


@torch.no_grad()
def _score_continuation(
    *,
    model: Any,
    tokenizer: Any,
    prefix_state: evaluate.PrefixState,
    continuation: str,
    device: str,
) -> dict[str, Any]:
    token_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if not token_ids:
        return {
            "score": float("-inf"),
            "sum_logprob": float("-inf"),
            "mean_logprob": float("-inf"),
            "tokens": 0,
        }
    past = prefix_state.past_key_values
    current = prefix_state.last_token
    logprobs: list[float] = []
    for token_id in token_ids:
        logits, past = evaluate._step_with_past(model, current, past, device)
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        logprobs.append(float(log_probs[int(token_id)].item()))
        current = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    total = float(sum(logprobs))
    return {
        "score": float(total / len(logprobs)),
        "sum_logprob": total,
        "mean_logprob": float(total / len(logprobs)),
        "tokens": len(logprobs),
    }


def _load_examples_by_id(eval_file: pathlib.Path) -> dict[str, evaluate.GenerationExample]:
    examples = evaluate.load_generation(str(eval_file))
    return {evaluate._generation_example_id(example): example for example in examples}


def _read_jsonl_if_exists(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_rows_by_id(path: pathlib.Path, ordered_ids: Sequence[str]) -> list[dict[str, Any]]:
    by_id = _by_id(_read_jsonl_if_exists(path))
    return [by_id[example_id] for example_id in ordered_ids if example_id in by_id]


def _prompt_for_example(
    example: evaluate.GenerationExample,
    *,
    prompt_mode: str,
    source_reasoning_mode: str,
) -> str:
    if prompt_mode == "source_reasoning":
        return evaluate._source_reasoning_prompt(example.prompt, source_reasoning_mode)
    return example.prompt


def collect(args: argparse.Namespace) -> dict[str, Any]:
    eval_file = _resolve(args.eval_file)
    output_jsonl = _resolve(args.output_jsonl)
    output_md = _resolve(args.output_md) if args.output_md else output_jsonl.with_suffix(".md")
    candidate_specs = [syndrome._parse_spec(spec) for spec in args.candidate]
    candidate_records = {
        spec.label: _by_id(syndrome._records_for_method(spec))
        for spec in candidate_specs
    }
    reference_label = args.reference_label or candidate_specs[0].label
    if reference_label not in candidate_records:
        raise ValueError(f"reference label {reference_label!r} not found")
    reference_ids = list(candidate_records[reference_label])
    if args.limit is not None:
        reference_ids = reference_ids[: int(args.limit)]
    for label, rows in candidate_records.items():
        missing = [example_id for example_id in reference_ids if example_id not in rows]
        if missing:
            raise ValueError(f"candidate {label!r} missing IDs: {missing[:5]}")
    examples_by_id = _load_examples_by_id(eval_file)
    missing_examples = [example_id for example_id in reference_ids if example_id not in examples_by_id]
    if missing_examples:
        raise ValueError(f"eval file missing candidate IDs: {missing_examples[:5]}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_by_id: dict[str, dict[str, Any]] = {}
    if args.resume:
        existing_by_id = _by_id(_read_jsonl_if_exists(output_jsonl))
    else:
        output_jsonl.write_text("", encoding="utf-8")
    completed_existing = [example_id for example_id in reference_ids if example_id in existing_by_id]

    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=_torch_dtype(args.dtype),
        trust_remote_code=True,
    ).to(args.device).eval()

    started = time.perf_counter()
    for index, example_id in enumerate(reference_ids):
        if example_id in existing_by_id:
            continue
        example = examples_by_id[example_id]
        prompt = _prompt_for_example(
            example,
            prompt_mode=args.prompt_mode,
            source_reasoning_mode=args.source_reasoning_mode,
        )
        prefix_state = evaluate._prepare_prefix_state(
            model,
            tokenizer,
            prompt,
            args.device,
            use_chat_template=bool(args.source_use_chat_template),
            enable_thinking=args.source_enable_thinking,
        )
        candidate_scores: list[dict[str, Any]] = []
        for label, by_id in candidate_records.items():
            candidate = by_id[example_id]
            raw_text = _candidate_text(candidate, args.candidate_text_field)
            text = _format_continuation(raw_text, args.continuation_template)
            score = _score_continuation(
                model=model,
                tokenizer=tokenizer,
                prefix_state=prefix_state,
                continuation=text,
                device=args.device,
            )
            candidate_scores.append(
                {
                    "label": label,
                    "score": score["score"],
                    "sum_logprob": score["sum_logprob"],
                    "mean_logprob": score["mean_logprob"],
                    "tokens": score["tokens"],
                    "candidate_text": text,
                    "candidate_raw_text": raw_text,
                    "candidate_correct": bool(candidate.get("correct")),
                }
            )
        ranked = sorted(candidate_scores, key=lambda item: (-float(item["score"]), item["label"]))
        row = {
            "index": index,
            "example_id": example_id,
            "method": "source_likelihood_sketch",
            "source_model": args.source_model,
            "prompt_mode": args.prompt_mode,
            "candidate_text_field": args.candidate_text_field,
            "continuation_template": args.continuation_template,
            "top_label": ranked[0]["label"] if ranked else None,
            "candidate_scores": candidate_scores,
        }
        with output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    rows = _ordered_rows_by_id(output_jsonl, reference_ids)
    if len(rows) != len(reference_ids):
        present = {str(row["example_id"]) for row in rows}
        missing_output = sorted(set(reference_ids) - present)
        raise RuntimeError(f"output JSONL incomplete; missing IDs: {missing_output[:5]}")
    ordered_ids_text = "\n".join(reference_ids) + "\n"
    payload = {
        "date": str(args.date),
        "status": "source_likelihood_sketch_collected",
        "command": getattr(args, "command", None),
        "git_commit": _git_commit(),
        "eval_file": _display_path(eval_file),
        "eval_file_sha256": _sha256_file(eval_file),
        "source_model": args.source_model,
        "candidate_text_field": args.candidate_text_field,
        "continuation_template": args.continuation_template,
        "candidate_specs": [
            {
                "label": spec.label,
                "path": _display_path(spec.path),
                "path_sha256": _sha256_file(spec.path),
                "method": spec.method,
            }
            for spec in candidate_specs
        ],
        "output_jsonl": _display_path(output_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "n": len(rows),
        "ordered_example_ids": reference_ids,
        "ordered_example_ids_sha256": _sha256_text(ordered_ids_text),
        "resume": bool(args.resume),
        "skipped_existing": len(completed_existing),
        "elapsed_sec": float(time.perf_counter() - started),
        "device": args.device,
        "dtype": args.dtype,
    }
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source Likelihood Sketch Collection",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- source model: `{payload['source_model']}`",
        f"- candidate text field: `{payload['candidate_text_field']}`",
        f"- continuation template: `{payload['continuation_template']}`",
        f"- git commit: `{payload['git_commit'] or 'unknown'}`",
        f"- eval file: `{payload['eval_file']}`",
        f"- eval file sha256: `{payload['eval_file_sha256']}`",
        f"- output JSONL: `{payload['output_jsonl']}`",
        f"- output JSONL sha256: `{payload['output_jsonl_sha256']}`",
        f"- rows: `{payload['n']}`",
        f"- ordered IDs sha256: `{payload['ordered_example_ids_sha256']}`",
        f"- resume: `{payload['resume']}`",
        f"- skipped existing: `{payload['skipped_existing']}`",
        f"- device: `{payload['device']}`",
        f"- dtype: `{payload['dtype']}`",
        "",
        "## Command",
        "",
        "```bash",
        str(payload.get("command") or "unknown"),
        "```",
        "",
        "## Candidate Inputs",
        "",
    ]
    for spec in payload["candidate_specs"]:
        lines.append(
            f"- `{spec['label']}`: `{spec['path']}` method `{spec['method']}` sha256 `{spec['path_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Ordered Example IDs",
            "",
        ]
    )
    lines.extend(f"- `{example_id}`" for example_id in payload["ordered_example_ids"])
    lines.extend([
        "",
        "The JSONL contains source-model continuation likelihoods over the "
        "candidate answer pool. The downstream gate transmits only a quantized "
        "top-label/margin sketch and compares it to source-destroyed controls.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--reference-label")
    parser.add_argument("--candidate-text-field", default="prediction")
    parser.add_argument("--continuation-template", default="{text}")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=evaluate.default_device())
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="brief_analysis",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["direct", "source_reasoning"],
        default="direct",
    )
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", type=_optional_bool, default=None)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/collect_source_likelihood_sketch.py", *argv]
    args.command = shlex.join(raw_argv)
    return collect(args)


if __name__ == "__main__":
    main()
