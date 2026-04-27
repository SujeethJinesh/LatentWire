#!/usr/bin/env python3
"""Collect frozen model-scored sidecars over SVAMP target-side candidates.

This producer is intentionally stricter than earlier likelihood sketches:
candidate values come from the target-side pool built by
`analyze_svamp_source_semantic_predicate_decoder`, and emitted rows contain no
gold labels or correctness fields.
"""

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
from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


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


def _load_examples_by_id(eval_file: pathlib.Path) -> dict[str, evaluate.GenerationExample]:
    examples = evaluate.load_generation(str(eval_file))
    return {evaluate._generation_example_id(example): example for example in examples}


def _prompt_for_example(
    example: evaluate.GenerationExample,
    *,
    prompt_mode: str,
    source_reasoning_mode: str,
) -> str:
    if prompt_mode == "source_reasoning":
        return evaluate._source_reasoning_prompt(example.prompt, source_reasoning_mode)
    return example.prompt


def _format_continuation(value: str, template: str) -> str:
    return template.format(text=value).strip()


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
        return {"score": float("-inf"), "sum_logprob": float("-inf"), "mean_logprob": float("-inf"), "tokens": 0}
    past = prefix_state.past_key_values
    current = prefix_state.last_token
    logprobs: list[float] = []
    for token_id in token_ids:
        logits, past = evaluate._step_with_past(model, current, past, device)
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        logprobs.append(float(log_probs[int(token_id)].item()))
        current = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    total = float(sum(logprobs))
    mean = float(total / len(logprobs))
    return {"score": mean, "sum_logprob": total, "mean_logprob": mean, "tokens": len(logprobs)}


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _ordered_rows(path: pathlib.Path, ordered_ids: Sequence[str]) -> list[dict[str, Any]]:
    by_id = _by_id(_read_jsonl(path))
    return [by_id[example_id] for example_id in ordered_ids if example_id in by_id]


def _candidate_label(candidate: decoder.Candidate) -> str:
    for label in candidate.labels:
        if label != "source":
            return str(label)
    return "target"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    target_set = _resolve(args.target_set_json)
    eval_file = _resolve(args.eval_file)
    output_jsonl = _resolve(args.output_jsonl)
    output_md = _resolve(args.output_md) if args.output_md else output_jsonl.with_suffix(".md")
    surface = decoder._load_surface("surface", target_set)
    reference_ids = list(surface.reference_ids)
    if args.limit is not None:
        reference_ids = reference_ids[: int(args.limit)]
    examples_by_id = _load_examples_by_id(eval_file)
    missing_examples = [example_id for example_id in reference_ids if example_id not in examples_by_id]
    if missing_examples:
        raise ValueError(f"eval file missing candidate IDs: {missing_examples[:5]}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_by_id: dict[str, dict[str, Any]] = {}
    if args.resume:
        existing_by_id = _by_id(_read_jsonl(output_jsonl))
    else:
        output_jsonl.write_text("", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.scorer_model,
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
            use_chat_template=bool(args.scorer_use_chat_template),
            enable_thinking=args.scorer_enable_thinking,
        )
        candidate_scores: list[dict[str, Any]] = []
        for candidate in decoder._candidate_pool(surface, example_id):
            continuation = _format_continuation(candidate.value, args.continuation_template)
            score = _score_continuation(
                model=model,
                tokenizer=tokenizer,
                prefix_state=prefix_state,
                continuation=continuation,
                device=args.device,
            )
            candidate_scores.append(
                {
                    "label": _candidate_label(candidate),
                    "value": candidate.value,
                    "score": score["score"],
                    "sum_logprob": score["sum_logprob"],
                    "mean_logprob": score["mean_logprob"],
                    "tokens": score["tokens"],
                }
            )
        candidate_scores.sort(key=lambda item: (-float(item["score"]), item["label"], item["value"]))
        top_score = float(candidate_scores[0]["score"]) if candidate_scores else 0.0
        second_score = float(candidate_scores[1]["score"]) if len(candidate_scores) > 1 else top_score
        row = {
            "index": index,
            "example_id": example_id,
            "method": "frozen_candidate_score_sidecar",
            "scorer_model": args.scorer_model,
            "candidate_scores": candidate_scores,
            "confidence": float(top_score - second_score),
            "sidecar_bits": int(args.sidecar_bits),
        }
        with output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

    rows = _ordered_rows(output_jsonl, reference_ids)
    if len(rows) != len(reference_ids):
        present = {str(row["example_id"]) for row in rows}
        missing_output = sorted(set(reference_ids) - present)
        raise RuntimeError(f"output JSONL incomplete; missing IDs: {missing_output[:5]}")

    ordered_ids_text = "\n".join(reference_ids) + "\n"
    payload = {
        "date": str(args.date),
        "status": "frozen_candidate_score_sidecar_collected",
        "command": getattr(args, "command", None),
        "git_commit": _git_commit(),
        "target_set_json": _display_path(target_set),
        "target_set_json_sha256": _sha256_file(target_set),
        "eval_file": _display_path(eval_file),
        "eval_file_sha256": _sha256_file(eval_file),
        "scorer_model": args.scorer_model,
        "prompt_mode": args.prompt_mode,
        "continuation_template": args.continuation_template,
        "candidate_pool": "target_side_only",
        "output_jsonl": _display_path(output_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "n": len(rows),
        "ordered_example_ids": reference_ids,
        "ordered_example_ids_sha256": _sha256_text(ordered_ids_text),
        "resume": bool(args.resume),
        "skipped_existing": sum(1 for example_id in reference_ids if example_id in existing_by_id),
        "elapsed_sec": float(time.perf_counter() - started),
        "sidecar_bits": int(args.sidecar_bits),
        "device": args.device,
        "dtype": args.dtype,
    }
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Frozen Candidate Score Sidecar Collection",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- scorer model: `{payload['scorer_model']}`",
        f"- candidate pool: `{payload['candidate_pool']}`",
        f"- continuation template: `{payload['continuation_template']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- target set: `{payload['target_set_json']}`",
        f"- target set sha256: `{payload['target_set_json_sha256']}`",
        f"- eval file: `{payload['eval_file']}`",
        f"- eval file sha256: `{payload['eval_file_sha256']}`",
        f"- output JSONL: `{payload['output_jsonl']}`",
        f"- output JSONL sha256: `{payload['output_jsonl_sha256']}`",
        f"- rows: `{payload['n']}`",
        f"- sidecar bits: `{payload['sidecar_bits']}`",
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
        "The JSONL contains model-scored sidecar preferences over target-side",
        "candidate values only. It intentionally omits gold answers, correctness",
        "labels, and source-only candidate values.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-model", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--continuation-template", default="Answer: {text}")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=evaluate.default_device())
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--sidecar-bits", type=int, default=32)
    parser.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="brief_analysis",
    )
    parser.add_argument("--prompt-mode", choices=["direct", "source_reasoning"], default="source_reasoning")
    parser.add_argument("--scorer-use-chat-template", action="store_true")
    parser.add_argument("--scorer-enable-thinking", type=_optional_bool, default=None)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/collect_svamp_frozen_candidate_score_sidecar.py", *argv]
    args.command = shlex.join(raw_argv)
    return collect(args)


if __name__ == "__main__":
    main()
