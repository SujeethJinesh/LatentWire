#!/usr/bin/env python3
"""Collect source-only generation confidence diagnostics without mutating baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_bridge import evaluate


@dataclass(frozen=True)
class StepStats:
    token_id: int
    chosen_logprob: float
    top1_logit: float
    top2_logit: float | None
    top1_prob: float
    top2_prob: float | None
    top1_top2_logit_margin: float | None
    entropy: float


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


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def _step_stats(logits: torch.Tensor, token_id: int) -> StepStats:
    vector = logits[0].float()
    log_probs = torch.log_softmax(vector, dim=-1)
    probs = log_probs.exp()
    topk = torch.topk(vector, k=min(2, vector.numel()))
    top_probs = torch.topk(probs, k=min(2, probs.numel()))
    entropy = float((-(probs * log_probs)).sum().item())
    top2_logit = float(topk.values[1].item()) if topk.values.numel() > 1 else None
    top2_prob = float(top_probs.values[1].item()) if top_probs.values.numel() > 1 else None
    margin = float(topk.values[0].item() - topk.values[1].item()) if topk.values.numel() > 1 else None
    return StepStats(
        token_id=int(token_id),
        chosen_logprob=float(log_probs[int(token_id)].item()),
        top1_logit=float(topk.values[0].item()),
        top2_logit=top2_logit,
        top1_prob=float(top_probs.values[0].item()),
        top2_prob=top2_prob,
        top1_top2_logit_margin=margin,
        entropy=entropy,
    )


def _summarize_steps(steps: list[StepStats]) -> dict[str, Any]:
    if not steps:
        return {
            "generated_token_ids": [],
            "generated_tokens": 0,
            "mean_chosen_logprob": None,
            "min_chosen_logprob": None,
            "final_chosen_logprob": None,
            "mean_entropy": None,
            "max_entropy": None,
            "mean_top1_top2_logit_margin": None,
            "min_top1_top2_logit_margin": None,
            "final_top1_top2_logit_margin": None,
            "mean_top1_prob": None,
            "min_top1_prob": None,
            "final_top1_prob": None,
        }
    logprobs = [step.chosen_logprob for step in steps]
    entropies = [step.entropy for step in steps]
    margins = [
        step.top1_top2_logit_margin
        for step in steps
        if step.top1_top2_logit_margin is not None
    ]
    top1_probs = [step.top1_prob for step in steps]
    return {
        "generated_token_ids": [step.token_id for step in steps],
        "generated_tokens": len(steps),
        "mean_chosen_logprob": float(sum(logprobs) / len(logprobs)),
        "min_chosen_logprob": float(min(logprobs)),
        "final_chosen_logprob": float(logprobs[-1]),
        "mean_entropy": float(sum(entropies) / len(entropies)),
        "max_entropy": float(max(entropies)),
        "mean_top1_top2_logit_margin": (
            float(sum(margins) / len(margins)) if margins else None
        ),
        "min_top1_top2_logit_margin": float(min(margins)) if margins else None,
        "final_top1_top2_logit_margin": steps[-1].top1_top2_logit_margin,
        "mean_top1_prob": float(sum(top1_probs) / len(top1_probs)),
        "min_top1_prob": float(min(top1_probs)),
        "final_top1_prob": float(top1_probs[-1]),
    }


@torch.no_grad()
def _generate_with_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
    use_chat_template: bool,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prefix_state = evaluate._prepare_prefix_state(
        model,
        tokenizer,
        prompt,
        device,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    prep_elapsed = time.perf_counter() - started
    logits, past = evaluate._step_with_past(
        model,
        prefix_state.last_token,
        prefix_state.past_key_values,
        device,
    )
    ttft_sec = time.perf_counter() - started
    generated: list[int] = []
    steps: list[StepStats] = []
    next_token = int(logits.argmax(dim=-1).item())
    generated.append(next_token)
    steps.append(_step_stats(logits, next_token))
    current = torch.tensor([[next_token]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break
        logits, past = evaluate._step_with_past(model, current, past, device)
        next_token = int(logits.argmax(dim=-1).item())
        generated.append(next_token)
        steps.append(_step_stats(logits, next_token))
        current = torch.tensor([[next_token]], dtype=torch.long, device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "prediction": text,
        "prep_elapsed_sec": float(prep_elapsed),
        "ttft_sec": float(ttft_sec),
        "elapsed_sec": float(time.perf_counter() - started),
        "eos_seen": bool(generated and generated[-1] == tokenizer.eos_token_id),
        **_summarize_steps(steps),
        "per_token_diagnostics": [step.__dict__ for step in steps],
    }


def _limit_examples(
    examples: list[evaluate.GenerationExample],
    limit: int | None,
) -> list[evaluate.GenerationExample]:
    return examples if limit is None else examples[: int(limit)]


def _optional_bool(value: str) -> bool | None:
    return evaluate._optional_bool_from_arg(value)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    eval_file = _resolve(args.eval_file)
    output_jsonl = _resolve(args.output_jsonl)
    output_md = _resolve(args.output_md) if args.output_md else output_jsonl.with_suffix(".md")
    examples = _limit_examples(evaluate.load_generation(str(eval_file)), args.limit)
    dtype = _torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device).eval()
    records: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        prompt = (
            evaluate._source_reasoning_prompt(example.prompt, args.source_reasoning_mode)
            if args.prompt_mode == "source_reasoning"
            else example.prompt
        )
        diagnostics = _generate_with_diagnostics(
            model,
            tokenizer,
            prompt,
            device=args.device,
            max_new_tokens=int(args.max_new_tokens),
            use_chat_template=bool(args.source_use_chat_template),
            enable_thinking=args.source_enable_thinking,
        )
        correct = evaluate._generation_match(diagnostics["prediction"], example.answers)
        records.append(
            {
                "index": idx,
                "example_id": evaluate._generation_example_id(example),
                "method": "source_generation_diagnostics",
                "source_model": args.source_model,
                "prompt_mode": args.prompt_mode,
                "source_reasoning_mode": args.source_reasoning_mode,
                "prediction": diagnostics["prediction"],
                "answer": example.answers,
                "correct": bool(correct),
                "normalized_prediction": evaluate._extract_prediction_numeric_answer(
                    diagnostics["prediction"]
                ),
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if key != "prediction"
                },
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    correct = sum(int(row["correct"]) for row in records)
    payload = {
        "date": str(date.today()),
        "status": "source_generation_diagnostics_collected",
        "eval_file": _display_path(eval_file),
        "eval_file_sha256": _sha256_file(eval_file),
        "output_jsonl": _display_path(output_jsonl),
        "output_jsonl_sha256": _sha256_file(output_jsonl),
        "source_model": args.source_model,
        "prompt_mode": args.prompt_mode,
        "source_reasoning_mode": args.source_reasoning_mode,
        "n": len(records),
        "correct": correct,
        "accuracy": float(correct / max(len(records), 1)),
        "max_new_tokens": int(args.max_new_tokens),
        "device": args.device,
        "dtype": args.dtype,
    }
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source Generation Diagnostics",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- source model: `{payload['source_model']}`",
        f"- prompt mode: `{payload['prompt_mode']}`",
        f"- eval file: `{payload['eval_file']}`",
        f"- output JSONL: `{payload['output_jsonl']}`",
        f"- output JSONL sha256: `{payload['output_jsonl_sha256']}`",
        f"- correct: `{payload['correct']}/{payload['n']}`",
        "",
        "This sidecar artifact records source-only greedy generation confidence "
        "signals: per-token chosen logprob, entropy, top-1 probability, and "
        "top-1/top-2 logit margin.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--device", default=evaluate.default_device())
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="brief_analysis",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["direct", "source_reasoning"],
        default="direct",
        help=(
            "Use direct example prompts to match source_alone baselines, or wrap "
            "with --source-reasoning-mode for text-relay/source-hint diagnostics."
        ),
    )
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", type=_optional_bool, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return collect(parse_args(argv))


if __name__ == "__main__":
    main()
