"""Use a target-model listwise verifier to rerank stochastic route candidates."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.evaluate import (  # noqa: E402
    _format_prompt_for_tokenizer,
    add_paired_prediction_summary,
    default_device,
    load_generation,
    write_prediction_records,
    write_prediction_sidecar,
)
from scripts.aggregate_stochastic_routes import load_records  # noqa: E402
from scripts.rerank_stochastic_routes import (  # noqa: E402
    _annotate_candidates,
    _choose,
    _reranked_record,
    _rows_by_index,
)


_CHOICE_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_LABELS = ("A", "B", "C", "D")


def _truncate_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def build_verifier_prompt(
    *,
    problem: str,
    candidates: list[dict[str, Any]],
    max_candidate_chars: int = 700,
) -> str:
    lines = [
        "You are a strict math answer verifier.",
        "Given the problem and four candidate model outputs, choose the candidate with the most likely correct final numeric answer.",
        "Check arithmetic, units, and whether the final answer follows from the solution. Do not prefer a candidate just because it is longer.",
        "Reply with exactly one letter: A, B, C, or D.",
        "",
        "Problem:",
        problem.strip(),
        "",
        "Candidates:",
    ]
    for label, row in zip(_LABELS, candidates):
        source = row.get("candidate_source", "")
        normalized = row.get("normalized_prediction") or ""
        prediction = _truncate_text(str(row.get("prediction", "")), max_candidate_chars)
        lines.extend(
            [
                f"{label}. source={source}; extracted_final_answer={normalized!r}",
                f"solution={prediction}",
            ]
        )
    lines.extend(["", "Best candidate letter:"])
    return "\n".join(lines)


def order_verifier_candidates(
    candidates: list[dict[str, Any]],
    *,
    example_index: int,
    shuffle_labels: bool,
    label_seed: int = 0,
) -> list[dict[str, Any]]:
    ordered = [dict(row) for row in candidates]
    if not shuffle_labels:
        return ordered
    rng = random.Random(f"{label_seed}:{example_index}")
    rng.shuffle(ordered)
    return ordered


def _target_label(prompt_candidates: list[dict[str, Any]]) -> str | None:
    for label, candidate in zip(_LABELS, prompt_candidates):
        if candidate.get("candidate_source") == "target":
            return label
    return None


def parse_verifier_choice(text: str, *, candidate_count: int) -> int | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    first = stripped[0].upper()
    if first in _LABELS[:candidate_count]:
        return _LABELS.index(first)
    match = _CHOICE_RE.search(stripped)
    if match:
        idx = _LABELS.index(match.group(1).upper())
        if idx < candidate_count:
            return idx
    lowered = stripped.lower()
    for idx, label in enumerate(_LABELS[:candidate_count]):
        if f"candidate {label.lower()}" in lowered or f"option {label.lower()}" in lowered:
            return idx
    return None


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


@torch.no_grad()
def _generate_verifier_response(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
    use_chat_template: bool,
    enable_thinking: bool | None,
) -> str:
    formatted = _format_prompt_for_tokenizer(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    encoded = tokenizer(formatted, return_tensors="pt").to(device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, encoded.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _indices_for_record_sets(
    record_sets: list[list[dict[str, Any]]],
    *,
    method: str,
    baseline_method: str,
) -> tuple[list[int], dict[int, dict[str, Any]], list[dict[int, dict[str, Any]]]]:
    baseline_rows = _rows_by_index(record_sets[0], baseline_method)
    seed_method_rows = [_rows_by_index(records, method) for records in record_sets]
    indices = sorted(set(baseline_rows).intersection(*(set(rows) for rows in seed_method_rows)))
    if not indices:
        raise ValueError(f"No paired examples for method={method!r} and baseline={baseline_method!r}")
    return indices, baseline_rows, seed_method_rows


def verifier_rerank_records(
    record_sets: list[list[dict[str, Any]]],
    *,
    examples: Sequence[Any],
    method: str,
    baseline_method: str = "target_alone",
    response_fn: Callable[[int, str, list[dict[str, Any]]], str],
    fallback_policy: str = "target_on_strict_format",
    max_candidate_chars: int = 700,
    shuffle_labels: bool = False,
    label_seed: int = 0,
) -> list[dict[str, Any]]:
    indices, baseline_rows, seed_method_rows = _indices_for_record_sets(
        record_sets,
        method=method,
        baseline_method=baseline_method,
    )
    output: list[dict[str, Any]] = []
    for idx in indices:
        baseline = dict(baseline_rows[idx])
        baseline["method"] = baseline_method
        output.append(baseline)

        seed_rows = [rows[idx] for rows in seed_method_rows]
        candidates = _annotate_candidates(baseline, seed_rows)
        if idx >= len(examples):
            raise IndexError(f"Example index {idx} is outside eval set of length {len(examples)}")
        prompt_candidates = order_verifier_candidates(
            candidates,
            example_index=idx,
            shuffle_labels=shuffle_labels,
            label_seed=label_seed,
        )
        prompt = build_verifier_prompt(
            problem=examples[idx].prompt,
            candidates=prompt_candidates,
            max_candidate_chars=max_candidate_chars,
        )
        raw_response = response_fn(idx, prompt, prompt_candidates)
        choice_idx = parse_verifier_choice(raw_response, candidate_count=len(prompt_candidates))
        fallback_used = choice_idx is None
        if choice_idx is None:
            chosen = _choose(candidates, fallback_policy)
        else:
            chosen = prompt_candidates[choice_idx]

        record = _reranked_record(
            method_name="rerank_target_model_verifier",
            policy="target_model_listwise_verifier_randomized" if shuffle_labels else "target_model_listwise_verifier",
            chosen=chosen,
            baseline=baseline,
            candidates=candidates,
        )
        record["verifier_raw_response"] = raw_response
        record["verifier_choice_index"] = choice_idx
        record["verifier_choice_label"] = None if choice_idx is None else _LABELS[choice_idx]
        record["verifier_choice_candidate_source"] = None if choice_idx is None else chosen.get("candidate_source")
        record["verifier_fallback_used"] = bool(fallback_used)
        record["verifier_fallback_policy"] = fallback_policy if fallback_used else None
        record["verifier_prompt_chars"] = len(prompt)
        record["verifier_labels_shuffled"] = bool(shuffle_labels)
        record["verifier_label_seed"] = int(label_seed)
        record["verifier_target_label"] = _target_label(prompt_candidates)
        record["verifier_label_sources"] = [
            {"label": label, "source": candidate.get("candidate_source")}
            for label, candidate in zip(_LABELS, prompt_candidates)
        ]
        record["verifier_label_predictions"] = [
            {
                "label": label,
                "source": candidate.get("candidate_source"),
                "normalized_prediction": candidate.get("normalized_prediction"),
            }
            for label, candidate in zip(_LABELS, prompt_candidates)
        ]
        output.append(record)
    return output


def summarize_results(records: list[dict[str, Any]]) -> dict[str, float]:
    results: dict[str, float] = {}
    for method in sorted({str(record["method"]) for record in records}):
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)
        if method == "rerank_target_model_verifier":
            results["rerank_target_model_verifier_fallback_rate"] = sum(
                bool(row.get("verifier_fallback_used")) for row in rows
            ) / max(len(rows), 1)
            results["rerank_target_model_verifier_target_selection_rate"] = sum(
                row.get("selected_candidate_source") == "target" for row in rows
            ) / max(len(rows), 1)
            results["rerank_target_model_verifier_choice_a_rate"] = sum(
                row.get("verifier_choice_label") == "A" for row in rows
            ) / max(len(rows), 1)
            results["rerank_target_model_verifier_target_was_a_rate"] = sum(
                row.get("verifier_target_label") == "A" for row in rows
            ) / max(len(rows), 1)
            results["rerank_target_model_verifier_label_shuffle_rate"] = sum(
                bool(row.get("verifier_labels_shuffled")) for row in rows
            ) / max(len(rows), 1)
    add_paired_prediction_summary(results, records)
    return results


def write_markdown_summary(results: dict[str, float], output_md: str | pathlib.Path) -> None:
    method = "rerank_target_model_verifier"
    target = float(results.get("target_alone", 0.0))
    prefix = f"paired_{method}_vs_target_alone"
    lines = [
        "# Target-Model Verifier Reranker Summary",
        "",
        "| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Fallback rate | Target selected | Choice A | Target was A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {method} | {acc:.4f} | {delta:+.4f} | {method_only:.0f} | {baseline_only:.0f} | {both_correct:.0f} | {both_wrong:.0f} | {fallback:.4f} | {target_selected:.4f} | {choice_a:.4f} | {target_a:.4f} |".format(
            method=method,
            acc=float(results.get(method, 0.0)),
            delta=float(results.get(f"{prefix}_delta_accuracy", float(results.get(method, 0.0)) - target)),
            method_only=float(results.get(f"{prefix}_method_only", 0.0)),
            baseline_only=float(results.get(f"{prefix}_baseline_only", 0.0)),
            both_correct=float(results.get(f"{prefix}_both_correct", 0.0)),
            both_wrong=float(results.get(f"{prefix}_both_wrong", 0.0)),
            fallback=float(results.get("rerank_target_model_verifier_fallback_rate", 0.0)),
            target_selected=float(results.get("rerank_target_model_verifier_target_selection_rate", 0.0)),
            choice_a=float(results.get("rerank_target_model_verifier_choice_a_rate", 0.0)),
            target_a=float(results.get("rerank_target_model_verifier_target_was_a_rate", 0.0)),
        ),
        "",
        "Interpretation:",
        "",
        "This is a non-oracle target-model listwise selector over the same stochastic candidate set. "
        "The raw verifier response, label order, fallback flag, and position-bias rates are logged per example "
        "so selection failures can be audited.",
    ]
    pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction JSONLs for different stochastic salts.")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--method", default="rotalign_kv_gate_0.10")
    parser.add_argument("--baseline-method", default="target_alone")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fallback-policy", default="target_on_strict_format")
    parser.add_argument("--max-candidate-chars", type=int, default=700)
    parser.add_argument("--shuffle-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-seed", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_sets = [load_records(path) for path in args.inputs]
    examples = load_generation(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
    ).to(args.device)
    model.eval()

    def response_fn(_idx: int, prompt: str, _candidates: list[dict[str, Any]]) -> str:
        return _generate_verifier_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=bool(args.use_chat_template),
            enable_thinking=bool(args.enable_thinking),
        )

    records = verifier_rerank_records(
        record_sets,
        examples=examples,
        method=args.method,
        baseline_method=args.baseline_method,
        response_fn=response_fn,
        fallback_policy=args.fallback_policy,
        max_candidate_chars=args.max_candidate_chars,
        shuffle_labels=bool(args.shuffle_labels),
        label_seed=int(args.label_seed),
    )
    results = summarize_results(records)
    write_prediction_records(args.output_jsonl, records)
    write_prediction_sidecar(
        args.output_jsonl,
        records,
        results,
        {
            "inputs": [str(path) for path in args.inputs],
            "eval_file": args.eval_file,
            "method": args.method,
            "baseline_method": args.baseline_method,
            "model": args.model,
            "fallback_policy": args.fallback_policy,
            "shuffle_labels": bool(args.shuffle_labels),
            "label_seed": int(args.label_seed),
        },
    )
    if args.output_md:
        write_markdown_summary(results, args.output_md)
    for key, value in sorted(results.items()):
        if not key.startswith("paired_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
