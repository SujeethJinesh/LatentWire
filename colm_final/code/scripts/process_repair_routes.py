"""Repair selected stochastic route candidates with a process-aware prompt."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.evaluate import (  # noqa: E402
    _extract_prediction_numeric_answer,
    _format_prompt_for_tokenizer,
    _generation_match,
    add_paired_prediction_summary,
    default_device,
    load_generation,
    write_prediction_records,
    write_prediction_sidecar,
)
from scripts.aggregate_stochastic_routes import load_records  # noqa: E402
from scripts.rerank_stochastic_routes import (  # noqa: E402
    _annotate_candidates,
    _candidate_metadata,
    _choose,
    _reranked_record,
    _rows_by_index,
)
from scripts.verifier_rerank_stochastic_routes import _truncate_text  # noqa: E402


def build_repair_prompt(
    *,
    problem: str,
    candidate: dict[str, Any],
    max_candidate_chars: int = 1200,
) -> str:
    candidate_text = _truncate_text(str(candidate.get("prediction", "")), max_candidate_chars)
    extracted = candidate.get("normalized_prediction") or ""
    return "\n".join(
        [
            "You are a careful math solution repairer.",
            "Audit the candidate solution step by step. If an arithmetic or reasoning step is wrong, correct the first wrong step and continue from the corrected value.",
            "If the candidate is already correct, keep the answer. Do not choose an answer just because it appeared in the candidate.",
            "End with exactly one line of the form: Final answer: <number>",
            "",
            "Problem:",
            problem.strip(),
            "",
            f"Candidate extracted final answer: {extracted!r}",
            "Candidate solution:",
            candidate_text,
            "",
            "Repair:",
        ]
    )


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


@torch.no_grad()
def _generate_repair_response(
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


def process_repair_records(
    record_sets: list[list[dict[str, Any]]],
    *,
    examples: Sequence[Any],
    method: str,
    response_fn: Callable[[int, str, dict[str, Any], list[dict[str, Any]]], str],
    baseline_method: str = "target_alone",
    selection_policy: str = "target_on_strict_format",
    max_candidate_chars: int = 1200,
    control_arms: Sequence[str] = (),
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
        selected = _choose(candidates, selection_policy)
        if idx >= len(examples):
            raise IndexError(f"Example index {idx} is outside eval set of length {len(examples)}")
        if "selected_no_repair" in control_arms:
            output.append(
                _make_repair_record(
                    idx=idx,
                    example=examples[idx],
                    method_name="selected_route_no_repair",
                    policy=f"selected_no_repair_after_{selection_policy}",
                    chosen=selected,
                    baseline=baseline,
                    candidates=candidates,
                    raw_response=str(selected.get("prediction", "")),
                    prompt_chars=0,
                    selection_policy=selection_policy,
                )
            )
        if "target_self_repair" in control_arms:
            target = next(row for row in candidates if row.get("candidate_source") == "target")
            target_prompt = build_repair_prompt(
                problem=examples[idx].prompt,
                candidate=target,
                max_candidate_chars=max_candidate_chars,
            )
            target_response = response_fn(idx, target_prompt, target, candidates)
            output.append(
                _make_repair_record(
                    idx=idx,
                    example=examples[idx],
                    method_name="target_self_repair",
                    policy="target_self_repair",
                    chosen=target,
                    baseline=baseline,
                    candidates=candidates,
                    raw_response=target_response,
                    prompt_chars=len(target_prompt),
                    selection_policy="target_only",
                )
            )

        prompt = build_repair_prompt(
            problem=examples[idx].prompt,
            candidate=selected,
            max_candidate_chars=max_candidate_chars,
        )
        raw_response = response_fn(idx, prompt, selected, candidates)
        output.append(
            _make_repair_record(
                idx=idx,
                example=examples[idx],
                method_name="process_repair_selected_route",
                policy=f"process_repair_after_{selection_policy}",
                chosen=selected,
                baseline=baseline,
                candidates=candidates,
                raw_response=raw_response,
                prompt_chars=len(prompt),
                selection_policy=selection_policy,
            )
        )
    return output


def _make_repair_record(
    *,
    idx: int,
    example: Any,
    method_name: str,
    policy: str,
    chosen: dict[str, Any],
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    raw_response: str,
    prompt_chars: int,
    selection_policy: str,
) -> dict[str, Any]:
    selected_normalized = chosen.get("normalized_prediction") or ""
    if method_name == "selected_route_no_repair":
        repaired_normalized = selected_normalized
        repaired_correct = bool(chosen.get("correct"))
    else:
        repaired_normalized = _extract_prediction_numeric_answer(raw_response)
        repaired_correct = _generation_match(raw_response, example.answers)
    full_meta = _candidate_metadata(candidates)

    record = _reranked_record(
        method_name=method_name,
        policy=policy,
        chosen=chosen,
        baseline=baseline,
        candidates=candidates,
    )
    record["index"] = int(idx)
    record["prediction"] = raw_response
    record["normalized_prediction"] = repaired_normalized
    record["correct"] = bool(repaired_correct)
    record["repair_raw_response"] = raw_response
    record["repair_prompt_chars"] = int(prompt_chars)
    record["repair_selection_policy"] = selection_policy
    record["repair_selected_candidate_source"] = chosen.get("candidate_source")
    record["repair_pre_prediction"] = chosen.get("prediction")
    record["repair_pre_normalized_prediction"] = selected_normalized
    record["repair_pre_correct"] = bool(chosen.get("correct"))
    record["repair_post_normalized_prediction"] = repaired_normalized
    record["repair_changed_answer"] = bool(str(repaired_normalized or "") != str(selected_normalized or ""))
    record["repair_full_oracle_correct"] = bool(full_meta["candidate_oracle_correct"])
    record["repair_full_seed_correct_count"] = int(full_meta["seed_correct_count"])
    record["repair_full_vote_entropy"] = float(full_meta["candidate_vote_entropy"])
    return record


def summarize_results(records: list[dict[str, Any]]) -> dict[str, float]:
    results: dict[str, float] = {}
    for method in sorted({str(record["method"]) for record in records}):
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)
        if any(any(str(key).startswith("repair_") for key in row) for row in rows):
            results[f"{method}_pre_repair_accuracy"] = sum(
                bool(row.get("repair_pre_correct")) for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_changed_answer_rate"] = sum(
                bool(row.get("repair_changed_answer")) for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_target_selection_rate"] = sum(
                row.get("repair_selected_candidate_source") == "target" for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_full_oracle_accuracy"] = sum(
                bool(row.get("repair_full_oracle_correct")) for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_repair_help_rate"] = sum(
                (not bool(row.get("repair_pre_correct"))) and bool(row.get("correct")) for row in rows
            ) / max(len(rows), 1)
            results[f"{method}_repair_harm_rate"] = sum(
                bool(row.get("repair_pre_correct")) and not bool(row.get("correct")) for row in rows
            ) / max(len(rows), 1)
    add_paired_prediction_summary(results, records)
    return results


def write_markdown_summary(results: dict[str, float], output_md: str | pathlib.Path) -> None:
    target = float(results.get("target_alone", 0.0))
    lines = [
        "# Process Repair Route Summary",
        "",
        "| Method | Accuracy | Delta vs target | Pre-repair accuracy | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("selected_route_no_repair", "target_self_repair", "process_repair_selected_route"):
        if method not in results:
            continue
        prefix = f"paired_{method}_vs_target_alone"
        lines.append("| {method} | {acc:.4f} | {delta:+.4f} | {pre:.4f} | {method_only:.0f} | {baseline_only:.0f} | {both_correct:.0f} | {both_wrong:.0f} | {changed:.4f} | {help_rate:.4f} | {harm_rate:.4f} | {target_selected:.4f} | {oracle:.4f} |".format(
            method=method,
            acc=float(results.get(method, 0.0)),
            delta=float(results.get(f"{prefix}_delta_accuracy", float(results.get(method, 0.0)) - target)),
            pre=float(results.get(f"{method}_pre_repair_accuracy", results.get(method, 0.0))),
            method_only=float(results.get(f"{prefix}_method_only", 0.0)),
            baseline_only=float(results.get(f"{prefix}_baseline_only", 0.0)),
            both_correct=float(results.get(f"{prefix}_both_correct", 0.0)),
            both_wrong=float(results.get(f"{prefix}_both_wrong", 0.0)),
            changed=float(results.get(f"{method}_changed_answer_rate", 0.0)),
            help_rate=float(results.get(f"{method}_repair_help_rate", 0.0)),
            harm_rate=float(results.get(f"{method}_repair_harm_rate", 0.0)),
            target_selected=float(results.get(f"{method}_target_selection_rate", 0.0)),
            oracle=float(results.get(f"{method}_full_oracle_accuracy", 0.0)),
        ))
    lines.extend([
        "",
        "Interpretation:",
        "",
        "This ablation selects a route with an existing non-oracle selector, then asks the target model to audit and repair the selected reasoning. Optional controls log selected-route no-repair and target self-repair under the same repair prompt, so gains can be attributed to candidate generation, target-side repair, or their combination.",
    ])
    pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair selected stochastic route candidates.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--method", default="rotalign_kv_gate_0.10")
    parser.add_argument("--baseline-method", default="target_alone")
    parser.add_argument("--selection-policy", default="target_on_strict_format")
    parser.add_argument(
        "--control-arms",
        nargs="*",
        choices=("selected_no_repair", "target_self_repair"),
        default=[],
        help="Optional same-prompt control arms to log alongside selected-route repair.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-candidate-chars", type=int, default=1200)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record_sets = [load_records(path) for path in args.inputs]
    examples = load_generation(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=_torch_dtype(args.dtype)).to(args.device)
    model.eval()

    def response_fn(_idx: int, prompt: str, _selected: dict[str, Any], _candidates: list[dict[str, Any]]) -> str:
        return _generate_repair_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=bool(args.use_chat_template),
            enable_thinking=bool(args.enable_thinking),
        )

    records = process_repair_records(
        record_sets,
        examples=examples,
        method=args.method,
        baseline_method=args.baseline_method,
        response_fn=response_fn,
        selection_policy=args.selection_policy,
        max_candidate_chars=args.max_candidate_chars,
        control_arms=tuple(args.control_arms),
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
            "selection_policy": args.selection_policy,
            "control_arms": list(args.control_arms),
            "model": args.model,
        },
    )
    if args.output_md:
        write_markdown_summary(results, args.output_md)
    for key, value in sorted(results.items()):
        if not key.startswith("paired_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
