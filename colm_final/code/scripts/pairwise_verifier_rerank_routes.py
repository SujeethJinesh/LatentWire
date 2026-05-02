"""Use a target-model pairwise tournament to rerank stochastic route candidates.

This ablation is intentionally bounded: it replaces listwise scoring with a
pairwise elimination tournament over a shuffled candidate order. Each pair is
presented in a randomized left/right orientation to reduce fixed-position bias.
Telemetry is kept explicit so failures can be audited later.
"""

from __future__ import annotations

import argparse
import collections
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
    _reranked_record,
    _rows_by_index,
)


_PAIRWISE_LEFT_RIGHT_RE = re.compile(r"\b(left|right)\b", re.IGNORECASE)
_PAIRWISE_LABELS = ("L", "R")


def _truncate_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def build_pairwise_prompt(
    *,
    problem: str,
    left_candidate: dict[str, Any],
    right_candidate: dict[str, Any],
    max_candidate_chars: int = 700,
) -> str:
    lines = [
        "You are a strict math answer verifier.",
        "Compare two candidate model outputs for the same problem and choose the one more likely to have the correct final numeric answer.",
        "Check arithmetic, units, and whether the final answer follows from the solution. Do not prefer a candidate just because it is longer.",
        "Reply with exactly one letter: L or R.",
        "",
        "Problem:",
        problem.strip(),
        "",
        "Left candidate:",
        f"source={left_candidate.get('candidate_source', '')}; extracted_final_answer={left_candidate.get('normalized_prediction') or ''!r}",
        f"solution={_truncate_text(str(left_candidate.get('prediction', '')), max_candidate_chars)}",
        "",
        "Right candidate:",
        f"source={right_candidate.get('candidate_source', '')}; extracted_final_answer={right_candidate.get('normalized_prediction') or ''!r}",
        f"solution={_truncate_text(str(right_candidate.get('prediction', '')), max_candidate_chars)}",
        "",
        "Best candidate letter:",
    ]
    return "\n".join(lines)


def order_pairwise_candidates(
    candidates: list[dict[str, Any]],
    *,
    example_index: int,
    pair_order_seed: int,
) -> list[dict[str, Any]]:
    ordered = [dict(row) for row in candidates]
    rng = random.Random(f"{pair_order_seed}:{example_index}")
    rng.shuffle(ordered)
    return ordered


def parse_pairwise_winner(text: str) -> int | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    first = stripped[0].upper()
    if first == "L":
        return 0
    if first == "R":
        return 1
    lowered = stripped.lower()
    if "left candidate" in lowered or lowered.startswith("left"):
        return 0
    if "right candidate" in lowered or lowered.startswith("right"):
        return 1
    match = _PAIRWISE_LEFT_RIGHT_RE.search(stripped)
    if match:
        return 0 if match.group(1).lower() == "left" else 1
    return None


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def _pairwise_fallback(
    left_candidate: dict[str, Any],
    right_candidate: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    if policy == "left_on_invalid_response":
        return left_candidate
    if policy == "target_on_invalid_response":
        if left_candidate.get("candidate_source") == "target":
            return left_candidate
        if right_candidate.get("candidate_source") == "target":
            return right_candidate
        return left_candidate
    raise ValueError(f"Unknown fallback policy: {policy}")


@torch.no_grad()
def _generate_pairwise_response(
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


def _select_by_win_counts(
    *,
    candidates: list[dict[str, Any]],
    win_counts: collections.Counter[str],
    order_rank: dict[str, int],
) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            int(win_counts.get(str(row.get("candidate_source")), 0)),
            row.get("candidate_source") == "target",
            -int(order_rank.get(str(row.get("candidate_source")), 0)),
        ),
    )


def pairwise_verifier_rerank_records(
    record_sets: list[list[dict[str, Any]]],
    *,
    examples: Sequence[Any],
    method: str,
    response_fn: Callable[[int, str, dict[str, Any], dict[str, Any]], str],
    baseline_method: str = "target_alone",
    fallback_policy: str = "target_on_invalid_response",
    max_candidate_chars: int = 700,
    pair_order_seed: int = 0,
    max_pairwise_comparisons: int | None = None,
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
        ordered_candidates = order_pairwise_candidates(
            candidates,
            example_index=idx,
            pair_order_seed=pair_order_seed,
        )
        order_rank = {str(row.get("candidate_source")): rank for rank, row in enumerate(ordered_candidates)}
        win_counts: collections.Counter[str] = collections.Counter(
            {str(row.get("candidate_source")): 0 for row in ordered_candidates}
        )
        comparison_logs: list[dict[str, Any]] = []
        active = [dict(row) for row in ordered_candidates]
        comparison_index = 0
        round_index = 0
        budget = max_pairwise_comparisons if max_pairwise_comparisons is not None else max(len(candidates) - 1, 0)
        budget_exhausted = False

        while len(active) > 1 and comparison_index < budget:
            next_round: list[dict[str, Any]] = []
            for pair_index in range(0, len(active), 2):
                if comparison_index >= budget:
                    budget_exhausted = True
                    next_round.extend(active[pair_index:])
                    break
                left_candidate = active[pair_index]
                if pair_index + 1 >= len(active):
                    next_round.append(left_candidate)
                    continue
                right_candidate = active[pair_index + 1]
                match_seed = random.Random(f"{pair_order_seed}:{idx}:{round_index}:{pair_index}")
                if match_seed.random() < 0.5:
                    prompt_left, prompt_right = left_candidate, right_candidate
                    prompt_left_label, prompt_right_label = "L", "R"
                else:
                    prompt_left, prompt_right = right_candidate, left_candidate
                    prompt_left_label, prompt_right_label = "L", "R"
                prompt = build_pairwise_prompt(
                    problem=examples[idx].prompt,
                    left_candidate=prompt_left,
                    right_candidate=prompt_right,
                    max_candidate_chars=max_candidate_chars,
                )
                raw_response = response_fn(idx, prompt, prompt_left, prompt_right)
                choice_idx = parse_pairwise_winner(raw_response)
                fallback_used = choice_idx is None
                if choice_idx is None:
                    chosen = _pairwise_fallback(prompt_left, prompt_right, fallback_policy)
                    parsed_winner_label = None
                else:
                    chosen = prompt_left if choice_idx == 0 else prompt_right
                    parsed_winner_label = _PAIRWISE_LABELS[choice_idx]
                loser = prompt_right if chosen is prompt_left else prompt_left
                chosen_source = str(chosen.get("candidate_source"))
                loser_source = str(loser.get("candidate_source"))
                win_counts[chosen_source] += 1
                next_round.append(chosen)
                comparison_logs.append(
                    {
                        "comparison_index": comparison_index,
                        "round_index": round_index,
                        "pair_index": pair_index // 2,
                        "pair_order_seed": int(pair_order_seed),
                        "pair_order_sources": [str(row.get("candidate_source")) for row in active],
                        "compared_sources": [str(prompt_left.get("candidate_source")), str(prompt_right.get("candidate_source"))],
                        "compared_source_ranks": [order_rank[str(prompt_left.get("candidate_source"))], order_rank[str(prompt_right.get("candidate_source"))]],
                        "compared_labels": [prompt_left_label, prompt_right_label],
                        "raw_response": raw_response,
                        "parsed_winner_label": parsed_winner_label,
                        "parsed_winner_source": None if choice_idx is None else chosen_source,
                        "fallback_used": bool(fallback_used),
                        "fallback_policy": fallback_policy if fallback_used else None,
                        "winner_source": chosen_source,
                        "loser_source": loser_source,
                        "winner_was_target": chosen_source == "target",
                        "target_was_left": prompt_left.get("candidate_source") == "target",
                        "target_was_right": prompt_right.get("candidate_source") == "target",
                    }
                )
                comparison_index += 1
            active = next_round
            round_index += 1

        selected = _select_by_win_counts(candidates=candidates, win_counts=win_counts, order_rank=order_rank)
        selected_source = str(selected.get("candidate_source"))
        record = _reranked_record(
            method_name="pairwise_verifier_tournament",
            policy="single_elimination_pairwise_tournament",
            chosen=selected,
            baseline=baseline,
            candidates=candidates,
        )
        record["pairwise_tournament_seed"] = int(pair_order_seed)
        record["pairwise_candidate_order_sources"] = [str(row.get("candidate_source")) for row in ordered_candidates]
        record["pairwise_candidate_order_ranks"] = order_rank
        record["pairwise_comparisons"] = comparison_logs
        record["pairwise_win_counts"] = {str(row.get("candidate_source")): int(win_counts.get(str(row.get("candidate_source")), 0)) for row in candidates}
        record["pairwise_total_comparisons"] = int(comparison_index)
        record["pairwise_budget_limit"] = None if max_pairwise_comparisons is None else int(max_pairwise_comparisons)
        record["pairwise_budget_exhausted"] = bool(budget_exhausted)
        record["pairwise_selected_candidate_source"] = selected_source
        record["pairwise_selected_winner_by_wins"] = selected_source == max(
            win_counts,
            key=lambda source: (
                int(win_counts[source]),
                source == "target",
                -int(order_rank.get(source, 0)),
            ),
        )
        record["pairwise_fallback_rate"] = (
            sum(bool(row["fallback_used"]) for row in comparison_logs) / max(len(comparison_logs), 1)
        )
        record["pairwise_target_was_left_rate"] = (
            sum(bool(row["target_was_left"]) for row in comparison_logs) / max(len(comparison_logs), 1)
        )
        record["pairwise_target_was_right_rate"] = (
            sum(bool(row["target_was_right"]) for row in comparison_logs) / max(len(comparison_logs), 1)
        )
        output.append(record)
    return output


def summarize_results(records: list[dict[str, Any]]) -> dict[str, float]:
    results: dict[str, float] = {}
    for method in sorted({str(record["method"]) for record in records}):
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)
        if method == "pairwise_verifier_tournament":
            results["pairwise_verifier_tournament_fallback_rate"] = sum(
                float(row.get("pairwise_fallback_rate", 0.0)) for row in rows
            ) / max(len(rows), 1)
            results["pairwise_verifier_tournament_target_selection_rate"] = sum(
                row.get("selected_candidate_source") == "target" for row in rows
            ) / max(len(rows), 1)
            results["pairwise_verifier_tournament_seed_selection_rate"] = sum(
                row.get("selected_candidate_source") != "target" for row in rows
            ) / max(len(rows), 1)
            results["pairwise_verifier_tournament_target_was_left_rate"] = sum(
                float(row.get("pairwise_target_was_left_rate", 0.0)) for row in rows
            ) / max(len(rows), 1)
            results["pairwise_verifier_tournament_target_was_right_rate"] = sum(
                float(row.get("pairwise_target_was_right_rate", 0.0)) for row in rows
            ) / max(len(rows), 1)
            results["pairwise_verifier_tournament_avg_comparisons"] = sum(
                float(row.get("pairwise_total_comparisons", 0.0)) for row in rows
            ) / max(len(rows), 1)
    add_paired_prediction_summary(results, records)
    return results


def write_markdown_summary(results: dict[str, float], output_md: str | pathlib.Path) -> None:
    method = "pairwise_verifier_tournament"
    target = float(results.get("target_alone", 0.0))
    prefix = f"paired_{method}_vs_target_alone"
    lines = [
        "# Pairwise Verifier Tournament Summary",
        "",
        "| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Fallback rate | Target selected | Seed selected | Target was left | Target was right | Avg comparisons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {method} | {acc:.4f} | {delta:+.4f} | {method_only:.0f} | {baseline_only:.0f} | {both_correct:.0f} | {both_wrong:.0f} | {fallback:.4f} | {target_selected:.4f} | {seed_selected:.4f} | {target_left:.4f} | {target_right:.4f} | {avg_comparisons:.2f} |".format(
            method=method,
            acc=float(results.get(method, 0.0)),
            delta=float(results.get(f"{prefix}_delta_accuracy", float(results.get(method, 0.0)) - target)),
            method_only=float(results.get(f"{prefix}_method_only", 0.0)),
            baseline_only=float(results.get(f"{prefix}_baseline_only", 0.0)),
            both_correct=float(results.get(f"{prefix}_both_correct", 0.0)),
            both_wrong=float(results.get(f"{prefix}_both_wrong", 0.0)),
            fallback=float(results.get("pairwise_verifier_tournament_fallback_rate", 0.0)),
            target_selected=float(results.get("pairwise_verifier_tournament_target_selection_rate", 0.0)),
            seed_selected=float(results.get("pairwise_verifier_tournament_seed_selection_rate", 0.0)),
            target_left=float(results.get("pairwise_verifier_tournament_target_was_left_rate", 0.0)),
            target_right=float(results.get("pairwise_verifier_tournament_target_was_right_rate", 0.0)),
            avg_comparisons=float(results.get("pairwise_verifier_tournament_avg_comparisons", 0.0)),
        ),
        "",
        "Interpretation:",
        "",
        "This ablation converts the verifier into a bounded pairwise tournament over a shuffled candidate order,",
        "with left/right orientation randomized per match. The raw responses, parsed winners, fallback flags,",
        "pair order seed, and win counts are logged so the remaining failure mode can be separated into candidate",
        "quality, pairwise order bias, and tournament aggregation error.",
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
    parser.add_argument("--fallback-policy", default="target_on_invalid_response")
    parser.add_argument("--max-candidate-chars", type=int, default=700)
    parser.add_argument("--pair-order-seed", type=int, default=0)
    parser.add_argument("--max-pairwise-comparisons", type=int, default=None)
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

    def response_fn(_idx: int, prompt: str, _left: dict[str, Any], _right: dict[str, Any]) -> str:
        return _generate_pairwise_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=bool(args.use_chat_template),
            enable_thinking=bool(args.enable_thinking),
        )

    records = pairwise_verifier_rerank_records(
        record_sets,
        examples=examples,
        method=args.method,
        baseline_method=args.baseline_method,
        response_fn=response_fn,
        fallback_policy=args.fallback_policy,
        max_candidate_chars=args.max_candidate_chars,
        pair_order_seed=int(args.pair_order_seed),
        max_pairwise_comparisons=args.max_pairwise_comparisons,
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
            "pair_order_seed": int(args.pair_order_seed),
            "max_pairwise_comparisons": args.max_pairwise_comparisons,
        },
    )
    if args.output_md:
        write_markdown_summary(results, args.output_md)
    for key, value in sorted(results.items()):
        if not key.startswith("paired_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
