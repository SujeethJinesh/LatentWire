#!/usr/bin/env python3
"""Audit source-side answer margins on frozen SVAMP32 residual IDs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.evaluate import (
    _extract_prediction_numeric_answer,
    _extract_reference_numeric_answer,
    _format_prompt_for_tokenizer,
    _generation_example_id,
    _source_reasoning_prompt,
    load_generation,
)


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def _method_records(path: pathlib.Path, method: str) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    selected = [row for row in records if str(row.get("method")) == method]
    if selected:
        return selected
    aliases = {"c2c_generate": "c2c"}
    alias = aliases.get(method)
    if alias is not None:
        selected = [row for row in records if str(row.get("method")) == alias]
    if not selected:
        available = sorted({str(row.get("method")) for row in records})
        raise KeyError(f"Method {method!r} not found in {path}; available={available}")
    return selected


def _load_target_ids(path: pathlib.Path) -> dict[str, list[str]]:
    payload = _read_json(path)
    ids = payload.get("ids", {})
    clean = [str(value) for value in ids.get("clean_residual_targets", [])]
    teacher_only = [str(value) for value in ids.get("teacher_only", [])]
    target_self = [str(value) for value in ids.get("target_self_repair", [])]
    if not clean:
        raise ValueError(f"No ids.clean_residual_targets in {path}")
    return {
        "clean_residual_targets": clean,
        "teacher_only": teacher_only,
        "target_self_repair": target_self,
    }


def _numeric_answer(answers: Sequence[str]) -> str:
    for answer in answers:
        numeric = _extract_reference_numeric_answer(str(answer))
        if numeric is not None:
            return numeric
    raise ValueError(f"Could not extract numeric answer from {answers!r}")


def _prediction_numeric(row: dict[str, Any]) -> str | None:
    normalized = str(row.get("normalized_prediction") or "").strip()
    if normalized:
        return normalized
    return _extract_prediction_numeric_answer(str(row.get("prediction", "")))


def _answer_continuation(answer: str, template: str) -> str:
    if "{answer}" not in template:
        raise ValueError("continuation template must contain {answer}")
    return template.format(answer=answer)


def _bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _score_continuation_logprob(
    *,
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: str,
    use_chat_template: bool,
    enable_thinking: bool | None,
) -> float:
    import torch

    context = _format_prompt_for_tokenizer(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    continuation_ids = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    if continuation_ids.shape[1] == 0:
        raise ValueError(f"Continuation tokenized to zero tokens: {continuation!r}")
    full_ids = torch.cat([context_ids, continuation_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits[0, :-1, :].log_softmax(dim=-1)
    targets = full_ids[0, 1:]
    start = max(context_ids.shape[1] - 1, 0)
    stop = start + continuation_ids.shape[1]
    token_logp = logits[start:stop].gather(1, targets[start:stop].unsqueeze(-1)).squeeze(-1)
    return float(token_logp.sum())


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    clean_ids: set[str],
    target_self_ids: set[str],
    min_margin_delta: float,
) -> dict[str, Any]:
    clean_rows = [row for row in rows if row["example_id"] in clean_ids]
    target_self_rows = [row for row in rows if row["example_id"] in target_self_ids]
    source_final_clean = [
        row for row in clean_rows if bool(row.get("source_alone_correct"))
    ]
    text_final_clean = [row for row in clean_rows if bool(row.get("text_to_text_correct"))]
    source_text_union_clean = [
        row
        for row in clean_rows
        if bool(row.get("source_alone_correct")) or bool(row.get("text_to_text_correct"))
    ]
    source_margin_positive = [
        row for row in clean_rows if float(row["source_margin"]) > 0.0
    ]
    target_margin_positive = [
        row for row in clean_rows if float(row["target_margin"]) > 0.0
    ]
    source_advantage = [
        row
        for row in clean_rows
        if float(row["source_margin"]) > float(row["target_margin"]) + min_margin_delta
    ]
    source_positive_advantage = [
        row
        for row in clean_rows
        if float(row["source_margin"]) > 0.0
        and float(row["source_margin"]) > float(row["target_margin"]) + min_margin_delta
    ]
    return {
        "clean_ids_scored": len(clean_rows),
        "target_self_ids_scored": len(target_self_rows),
        "source_final_clean_correct_count": len(source_final_clean),
        "source_final_clean_correct_ids": [row["example_id"] for row in source_final_clean],
        "text_final_clean_correct_count": len(text_final_clean),
        "text_final_clean_correct_ids": [row["example_id"] for row in text_final_clean],
        "source_text_union_clean_correct_count": len(source_text_union_clean),
        "source_text_union_clean_correct_ids": [
            row["example_id"] for row in source_text_union_clean
        ],
        "source_margin_positive_clean_count": len(source_margin_positive),
        "source_margin_positive_clean_ids": [
            row["example_id"] for row in source_margin_positive
        ],
        "target_margin_positive_clean_count": len(target_margin_positive),
        "target_margin_positive_clean_ids": [
            row["example_id"] for row in target_margin_positive
        ],
        "source_margin_advantage_clean_count": len(source_advantage),
        "source_margin_advantage_clean_ids": [row["example_id"] for row in source_advantage],
        "source_margin_positive_advantage_clean_count": len(source_positive_advantage),
        "source_margin_positive_advantage_clean_ids": [
            row["example_id"] for row in source_positive_advantage
        ],
        "mean_source_margin_clean": _mean(
            [float(row["source_margin"]) for row in clean_rows]
        ),
        "mean_target_margin_clean": _mean(
            [float(row["target_margin"]) for row in clean_rows]
        ),
        "mean_source_minus_target_margin_clean": _mean(
            [float(row["source_minus_target_margin"]) for row in clean_rows]
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# SVAMP32 Source Margin Audit",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- clean IDs scored: `{summary['clean_ids_scored']}`",
        f"- source/text final clean correct: `{summary['source_text_union_clean_correct_count']}/{summary['clean_ids_scored']}`",
        f"- source-margin positive clean IDs: `{summary['source_margin_positive_clean_count']}/{summary['clean_ids_scored']}`",
        f"- source-margin positive+advantage clean IDs: `{summary['source_margin_positive_advantage_clean_count']}/{summary['clean_ids_scored']}`",
        f"- mean source margin: `{summary['mean_source_margin_clean']:.6f}`",
        f"- mean target margin: `{summary['mean_target_margin_clean']:.6f}`",
        f"- mean source-minus-target margin: `{summary['mean_source_minus_target_margin_clean']:.6f}`",
        "",
        "## Clean Residual Rows",
        "",
        "| Example ID | Gold | Distractor | Source Pred | Text Pred | Source Margin | Target Margin | Source - Target | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        if "clean_c2c_residual_target" not in row["labels"]:
            continue
        lines.append(
            "| {example_id} | {gold} | {distractor} | {source_pred} | {text_pred} | {source_margin:.6f} | {target_margin:.6f} | {delta:.6f} | {status} |".format(
                example_id=row["example_id"],
                gold=row["gold_answer"],
                distractor=row["distractor_answer"],
                source_pred=row.get("source_alone_prediction") or "n/a",
                text_pred=row.get("text_to_text_prediction") or "n/a",
                source_margin=float(row["source_margin"]),
                target_margin=float(row["target_margin"]),
                delta=float(row["source_minus_target_margin"]),
                status=row["status"],
            )
        )
    target_self_rows = [
        row for row in payload["rows"] if "target_self_repair" in row["labels"]
    ]
    if target_self_rows:
        lines.extend(
            [
                "",
                "## Target-Self-Repair Rows",
                "",
                "| Example ID | Gold | Distractor | Source Margin | Target Margin | Source - Target | Status |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in target_self_rows:
            lines.append(
                "| {example_id} | {gold} | {distractor} | {source_margin:.6f} | {target_margin:.6f} | {delta:.6f} | {status} |".format(
                    example_id=row["example_id"],
                    gold=row["gold_answer"],
                    distractor=row["distractor_answer"],
                    source_margin=float(row["source_margin"]),
                    target_margin=float(row["target_margin"]),
                    delta=float(row["source_minus_target_margin"]),
                    status=row["status"],
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--text-jsonl", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--target-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--target-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--continuation-template", default=" {answer}")
    parser.add_argument("--min-margin-delta", type=float, default=0.0)
    parser.add_argument(
        "--score-target-self",
        action="store_true",
        help="Also score ids.target_self_repair in addition to clean residual IDs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    eval_path = _resolve(args.eval_file)
    target_path = _resolve(args.target_jsonl)
    teacher_path = _resolve(args.teacher_jsonl)
    source_path = _resolve(args.source_jsonl)
    text_path = _resolve(args.text_jsonl)
    target_set_path = _resolve(args.target_set_json)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)

    examples = load_generation(str(eval_path))
    example_ids = [_generation_example_id(example) for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("eval file has duplicate stable example IDs")
    by_example_id = {
        example_id: (idx, example)
        for idx, (example_id, example) in enumerate(zip(example_ids, examples))
    }

    target_records = _by_id(_method_records(target_path, "target_alone"))
    teacher_records = _by_id(_method_records(teacher_path, "c2c_generate"))
    source_records = _by_id(_method_records(source_path, "source_alone"))
    text_records = _by_id(_method_records(text_path, "text_to_text"))
    target_ids = _load_target_ids(target_set_path)
    scored_ids = list(target_ids["clean_residual_targets"])
    if args.score_target_self:
        seen = set(scored_ids)
        scored_ids.extend(
            example_id
            for example_id in target_ids["target_self_repair"]
            if example_id not in seen
        )
    missing = [example_id for example_id in scored_ids if example_id not in by_example_id]
    if missing:
        raise ValueError(f"target-set IDs missing from eval file: {missing}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_enable_thinking = _bool_arg(args.source_enable_thinking)
    target_enable_thinking = _bool_arg(args.target_enable_thinking)
    print(f"Loading source model: {args.source_model}", flush=True)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    source_model = (
        AutoModelForCausalLM.from_pretrained(args.source_model).to(args.device).eval()
    )
    print(f"Loading target model: {args.target_model}", flush=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = (
        AutoModelForCausalLM.from_pretrained(args.target_model).to(args.device).eval()
    )

    rows: list[dict[str, Any]] = []
    for example_id in scored_ids:
        idx, example = by_example_id[example_id]
        target_row = target_records.get(example_id)
        teacher_row = teacher_records.get(example_id)
        source_row = source_records.get(example_id)
        text_row = text_records.get(example_id)
        if target_row is None or teacher_row is None:
            raise ValueError(f"Missing target/teacher record for {example_id}")
        if source_row is None or text_row is None:
            raise ValueError(f"Missing source/text record for {example_id}")
        gold_answer = _numeric_answer(example.answers)
        distractor_answer = _prediction_numeric(target_row) or ""
        if not distractor_answer or distractor_answer == gold_answer:
            distractor_answer = _prediction_numeric(teacher_row) or "0"
        if distractor_answer == gold_answer:
            raise ValueError(f"Could not find non-gold distractor for {example_id}")

        gold_continuation = _answer_continuation(gold_answer, args.continuation_template)
        distractor_continuation = _answer_continuation(
            distractor_answer,
            args.continuation_template,
        )
        source_prompt = _source_reasoning_prompt(
            example.prompt,
            args.source_reasoning_mode,
        )
        source_gold = _score_continuation_logprob(
            model=source_model,
            tokenizer=source_tokenizer,
            prompt=source_prompt,
            continuation=gold_continuation,
            device=args.device,
            use_chat_template=args.source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        source_distractor = _score_continuation_logprob(
            model=source_model,
            tokenizer=source_tokenizer,
            prompt=source_prompt,
            continuation=distractor_continuation,
            device=args.device,
            use_chat_template=args.source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        target_gold = _score_continuation_logprob(
            model=target_model,
            tokenizer=target_tokenizer,
            prompt=example.prompt,
            continuation=gold_continuation,
            device=args.device,
            use_chat_template=args.target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        target_distractor = _score_continuation_logprob(
            model=target_model,
            tokenizer=target_tokenizer,
            prompt=example.prompt,
            continuation=distractor_continuation,
            device=args.device,
            use_chat_template=args.target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        source_margin = float(source_gold - source_distractor)
        target_margin = float(target_gold - target_distractor)
        source_minus_target = float(source_margin - target_margin)
        labels = []
        if example_id in target_ids["clean_residual_targets"]:
            labels.append("clean_c2c_residual_target")
        if example_id in target_ids["target_self_repair"]:
            labels.append("target_self_repair")
        status = (
            "source_positive_advantage"
            if source_margin > 0.0
            and source_margin > target_margin + float(args.min_margin_delta)
            else "no_source_margin_advantage"
        )
        rows.append(
            {
                "index": idx,
                "example_id": example_id,
                "labels": labels,
                "gold_answer": gold_answer,
                "distractor_answer": distractor_answer,
                "target_prediction": _prediction_numeric(target_row),
                "teacher_prediction": _prediction_numeric(teacher_row),
                "source_alone_prediction": _prediction_numeric(source_row),
                "text_to_text_prediction": _prediction_numeric(text_row),
                "target_alone_correct": bool(target_row.get("correct")),
                "teacher_correct": bool(teacher_row.get("correct")),
                "source_alone_correct": bool(source_row.get("correct")),
                "text_to_text_correct": bool(text_row.get("correct")),
                "source_gold_logprob": float(source_gold),
                "source_distractor_logprob": float(source_distractor),
                "source_margin": source_margin,
                "target_gold_logprob": float(target_gold),
                "target_distractor_logprob": float(target_distractor),
                "target_margin": target_margin,
                "source_minus_target_margin": source_minus_target,
                "status": status,
            }
        )
        print(
            f"{example_id}: source_margin={source_margin:.4f} "
            f"target_margin={target_margin:.4f} delta={source_minus_target:.4f}",
            flush=True,
        )

    summary = _summarize_rows(
        rows,
        clean_ids=set(target_ids["clean_residual_targets"]),
        target_self_ids=set(target_ids["target_self_repair"]),
        min_margin_delta=float(args.min_margin_delta),
    )
    status = (
        "source_margin_signal_candidate"
        if summary["source_margin_positive_advantage_clean_count"] >= 2
        else "no_source_margin_signal"
    )
    payload = {
        "date": date.today().isoformat(),
        "status": status,
        "artifacts": {
            "eval_file": _display_path(eval_path),
            "target_jsonl": _display_path(target_path),
            "teacher_jsonl": _display_path(teacher_path),
            "source_jsonl": _display_path(source_path),
            "text_jsonl": _display_path(text_path),
            "target_set_json": _display_path(target_set_path),
        },
        "config": {
            "source_model": args.source_model,
            "target_model": args.target_model,
            "source_reasoning_mode": args.source_reasoning_mode,
            "source_use_chat_template": bool(args.source_use_chat_template),
            "target_use_chat_template": bool(args.target_use_chat_template),
            "source_enable_thinking": source_enable_thinking,
            "target_enable_thinking": target_enable_thinking,
            "continuation_template": args.continuation_template,
            "min_margin_delta": float(args.min_margin_delta),
        },
        "ids": target_ids,
        "summary": summary,
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_md, payload)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
