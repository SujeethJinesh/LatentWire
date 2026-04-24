#!/usr/bin/env python3
"""Teacher-forced source-control diagnostic for SVAMP32 connector checkpoints."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_bridge import RotAlignKVTranslator
from latent_bridge.evaluate import (
    _build_rotalign_prefix_state,
    _extract_prediction_numeric_answer,
    _extract_reference_numeric_answer,
    _generation_example_id,
    _logprob_tokens_from_prefix_state,
    _source_control_index,
    _source_reasoning_prompt,
    load_generation,
)


@dataclass(frozen=True)
class ControlRow:
    name: str
    source_prompt_control: str = "real"
    source_kv_control: str = "real"
    innovation_memory_control: str | None = None
    random_salt: int = 0


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


def _numeric_answer(answers: Sequence[str]) -> str:
    for answer in answers:
        numeric = _extract_reference_numeric_answer(str(answer))
        if numeric is not None:
            return numeric
    raise ValueError(f"Could not extract numeric answer from {answers!r}")


def _answer_continuation(answer: str, template: str) -> str:
    if "{answer}" not in template:
        raise ValueError("continuation template must contain {answer}")
    return template.format(answer=answer)


def _method_records(path: pathlib.Path, method: str) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    selected = [row for row in records if str(row.get("method")) == method]
    if selected:
        return selected
    normalized = method
    if method == "c2c_generate":
        normalized = "c2c"
    selected = [row for row in records if str(row.get("method")) == normalized]
    if not selected:
        available = sorted({str(row.get("method")) for row in records})
        raise KeyError(f"Method {method!r} not found in {path}; available={available}")
    return selected


def _load_clean_ids(path: pathlib.Path) -> list[str]:
    payload = _read_json(path)
    ids = payload.get("ids", {}).get("clean_residual_targets", [])
    if not ids:
        raise ValueError(f"No ids.clean_residual_targets in {path}")
    return [str(value) for value in ids]


def _load_target_self_ids(path: pathlib.Path) -> list[str]:
    payload = _read_json(path)
    return [str(value) for value in payload.get("ids", {}).get("target_self_repair", [])]


def _continuation_ids(tokenizer, text: str, device: str):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if encoded.shape[1] == 0:
        raise ValueError(f"Continuation tokenized to zero tokens: {text!r}")
    return encoded.to(device)


def _summarize_clean_rows(
    rows: Sequence[dict[str, Any]],
    *,
    clean_ids: set[str],
    target_self_ids: set[str],
    min_margin_delta: float,
) -> dict[str, Any]:
    clean_rows = [row for row in rows if row["example_id"] in clean_ids]
    target_self_rows = [row for row in rows if row["example_id"] in target_self_ids]
    matched_only = [
        row
        for row in clean_rows
        if row["matched_margin"] > 0.0
        and row["matched_minus_best_control_margin"] > min_margin_delta
    ]
    control_leak = [
        row
        for row in clean_rows
        if row["best_control_margin"] > 0.0
        and row["matched_margin"] <= row["best_control_margin"] + min_margin_delta
    ]
    return {
        "clean_ids_scored": len(clean_rows),
        "target_self_ids_scored": len(target_self_rows),
        "matched_positive_clean_count": int(sum(row["matched_margin"] > 0.0 for row in clean_rows)),
        "matched_only_clean_count": len(matched_only),
        "matched_only_clean_ids": [row["example_id"] for row in matched_only],
        "control_leak_clean_count": len(control_leak),
        "control_leak_clean_ids": [row["example_id"] for row in control_leak],
        "mean_matched_margin_clean": (
            sum(row["matched_margin"] for row in clean_rows) / max(len(clean_rows), 1)
        ),
        "mean_best_control_margin_clean": (
            sum(row["best_control_margin"] for row in clean_rows) / max(len(clean_rows), 1)
        ),
        "mean_matched_minus_best_control_clean": (
            sum(row["matched_minus_best_control_margin"] for row in clean_rows)
            / max(len(clean_rows), 1)
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# SVAMP32 Teacher-Forced Connector Diagnostic",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- checkpoint: `{payload['artifacts']['translator']}`",
        f"- clean IDs scored: `{summary['clean_ids_scored']}`",
        f"- matched-positive clean IDs: `{summary['matched_positive_clean_count']}`",
        f"- matched-only clean IDs: `{summary['matched_only_clean_count']}`",
        f"- control-leak clean IDs: `{summary['control_leak_clean_count']}`",
        f"- mean matched margin: `{summary['mean_matched_margin_clean']:.6f}`",
        f"- mean best-control margin: `{summary['mean_best_control_margin_clean']:.6f}`",
        f"- mean matched-minus-control margin: `{summary['mean_matched_minus_best_control_clean']:.6f}`",
        "",
        "## Clean Residual Rows",
        "",
        "| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        if "clean_c2c_residual_target" not in row["labels"]:
            continue
        lines.append(
            "| {example_id} | {gold} | {distractor} | {matched:.6f} | {control:.6f} | {delta:.6f} | {best_control} | {status} |".format(
                example_id=row["example_id"],
                gold=row["gold_answer"],
                distractor=row["distractor_answer"],
                matched=row["matched_margin"],
                control=row["best_control_margin"],
                delta=row["matched_minus_best_control_margin"],
                best_control=row["best_control"],
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
                "| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in target_self_rows:
            lines.append(
                "| {example_id} | {gold} | {distractor} | {matched:.6f} | {control:.6f} | {delta:.6f} | {best_control} | {status} |".format(
                    example_id=row["example_id"],
                    gold=row["gold_answer"],
                    distractor=row["distractor_answer"],
                    matched=row["matched_margin"],
                    control=row["best_control_margin"],
                    delta=row["matched_minus_best_control_margin"],
                    best_control=row["best_control"],
                    status=row["status"],
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--translator", required=True)
    p.add_argument("--source-model", required=True)
    p.add_argument("--target-model", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--target-jsonl", required=True)
    p.add_argument("--teacher-jsonl", required=True)
    p.add_argument("--target-set-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--source-reasoning-mode", default="brief_analysis")
    p.add_argument("--source-use-chat-template", action="store_true")
    p.add_argument("--target-use-chat-template", action="store_true")
    p.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    p.add_argument("--target-enable-thinking", choices=["true", "false"], default=None)
    p.add_argument("--fixed-gate", type=float, default=0.15)
    p.add_argument("--kv-transport", default="k_only")
    p.add_argument("--position-selection-ratio", type=float, default=0.5)
    p.add_argument("--position-selection-metric", default="attention")
    p.add_argument("--random-salt", type=int, default=1)
    p.add_argument("--continuation-template", default=" {answer}")
    p.add_argument("--min-margin-delta", type=float, default=0.0)
    p.add_argument(
        "--score-target-self",
        action="store_true",
        help="Also score ids.target_self_repair in addition to clean residual IDs.",
    )
    return p.parse_args()


def _bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def main() -> None:
    args = parse_args()
    translator_path = _resolve(args.translator)
    eval_path = _resolve(args.eval_file)
    target_path = _resolve(args.target_jsonl)
    teacher_path = _resolve(args.teacher_jsonl)
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
    clean_ids = _load_clean_ids(target_set_path)
    target_self_ids = _load_target_self_ids(target_set_path)
    scored_ids = list(clean_ids)
    if args.score_target_self:
        seen = set(scored_ids)
        scored_ids.extend(
            example_id for example_id in target_self_ids if example_id not in seen
        )
    missing = [example_id for example_id in scored_ids if example_id not in by_example_id]
    if missing:
        raise ValueError(f"target-set IDs missing from eval file: {missing}")

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
    print(f"Loading translator: {translator_path}", flush=True)
    translator = (
        RotAlignKVTranslator.load(str(translator_path), map_location=args.device)
        .to(args.device)
        .eval()
    )
    translator.set_fixed_gates(float(args.fixed_gate))
    base_memory_control = translator.config.innovation_memory_control

    controls = [
        ControlRow("matched"),
        ControlRow("zero_source", source_kv_control="zero"),
        ControlRow(
            "shuffled_source",
            source_prompt_control="shuffle_examples",
            random_salt=args.random_salt,
        ),
        ControlRow(
            "target_only",
            source_kv_control="zero",
            innovation_memory_control="target_only",
        ),
        ControlRow(
            "slots_only",
            source_kv_control="zero",
            innovation_memory_control="slots_only",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for example_id in scored_ids:
        idx, example = by_example_id[example_id]
        target_row = target_records.get(example_id)
        teacher_row = teacher_records.get(example_id)
        if target_row is None or teacher_row is None:
            raise ValueError(f"Missing target/teacher record for {example_id}")
        gold_answer = _numeric_answer(example.answers)
        distractor_answer = str(target_row.get("normalized_prediction") or "")
        if not distractor_answer or distractor_answer == gold_answer:
            distractor_answer = (
                _extract_prediction_numeric_answer(str(target_row.get("prediction", "")))
                or ""
            )
        if not distractor_answer or distractor_answer == gold_answer:
            distractor_answer = str(teacher_row.get("normalized_prediction") or "0")
        if distractor_answer == gold_answer:
            raise ValueError(f"Could not find non-gold distractor for {example_id}")
        gold_ids = _continuation_ids(
            target_tokenizer,
            _answer_continuation(gold_answer, args.continuation_template),
            args.device,
        )
        distractor_ids = _continuation_ids(
            target_tokenizer,
            _answer_continuation(distractor_answer, args.continuation_template),
            args.device,
        )

        control_scores: dict[str, dict[str, Any]] = {}
        for control in controls:
            translator.config.innovation_memory_control = (
                control.innovation_memory_control or base_memory_control
            )
            source_idx = _source_control_index(
                idx,
                len(examples),
                control.source_prompt_control,
                control.random_salt,
            )
            source_example = examples[source_idx]
            source_prompt = _source_reasoning_prompt(
                source_example.prompt,
                args.source_reasoning_mode,
            )
            prefix_state, stats = _build_rotalign_prefix_state(
                source_model,
                source_tokenizer,
                target_model,
                target_tokenizer,
                translator,
                source_prompt,
                example.prompt,
                args.device,
                True,
                "fused",
                source_kv_control=control.source_kv_control,
                kv_transport=args.kv_transport,
                position_selection_ratio=args.position_selection_ratio,
                position_selection_metric=args.position_selection_metric,
                random_salt=control.random_salt,
                source_use_chat_template=args.source_use_chat_template,
                target_use_chat_template=args.target_use_chat_template,
                source_enable_thinking=source_enable_thinking,
                target_enable_thinking=target_enable_thinking,
            )
            gold_logprob = _logprob_tokens_from_prefix_state(
                target_model,
                prefix_state,
                gold_ids,
                args.device,
            )
            distractor_logprob = _logprob_tokens_from_prefix_state(
                target_model,
                prefix_state,
                distractor_ids,
                args.device,
            )
            control_scores[control.name] = {
                "gold_logprob": float(gold_logprob),
                "distractor_logprob": float(distractor_logprob),
                "margin": float(gold_logprob - distractor_logprob),
                "source_example_id": example_ids[source_idx],
                "source_kv_control": control.source_kv_control,
                "source_prompt_control": control.source_prompt_control,
                "innovation_memory_control": translator.config.innovation_memory_control,
                "bits": float(stats.get("bits", 0.0)),
                "bytes": float(
                    stats.get("bytes", float(stats.get("bits", 0.0)) / 8.0)
                ),
            }
        translator.config.innovation_memory_control = base_memory_control

        matched_margin = control_scores["matched"]["margin"]
        control_margins = {
            name: value["margin"]
            for name, value in control_scores.items()
            if name != "matched"
        }
        best_control, best_control_margin = max(
            control_margins.items(), key=lambda item: item[1]
        )
        matched_minus_control = float(matched_margin - best_control_margin)
        labels = []
        if example_id in clean_ids:
            labels.append("clean_c2c_residual_target")
        if example_id in target_self_ids:
            labels.append("target_self_repair")
        status = (
            "matched_only_positive"
            if matched_margin > 0.0 and matched_minus_control > args.min_margin_delta
            else "control_or_negative"
        )
        rows.append(
            {
                "index": idx,
                "example_id": example_id,
                "labels": labels,
                "gold_answer": gold_answer,
                "distractor_answer": distractor_answer,
                "target_prediction": target_row.get("normalized_prediction"),
                "teacher_prediction": teacher_row.get("normalized_prediction"),
                "matched_margin": float(matched_margin),
                "best_control": best_control,
                "best_control_margin": float(best_control_margin),
                "matched_minus_best_control_margin": matched_minus_control,
                "status": status,
                "scores": control_scores,
            }
        )
        print(
            f"{example_id}: matched_margin={matched_margin:.4f} "
            f"best_control={best_control}:{best_control_margin:.4f} "
            f"delta={matched_minus_control:.4f}",
            flush=True,
        )

    summary = _summarize_clean_rows(
        rows,
        clean_ids=set(clean_ids),
        target_self_ids=set(target_self_ids),
        min_margin_delta=float(args.min_margin_delta),
    )
    status = (
        "teacher_forced_source_signal_candidate"
        if summary["matched_only_clean_count"] >= 2
        else "no_teacher_forced_source_signal"
    )
    payload = {
        "date": date.today().isoformat(),
        "status": status,
        "artifacts": {
            "translator": _display_path(translator_path),
            "eval_file": _display_path(eval_path),
            "target_jsonl": _display_path(target_path),
            "teacher_jsonl": _display_path(teacher_path),
            "target_set_json": _display_path(target_set_path),
        },
        "config": {
            "source_model": args.source_model,
            "target_model": args.target_model,
            "fixed_gate": float(args.fixed_gate),
            "kv_transport": args.kv_transport,
            "position_selection_ratio": float(args.position_selection_ratio),
            "position_selection_metric": args.position_selection_metric,
            "continuation_template": args.continuation_template,
            "min_margin_delta": float(args.min_margin_delta),
            "controls": [control.__dict__ for control in controls],
        },
        "ids": {
            "clean_residual_targets": clean_ids,
            "target_self_repair": target_self_ids,
        },
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
