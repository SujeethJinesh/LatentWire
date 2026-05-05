#!/usr/bin/env python3
"""Evaluate open-loop C2C candidate-pool delta packets on SVAMP32.

This is a C2C-distillation capacity gate, not a deployable source-private
receiver. It scores public numeric answer candidates with the target model and
the repaired C2C teacher without conditioning on the teacher-generated prefix.
The packet is a few quantized C2C-minus-target candidate-score deltas in public
candidate order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.c2c_eval import build_c2c_messages, load_c2c_model
from latent_bridge.evaluate import (
    _extract_prediction_numeric_answer,
    _extract_reference_numeric_answer,
    _generation_example_id,
    _generation_match,
    load_generation,
)


@dataclass(frozen=True)
class Candidate:
    value: str
    numeric_value: float
    origins: tuple[str, ...]


@dataclass(frozen=True)
class RowScores:
    index: int
    example_id: str
    answers: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    target_scores: tuple[float, ...]
    teacher_scores: tuple[float, ...]
    packet_values: tuple[float, ...]
    packet_quantized: tuple[int, ...]
    packet_scale: float


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


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example IDs: {sorted(duplicates)}")
    return out


def _load_method_records(path: pathlib.Path, method: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    selected = [row for row in rows if str(row.get("method")) == method]
    if not selected and method == "c2c_generate":
        selected = [row for row in rows if str(row.get("method")) == "c2c"]
    if not selected:
        available = sorted({str(row.get("method")) for row in rows})
        raise KeyError(f"Method {method!r} not found in {path}; available={available}")
    return _by_id(selected)


def _load_target_ids(path: pathlib.Path) -> dict[str, set[str]]:
    payload = _read_json(path)
    ids = payload.get("ids", {})
    return {
        "teacher_only": {str(value) for value in ids.get("teacher_only", [])},
        "clean_residual_targets": {
            str(value) for value in ids.get("clean_residual_targets", [])
        },
    }


def _canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        dec = Decimal(raw)
    except InvalidOperation:
        return None
    if dec == dec.to_integral_value():
        return str(int(dec))
    normalized = format(dec.normalize(), "f")
    return normalized.rstrip("0").rstrip(".")


def _numeric_float(value: str) -> float:
    return float(Decimal(str(value)))


def _prediction_number(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    normalized = _canonical_number(str(row.get("normalized_prediction") or ""))
    if normalized is not None:
        return normalized
    return _canonical_number(_extract_prediction_numeric_answer(str(row.get("prediction", ""))))


def _gold_number(answers: Sequence[str]) -> str:
    for answer in answers:
        normalized = _canonical_number(_extract_reference_numeric_answer(str(answer)))
        if normalized is not None:
            return normalized
    raise ValueError(f"Could not find numeric gold answer in {answers!r}")


def _candidate_pool(
    *,
    answers: Sequence[str],
    rows: dict[str, dict[str, Any] | None],
) -> tuple[Candidate, ...]:
    by_value: dict[str, set[str]] = {}
    gold = _gold_number(answers)
    by_value.setdefault(gold, set()).add("gold")
    for origin, row in rows.items():
        value = _prediction_number(row)
        if value is not None:
            by_value.setdefault(value, set()).add(origin)
    candidates = [
        Candidate(
            value=value,
            numeric_value=_numeric_float(value),
            origins=tuple(sorted(origins)),
        )
        for value, origins in by_value.items()
    ]
    return tuple(sorted(candidates, key=lambda item: (item.numeric_value, item.value)))


def _continuation_ids(tokenizer: Any, value: str, *, template: str, device: str) -> torch.Tensor:
    if "{answer}" not in template:
        raise ValueError("continuation template must include {answer}")
    text = template.format(answer=value)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if encoded.shape[1] == 0:
        raise ValueError(f"Continuation tokenized to zero tokens: {text!r}")
    return encoded.to(device)


def _row_zscores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    mean = statistics.fmean(float(score) for score in scores)
    if len(scores) == 1:
        return [0.0]
    variance = statistics.fmean((float(score) - mean) ** 2 for score in scores)
    std = math.sqrt(max(variance, 1e-12))
    return [(float(score) - mean) / std for score in scores]


def _quantize(values: Sequence[float], *, coeff_bits: int) -> tuple[list[int], list[float], float]:
    if not values:
        return [], [], 0.0
    if coeff_bits <= 0:
        return [0 for _ in values], [float(value) for value in values], 0.0
    levels = max((1 << (int(coeff_bits) - 1)) - 1, 1)
    max_abs = max(abs(float(value)) for value in values)
    scale = max_abs / levels if max_abs > 0.0 else 1.0
    quantized = [
        max(-levels, min(levels, int(round(float(value) / scale))))
        for value in values
    ]
    decoded = [float(value * scale) for value in quantized]
    return quantized, decoded, float(scale)


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        build_c2c_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _c2c_kv_index_for_scoring(
    *,
    prompt_len: int,
    continuation_len: int,
    device: str | torch.device,
) -> list[torch.Tensor]:
    if prompt_len <= 1:
        return [
            torch.tensor([[-1, 0]], dtype=torch.long)
            .repeat(1, max(int(prompt_len) + int(continuation_len), 1), 1)
            .to(device)
        ]
    instruction = (
        torch.tensor([1, 0], dtype=torch.long)
        .repeat(int(prompt_len) - 1, 1)
        .unsqueeze(0)
        .to(device)
    )
    tail = (
        torch.tensor([-1, 0], dtype=torch.long)
        .repeat(1 + int(continuation_len), 1)
        .unsqueeze(0)
        .to(device)
    )
    return [instruction, tail]


def _sequence_mean_logprob(logits: torch.Tensor, full_ids: torch.Tensor, *, start: int, length: int) -> float:
    logprobs = logits[0, :-1, :].float().log_softmax(dim=-1)
    targets = full_ids[0, 1:]
    stop = int(start) + int(length)
    token_logp = logprobs[start:stop].gather(1, targets[start:stop].unsqueeze(-1)).squeeze(-1)
    return float(token_logp.mean().detach().cpu().item())


def _section_mean_logprob(logits: torch.Tensor, continuation_ids: torch.Tensor) -> float:
    length = int(continuation_ids.shape[1])
    logprobs = logits[0, :length, :].float().log_softmax(dim=-1)
    targets = continuation_ids[0, :length]
    token_logp = logprobs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logp.mean().detach().cpu().item())


@torch.no_grad()
def _score_target_candidate(
    *,
    target_model: Any,
    input_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> float:
    full_ids = torch.cat([input_ids, continuation_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    output = target_model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    return _sequence_mean_logprob(
        output.logits,
        full_ids,
        start=max(int(input_ids.shape[1]) - 1, 0),
        length=int(continuation_ids.shape[1]),
    )


@torch.no_grad()
def _score_c2c_candidate(
    *,
    model: Any,
    input_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    device: str,
) -> float:
    full_ids = torch.cat([input_ids, continuation_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    output = model(
        kv_cache_index=_c2c_kv_index_for_scoring(
            prompt_len=int(input_ids.shape[1]),
            continuation_len=int(continuation_ids.shape[1]),
            device=device,
        ),
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    return _section_mean_logprob(output.logits, continuation_ids)


def _argmax(values: Sequence[float]) -> int:
    if not values:
        return -1
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def _candidate_correct(candidate: Candidate, answers: Sequence[str]) -> bool:
    return _generation_match(candidate.value, list(answers))


def _choose_wrong_row(rows: Sequence[RowScores], row_index: int, *, same_top: bool) -> RowScores:
    current_top = _argmax(rows[row_index].packet_values)
    for offset in range(1, len(rows)):
        candidate = rows[(row_index + offset) % len(rows)]
        if same_top and _argmax(candidate.packet_values) != current_top:
            continue
        if candidate.example_id != rows[row_index].example_id:
            return candidate
    return rows[(row_index + 1) % len(rows)]


def _condition_packet(
    rows: Sequence[RowScores],
    row_index: int,
    *,
    condition: str,
    rng: random.Random,
) -> list[float]:
    row = rows[row_index]
    values = list(row.packet_values)
    if condition in {"target_only", "zero_delta"}:
        return [0.0 for _ in values]
    if condition == "matched":
        return values
    if condition == "row_shuffle":
        other = _choose_wrong_row(rows, row_index, same_top=False)
        source = list(other.packet_values)
    elif condition == "same_top_wrong_row":
        other = _choose_wrong_row(rows, row_index, same_top=True)
        source = list(other.packet_values)
    elif condition == "candidate_roll":
        return values[1:] + values[:1] if values else values
    elif condition == "candidate_derangement":
        return list(reversed(values))
    elif condition == "coeff_shuffle":
        del rng
        return values[1:] + values[:1] if len(values) > 1 else values
    elif condition == "coeff_sign_flip":
        return [-float(value) for value in values]
    elif condition == "target_derived_packet":
        return _row_zscores(row.target_scores)
    elif condition == "teacher_top_index":
        selected = _argmax(row.teacher_scores)
        return [4.0 if index == selected else 0.0 for index in range(len(values))]
    else:
        raise ValueError(f"Unsupported condition: {condition!r}")
    if not values:
        return []
    if not source:
        return [0.0 for _ in values]
    return [float(source[index % len(source)]) for index in range(len(values))]


def evaluate_conditions(
    rows: Sequence[RowScores],
    *,
    target_ids: dict[str, set[str]],
    conditions: Sequence[str],
    rng_seed: int,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    row_outputs: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    for condition in conditions:
        correct_ids: set[str] = set()
        teacher_only_ids: set[str] = set()
        clean_ids: set[str] = set()
        selected_indices: list[int] = []
        helps = 0
        harms = 0
        rng = random.Random(int(rng_seed))
        for row_index, row in enumerate(rows):
            target_z = _row_zscores(row.target_scores)
            packet = _condition_packet(rows, row_index, condition=condition, rng=rng)
            adjusted = [
                float(score) + float(delta)
                for score, delta in zip(target_z, packet, strict=True)
            ]
            selected = _argmax(adjusted)
            target_selected = _argmax(target_z)
            correct = selected >= 0 and _candidate_correct(row.candidates[selected], row.answers)
            target_correct = target_selected >= 0 and _candidate_correct(row.candidates[target_selected], row.answers)
            helps += int(correct and not target_correct)
            harms += int(target_correct and not correct)
            selected_indices.append(int(selected))
            if correct:
                correct_ids.add(row.example_id)
                if row.example_id in target_ids["teacher_only"]:
                    teacher_only_ids.add(row.example_id)
                if row.example_id in target_ids["clean_residual_targets"]:
                    clean_ids.add(row.example_id)
            row_outputs[condition].append(
                {
                    "example_id": row.example_id,
                    "selected_index": int(selected),
                    "selected_value": row.candidates[selected].value if selected >= 0 else None,
                    "target_selected_index": int(target_selected),
                    "target_selected_value": row.candidates[target_selected].value if target_selected >= 0 else None,
                    "teacher_selected_index": int(_argmax(row.teacher_scores)),
                    "teacher_selected_value": row.candidates[_argmax(row.teacher_scores)].value,
                    "correct": bool(correct),
                    "target_correct": bool(target_correct),
                    "packet": [float(value) for value in packet],
                    "adjusted_scores": [float(value) for value in adjusted],
                }
            )
        summaries[condition] = {
            "condition": condition,
            "correct_count": len(correct_ids),
            "correct_ids": sorted(correct_ids),
            "teacher_only_correct_count": len(teacher_only_ids),
            "teacher_only_correct_ids": sorted(teacher_only_ids),
            "clean_correct_count": len(clean_ids),
            "clean_correct_ids": sorted(clean_ids),
            "helps_vs_target": int(helps),
            "harms_vs_target": int(harms),
            "selection_histogram": {
                str(index): selected_indices.count(index) for index in sorted(set(selected_indices))
            },
        }
    matched_clean = set(summaries["matched"]["clean_correct_ids"])
    control_clean_union = set().union(
        *[
            set(summary["clean_correct_ids"])
            for condition, summary in summaries.items()
            if condition != "matched" and condition != "teacher_top_index"
        ]
    )
    return {
        "condition_summaries": summaries,
        "source_necessary_clean_ids": sorted(matched_clean - control_clean_union),
        "control_clean_union_ids": sorted(control_clean_union),
        "rows": row_outputs,
    }


def _packet_bits(*, candidate_count: int, coeff_bits: int, include_scale_bits: int) -> int:
    return int(candidate_count) * max(int(coeff_bits), 0) + max(int(include_scale_bits), 0)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SVAMP32 C2C Candidate-Pool Delta Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- average candidate count: `{payload['candidate_pool']['avg_candidate_count']:.2f}`",
        f"- average packet bytes per row: `{payload['packet_contract']['avg_packet_bytes_per_row']:.2f}`",
        f"- clean source-necessary IDs: `{len(payload['run']['source_necessary_clean_ids'])}`",
        "",
        "## Summary",
        "",
        "| Condition | Correct | Teacher-only | Clean | Helps | Harms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, summary in payload["run"]["condition_summaries"].items():
        lines.append(
            f"| `{condition}` | {summary['correct_count']}/{payload['reference_n']} | "
            f"{summary['teacher_only_correct_count']} | {summary['clean_correct_count']} | "
            f"{summary['helps_vs_target']} | {summary['harms_vs_target']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
            "## Claim Boundary",
            "",
            "- This is a dense-C2C-teacher capacity gate, not a deployable source-causal receiver.",
            "- The target is not conditioned on the C2C teacher-generated prefix.",
            "- The packet is derived from C2C teacher candidate scores and can expose answer-candidate preferences.",
            "- A pass would justify training a source-side or representation-side predictor for the same candidate packet.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@torch.no_grad()
def capture_rows(
    *,
    source_model: str,
    target_model: str,
    eval_file: pathlib.Path,
    target_jsonl: pathlib.Path,
    source_jsonl: pathlib.Path,
    text_jsonl: pathlib.Path,
    teacher_jsonl: pathlib.Path,
    device: str,
    max_new_tokens: int,
    coeff_bits: int,
    continuation_template: str,
    limit: int | None,
) -> tuple[list[RowScores], dict[str, Any]]:
    examples = load_generation(str(eval_file))
    if limit is not None:
        examples = examples[: int(limit)]
    target_records = _load_method_records(target_jsonl, "target_alone")
    source_records = _load_method_records(source_jsonl, "source_alone")
    text_records = _load_method_records(text_jsonl, "text_to_text")
    teacher_records = _load_method_records(teacher_jsonl, "c2c_generate")
    model, tokenizer, artifact = load_c2c_model(
        source_model=source_model,
        target_model=target_model,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    target_model_obj = model.model_list[int(getattr(model, "base_model_idx", 0))]
    rows: list[RowScores] = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        example_id = _generation_example_id(example)
        print(f"[candidate-gate] row {index + 1}/{len(examples)} {example_id}", file=sys.stderr, flush=True)
        candidates = _candidate_pool(
            answers=example.answers,
            rows={
                "target_alone": target_records.get(example_id),
                "source_alone": source_records.get(example_id),
                "text_to_text": text_records.get(example_id),
                "c2c_teacher": teacher_records.get(example_id),
            },
        )
        prompt_text = _format_prompt(tokenizer, example.prompt)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        target_scores: list[float] = []
        teacher_scores: list[float] = []
        for candidate in candidates:
            continuation_ids = _continuation_ids(
                tokenizer,
                candidate.value,
                template=continuation_template,
                device=device,
            )
            target_scores.append(
                _score_target_candidate(
                    target_model=target_model_obj,
                    input_ids=input_ids,
                    continuation_ids=continuation_ids,
                )
            )
            teacher_scores.append(
                _score_c2c_candidate(
                    model=model,
                    input_ids=input_ids,
                    continuation_ids=continuation_ids,
                    device=device,
                )
            )
        target_z = _row_zscores(target_scores)
        teacher_z = _row_zscores(teacher_scores)
        raw_delta = [teacher - target for teacher, target in zip(teacher_z, target_z, strict=True)]
        quantized, decoded, scale = _quantize(raw_delta, coeff_bits=int(coeff_bits))
        rows.append(
            RowScores(
                index=int(index),
                example_id=example_id,
                answers=tuple(str(answer) for answer in example.answers),
                candidates=candidates,
                target_scores=tuple(float(value) for value in target_scores),
                teacher_scores=tuple(float(value) for value in teacher_scores),
                packet_values=tuple(float(value) for value in decoded),
                packet_quantized=tuple(int(value) for value in quantized),
                packet_scale=float(scale),
            )
        )
    run_config = {
        "source_model": source_model,
        "target_model": target_model,
        "eval_file": _display_path(eval_file),
        "target_jsonl": _display_path(target_jsonl),
        "source_jsonl": _display_path(source_jsonl),
        "text_jsonl": _display_path(text_jsonl),
        "teacher_jsonl": _display_path(teacher_jsonl),
        "device": device,
        "max_new_tokens": int(max_new_tokens),
        "coeff_bits": int(coeff_bits),
        "continuation_template": continuation_template,
        "limit": limit,
        "elapsed_sec": float(time.perf_counter() - started),
        "published_repo_id": artifact.repo_id,
        "published_subdir": artifact.subdir,
        "published_config_path": artifact.config_path,
        "published_checkpoint_dir": artifact.checkpoint_dir,
        "local_root": artifact.local_root,
    }
    return rows, run_config


def analyze(
    *,
    rows: Sequence[RowScores],
    target_set_path: pathlib.Path,
    run_config: dict[str, Any],
    run_date: str,
    output_json: pathlib.Path,
    output_md: pathlib.Path,
) -> dict[str, Any]:
    target_ids = _load_target_ids(target_set_path)
    conditions = (
        "matched",
        "target_only",
        "zero_delta",
        "row_shuffle",
        "same_top_wrong_row",
        "candidate_roll",
        "candidate_derangement",
        "coeff_shuffle",
        "coeff_sign_flip",
        "target_derived_packet",
        "teacher_top_index",
    )
    run = evaluate_conditions(
        rows,
        target_ids=target_ids,
        conditions=conditions,
        rng_seed=52060505,
    )
    matched = run["condition_summaries"]["matched"]
    control_names = [condition for condition in conditions if condition not in {"matched", "teacher_top_index"}]
    best_control_correct = max(run["condition_summaries"][condition]["correct_count"] for condition in control_names)
    best_control_clean = max(run["condition_summaries"][condition]["clean_correct_count"] for condition in control_names)
    source_necessary_count = len(run["source_necessary_clean_ids"])
    clears = (
        matched["correct_count"] > best_control_correct
        and matched["clean_correct_count"] > best_control_clean
        and source_necessary_count >= 2
    )
    status = (
        "c2c_candidate_pool_delta_packet_capacity_clears_controls_not_deployable"
        if clears
        else "c2c_candidate_pool_delta_packet_capacity_fails_controls"
    )
    candidate_counts = [len(row.candidates) for row in rows]
    total_bits = sum(
        _packet_bits(
            candidate_count=len(row.candidates),
            coeff_bits=int(run_config["coeff_bits"]),
            include_scale_bits=8,
        )
        for row in rows
    )
    avg_bytes = float((total_bits / 8.0) / max(len(rows), 1))
    packet_contract = {
        "kind": "open_loop_public_candidate_c2c_minus_target_zscore_delta",
        "coeff_bits": int(run_config["coeff_bits"]),
        "scale_bits_per_row": 8,
        "avg_packet_bits_per_row": float(total_bits / max(len(rows), 1)),
        "avg_packet_bytes_per_row": avg_bytes,
        "avg_cacheline_rounded_bytes_per_row": float(math.ceil(avg_bytes / 64.0) * 64.0) if avg_bytes else 0.0,
        "source_state_private": True,
        "teacher_derived_not_deployable": True,
        "answer_candidate_exposure": "quantized score delta over public numeric candidates",
    }
    payload = {
        "date": run_date,
        "status": status,
        "reference_n": len(rows),
        "run_config": run_config,
        "target_set_json": _display_path(target_set_path),
        "candidate_pool": {
            "kind": "gold_plus_baseline_numeric_predictions_sorted_by_numeric_value",
            "min_candidate_count": min(candidate_counts) if candidate_counts else 0,
            "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
            "avg_candidate_count": float(statistics.fmean(candidate_counts)) if candidate_counts else 0.0,
            "gold_recall": float(
                sum(any(_candidate_correct(candidate, row.answers) for candidate in row.candidates) for row in rows)
                / max(len(rows), 1)
            ),
        },
        "packet_contract": packet_contract,
        "headline": {
            "matched_correct": matched["correct_count"],
            "best_control_correct": best_control_correct,
            "matched_clean": matched["clean_correct_count"],
            "best_control_clean": best_control_clean,
            "source_necessary_clean_count": source_necessary_count,
            "capacity_pass": bool(clears),
        },
        "run": run,
        "rows": [
            {
                "index": row.index,
                "example_id": row.example_id,
                "answers": list(row.answers),
                "candidates": [
                    {
                        "value": candidate.value,
                        "numeric_value": candidate.numeric_value,
                        "origins": list(candidate.origins),
                    }
                    for candidate in row.candidates
                ],
                "target_scores": list(row.target_scores),
                "teacher_scores": list(row.teacher_scores),
                "packet_values": list(row.packet_values),
                "packet_quantized": list(row.packet_quantized),
                "packet_scale": row.packet_scale,
            }
            for row in rows
        ],
        "decision": (
            "Promote candidate-pool delta packets to a source-predictor gate."
            if clears
            else "Do not train a source predictor for this exact candidate-delta packet unless a follow-up removes the winning control."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    manifest_md = output_json.parent / "manifest.md"
    manifest_json = output_json.parent / "manifest.json"
    manifest_md.write_text(
        "\n".join(
            [
                "# SVAMP32 C2C Candidate-Pool Delta Packet Gate Manifest",
                "",
                f"- date: `{run_date}`",
                f"- status: `{status}`",
                f"- output json: `{_display_path(output_json)}`",
                f"- output md: `{_display_path(output_md)}`",
                f"- manifest json: `{_display_path(manifest_json)}`",
                "",
                "## Interpretation",
                "",
                "- This artifact removes teacher-generated-prefix conditioning and scores numeric candidates directly.",
                "- It is still a dense-teacher capacity gate because packet values are computed from C2C candidate scores.",
                "- If matched fails against candidate-roll or wrong-row controls, the packet is not a source-causal method target.",
            ]
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "date": run_date,
        "status": status,
        "artifacts": {
            _display_path(output_json): {"sha256": _sha256(output_json)},
            _display_path(output_md): {"sha256": _sha256(output_md)},
            _display_path(manifest_md): {"sha256": _sha256(manifest_md)},
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target-jsonl", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--text-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--coeff-bits", type=int, default=4)
    parser.add_argument("--continuation-template", default=" {answer}")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    rows, run_config = capture_rows(
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        eval_file=_resolve(args.eval_file),
        target_jsonl=_resolve(args.target_jsonl),
        source_jsonl=_resolve(args.source_jsonl),
        text_jsonl=_resolve(args.text_jsonl),
        teacher_jsonl=_resolve(args.teacher_jsonl),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        coeff_bits=int(args.coeff_bits),
        continuation_template=str(args.continuation_template),
        limit=args.limit,
    )
    payload = analyze(
        rows=rows,
        target_set_path=_resolve(args.target_set_json),
        run_config=run_config,
        run_date=str(args.date),
        output_json=_resolve(args.output_json),
        output_md=_resolve(args.output_md),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_json": _display_path(_resolve(args.output_json)),
                "headline": payload["headline"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
