#!/usr/bin/env python3
"""Evaluate sparse C2C teacher-logit delta packets on SVAMP32.

This is a packet-capacity gate, not a deployable source-causal receiver. It
uses the repaired C2C teacher to build per-step sparse logit-delta packets, then
checks whether those packets recover teacher behavior beyond destructive
controls. Passing this gate would justify the next source-side predictor; failing
it rules out this packet format before spending more method effort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.c2c_eval import build_c2c_kv_cache_index, build_c2c_messages, load_c2c_model
from latent_bridge.evaluate import (
    _generation_example_id,
    _generation_match,
    load_generation,
)


@dataclass(frozen=True)
class StepPacket:
    token_ids: tuple[int, ...]
    values: tuple[float, ...]
    quantized: tuple[int, ...]
    scale: float


@dataclass
class RowCapture:
    index: int
    example_id: str
    answers: list[str]
    c2c_tokens: list[int]
    c2c_text: str
    target_logits: torch.Tensor
    packets: list[StepPacket]


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


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_target_ids(path: pathlib.Path) -> dict[str, set[str]]:
    payload = _read_json(path)
    ids = payload.get("ids", {})
    return {
        "teacher_only": {str(value) for value in ids.get("teacher_only", [])},
        "clean_residual_targets": {
            str(value) for value in ids.get("clean_residual_targets", [])
        },
    }


def _quantize(values: torch.Tensor, *, coeff_bits: int) -> tuple[list[int], list[float], float]:
    if values.numel() == 0:
        return [], [], 0.0
    if coeff_bits <= 0:
        decoded = [float(value) for value in values.detach().cpu().tolist()]
        return [0 for _ in decoded], decoded, 0.0
    levels = max((1 << (int(coeff_bits) - 1)) - 1, 1)
    max_abs = float(values.detach().abs().max().cpu().item())
    scale = float(max_abs / levels) if max_abs > 0.0 else 1.0
    q = torch.round(values.detach().cpu() / scale).clamp(-levels, levels).to(torch.int64)
    decoded = (q.float() * scale).tolist()
    return [int(value) for value in q.tolist()], [float(value) for value in decoded], scale


def build_step_packet(
    target_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    coeff_bits: int,
    mode: str,
) -> StepPacket:
    delta = teacher_logits.detach().float().cpu() - target_logits.detach().float().cpu()
    k = min(max(int(top_k), 0), int(delta.numel()))
    if k == 0:
        return StepPacket(token_ids=(), values=(), quantized=(), scale=0.0)
    if mode == "positive":
        top = torch.topk(delta, k=k)
    elif mode == "absolute":
        top_abs = torch.topk(delta.abs(), k=k)
        top = type("TopK", (), {"indices": top_abs.indices, "values": delta[top_abs.indices]})()
    else:
        raise ValueError(f"Unsupported packet mode: {mode!r}")
    quantized, decoded, scale = _quantize(top.values, coeff_bits=int(coeff_bits))
    return StepPacket(
        token_ids=tuple(int(value) for value in top.indices.tolist()),
        values=tuple(decoded),
        quantized=tuple(quantized),
        scale=float(scale),
    )


def transform_packet(packet: StepPacket, *, condition: str, rng: random.Random) -> StepPacket:
    ids = list(packet.token_ids)
    values = list(packet.values)
    quantized = list(packet.quantized)
    if condition == "matched" or not ids:
        return packet
    if condition == "atom_shuffle":
        ids = ids[1:] + ids[:1]
    elif condition == "coeff_shuffle":
        values = values[1:] + values[:1]
        quantized = quantized[1:] + quantized[:1]
    elif condition == "coeff_sign_flip":
        values = [-float(value) for value in values]
        quantized = [-int(value) for value in quantized]
    elif condition == "random_atom_ids":
        ids = [rng.randrange(0, max(max(ids) + 1, 2)) for _ in ids]
    else:
        raise ValueError(f"Unsupported packet transform: {condition!r}")
    return StepPacket(
        token_ids=tuple(ids),
        values=tuple(float(value) for value in values),
        quantized=tuple(int(value) for value in quantized),
        scale=packet.scale,
    )


def apply_packet(logits: torch.Tensor, packet: StepPacket | None) -> int:
    adjusted = logits.detach().float().clone()
    if packet is not None:
        for token_id, value in zip(packet.token_ids, packet.values, strict=True):
            if 0 <= int(token_id) < int(adjusted.numel()):
                adjusted[int(token_id)] += float(value)
    return int(torch.argmax(adjusted).item())


def _packet_bits(*, vocab_size: int, steps: int, top_k: int, coeff_bits: int) -> int:
    id_bits = int(math.ceil(math.log2(max(int(vocab_size), 2))))
    scale_bits = 16 if int(coeff_bits) > 0 and int(top_k) > 0 else 0
    return int(steps) * (int(top_k) * (id_bits + max(int(coeff_bits), 0)) + scale_bits)


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        build_c2c_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def _target_logits_on_teacher_prefix(
    model: Any,
    input_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
    *,
    base_model_idx: int,
) -> torch.Tensor:
    if teacher_tokens.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    prefix = torch.cat([input_ids, teacher_tokens[:-1].view(1, -1)], dim=1)
    attention_mask = torch.ones_like(prefix)
    target_model = model.model_list[base_model_idx]
    output = target_model(
        input_ids=prefix,
        attention_mask=attention_mask,
        use_cache=False,
    )
    prompt_len = int(input_ids.shape[1])
    positions = torch.arange(
        prompt_len - 1,
        prompt_len - 1 + int(teacher_tokens.numel()),
        device=output.logits.device,
    )
    return output.logits[0, positions, :].detach().float().cpu()


@torch.no_grad()
def capture_rows(
    *,
    source_model: str,
    target_model: str,
    eval_file: pathlib.Path,
    device: str,
    max_new_tokens: int,
    top_k: int,
    coeff_bits: int,
    packet_mode: str,
    limit: int | None,
) -> tuple[list[RowCapture], dict[str, Any], Any]:
    examples = load_generation(str(eval_file))
    if limit is not None:
        examples = examples[: int(limit)]
    model, tokenizer, artifact = load_c2c_model(
        source_model=source_model,
        target_model=target_model,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    rows: list[RowCapture] = []
    started = time.perf_counter()
    total = len(examples)
    for index, example in enumerate(examples):
        print(f"[capture] row {index + 1}/{total}", file=sys.stderr, flush=True)
        text = _format_prompt(tokenizer, example.prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_len = int(inputs["input_ids"].shape[1])
        outputs = model.generate(
            **inputs,
            kv_cache_index=build_c2c_kv_cache_index(prompt_len, device=device),
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs["sequences"]
        scores = list(outputs.scores or []) if hasattr(outputs, "scores") else list(outputs.get("scores") or [])
        teacher_tokens = sequences[0, prompt_len:].detach()
        target_logits = _target_logits_on_teacher_prefix(
            model,
            inputs["input_ids"],
            teacher_tokens,
            base_model_idx=int(getattr(model, "base_model_idx", 0)),
        )
        step_count = min(int(target_logits.shape[0]), len(scores), int(teacher_tokens.numel()))
        packets = [
            build_step_packet(
                target_logits[step],
                scores[step][0].detach().float().cpu(),
                top_k=int(top_k),
                coeff_bits=int(coeff_bits),
                mode=str(packet_mode),
            )
            for step in range(step_count)
        ]
        rows.append(
            RowCapture(
                index=int(index),
                example_id=_generation_example_id(example),
                answers=list(example.answers),
                c2c_tokens=[int(value) for value in teacher_tokens[:step_count].detach().cpu().tolist()],
                c2c_text=tokenizer.decode(teacher_tokens[:step_count], skip_special_tokens=True).strip(),
                target_logits=target_logits[:step_count].contiguous(),
                packets=packets,
            )
        )
    run_config = {
        "source_model": source_model,
        "target_model": target_model,
        "eval_file": _display_path(eval_file),
        "device": device,
        "max_new_tokens": int(max_new_tokens),
        "top_k": int(top_k),
        "coeff_bits": int(coeff_bits),
        "packet_mode": packet_mode,
        "limit": limit,
        "elapsed_sec": float(time.perf_counter() - started),
        "published_repo_id": artifact.repo_id,
        "published_subdir": artifact.subdir,
        "published_config_path": artifact.config_path,
        "published_checkpoint_dir": artifact.checkpoint_dir,
        "local_root": artifact.local_root,
    }
    return rows, run_config, tokenizer


def _condition_packet(
    rows: Sequence[RowCapture],
    row_index: int,
    step_index: int,
    *,
    condition: str,
    rng: random.Random,
) -> StepPacket | None:
    if condition in {"target_only", "zero_delta"}:
        return None
    if condition == "row_shuffle":
        other = rows[(row_index + 1) % len(rows)]
        if not other.packets:
            return None
        return other.packets[step_index % len(other.packets)]
    packet = rows[row_index].packets[step_index]
    if condition == "matched":
        return packet
    return transform_packet(packet, condition=condition, rng=rng)


def evaluate_conditions(
    rows: Sequence[RowCapture],
    *,
    tokenizer: Any,
    target_ids: dict[str, set[str]],
    conditions: Sequence[str],
    rng_seed: int,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    row_outputs: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    for condition in conditions:
        correct_ids: set[str] = set()
        exact_replay_ids: set[str] = set()
        teacher_only_correct: set[str] = set()
        clean_correct: set[str] = set()
        token_match = 0
        token_total = 0
        rng = random.Random(int(rng_seed))
        for row_index, row in enumerate(rows):
            selected_tokens: list[int] = []
            row_token_match = 0
            for step_index, teacher_token in enumerate(row.c2c_tokens):
                packet = _condition_packet(
                    rows,
                    row_index,
                    step_index,
                    condition=condition,
                    rng=rng,
                )
                pred = apply_packet(row.target_logits[step_index], packet)
                selected_tokens.append(pred)
                row_token_match += int(pred == int(teacher_token))
            decoded = tokenizer.decode(selected_tokens, skip_special_tokens=True).strip()
            correct = _generation_match(decoded, row.answers)
            exact_replay = selected_tokens == list(row.c2c_tokens)
            token_match += row_token_match
            token_total += len(row.c2c_tokens)
            if correct:
                correct_ids.add(row.example_id)
            if exact_replay:
                exact_replay_ids.add(row.example_id)
            if correct and row.example_id in target_ids["teacher_only"]:
                teacher_only_correct.add(row.example_id)
            if correct and row.example_id in target_ids["clean_residual_targets"]:
                clean_correct.add(row.example_id)
            row_outputs[condition].append(
                {
                    "example_id": row.example_id,
                    "correct": bool(correct),
                    "exact_teacher_replay": bool(exact_replay),
                    "token_match_count": int(row_token_match),
                    "token_count": int(len(row.c2c_tokens)),
                    "token_match_rate": float(row_token_match / max(len(row.c2c_tokens), 1)),
                    "prediction": decoded,
                }
            )
        summaries[condition] = {
            "condition": condition,
            "correct_count": len(correct_ids),
            "correct_ids": sorted(correct_ids),
            "teacher_only_correct_count": len(teacher_only_correct),
            "teacher_only_correct_ids": sorted(teacher_only_correct),
            "clean_correct_count": len(clean_correct),
            "clean_correct_ids": sorted(clean_correct),
            "exact_teacher_replay_count": len(exact_replay_ids),
            "exact_teacher_replay_ids": sorted(exact_replay_ids),
            "token_match_count": int(token_match),
            "token_count": int(token_total),
            "token_match_rate": float(token_match / max(token_total, 1)),
        }
    matched_clean = set(summaries["matched"]["clean_correct_ids"])
    control_clean_union = set().union(
        *[
            set(summaries[condition]["clean_correct_ids"])
            for condition in conditions
            if condition != "matched"
        ]
    )
    source_necessary = matched_clean - control_clean_union
    return {
        "condition_summaries": summaries,
        "source_necessary_clean_ids": sorted(source_necessary),
        "control_clean_union_ids": sorted(control_clean_union),
        "rows": row_outputs,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# SVAMP32 C2C Teacher-Delta Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- packet top-k: `{payload['packet_contract']['top_k']}`",
        f"- coeff bits: `{payload['packet_contract']['coeff_bits']}`",
        f"- average packet bytes per row: `{payload['packet_contract']['avg_packet_bytes_per_row']:.2f}`",
        f"- clean source-necessary IDs: `{len(payload['run']['source_necessary_clean_ids'])}`",
        "",
        "## Summary",
        "",
        "| Condition | Correct | Teacher-only | Clean | Exact teacher replay | Token match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, summary in payload["run"]["condition_summaries"].items():
        lines.append(
            f"| `{condition}` | {summary['correct_count']}/{payload['reference_n']} | "
            f"{summary['teacher_only_correct_count']} | {summary['clean_correct_count']} | "
            f"{summary['exact_teacher_replay_count']} | {summary['token_match_rate']:.3f} |"
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
            "- This is a C2C-teacher packet-capacity gate, not a deployable source-causal receiver.",
            "- It transmits sparse token-logit deltas from the dense teacher and therefore cannot by itself prove source-private latent communication.",
            "- Target-only under the teacher-generated prefix is a strong target-cache control; matching that control is not source-causal evidence.",
            "- A pass would only justify training a source-side predictor for the same packet; a fail rules out this packet format.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze(
    *,
    rows: Sequence[RowCapture],
    tokenizer: Any,
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
        "atom_shuffle",
        "coeff_shuffle",
        "coeff_sign_flip",
    )
    run = evaluate_conditions(
        rows,
        tokenizer=tokenizer,
        target_ids=target_ids,
        conditions=conditions,
        rng_seed=52060505,
    )
    matched = run["condition_summaries"]["matched"]
    best_control_correct = max(
        summary["correct_count"]
        for condition, summary in run["condition_summaries"].items()
        if condition != "matched"
    )
    best_control_clean = max(
        summary["clean_correct_count"]
        for condition, summary in run["condition_summaries"].items()
        if condition != "matched"
    )
    source_necessary_count = len(run["source_necessary_clean_ids"])
    clears = (
        matched["correct_count"] > best_control_correct
        and matched["clean_correct_count"] > best_control_clean
        and source_necessary_count >= 2
    )
    status = (
        "teacher_delta_packet_capacity_clears_controls_not_deployable"
        if clears
        else "teacher_delta_packet_capacity_fails_controls"
    )
    vocab_size = int(rows[0].target_logits.shape[-1]) if rows else 0
    total_bits = sum(
        _packet_bits(
            vocab_size=vocab_size,
            steps=len(row.packets),
            top_k=int(run_config["top_k"]),
            coeff_bits=int(run_config["coeff_bits"]),
        )
        for row in rows
    )
    avg_bytes = float((total_bits / 8.0) / max(len(rows), 1))
    packet_contract = {
        "kind": "teacher_forced_sparse_c2c_minus_target_logit_delta",
        "top_k": int(run_config["top_k"]),
        "coeff_bits": int(run_config["coeff_bits"]),
        "packet_mode": str(run_config["packet_mode"]),
        "vocab_size": vocab_size,
        "avg_packet_bits_per_row": float(total_bits / max(len(rows), 1)),
        "avg_packet_bytes_per_row": avg_bytes,
        "avg_cacheline_rounded_bytes_per_row": float(math.ceil(avg_bytes / 64.0) * 64.0) if avg_bytes else 0.0,
        "source_private": False,
        "not_deployable_reason": "packet is derived from dense C2C teacher logits",
    }
    payload = {
        "date": run_date,
        "status": status,
        "reference_n": len(rows),
        "run_config": run_config,
        "target_set_json": _display_path(target_set_path),
        "packet_contract": packet_contract,
        "headline": {
            "matched_correct": matched["correct_count"],
            "best_control_correct": best_control_correct,
            "matched_clean": matched["clean_correct_count"],
            "best_control_clean": best_control_clean,
            "source_necessary_clean_count": source_necessary_count,
            "capacity_pass": clears,
        },
        "run": run,
        "decision": (
            "Promote the packet format to a source-predictor gate."
            if clears
            else "Do not train a source predictor for this exact packet format; the matched packet does not separate from destructive controls."
        ),
        "interpretation": [
            "This is a C2C-teacher packet-capacity gate, not a deployable source-causal receiver.",
            "The packet is computed from dense C2C teacher logits and is not source-private.",
            "Target-only under the teacher-generated prefix is a target-cache control; matching it is not source-causal evidence.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    manifest_path = output_json.parent / "manifest.json"
    manifest_md = output_json.parent / "manifest.md"
    manifest_md.write_text(
        "\n".join(
            [
                "# SVAMP32 C2C Teacher-Delta Packet Gate Manifest",
                "",
                f"- date: `{run_date}`",
                f"- status: `{status}`",
                f"- output json: `{_display_path(output_json)}`",
                f"- output md: `{_display_path(output_md)}`",
                f"- manifest json: `{_display_path(manifest_path)}`",
                "",
                "## Interpretation",
                "",
                "- This artifact tests whether sparse top-k logit deltas from the dense C2C teacher survive destructive controls.",
                "- The gate is not source-private because the packet is computed from dense C2C teacher logits.",
                "- If matched does not beat target-only under the teacher-generated prefix, the apparent signal is a teacher-prefix/target-cache effect.",
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
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--coeff-bits", type=int, default=4)
    parser.add_argument("--packet-mode", choices=["positive", "absolute"], default="positive")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    rows, run_config, tokenizer = capture_rows(
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        eval_file=_resolve(args.eval_file),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        top_k=int(args.top_k),
        coeff_bits=int(args.coeff_bits),
        packet_mode=str(args.packet_mode),
        limit=args.limit,
    )
    payload = analyze(
        rows=rows,
        tokenizer=tokenizer,
        target_set_path=_resolve(args.target_set_json),
        run_config=run_config,
        run_date=str(args.date),
        output_json=_resolve(args.output_json),
        output_md=_resolve(args.output_md),
    )
    print(
        json.dumps(
            {"status": payload["status"], "output_json": _display_path(_resolve(args.output_json))},
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
