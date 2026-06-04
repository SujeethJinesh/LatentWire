#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmc.cache_split import split_for_key
from pmc.stage1_screens import write_csv, write_jsonl


OUT_DIR = ROOT / "results" / "mac_continue" / "fresh_mmlu_pro"
FRESH_LEADERBOARD = OUT_DIR / "leaderboard.csv"
RAW_ROWS = OUT_DIR / "fresh_mmlu_pro_rows.jsonl"
SPLIT_MANIFEST = OUT_DIR / "fresh_split_manifest.jsonl"
SUMMARY_JSON = OUT_DIR / "summary.json"
SUMMARY_MD = OUT_DIR / "summary.md"
RERANK_JSONL = OUT_DIR / "rerank_generation_rows.jsonl"
BOOTSTRAP_SEED = 20260604
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class FreshExample:
    row_id: str
    question: str
    options: list[str]
    answer_index: int
    category: str
    split: str


def stable_split(row_id: str) -> str:
    return split_for_key(f"mac_continue|fresh_mmlu_pro|{row_id}")


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.mean(values))) / std


def top1(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def margin(scores: np.ndarray) -> float:
    ordered = np.sort(scores)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[-1] - ordered[-2])


def format_prompt(example: FreshExample) -> str:
    choices = "\n".join(
        f"{LETTERS[index]}. {option}" for index, option in enumerate(example.options)
    )
    return (
        "Answer the multiple-choice question. Choose the single best option.\n\n"
        f"Question: {example.question}\n"
        f"Options:\n{choices}\n\n"
        "Answer:"
    )


def load_mmlu_pro_examples(max_scan: int, max_score_rows: int) -> list[FreshExample]:
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split=f"test[:{max_scan}]",
        cache_dir=str(ROOT / ".hf_home" / "datasets"),
    )
    examples: list[FreshExample] = []
    scored = 0
    for row in dataset:
        options = list(row["options"])
        answer_index = int(row["answer_index"])
        if len(options) != 10 or answer_index < 0 or answer_index >= len(options):
            continue
        row_id = f"mmlu_pro_{row['question_id']}"
        split = stable_split(row_id)
        examples.append(
            FreshExample(
                row_id=row_id,
                question=str(row["question"]),
                options=[str(option) for option in options],
                answer_index=answer_index,
                category=str(row.get("category", "")),
                split=split,
            )
        )
        if split in {"dev", "gate"}:
            scored += 1
        if scored >= max_score_rows:
            break
    return examples


def continuation_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    continuations: list[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_continuations = continuations[start : start + batch_size]
        encoded_full = tokenizer(
            [prompt + continuation for prompt, continuation in zip(batch_prompts, batch_continuations, strict=True)],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_prompt = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded_full["input_ids"].to(device)
        attention_mask = encoded_full["attention_mask"].to(device)
        prompt_lengths = encoded_prompt["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        for row_index, prompt_len in enumerate(prompt_lengths):
            token_scores: list[float] = []
            for label_pos in range(max(0, int(prompt_len) - 1), labels.shape[1]):
                if attention_mask[row_index, label_pos + 1].item() == 0:
                    continue
                label = int(labels[row_index, label_pos].item())
                if label == pad_id:
                    continue
                token_scores.append(float(log_probs[row_index, label_pos, label].item()))
            scores.append(float(np.mean(token_scores)) if token_scores else -1e9)
    return scores


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".hf_home" / "transformers"),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".hf_home" / "transformers"),
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def score_examples(
    examples: list[FreshExample],
    *,
    source_model_name: str,
    target_model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    scored_examples = [example for example in examples if example.split in {"dev", "gate"}]
    prompts: list[str] = []
    continuations: list[str] = []
    owner: list[FreshExample] = []
    for example in scored_examples:
        prompt = format_prompt(example)
        for option_index, option in enumerate(example.options):
            prompts.append(prompt)
            continuations.append(f" {LETTERS[option_index]}. {option}")
            owner.append(example)

    source_tokenizer, source_model = load_model(source_model_name, device)
    source_flat = continuation_logprobs(
        source_model,
        source_tokenizer,
        prompts,
        continuations,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    del source_model, source_tokenizer
    if device == "mps":
        torch.mps.empty_cache()

    target_tokenizer, target_model = load_model(target_model_name, device)
    target_flat = continuation_logprobs(
        target_model,
        target_tokenizer,
        prompts,
        continuations,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    del target_model, target_tokenizer
    if device == "mps":
        torch.mps.empty_cache()

    by_row: dict[str, dict[str, Any]] = {}
    for example in scored_examples:
        by_row[example.row_id] = {
            "row_id": example.row_id,
            "split": example.split,
            "category": example.category,
            "answer_index": example.answer_index,
            "option_count": len(example.options),
            "source_scores": [],
            "target_scores": [],
        }
    for example, source_score, target_score in zip(owner, source_flat, target_flat, strict=True):
        by_row[example.row_id]["source_scores"].append(float(source_score))
        by_row[example.row_id]["target_scores"].append(float(target_score))

    rows = []
    for row in by_row.values():
        source = np.asarray(row["source_scores"], dtype=np.float64)
        target = np.asarray(row["target_scores"], dtype=np.float64)
        row["source_top1"] = top1(source)
        row["target_top1"] = top1(target)
        row["source_correct"] = row["source_top1"] == row["answer_index"]
        row["target_correct"] = row["target_top1"] == row["answer_index"]
        row["source_margin"] = margin(source)
        row["target_margin"] = margin(target)
        rows.append(row)
    return rows


def empirical_mi(x_values: list[Any], y_values: list[Any]) -> float:
    n = len(x_values)
    if n == 0:
        return 0.0
    joint = Counter(zip(x_values, y_values, strict=True))
    x_counts = Counter(x_values)
    y_counts = Counter(y_values)
    total = float(n)
    mi = 0.0
    for (x_value, y_value), count in joint.items():
        pxy = count / total
        px = x_counts[x_value] / total
        py = y_counts[y_value] / total
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log2(pxy / (px * py))
    return float(mi)


def source_score_signature(row: dict[str, Any], bins: int = 4) -> tuple[int, ...]:
    scores = zscore(np.asarray(row["source_scores"], dtype=np.float64))
    quantized = []
    for value in scores:
        clipped = max(-2.0, min(2.0, float(value)))
        quantized.append(int(round((clipped + 2.0) * (bins - 1) / 4.0)))
    return tuple(quantized)


def information_beyond(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["split"] in {"dev", "gate"}]
    correct = [int(row["answer_index"]) for row in valid]
    source_top = [int(row["source_top1"]) for row in valid]
    source_signature = [source_score_signature(row) for row in valid]
    source_full = [(source_top[index], source_signature[index]) for index in range(len(valid))]
    mi_top = empirical_mi(source_top, correct)
    mi_full = empirical_mi(source_full, correct)
    return {
        "n": len(valid),
        "mi_source_top1_correct_bits": mi_top,
        "mi_source_scores_correct_bits": mi_full,
        "i_beyond_bits": max(0.0, mi_full - mi_top),
    }


def accuracy(rows: list[dict[str, Any]], pred_key: str, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(
        sum(int(rows[index][pred_key]) == int(rows[index]["answer_index"]) for index in indices) / len(indices)
    )


def paired_delta_ci(a_correct: list[int], b_correct: list[int], samples: int = 1000) -> dict[str, Any]:
    if len(a_correct) != len(b_correct):
        raise ValueError("paired vectors must have same length")
    if not a_correct:
        return {"n": 0, "delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    diffs = [a - b for a, b in zip(a_correct, b_correct, strict=True)]
    rng = random.Random(BOOTSTRAP_SEED)
    boot = []
    for _ in range(samples):
        draw = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(sum(draw) / len(draw))
    boot.sort()
    return {
        "n": len(diffs),
        "delta": sum(diffs) / len(diffs),
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def choose_alpha(rows: list[dict[str, Any]], dev_indices: list[int]) -> float:
    candidates = [step / 10.0 for step in range(-10, 21)]
    best = (0.0, -1.0)
    for alpha in candidates:
        correct = 0
        for index in dev_indices:
            row = rows[index]
            source = zscore(np.asarray(row["source_scores"], dtype=np.float64))
            target = zscore(np.asarray(row["target_scores"], dtype=np.float64))
            pred = top1(target + alpha * source)
            correct += int(pred == row["answer_index"])
        acc = correct / len(dev_indices) if dev_indices else 0.0
        if acc > best[1]:
            best = (alpha, acc)
    return best[0]


def quantize_source_packet(row: dict[str, Any]) -> np.ndarray:
    source = zscore(np.asarray(row["source_scores"], dtype=np.float64))
    quantized = []
    for value in source:
        clipped = max(-2.0, min(2.0, float(value)))
        level = int(round((clipped + 2.0) * 15.0 / 4.0))
        level = max(0, min(15, level))
        quantized.append((level * 4.0 / 15.0) - 2.0)
    return np.asarray(quantized, dtype=np.float64)


def deranged(values: np.ndarray) -> np.ndarray:
    return np.roll(values, 1)


def apply_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev_indices = [index for index, row in enumerate(rows) if row["split"] == "dev"]
    gate_indices = [index for index, row in enumerate(rows) if row["split"] == "gate"]
    alpha = choose_alpha(rows, dev_indices)
    rng = random.Random(BOOTSTRAP_SEED)
    source_packets = [quantize_source_packet(row) for row in rows]
    shuffled_packets = source_packets[:]
    rng.shuffle(shuffled_packets)

    for index, row in enumerate(rows):
        target = zscore(np.asarray(row["target_scores"], dtype=np.float64))
        source = source_packets[index]
        shuffled = shuffled_packets[index]
        row["source_index_pred"] = int(row["source_top1"])
        row["target_only_pred"] = int(row["target_top1"])
        row["wz_pred"] = top1(target + alpha * source)
        row["wrong_row_pred"] = top1(target + alpha * shuffled)
        row["derangement_pred"] = top1(target + alpha * deranged(source))
        row["random_same_byte_pred"] = int(rng.randrange(row["option_count"]))
        row["label_shuffle_pred"] = (int(row["source_top1"]) + 1) % int(row["option_count"])

    def gate_correct(key: str) -> list[int]:
        return [int(rows[index][key] == rows[index]["answer_index"]) for index in gate_indices]

    wz = gate_correct("wz_pred")
    target = gate_correct("target_only_pred")
    source = gate_correct("source_index_pred")
    wrong = gate_correct("wrong_row_pred")
    derange = gate_correct("derangement_pred")
    random_control = gate_correct("random_same_byte_pred")
    label_shuffle = gate_correct("label_shuffle_pred")
    baseline_vectors = {
        "target_only": target,
        "source_index": source,
        "wrong_row": wrong,
        "derangement": derange,
        "random_same_byte": random_control,
        "label_shuffle": label_shuffle,
    }
    baseline_acc = {key: sum(value) / len(value) if value else 0.0 for key, value in baseline_vectors.items()}
    best_baseline = max(baseline_acc, key=baseline_acc.get)
    return {
        "alpha": alpha,
        "dev_n": len(dev_indices),
        "gate_n": len(gate_indices),
        "gate_accuracy": {
            "wz": sum(wz) / len(wz) if wz else 0.0,
            **baseline_acc,
        },
        "delta_beyond_score": paired_delta_ci(wz, source),
        "delta_beyond_label": paired_delta_ci(wz, label_shuffle),
        "delta_vs_best_baseline": paired_delta_ci(wz, baseline_vectors[best_baseline]),
        "best_baseline": best_baseline,
        "control_collapse": {
            "wrong_row": baseline_acc["wrong_row"],
            "derangement": baseline_acc["derangement"],
            "random_same_byte": baseline_acc["random_same_byte"],
        },
    }


def choose_l_b1_threshold(rows: list[dict[str, Any]], dev_indices: list[int]) -> float:
    candidates = sorted(
        {
            float(rows[index]["source_margin"] - rows[index]["target_margin"])
            for index in dev_indices
        }
    )
    if not candidates:
        return float("inf")
    grid = [candidates[0] - 1e-6, *candidates, candidates[-1] + 1e-6]
    best = (float("inf"), -1.0)
    for threshold in grid:
        correct = 0
        for index in dev_indices:
            row = rows[index]
            use_source = (float(row["source_margin"]) - float(row["target_margin"])) >= threshold
            pred = row["source_top1"] if use_source else row["target_top1"]
            correct += int(pred == row["answer_index"])
        acc = correct / len(dev_indices)
        if acc > best[1]:
            best = (threshold, acc)
    return best[0]


def aurc(correct: list[int], confidence: list[float]) -> float:
    if not correct:
        return 1.0
    ordered = sorted(zip(confidence, correct, strict=True), reverse=True)
    risks = []
    hits = 0
    for rank, (_, ok) in enumerate(ordered, start=1):
        hits += int(ok)
        risks.append(1.0 - hits / rank)
    return float(sum(risks) / len(risks))


def risk_at_coverage(correct: list[int], confidence: list[float], coverage: float) -> float:
    if not correct:
        return 1.0
    k = max(1, int(math.ceil(len(correct) * coverage)))
    ordered = sorted(zip(confidence, correct, strict=True), reverse=True)[:k]
    return float(1.0 - sum(ok for _, ok in ordered) / len(ordered))


def l_b1_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev_indices = [index for index, row in enumerate(rows) if row["split"] == "dev"]
    gate_indices = [index for index, row in enumerate(rows) if row["split"] == "gate"]
    threshold = choose_l_b1_threshold(rows, dev_indices)
    for row in rows:
        use_source = (float(row["source_margin"]) - float(row["target_margin"])) >= threshold
        row["l_b1_pred"] = int(row["source_top1"] if use_source else row["target_top1"])
        row["l_b1_confidence"] = abs(float(row["source_margin"]) - float(row["target_margin"]))
        row["source_confidence"] = float(row["source_margin"])

    l_b1_correct = [int(rows[index]["l_b1_pred"] == rows[index]["answer_index"]) for index in gate_indices]
    source_correct = [int(rows[index]["source_top1"] == rows[index]["answer_index"]) for index in gate_indices]
    target_correct = [int(rows[index]["target_top1"] == rows[index]["answer_index"]) for index in gate_indices]
    l_b1_conf = [float(rows[index]["l_b1_confidence"]) for index in gate_indices]
    source_conf = [float(rows[index]["source_confidence"]) for index in gate_indices]
    damage_den = [
        index
        for index in gate_indices
        if rows[index]["target_top1"] == rows[index]["answer_index"]
        and rows[index]["source_top1"] != rows[index]["answer_index"]
    ]
    repair_den = [
        index
        for index in gate_indices
        if rows[index]["target_top1"] != rows[index]["answer_index"]
        and rows[index]["source_top1"] == rows[index]["answer_index"]
    ]
    l_b1_damage = sum(int(rows[index]["l_b1_pred"] != rows[index]["answer_index"]) for index in damage_den)
    source_damage = len(damage_den)
    l_b1_repair = sum(int(rows[index]["l_b1_pred"] == rows[index]["answer_index"]) for index in repair_den)
    source_repair = len(repair_den)
    packet_top1 = [int(rows[index]["source_top1"]) for index in gate_indices]
    answer = [int(rows[index]["answer_index"]) for index in gate_indices]
    packet_mi = empirical_mi(packet_top1, [int(rows[index]["wz_pred"]) for index in gate_indices])
    return {
        "threshold": threshold,
        "gate_n": len(gate_indices),
        "accuracy": sum(l_b1_correct) / len(l_b1_correct) if l_b1_correct else 0.0,
        "source_index_accuracy": sum(source_correct) / len(source_correct) if source_correct else 0.0,
        "target_accuracy": sum(target_correct) / len(target_correct) if target_correct else 0.0,
        "aurc": aurc(l_b1_correct, l_b1_conf),
        "source_index_confidence_aurc": aurc(source_correct, source_conf),
        "aurc_delta_vs_source_index_confidence": aurc(source_correct, source_conf) - aurc(l_b1_correct, l_b1_conf),
        "risk_at_coverage_50": risk_at_coverage(l_b1_correct, l_b1_conf, 0.5),
        "source_risk_at_coverage_50": risk_at_coverage(source_correct, source_conf, 0.5),
        "accuracy_at_fixed_coverage_50": 1.0 - risk_at_coverage(l_b1_correct, l_b1_conf, 0.5),
        "damage_den": len(damage_den),
        "source_damage": source_damage,
        "l_b1_damage": l_b1_damage,
        "damage_reduction": source_damage - l_b1_damage,
        "repair_den": len(repair_den),
        "source_repair": source_repair,
        "l_b1_repair": l_b1_repair,
        "repair_rate": l_b1_repair / len(repair_den) if repair_den else 0.0,
        "candidate_leakage_accuracy": sum(int(p == a) for p, a in zip(packet_top1, answer, strict=True)) / len(answer)
        if answer
        else 0.0,
        "packet_source_top1_mi_bits": packet_mi,
        "paired_delta_vs_source_index": paired_delta_ci(l_b1_correct, source_correct),
    }


def generate_rerank_slice(
    examples: list[FreshExample],
    *,
    target_model_name: str,
    device: str,
    max_rows: int,
    max_new_tokens: int,
    max_seconds: float,
) -> dict[str, Any]:
    start = time.time()
    rows = [example for example in examples if example.split in {"dev", "gate"}][:max_rows]
    if not rows:
        return {"status": "MAC_DONE", "rows": 0, "generated": 0, "parked": False}
    tokenizer, model = load_model(target_model_name, device)
    generated_rows: list[dict[str, Any]] = []
    parked = False
    for example in rows:
        if time.time() - start > max_seconds:
            parked = True
            break
        prompt = (
            f"{format_prompt(example)}\nGive only the option letter and one short reason."
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        generated_rows.append(
            {
                "row_id": example.row_id,
                "split": example.split,
                "answer_index": example.answer_index,
                "generated_text": text.strip(),
            }
        )
    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    write_jsonl(RERANK_JSONL, generated_rows)
    return {
        "status": "PARKED_NEEDS_GPU_BACKFILL" if parked else "MAC_DONE",
        "rows": len(rows),
        "generated": len(generated_rows),
        "parked": parked,
        "path": str(RERANK_JSONL.relative_to(ROOT)),
    }


def build_outputs(
    *,
    examples: list[FreshExample],
    scored_rows: list[dict[str, Any]],
    info: dict[str, Any],
    wz: dict[str, Any],
    l_b1: dict[str, Any],
    rerank: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gate_n = wz["gate_n"]
    i_beyond = float(info["i_beyond_bits"])
    wz_gate_delta = float(wz["delta_vs_best_baseline"]["delta"])
    wz_gate_lb = float(wz["delta_vs_best_baseline"]["ci95_low"])
    l_b1_delta = float(l_b1["paired_delta_vs_source_index"]["delta"])
    l_b1_lb = float(l_b1["paired_delta_vs_source_index"]["ci95_low"])
    controls_ok = (
        wz["gate_accuracy"]["wz"] > wz["control_collapse"]["wrong_row"]
        and wz["gate_accuracy"]["wz"] > wz["control_collapse"]["derangement"]
    )
    provisional = bool(
        gate_n >= args.min_gate_for_promote
        and i_beyond > args.i_beyond_floor
        and wz_gate_lb > 0
        and l_b1_lb > 0
        and controls_ok
    )
    if i_beyond <= args.i_beyond_floor:
        latentwire_status = "MAC_DONE"
        headline = "BOUNDED_NEGATIVE"
    elif provisional:
        latentwire_status = "PROVISIONAL_PROMOTE_TO_GPU"
        headline = "PROVISIONAL_PROMOTE_TO_GPU"
    else:
        latentwire_status = "MAC_DONE"
        headline = "BOUNDED_NEGATIVE"

    split_counts = Counter(example.split for example in examples)
    summary = {
        "status": latentwire_status,
        "headline": headline,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "TIGER-Lab/MMLU-Pro",
        "source_model": args.source_model,
        "target_model": args.target_model,
        "split_counts": dict(split_counts),
        "scored_dev_gate_rows": len(scored_rows),
        "confirm_rows_scored": 0,
        "information": info,
        "wz": wz,
        "l_b1": l_b1,
        "rerank": rerank,
        "terminal_classification": {
            "latentwire_mmlu_pro_info_ceiling": "MAC_DONE",
            "latentwire_fresh_wz_high_entropy": latentwire_status,
            "latentwire_l_b1_damage_avoidance": latentwire_status,
            "latentwire_l_a2_rerank_pool": rerank["status"],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_MANIFEST.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "row_id": example.row_id,
                        "split": example.split,
                        "category": example.category,
                        "option_count": len(example.options),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    serializable_rows: list[dict[str, Any]] = []
    for row in scored_rows:
        serializable = dict(row)
        serializable["source_scores"] = [float(value) for value in serializable["source_scores"]]
        serializable["target_scores"] = [float(value) for value in serializable["target_scores"]]
        serializable_rows.append(serializable)
    write_jsonl(RAW_ROWS, serializable_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(summary)
    write_fresh_leaderboard(summary)
    update_dashboard_files(summary)
    update_queue_files(summary)
    update_lessons(summary)
    return summary


def write_summary_markdown(summary: dict[str, Any]) -> None:
    info = summary["information"]
    wz = summary["wz"]
    l_b1 = summary["l_b1"]
    lines = [
        "# Fresh MMLU-Pro Mac Continue Summary",
        "",
        f"- headline: `{summary['headline']}`",
        f"- status: `{summary['status']}`",
        f"- source model: `{summary['source_model']}`",
        f"- target model: `{summary['target_model']}`",
        f"- split counts: `{summary['split_counts']}`",
        f"- scored dev/gate rows: `{summary['scored_dev_gate_rows']}`",
        "- confirm rows scored: `0`",
        f"- I_beyond bits: `{info['i_beyond_bits']:.6f}`",
        f"- source_top1 MI bits: `{info['mi_source_top1_correct_bits']:.6f}`",
        f"- source_scores MI bits: `{info['mi_source_scores_correct_bits']:.6f}`",
        "",
        "## WZ / Score Packet",
        "",
        f"- selected alpha: `{wz['alpha']}`",
        f"- gate accuracy: `{wz['gate_accuracy']}`",
        f"- best baseline: `{wz['best_baseline']}`",
        f"- delta_beyond_score: `{wz['delta_beyond_score']}`",
        f"- delta_beyond_label: `{wz['delta_beyond_label']}`",
        f"- delta_vs_best_baseline: `{wz['delta_vs_best_baseline']}`",
        f"- controls: `{wz['control_collapse']}`",
        "",
        "## L-B1 Damage Avoidance",
        "",
        f"- gate accuracy: `{l_b1['accuracy']:.6f}`",
        f"- source-index accuracy: `{l_b1['source_index_accuracy']:.6f}`",
        f"- target accuracy: `{l_b1['target_accuracy']:.6f}`",
        f"- AURC delta vs source-index+confidence: `{l_b1['aurc_delta_vs_source_index_confidence']:.6f}`",
        f"- risk@coverage50: `{l_b1['risk_at_coverage_50']:.6f}`",
        f"- accuracy@coverage50: `{l_b1['accuracy_at_fixed_coverage_50']:.6f}`",
        f"- damage reduction: `{l_b1['damage_reduction']}` of `{l_b1['damage_den']}`",
        f"- repair rate: `{l_b1['repair_rate']:.6f}`",
        f"- candidate leakage accuracy: `{l_b1['candidate_leakage_accuracy']:.6f}`",
        f"- packet<->source-top1 MI bits: `{l_b1['packet_source_top1_mi_bits']:.6f}`",
        "",
        "## Rerank Slice",
        "",
        f"- status: `{summary['rerank']['status']}`",
        f"- generated rows: `{summary['rerank']['generated']}` / `{summary['rerank']['rows']}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_fresh_leaderboard(summary: dict[str, Any]) -> None:
    rows = [
        {
            "method_id": "L_A1_mmlu_pro_info_ceiling",
            "paper": "latentwire",
            "split": "dev_gate",
            "status": "MAC_DONE",
            "source_path": str(RAW_ROWS.relative_to(ROOT)),
            "matched_condition": "source_scores_given_source_top1",
            "matched_accuracy": "",
            "best_baseline_condition": "source_top1",
            "best_baseline_accuracy": "",
            "delta_vs_best_baseline": summary["information"]["i_beyond_bits"],
            "ci95_low_vs_best_baseline": "",
            "ci95_high_vs_best_baseline": "",
            "paired_n": summary["information"]["n"],
            "regime": "fresh_mmlu_pro_info_ceiling",
            "n": summary["information"]["n"],
            "median_recovery": "",
            "cvar25_recovery": "",
            "worst_recovery": "",
            "no_gap_rows": "",
            "total_rows": summary["information"]["n"],
            "note": "fresh dev/gate only; confirm rows scored 0",
        },
        {
            "method_id": "L_SCORECOMP_fresh_wz_high_entropy",
            "paper": "latentwire",
            "split": "gate",
            "status": summary["status"],
            "source_path": str(RAW_ROWS.relative_to(ROOT)),
            "matched_condition": "deployable_source_score_wz",
            "matched_accuracy": summary["wz"]["gate_accuracy"]["wz"],
            "best_baseline_condition": summary["wz"]["best_baseline"],
            "best_baseline_accuracy": summary["wz"]["gate_accuracy"][summary["wz"]["best_baseline"]],
            "delta_vs_best_baseline": summary["wz"]["delta_vs_best_baseline"]["delta"],
            "ci95_low_vs_best_baseline": summary["wz"]["delta_vs_best_baseline"]["ci95_low"],
            "ci95_high_vs_best_baseline": summary["wz"]["delta_vs_best_baseline"]["ci95_high"],
            "paired_n": summary["wz"]["delta_vs_best_baseline"]["n"],
            "regime": "fresh_mmlu_pro_wz",
            "n": summary["wz"]["gate_n"],
            "median_recovery": "",
            "cvar25_recovery": "",
            "worst_recovery": "",
            "no_gap_rows": "",
            "total_rows": summary["wz"]["gate_n"],
            "note": "Mac result; no PASSED status",
        },
        {
            "method_id": "L_B1_damage_avoidance_fresh",
            "paper": "latentwire",
            "split": "gate",
            "status": summary["status"],
            "source_path": str(RAW_ROWS.relative_to(ROOT)),
            "matched_condition": "l_b1_damage_avoidance",
            "matched_accuracy": summary["l_b1"]["accuracy"],
            "best_baseline_condition": "source_index_confidence",
            "best_baseline_accuracy": summary["l_b1"]["source_index_accuracy"],
            "delta_vs_best_baseline": summary["l_b1"]["paired_delta_vs_source_index"]["delta"],
            "ci95_low_vs_best_baseline": summary["l_b1"]["paired_delta_vs_source_index"]["ci95_low"],
            "ci95_high_vs_best_baseline": summary["l_b1"]["paired_delta_vs_source_index"]["ci95_high"],
            "paired_n": summary["l_b1"]["paired_delta_vs_source_index"]["n"],
            "regime": "fresh_mmlu_pro_damage_avoidance",
            "n": summary["l_b1"]["gate_n"],
            "median_recovery": "",
            "cvar25_recovery": "",
            "worst_recovery": "",
            "no_gap_rows": "",
            "total_rows": summary["l_b1"]["gate_n"],
            "note": "Mac result; no PASSED status",
        },
        {
            "method_id": "L_A2_verifier_rerank",
            "paper": "latentwire",
            "split": "dev_gate",
            "status": summary["rerank"]["status"],
            "source_path": summary["rerank"].get("path", ""),
            "matched_condition": "bounded_generation_slice",
            "matched_accuracy": "",
            "best_baseline_condition": "",
            "best_baseline_accuracy": "",
            "delta_vs_best_baseline": "",
            "ci95_low_vs_best_baseline": "",
            "ci95_high_vs_best_baseline": "",
            "paired_n": summary["rerank"].get("generated", 0),
            "regime": "fresh_generated_solution_rerank_slice",
            "n": summary["rerank"].get("generated", 0),
            "median_recovery": "",
            "cvar25_recovery": "",
            "worst_recovery": "",
            "no_gap_rows": "",
            "total_rows": summary["rerank"].get("rows", 0),
            "note": "generation capped on Mac; no confirm rows",
        },
    ]
    write_csv(FRESH_LEADERBOARD, rows)
    dashboard_leaderboard = ROOT / "dashboard" / "leaderboard.csv"
    if dashboard_leaderboard.exists():
        existing = list(csv.DictReader(dashboard_leaderboard.open("r", encoding="utf-8", newline="")))
        existing = [
            row
            for row in existing
            if row.get("method_id")
            not in {
                "L_A1_mmlu_pro_info_ceiling",
                "L_SCORECOMP_fresh_wz_high_entropy",
                "L_B1_damage_avoidance_fresh",
                "L_A2_verifier_rerank",
            }
        ]
        write_csv(dashboard_leaderboard, existing + rows)


def update_dashboard_files(summary: dict[str, Any]) -> None:
    status = summary["status"]
    headline = summary["headline"]
    mac_report = [
        "# Mac Continue Report",
        "",
        f"- terminal headline: `{headline}`",
        f"- LatentWire status: `{status}`",
        f"- mac_continue drained: `true`",
        f"- confirm rows scored: `0`",
        f"- fresh-data I_beyond bits: `{summary['information']['i_beyond_bits']:.6f}`",
        f"- WZ gate delta vs best baseline: `{summary['wz']['delta_vs_best_baseline']}`",
        f"- WZ delta_beyond_score: `{summary['wz']['delta_beyond_score']}`",
        f"- WZ delta_beyond_label: `{summary['wz']['delta_beyond_label']}`",
        f"- L-B1 AURC delta vs source-index+confidence: `{summary['l_b1']['aurc_delta_vs_source_index_confidence']:.6f}`",
        f"- L-B1 paired delta vs source-index: `{summary['l_b1']['paired_delta_vs_source_index']}`",
        f"- L-B1 damage reduction: `{summary['l_b1']['damage_reduction']}` / `{summary['l_b1']['damage_den']}`",
        f"- L-B1 repair rate: `{summary['l_b1']['repair_rate']:.6f}`",
        f"- L-A2 rerank slice: `{summary['rerank']}`",
        "",
        "Channel-Set remains GPU-backfill only from prior triage: C_A1 has limited offline headroom, C_F is mixed/control-contaminated, CE13 has no parseable cache, C_D1 is killed, and C_A2 is tests-only.",
    ]
    (ROOT / "dashboard" / "mac_continue_report.md").write_text("\n".join(mac_report).rstrip() + "\n", encoding="utf-8")
    paper_assessment = [
        "# Paper Path Assessment",
        "",
        f"- current answer: `{headline}`",
        "- target venue calibration: COLM Efficient Reasoning workshop; bounded negatives are acceptable evidence only if framed as limits, not a positive method.",
        f"- fresh I_beyond: `{summary['information']['i_beyond_bits']:.6f}` bits",
        f"- LatentWire positive: `{'yes' if summary['status'] == 'PROVISIONAL_PROMOTE_TO_GPU' else 'no'}`",
        f"- L-B1 AURC signal: `{'yes' if summary['l_b1']['aurc_delta_vs_source_index_confidence'] > 0 else 'no'}`",
        "- C_A1/C_F/CE13 offline signal: C_A1 limited, C_F mixed, CE13 none.",
        "- next paper path: write a bounded-negative workshop story unless a future fresh branch creates a real provisional promote.",
    ]
    (ROOT / "dashboard" / "paper_path_assessment.md").write_text("\n".join(paper_assessment).rstrip() + "\n", encoding="utf-8")
    gpu_plan = [
        "# GPU Handoff Plan",
        "",
        "No foreground GPU job is authorized from this Mac continuation.",
        "",
        f"- PROVISIONAL_PROMOTE_TO_GPU rows: `{1 if summary['status'] == 'PROVISIONAL_PROMOTE_TO_GPU' else 0}`",
        "- Foreground confirmation queue: empty unless a future row is explicitly promoted.",
        "- LatentWire remains Mac-complete for this pass.",
        "",
        "## Allowed Backfill Only",
        "",
        "- C_A1 ParoQuant parity: replay exact cached gate packets with native W4A16 forwards; est 2-4 GPU hours; no promotion allowed.",
        "- C_F hazard controls: backfill paired static/random controls before any native forward confirmation; est 2-6 GPU hours; no promotion allowed.",
        "- C_D1 OSC/DecDEC: only materialize a preregistered replacement shard for the current underpowered negative row; est 2-4 GPU hours; no promotion allowed.",
        "- CE13 warmup policy: no parseable cache exists; spend 0 GPU hours now and only consider 2-4 GPU hours after a tiny dev/gate cache exists.",
        "",
        "## Hard Stop",
        "",
        "Do not run foreground GPU confirmation until a future dashboard contains at least one `PROVISIONAL_PROMOTE_TO_GPU` row or a Channel-Set row with positive offline gate headroom, matched-budget baselines, and control collapse.",
    ]
    (ROOT / "dashboard" / "gpu_handoff_plan.md").write_text("\n".join(gpu_plan).rstrip() + "\n", encoding="utf-8")
    if summary["status"] == "PROVISIONAL_PROMOTE_TO_GPU":
        breakthrough = [
            "# Breakthrough Board",
            "",
            "| method_id | paper | source_path | delta_vs_best_baseline | ci95_low_vs_best_baseline | median_recovery | note |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| L_SCORECOMP_fresh_wz_high_entropy | latentwire | {RAW_ROWS.relative_to(ROOT)} | {summary['wz']['delta_vs_best_baseline']['delta']} | {summary['wz']['delta_vs_best_baseline']['ci95_low']} |  | fresh Mac provisional only |",
        ]
    else:
        breakthrough = ["# Breakthrough Board", "", "_No rows._"]
    (ROOT / "dashboard" / "breakthrough_board.md").write_text("\n".join(breakthrough).rstrip() + "\n", encoding="utf-8")
    kill_path = ROOT / "dashboard" / "kill_board.md"
    existing = kill_path.read_text(encoding="utf-8").rstrip() if kill_path.exists() else "# Kill Board"
    existing_lines = [
        line for line in existing.splitlines() if "L_SCORECOMP_fresh_wz_high_entropy" not in line
    ]
    existing = "\n".join(existing_lines).rstrip()
    if summary["headline"] == "BOUNDED_NEGATIVE":
        existing += (
            "\n| L_SCORECOMP_fresh_wz_high_entropy | latentwire | "
            f"{RAW_ROWS.relative_to(ROOT)} | {summary['wz']['delta_vs_best_baseline']['delta']} | "
            f"{summary['wz']['delta_vs_best_baseline']['ci95_high']} | fresh high-entropy Mac bounded negative |"
        )
    kill_path.write_text(existing.rstrip() + "\n", encoding="utf-8")


def update_queue_files(summary: dict[str, Any]) -> None:
    (ROOT / "queues" / "mac_continue.yaml").write_text(
        "mac_continue: []\ncompleted:\n"
        "  - latentwire_mmlu_pro_info_ceiling\n"
        "  - latentwire_fresh_wz_high_entropy\n"
        "  - latentwire_l_b1_damage_avoidance\n"
        "  - latentwire_l_a2_rerank_pool\n",
        encoding="utf-8",
    )
    foreground = "foreground: []\nnotes:\n  - \"No foreground GPU confirmation authorized by Mac continuation.\"\n"
    (ROOT / "queues" / "gpu_foreground.yaml").write_text(foreground, encoding="utf-8")
    backfill = """backfill:
  - id: channel_set_c_a1_paroquant_parity_card
    reason: limited_offline_gate_headroom_needs_native_forward_parity
    command: "prepare a user-run packet for C_A1 cached gate rows only; no foreground confirmation claim"
    required_cache_or_model: "real W4A16-capable checkpoint plus exact cached C_A1 gate identifiers"
    est_gpu_hours: 2-4
    promotion_allowed: false
  - id: channel_set_c_d1_osc_decdec_cache_shard
    reason: current_offline_rows_negative_and_underpowered
    command: "materialize a tiny OSC/DecDEC dev/gate shard only after preregistering the new denominator"
    required_cache_or_model: "reasoning-model traces for OSC/DecDEC stress"
    est_gpu_hours: 2-4
    promotion_allowed: false
  - id: channel_set_ce13_warmup_policy_cache_materialization
    reason: no_parseable_warmup_policy_cache_offline
    command: "find or build a tiny CE13 dev/gate warmup-policy cache before any GPU forward"
    required_cache_or_model: "warmup-policy traces and locked dev/gate row ids"
    est_gpu_hours: "0 now; 2-4 only after cache exists"
    promotion_allowed: false
  - id: channel_set_c_f_hazard_control_shard
    reason: mixed_gate_signal_and_random_control_contamination
    command: "backfill paired static/random controls before any native forward confirmation"
    required_cache_or_model: "matched-budget static top-K and random-control trace cache"
    est_gpu_hours: 2-6
    promotion_allowed: false
"""
    if summary["rerank"].get("parked"):
        backfill += """  - id: latentwire_rerank_generation_overflow
    reason: mac_generation_timebox_exceeded
    command: "continue capped generated-solution rerank shard from results/mac_continue/fresh_mmlu_pro"
    required_cache_or_model: "small-model candidate generation"
    est_gpu_hours: 0-2
    promotion_allowed: false
"""
    (ROOT / "queues" / "gpu_backfill.yaml").write_text(backfill, encoding="utf-8")


def update_lessons(summary: dict[str, Any]) -> None:
    path = ROOT / "lessons" / "LESSONS_LEDGER.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Lessons Ledger\n"
    marker = "2026-06-03 - Fresh MMLU-Pro Mac continuation"
    entry = f"""## 2026-06-03 - Fresh MMLU-Pro Mac continuation

- Fresh split guard: MMLU-Pro rows were split before scoring; confirm rows scored `0`.
- Fresh I_beyond: `{summary['information']['i_beyond_bits']:.6f}` bits on scored dev/gate rows.
- WZ result: status `{summary['status']}`, gate delta vs best baseline `{summary['wz']['delta_vs_best_baseline']['delta']:.6f}` with CI-low `{summary['wz']['delta_vs_best_baseline']['ci95_low']:.6f}`.
- L-B1 result: AURC delta vs source-index+confidence `{summary['l_b1']['aurc_delta_vs_source_index_confidence']:.6f}`, paired delta vs source-index `{summary['l_b1']['paired_delta_vs_source_index']['delta']:.6f}`.
- Decision: `{summary['headline']}` for this Mac continuation; no foreground GPU unless a future row earns `PROVISIONAL_PROMOTE_TO_GPU`.
"""
    if marker in existing:
        start = existing.index(f"## {marker}")
        next_start = existing.find("\n## ", start + 1)
        if next_start == -1:
            updated = existing[:start].rstrip() + "\n\n" + entry.rstrip() + "\n"
        else:
            updated = existing[:start].rstrip() + "\n\n" + entry.rstrip() + "\n" + existing[next_start:]
    else:
        updated = existing.rstrip() + "\n\n" + entry.rstrip() + "\n"
    path.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded Mac-only LatentWire continuation")
    parser.add_argument("--max-scan-rows", type=int, default=120)
    parser.add_argument("--max-score-rows", type=int, default=36)
    parser.add_argument("--max-generation-rows", type=int, default=4)
    parser.add_argument("--generation-timebox-seconds", type=float, default=180.0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--source-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--i-beyond-floor", type=float, default=0.01)
    parser.add_argument("--min-gate-for-promote", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf_home" / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_home" / "transformers"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    examples = load_mmlu_pro_examples(args.max_scan_rows, args.max_score_rows)
    scored_rows = score_examples(
        examples,
        source_model_name=args.source_model,
        target_model_name=args.target_model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    info = information_beyond(scored_rows)
    wz = apply_predictions(scored_rows)
    l_b1 = l_b1_metrics(scored_rows)
    rerank = generate_rerank_slice(
        examples,
        target_model_name=args.target_model,
        device=device,
        max_rows=args.max_generation_rows,
        max_new_tokens=args.max_new_tokens,
        max_seconds=args.generation_timebox_seconds,
    )
    summary = build_outputs(
        examples=examples,
        scored_rows=scored_rows,
        info=info,
        wz=wz,
        l_b1=l_b1,
        rerank=rerank,
        args=args,
    )
    print(
        "mac continue complete: "
        f"headline={summary['headline']} status={summary['status']} "
        f"i_beyond={summary['information']['i_beyond_bits']:.6f} "
        f"gate_n={summary['wz']['gate_n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
