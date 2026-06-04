#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
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


OUT_DIR = ROOT / "results" / "mac_continue" / "latentwire_oracle_ladder"
RAW_ROWS = OUT_DIR / "oracle_ladder_rows.jsonl"
SPLIT_MANIFEST = OUT_DIR / "fresh_split_manifest.jsonl"
SUMMARY_JSON = OUT_DIR / "summary.json"
SUMMARY_MD = OUT_DIR / "summary.md"
LEADERBOARD = OUT_DIR / "leaderboard.csv"
LETTERS = "ABCDEFGHIJ"
SEED = 20260604


@dataclass(frozen=True)
class Example:
    row_id: str
    question: str
    options: list[str]
    answer_index: int
    category: str
    split: str


def stable_split(row_id: str) -> str:
    return split_for_key(f"latentwire_oracle_ladder|mmlu_pro|{row_id}")


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.mean(values))) / std


def top1(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def margin(scores: np.ndarray) -> float:
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0


def prompt_for(example: Example) -> str:
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(example.options))
    return (
        "Answer the multiple-choice question. Reply with only the option letter.\n\n"
        f"Question: {example.question}\n"
        f"Options:\n{options}\n\n"
        "Answer:"
    )


def load_examples(max_scan_rows: int, target_scored_rows: int) -> list[Example]:
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split=f"test[:{max_scan_rows}]",
        cache_dir=str(ROOT / ".hf_home" / "datasets"),
    )
    examples: list[Example] = []
    scored = 0
    for row in dataset:
        options = [str(option) for option in row["options"]]
        answer_index = int(row["answer_index"])
        if len(options) != 10 or answer_index < 0 or answer_index >= len(options):
            continue
        row_id = f"mmlu_pro_{row['question_id']}"
        split = stable_split(row_id)
        examples.append(
            Example(
                row_id=row_id,
                question=str(row["question"]),
                options=options,
                answer_index=answer_index,
                category=str(row.get("category", "")),
                split=split,
            )
        )
        if split in {"dev", "gate"}:
            scored += 1
        if scored >= target_scored_rows:
            break
    return examples


def load_done_rows() -> dict[str, dict[str, Any]]:
    if not RAW_ROWS.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with RAW_ROWS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["row_id"])] = row
    return rows


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".hf_home" / "transformers"),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32
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


def label_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for letter in LETTERS:
        encoded = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(encoded) != 1:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"label {letter!r} is not single-token for {tokenizer.name_or_path}")
        ids.append(int(encoded[0]))
    return ids


def score_label_logits(model, tokenizer, examples: list[Example], *, device: str, batch_size: int, max_length: int) -> list[list[float]]:
    label_ids = label_token_ids(tokenizer)
    prompts = [prompt_for(example) for example in examples]
    scores: list[list[float]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            last_positions = attention_mask.sum(dim=1) - 1
            next_logits = logits[torch.arange(input_ids.shape[0], device=device), last_positions, :]
            log_probs = torch.log_softmax(next_logits, dim=-1)
            selected = log_probs[:, label_ids].detach().cpu().numpy()
        scores.extend([[float(value) for value in row] for row in selected])
    return scores


def score_missing(
    examples: list[Example],
    done_rows: dict[str, dict[str, Any]],
    *,
    source_model_name: str,
    target_model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    todo = [example for example in examples if example.split in {"dev", "gate"} and example.row_id not in done_rows]
    if not todo:
        return sorted(done_rows.values(), key=lambda row: row["row_id"])

    source_tokenizer, source_model = load_model(source_model_name, device)
    source_scores = score_label_logits(source_model, source_tokenizer, todo, device=device, batch_size=batch_size, max_length=max_length)
    del source_model, source_tokenizer

    target_tokenizer, target_model = load_model(target_model_name, device)
    target_scores = score_label_logits(target_model, target_tokenizer, todo, device=device, batch_size=batch_size, max_length=max_length)
    del target_model, target_tokenizer

    new_rows: list[dict[str, Any]] = []
    for example, source, target in zip(todo, source_scores, target_scores, strict=True):
        source_arr = np.asarray(source, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        new_rows.append(
            {
                "row_id": example.row_id,
                "split": example.split,
                "category": example.category,
                "answer_index": example.answer_index,
                "option_count": len(example.options),
                "score_surface": "next_token_label_logprob",
                "source_scores": source,
                "target_scores": target,
                "source_top1": top1(source_arr),
                "target_top1": top1(target_arr),
                "source_correct": top1(source_arr) == example.answer_index,
                "target_correct": top1(target_arr) == example.answer_index,
                "source_margin": margin(source_arr),
                "target_margin": margin(target_arr),
            }
        )
    merged = {**done_rows, **{row["row_id"]: row for row in new_rows}}
    ordered = sorted(merged.values(), key=lambda row: row["row_id"])
    write_jsonl(RAW_ROWS, ordered)
    return ordered


def quantize(values: np.ndarray, bits: int) -> np.ndarray:
    values = zscore(values)
    if bits >= 8:
        return values
    levels = (2**bits) - 1
    clipped = np.clip(values, -2.0, 2.0)
    quantized = np.rint((clipped + 2.0) * levels / 4.0)
    return (quantized * 4.0 / levels) - 2.0


def empirical_mi(x_values: list[Any], y_values: list[Any]) -> float:
    n = len(x_values)
    if n == 0:
        return 0.0
    joint = Counter(zip(x_values, y_values, strict=True))
    x_counts = Counter(x_values)
    y_counts = Counter(y_values)
    mi = 0.0
    for (x_value, y_value), count in joint.items():
        pxy = count / n
        px = x_counts[x_value] / n
        py = y_counts[y_value] / n
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log2(pxy / (px * py))
    return float(mi)


def conditional_mi(x_values: list[Any], y_values: list[Any], z_values: list[Any]) -> float:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, z_value in enumerate(z_values):
        groups[z_value].append(index)
    n = len(x_values)
    total = 0.0
    for indices in groups.values():
        xs = [x_values[index] for index in indices]
        ys = [y_values[index] for index in indices]
        total += (len(indices) / n) * empirical_mi(xs, ys)
    return float(total)


def score_signature(row: dict[str, Any], key: str, bins: int = 4) -> tuple[int, ...]:
    scores = zscore(np.asarray(row[key], dtype=np.float64))
    levels = bins - 1
    out = []
    for value in scores:
        clipped = max(-2.0, min(2.0, float(value)))
        out.append(int(round((clipped + 2.0) * levels / 4.0)))
    return tuple(out)


def paired_delta_ci(a_correct: list[int], b_correct: list[int], samples: int = 1000) -> dict[str, Any]:
    if len(a_correct) != len(b_correct):
        raise ValueError("paired vectors must have same length")
    if not a_correct:
        return {"n": 0, "delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "mde_half_width": 1.0}
    diffs = [a - b for a, b in zip(a_correct, b_correct, strict=True)]
    rng = random.Random(SEED)
    boot = []
    for _ in range(samples):
        boot.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    boot.sort()
    delta = float(sum(diffs) / len(diffs))
    low = float(boot[int(0.025 * (len(boot) - 1))])
    high = float(boot[int(0.975 * (len(boot) - 1))])
    return {
        "n": len(diffs),
        "delta": delta,
        "ci95_low": low,
        "ci95_high": high,
        "mde_half_width": max(abs(delta - low), abs(high - delta)),
    }


def choose_alpha(rows: list[dict[str, Any]], dev: list[int], source_key: str) -> float:
    best_alpha = 0.0
    best_acc = -1.0
    for step in range(-20, 41):
        alpha = step / 10.0
        correct = 0
        for index in dev:
            row = rows[index]
            pred = top1(zscore(np.asarray(row["target_scores"])) + alpha * np.asarray(row[source_key]))
            correct += int(pred == row["answer_index"])
        acc = correct / len(dev) if dev else 0.0
        if acc > best_acc:
            best_alpha, best_acc = alpha, acc
    return best_alpha


def choose_switch_threshold(rows: list[dict[str, Any]], dev: list[int]) -> float:
    values = sorted({float(rows[i]["source_margin"]) - float(rows[i]["target_margin"]) for i in dev})
    if not values:
        return float("inf")
    best_threshold = values[0]
    best_acc = -1.0
    for threshold in [values[0] - 1e-9, *values, values[-1] + 1e-9]:
        correct = 0
        for index in dev:
            row = rows[index]
            use_source = (float(row["source_margin"]) - float(row["target_margin"])) >= threshold
            pred = int(row["source_top1"] if use_source else row["target_top1"])
            correct += int(pred == row["answer_index"])
        acc = correct / len(dev)
        if acc > best_acc:
            best_threshold, best_acc = threshold, acc
    return best_threshold


def apply_ladder(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = [i for i, row in enumerate(rows) if row["split"] == "dev"]
    gate = [i for i, row in enumerate(rows) if row["split"] == "gate"]
    rng = random.Random(SEED)

    for row in rows:
        source = np.asarray(row["source_scores"], dtype=np.float64)
        target = np.asarray(row["target_scores"], dtype=np.float64)
        row["source_full"] = zscore(source)
        row["source_q1"] = quantize(source, 1)
        row["source_q2"] = quantize(source, 2)
        row["source_q4"] = quantize(source, 4)
        row["source_q8"] = quantize(source, 8)

    budget_defs = {
        "wz_1bit": "source_q1",
        "wz_2bit": "source_q2",
        "current_packet_4bit": "source_q4",
        "wz_8bit": "source_q8",
        "full_source_score_fusion_oracle": "source_full",
    }
    alphas = {name: choose_alpha(rows, dev, key) for name, key in budget_defs.items()}
    switch_threshold = choose_switch_threshold(rows, dev)

    dev_acc: dict[str, float] = {}
    predictions: dict[str, list[int]] = defaultdict(list)
    correct: dict[str, list[int]] = defaultdict(list)
    for split_name, indices in (("dev", dev), ("gate", gate)):
        for index in indices:
            row = rows[index]
            answer = int(row["answer_index"])
            target = zscore(np.asarray(row["target_scores"], dtype=np.float64))
            row_preds: dict[str, int] = {
                "target_only": int(row["target_top1"]),
                "source_index": int(row["source_top1"]),
                "same_byte_text_proxy": int(row["source_top1"]),
                "random_same_byte": rng.randrange(int(row["option_count"])),
            }
            use_source = (float(row["source_margin"]) - float(row["target_margin"])) >= switch_threshold
            row_preds["source_index_confidence"] = int(row["source_top1"] if use_source else row["target_top1"])
            for name, source_key in budget_defs.items():
                row_preds[name] = top1(target + alphas[name] * np.asarray(row[source_key], dtype=np.float64))
            row_preds["deployable_wz"] = max(
                ("wz_1bit", "wz_2bit", "current_packet_4bit", "wz_8bit"),
                key=lambda name: dev_acc.get(name, 0.0),
            )
            row_preds["deployable_wz"] = row_preds[row_preds["deployable_wz"]]
            row_preds["best_equal_byte_score_sketch"] = row_preds[
                max(("wz_1bit", "wz_2bit", "current_packet_4bit"), key=lambda name: dev_acc.get(name, 0.0))
            ]
            row_preds["source_target_at_encoder_upper_bound"] = answer if (
                row_preds["target_only"] == answer
                or row_preds["source_index"] == answer
                or row_preds["full_source_score_fusion_oracle"] == answer
            ) else row_preds["full_source_score_fusion_oracle"]
            for name, pred in row_preds.items():
                if split_name == "dev":
                    predictions[f"dev:{name}"].append(pred)
                    correct[f"dev:{name}"].append(int(pred == answer))
                else:
                    predictions[name].append(pred)
                    correct[name].append(int(pred == answer))
        if split_name == "dev":
            for key, values in list(correct.items()):
                if key.startswith("dev:"):
                    dev_acc[key.removeprefix("dev:")] = sum(values) / len(values) if values else 0.0

    gate_acc = {name: sum(values) / len(values) if values else 0.0 for name, values in correct.items() if not name.startswith("dev:")}
    non_oracle_baselines = [
        "target_only",
        "source_index",
        "source_index_confidence",
        "best_equal_byte_score_sketch",
        "same_byte_text_proxy",
        "current_packet_4bit",
        "random_same_byte",
    ]
    best_baseline = max(non_oracle_baselines, key=lambda name: gate_acc.get(name, 0.0))
    deltas = {
        name: paired_delta_ci(correct[name], correct[best_baseline])
        for name in [
            "deployable_wz",
            "current_packet_4bit",
            "full_source_score_fusion_oracle",
            "source_target_at_encoder_upper_bound",
        ]
    }
    return {
        "dev_n": len(dev),
        "gate_n": len(gate),
        "dev_accuracy": dev_acc,
        "gate_accuracy": gate_acc,
        "best_baseline": best_baseline,
        "deltas_vs_best_baseline": deltas,
        "delta_beyond_score": paired_delta_ci(correct["deployable_wz"], correct["source_index"]),
        "budget_alphas": alphas,
        "switch_threshold": switch_threshold,
        "correct_vectors": correct,
    }


def information_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["split"] in {"dev", "gate"}]
    y = [int(row["answer_index"]) for row in valid]
    source_top = [int(row["source_top1"]) for row in valid]
    source_sig = [score_signature(row, "source_scores") for row in valid]
    target_sig = [score_signature(row, "target_scores") for row in valid]
    return {
        "n": len(valid),
        "i_source_scores_answer_given_source_top1_bits": conditional_mi(source_sig, y, source_top),
        "i_source_scores_answer_given_source_top1_target_scores_bits": conditional_mi(
            source_sig, y, list(zip(source_top, target_sig, strict=True))
        ),
    }


def classify(summary: dict[str, Any], mde_target: float) -> str:
    ladder = summary["ladder"]
    info = summary["information"]
    achieved_mde = float(ladder["delta_beyond_score"]["mde_half_width"])
    if achieved_mde > mde_target:
        return "INCONCLUSIVE_UNDERPOWERED"
    full_source = ladder["deltas_vs_best_baseline"]["full_source_score_fusion_oracle"]
    deployable = ladder["deltas_vs_best_baseline"]["deployable_wz"]
    upper = ladder["deltas_vs_best_baseline"]["source_target_at_encoder_upper_bound"]
    receiver_cmi = float(info["i_source_scores_answer_given_source_top1_target_scores_bits"])
    if full_source["ci95_high"] <= 0 and receiver_cmi <= 0.01:
        return "information-limited"
    if full_source["ci95_low"] > 0 and deployable["ci95_low"] <= 0:
        return "codec-limited"
    gate_acc = ladder["gate_accuracy"]
    if gate_acc.get("wz_8bit", 0.0) > gate_acc.get("current_packet_4bit", 0.0) and deployable["ci95_low"] > 0:
        return "rate/phase-transition"
    if upper["ci95_low"] > 0 and full_source["ci95_low"] <= 0:
        return "not-deployable"
    return "decoder-limited"


def update_dashboards(summary: dict[str, Any]) -> None:
    ladder = summary["ladder"]
    info = summary["information"]
    next_scored = "not needed"
    if not summary["powered"]:
        next_gate = math.ceil(
            ladder["gate_n"]
            * (ladder["delta_beyond_score"]["mde_half_width"] / summary["mde_target"]) ** 2
        )
        next_scored = str(math.ceil(next_gate / 0.20))
    lines = [
        "# LatentWire Oracle Ladder",
        "",
        f"- classification: `{summary['classification']}`",
        f"- powered: `{summary['powered']}`",
        f"- achieved delta_beyond_score MDE half-width: `{ladder['delta_beyond_score']['mde_half_width']:.6f}`",
        f"- MDE target: `{summary['mde_target']:.6f}`",
        f"- scored dev/gate rows: `{summary['scored_dev_gate_rows']}`",
        f"- split counts: `{summary['split_counts']}`",
        f"- confirm rows scored: `{summary['confirm_rows_scored']}`",
        f"- next estimated scored rows if underpowered: `{next_scored}`",
        "",
        "## Mutual Information",
        "",
        f"- I(source_scores; correct_option | source_top1): `{info['i_source_scores_answer_given_source_top1_bits']:.6f}` bits",
        f"- I(source_scores; correct_option | source_top1, target_scores): `{info['i_source_scores_answer_given_source_top1_target_scores_bits']:.6f}` bits",
        "",
        "## Gate Ladder",
        "",
        f"- best non-oracle baseline: `{ladder['best_baseline']}`",
        f"- gate accuracy: `{ladder['gate_accuracy']}`",
        f"- deployable WZ delta vs best baseline: `{ladder['deltas_vs_best_baseline']['deployable_wz']}`",
        f"- current packet delta vs best baseline: `{ladder['deltas_vs_best_baseline']['current_packet_4bit']}`",
        f"- full source-score oracle delta vs best baseline: `{ladder['deltas_vs_best_baseline']['full_source_score_fusion_oracle']}`",
        f"- source+target-at-encoder upper bound delta vs best baseline: `{ladder['deltas_vs_best_baseline']['source_target_at_encoder_upper_bound']}`",
        "",
        "The source+target-at-encoder row is an upper-bound diagnostic only; it is not deployable and must not be reported as a method.",
        "",
        "No confirm rows were scored; this is dev/gate screening evidence only.",
    ]
    (ROOT / "dashboard" / "latentwire_oracle_ladder.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    SUMMARY_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_summary(summary: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(summary, default=lambda value: None))
    SUMMARY_JSON.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        {
            "method_id": "L_ORACLE_POWERED_LADDER",
            "paper": "latentwire",
            "split": "dev_gate",
            "status": summary["classification"],
            "source_path": str(RAW_ROWS.relative_to(ROOT)),
            "matched_condition": "deployable_wz",
            "matched_accuracy": summary["ladder"]["gate_accuracy"].get("deployable_wz"),
            "best_baseline_condition": summary["ladder"]["best_baseline"],
            "best_baseline_accuracy": summary["ladder"]["gate_accuracy"].get(summary["ladder"]["best_baseline"]),
            "delta_vs_best_baseline": summary["ladder"]["deltas_vs_best_baseline"]["deployable_wz"]["delta"],
            "ci95_low_vs_best_baseline": summary["ladder"]["deltas_vs_best_baseline"]["deployable_wz"]["ci95_low"],
            "ci95_high_vs_best_baseline": summary["ladder"]["deltas_vs_best_baseline"]["deployable_wz"]["ci95_high"],
            "paired_n": summary["ladder"]["gate_n"],
            "regime": "fresh_mmlu_pro_oracle_ladder",
            "n": summary["scored_dev_gate_rows"],
            "median_recovery": "",
            "cvar25_recovery": "",
            "worst_recovery": "",
            "no_gap_rows": "",
            "total_rows": summary["scored_dev_gate_rows"],
            "note": "CPU next-token label scores; no confirm rows; no PASSED status",
        }
    ]
    write_csv(LEADERBOARD, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-only LatentWire powered oracle ladder")
    parser.add_argument("--target-scored-rows", type=int, default=900)
    parser.add_argument("--max-scan-rows", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--source-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--mde-target", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.max_scan_rows, args.target_scored_rows)
    manifest = [
        {
            "row_id": example.row_id,
            "split": example.split,
            "category": example.category,
            "option_count": len(example.options),
        }
        for example in examples
    ]
    write_jsonl(SPLIT_MANIFEST, manifest)
    rows = score_missing(
        examples,
        load_done_rows(),
        source_model_name=args.source_model,
        target_model_name=args.target_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    rows = [row for row in rows if row["row_id"] in {example.row_id for example in examples}]
    ladder = apply_ladder(rows)
    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "TIGER-Lab/MMLU-Pro",
        "score_surface": "next_token_label_logprob",
        "source_model": args.source_model,
        "target_model": args.target_model,
        "split_counts": dict(Counter(example.split for example in examples)),
        "confirm_rows_scored": 0,
        "scored_dev_gate_rows": len(rows),
        "mde_target": args.mde_target,
        "information": information_metrics(rows),
        "ladder": {key: value for key, value in ladder.items() if key != "correct_vectors"},
    }
    summary["classification"] = classify(summary, args.mde_target)
    summary["powered"] = summary["classification"] != "INCONCLUSIVE_UNDERPOWERED"
    update_dashboards(summary)
    write_summary(summary)
    print(
        "oracle ladder complete:",
        f"classification={summary['classification']}",
        f"powered={summary['powered']}",
        f"gate_n={summary['ladder']['gate_n']}",
        f"mde={summary['ladder']['delta_beyond_score']['mde_half_width']:.6f}",
    )


if __name__ == "__main__":
    main()
