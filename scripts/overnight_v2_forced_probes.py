#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
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

from pmc.stage1_screens import write_jsonl

RUN_ID = "20260604_forced_powered_probes"
DEFAULT_OUT = ROOT / "results" / "overnight_v2" / RUN_ID
LETTERS = "ABCDEFGHIJ"
SEED = 20260604


@dataclass(frozen=True)
class McqaExample:
    row_id: str
    question: str
    options: list[str]
    answer_index: int
    category: str
    split: str


@dataclass(frozen=True)
class MathExample:
    row_id: str
    prompt: str
    question: str
    answer_text: str
    split: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def split_dev_gate(key: str, gate_mod: int = 10) -> str:
    # Deterministic dev/gate-only split. No confirmation rows are created or scored.
    value = sum(ord(ch) for ch in key) % gate_mod
    return "gate" if value in {0, 1, 2} else "dev"


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.mean(values))) / std


def top1(values: np.ndarray) -> int:
    return int(np.argmax(values))


def prompt_for_mcqa(example: McqaExample) -> str:
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(example.options))
    return (
        "Answer the multiple-choice question. Reply with only the option letter.\n\n"
        f"Question: {example.question}\n"
        f"Options:\n{options}\n\n"
        "Answer:"
    )


def load_mmlu_examples(max_scan_rows: int, target_rows: int) -> list[McqaExample]:
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split=f"test[:{max_scan_rows}]",
        cache_dir=str(ROOT / ".hf_home" / "datasets"),
    )
    rows: list[McqaExample] = []
    for raw in dataset:
        options = [str(option) for option in raw["options"]]
        answer_index = int(raw["answer_index"])
        if len(options) != 10 or not 0 <= answer_index < len(options):
            continue
        row_id = f"mmlu_pro_{raw['question_id']}"
        rows.append(
            McqaExample(
                row_id=row_id,
                question=str(raw["question"]),
                options=options,
                answer_index=answer_index,
                category=str(raw.get("category", "")),
                split=split_dev_gate(f"exp1|{row_id}"),
            )
        )
        if len(rows) >= target_rows:
            break
    return rows


def load_gsm8k_examples(path: Path, prompts: int) -> list[MathExample]:
    rows = read_jsonl(path)[:prompts]
    out: list[MathExample] = []
    for index, row in enumerate(rows):
        row_id = f"gsm8k_{index:04d}"
        out.append(
            MathExample(
                row_id=row_id,
                prompt=str(row["prompt"]),
                question=str(row.get("source_question", row["prompt"])),
                answer_text=str(row["answer_text"]),
                split=split_dev_gate(f"exp4|{row_id}", gate_mod=10),
            )
        )
    return out


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".hf_home" / "transformers"),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".hf_home" / "transformers"),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def clear_device(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


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


def extract_hidden_features(
    *,
    model_name: str,
    examples: list[McqaExample],
    device: str,
    batch_size: int,
    max_length: int,
    output_npz: Path,
    role: str,
) -> dict[str, Any]:
    if output_npz.exists():
        data = np.load(output_npz, allow_pickle=True)
        return {
            "path": str(output_npz.relative_to(ROOT)),
            "rows": int(data["features"].shape[0]),
            "feature_dim": int(data["features"].shape[1]),
            "loaded_existing": True,
        }
    tokenizer, model = load_model(model_name, device)
    label_ids = label_token_ids(tokenizer)
    prompts = [prompt_for_mcqa(example) for example in examples]
    features: list[np.ndarray] = []
    label_scores: list[np.ndarray] = []
    row_ids: list[str] = []
    started = time.time()
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
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]
            lengths = attention_mask.sum(dim=1)
            last_pos = lengths - 1
            last = hidden[torch.arange(hidden.shape[0], device=device), last_pos, :]
            masked = hidden * attention_mask.unsqueeze(-1)
            mean = masked.sum(dim=1) / lengths.unsqueeze(-1).clamp_min(1)
            feat = torch.cat([last, mean], dim=1).detach().cpu().numpy().astype(np.float32)
            logits = out.logits[torch.arange(input_ids.shape[0], device=device), last_pos, :]
            scores = torch.log_softmax(logits[:, label_ids], dim=-1).detach().cpu().numpy().astype(np.float32)
        features.extend(feat)
        label_scores.extend(scores)
        row_ids.extend(example.row_id for example in examples[start : start + len(batch)])
        print(
            f"[EXP1] {role} hidden rows {len(features)}/{len(examples)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    del model, tokenizer
    clear_device(device)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        row_ids=np.asarray(row_ids),
        features=np.vstack(features),
        label_scores=np.vstack(label_scores),
    )
    return {
        "path": str(output_npz.relative_to(ROOT)),
        "rows": len(row_ids),
        "feature_dim": int(np.vstack(features).shape[1]),
        "loaded_existing": False,
    }


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    classes: int,
    lam: float = 10.0,
) -> np.ndarray:
    train_x, test_x = standardize(train_x.astype(np.float64), test_x.astype(np.float64))
    y = np.zeros((train_x.shape[0], classes), dtype=np.float64)
    y[np.arange(train_x.shape[0]), train_y] = 1.0
    kernel = train_x @ train_x.T
    alpha = np.linalg.solve(kernel + lam * np.eye(kernel.shape[0]), y)
    scores = test_x @ train_x.T @ alpha
    return np.argmax(scores, axis=1).astype(np.int64)


def paired_ci(a: list[int], b: list[int], samples: int = 2000) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("paired vectors must have same length")
    if not a:
        return {"n": 0, "delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "mde_half_width": 1.0}
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(SEED)
    boot = []
    for _ in range(samples):
        boot.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    boot.sort()
    delta = sum(diffs) / len(diffs)
    low = boot[int(0.025 * (samples - 1))]
    high = boot[int(0.975 * (samples - 1))]
    return {
        "n": len(diffs),
        "delta": float(delta),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "mde_half_width": float(max(abs(delta - low), abs(high - delta))),
    }


def choose_alpha(dev_target: np.ndarray, dev_source: np.ndarray, dev_y: np.ndarray) -> float:
    best_alpha = 0.0
    best_acc = -1.0
    for step in range(-20, 41):
        alpha = step / 10.0
        pred = np.argmax(np.apply_along_axis(zscore, 1, dev_target) + alpha * np.apply_along_axis(zscore, 1, dev_source), axis=1)
        acc = float(np.mean(pred == dev_y)) if len(dev_y) else 0.0
        if acc > best_acc:
            best_alpha = alpha
            best_acc = acc
    return best_alpha


def empirical_mi(x_values: list[Any], y_values: list[Any]) -> float:
    n = len(x_values)
    if n == 0:
        return 0.0
    joint = Counter(zip(x_values, y_values, strict=True))
    x_counts = Counter(x_values)
    y_counts = Counter(y_values)
    out = 0.0
    for (x, y), count in joint.items():
        pxy = count / n
        px = x_counts[x] / n
        py = y_counts[y] / n
        out += pxy * math.log2(pxy / (px * py))
    return float(out)


def conditional_mi(x_values: list[Any], y_values: list[Any], z_values: list[Any]) -> float:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, z_value in enumerate(z_values):
        groups[z_value].append(index)
    total = 0.0
    n = len(x_values)
    for indices in groups.values():
        xs = [x_values[i] for i in indices]
        ys = [y_values[i] for i in indices]
        total += (len(indices) / n) * empirical_mi(xs, ys)
    return float(total)


def score_signature(values: np.ndarray, bins: int = 4) -> tuple[int, ...]:
    values = zscore(values)
    levels = bins - 1
    out = []
    for value in values:
        clipped = max(-2.0, min(2.0, float(value)))
        out.append(int(round((clipped + 2.0) * levels / 4.0)))
    return tuple(out)


def run_exp1(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    exp_dir = out_dir / "exp1_cachewire_powered_ceiling"
    exp_dir.mkdir(parents=True, exist_ok=True)
    examples = load_mmlu_examples(args.exp1_max_scan_rows, args.exp1_rows)
    write_jsonl(
        exp_dir / "split_manifest.jsonl",
        [
            {
                "row_id": example.row_id,
                "split": example.split,
                "category": example.category,
                "option_count": len(example.options),
            }
            for example in examples
        ],
    )
    target_meta = extract_hidden_features(
        model_name=args.target_model,
        examples=examples,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.exp1_max_length,
        output_npz=exp_dir / "receiver_target_features.npz",
        role="receiver",
    )
    source_meta = extract_hidden_features(
        model_name=args.source_model,
        examples=examples,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.exp1_max_length,
        output_npz=exp_dir / "source_features.npz",
        role="source",
    )
    target = np.load(exp_dir / "receiver_target_features.npz", allow_pickle=True)
    source = np.load(exp_dir / "source_features.npz", allow_pickle=True)
    target_x = target["features"]
    source_x = source["features"]
    target_scores = target["label_scores"]
    source_scores = source["label_scores"]
    y = np.asarray([example.answer_index for example in examples], dtype=np.int64)
    dev_idx = np.asarray([i for i, example in enumerate(examples) if example.split == "dev"], dtype=np.int64)
    gate_idx = np.asarray([i for i, example in enumerate(examples) if example.split == "gate"], dtype=np.int64)
    receiver_pred = ridge_predict(target_x[dev_idx], y[dev_idx], target_x[gate_idx], classes=10)
    combo_pred = ridge_predict(
        np.concatenate([target_x[dev_idx], source_x[dev_idx]], axis=1),
        y[dev_idx],
        np.concatenate([target_x[gate_idx], source_x[gate_idx]], axis=1),
        classes=10,
    )
    receiver_correct = [int(pred == answer) for pred, answer in zip(receiver_pred, y[gate_idx], strict=True)]
    combo_correct = [int(pred == answer) for pred, answer in zip(combo_pred, y[gate_idx], strict=True)]
    feature_gain = paired_ci(combo_correct, receiver_correct)

    alpha = choose_alpha(target_scores[dev_idx], source_scores[dev_idx], y[dev_idx])
    dense_pred = np.argmax(
        np.apply_along_axis(zscore, 1, target_scores[gate_idx])
        + alpha * np.apply_along_axis(zscore, 1, source_scores[gate_idx]),
        axis=1,
    )
    target_pred = np.argmax(target_scores[gate_idx], axis=1)
    dense_correct = [int(pred == answer) for pred, answer in zip(dense_pred, y[gate_idx], strict=True)]
    target_correct = [int(pred == answer) for pred, answer in zip(target_pred, y[gate_idx], strict=True)]
    dense_gain = paired_ci(dense_correct, target_correct)

    powered = len(examples) >= args.exp1_min_rows and feature_gain["mde_half_width"] <= args.mde_target
    if powered and feature_gain["ci95_low"] > 0:
        verdict = "CACHEWIRE_HAS_HEADROOM"
    elif powered:
        verdict = "RECEIVER_LIMITED"
    else:
        verdict = "INCONCLUSIVE_UNDERPOWERED"

    raw_rows = []
    for local_index, global_index in enumerate(gate_idx.tolist()):
        raw_rows.append(
            {
                "row_id": examples[global_index].row_id,
                "split": "gate",
                "answer_index": int(y[global_index]),
                "receiver_feature_pred": int(receiver_pred[local_index]),
                "receiver_source_feature_pred": int(combo_pred[local_index]),
                "receiver_feature_correct": bool(receiver_correct[local_index]),
                "receiver_source_feature_correct": bool(combo_correct[local_index]),
                "target_score_pred": int(target_pred[local_index]),
                "dense_fusion_score_pred": int(dense_pred[local_index]),
                "target_score_correct": bool(target_correct[local_index]),
                "dense_fusion_score_correct": bool(dense_correct[local_index]),
            }
        )
    write_jsonl(exp_dir / "raw_gate_rows.jsonl", raw_rows)
    summary = {
        "experiment": "EXP1_cachewire_powered_ceiling",
        "created_utc": now_utc(),
        "wall_clock_seconds": time.time() - started,
        "status": verdict,
        "verdict": verdict,
        "powered": powered,
        "mde_target": args.mde_target,
        "achieved_n_dev": int(len(dev_idx)),
        "achieved_n_gate": int(len(gate_idx)),
        "achieved_n_dev_gate": int(len(examples)),
        "minimum_n_dev_gate": args.exp1_min_rows,
        "receiver_feature_accuracy": float(np.mean(receiver_correct)) if receiver_correct else 0.0,
        "receiver_source_feature_accuracy": float(np.mean(combo_correct)) if combo_correct else 0.0,
        "feature_gain_receiver_source_vs_receiver": feature_gain,
        "dense_fusion_oracle_gain_vs_target_scores": dense_gain,
        "dense_fusion_alpha": alpha,
        "target_feature_artifact": target_meta,
        "source_feature_artifact": source_meta,
        "confirm_rows_scored": 0,
        "gpu_command_if_underpowered": (
            "venv_arm64/bin/python scripts/overnight_v2_forced_probes.py --device cpu "
            "--run-exp exp1 --exp1-rows 2500 --exp1-min-rows 500 --batch-size 2 --no-confirm"
        ),
    }
    write_json(exp_dir / "summary.json", summary)
    print(
        f"[EXP1] complete verdict={verdict} n={len(examples)} gate={len(gate_idx)} "
        f"mde={feature_gain['mde_half_width']:.4f} wall={summary['wall_clock_seconds']:.1f}s",
        flush=True,
    )
    return summary


def normalize_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def generate_candidates(args: argparse.Namespace, examples: list[MathExample], exp_dir: Path) -> list[dict[str, Any]]:
    path = exp_dir / "generated_candidates.jsonl"
    existing = read_jsonl(path)
    done = {(row["row_id"], int(row["candidate_index"])) for row in existing}
    if len(done) >= len(examples) * args.exp4_candidates:
        return existing
    tokenizer, model = load_model(args.target_model, args.device)
    rows = existing[:]
    started = time.time()
    for prompt_index, example in enumerate(examples):
        missing = [
            candidate_index
            for candidate_index in range(args.exp4_candidates)
            if (example.row_id, candidate_index) not in done
        ]
        prompt = f"{example.prompt}\nWork briefly, then end with a line of the form #### <number>."
        for batch_start in range(0, len(missing), args.exp4_generation_batch):
            batch_indices = missing[batch_start : batch_start + args.exp4_generation_batch]
            seed = SEED + prompt_index * 1000 + batch_indices[0]
            torch.manual_seed(seed)
            random.seed(seed)
            encoded = tokenizer(
                [prompt] * len(batch_indices),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.exp4_prompt_max_length,
            ).to(args.device)
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.exp4_max_new_tokens,
                    do_sample=True,
                    temperature=args.exp4_temperature,
                    top_p=args.exp4_top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for row_index, candidate_index in enumerate(batch_indices):
                text = tokenizer.decode(
                    output[row_index][encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                pred_answer = normalize_number(text)
                gold = normalize_number(example.answer_text)
                row = {
                    "row_id": example.row_id,
                    "split": example.split,
                    "candidate_index": candidate_index,
                    "prompt": prompt,
                    "generation_seed": seed,
                    "generated_text": text,
                    "pred_answer": pred_answer,
                    "gold_answer": gold,
                    "correct": bool(pred_answer is not None and gold is not None and pred_answer == gold),
                }
                rows.append(row)
                done.add((example.row_id, candidate_index))
                append_jsonl(path, row)
                print(
                    f"[EXP4] generated prompts={prompt_index + 1}/{len(examples)} "
                    f"candidate={candidate_index + 1}/{args.exp4_candidates} total={len(rows)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    del model, tokenizer
    clear_device(args.device)
    return rows


def continuation_logprobs(
    *,
    model,
    tokenizer,
    prompts: list[str],
    continuations: list[str],
    device: str,
    batch_size: int,
    max_length: int,
    role: str,
) -> list[float]:
    scores: list[float] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.time()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_continuations = continuations[start : start + batch_size]
        full = tokenizer(
            [p + c for p, c in zip(batch_prompts, batch_continuations, strict=True)],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_only = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = full["input_ids"].to(device)
        attention_mask = full["attention_mask"].to(device)
        prompt_lengths = prompt_only["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        for row_index, prompt_len in enumerate(prompt_lengths):
            token_scores = []
            for label_pos in range(max(0, int(prompt_len) - 1), labels.shape[1]):
                if attention_mask[row_index, label_pos + 1].item() == 0:
                    continue
                label = int(labels[row_index, label_pos].item())
                if label == pad_id:
                    continue
                token_scores.append(float(log_probs[row_index, label_pos, label].item()))
            scores.append(float(np.mean(token_scores)) if token_scores else -1e9)
        print(f"[EXP4] scored {role} candidates {len(scores)}/{len(prompts)} elapsed={time.time() - started:.1f}s", flush=True)
    return scores


def score_candidates(args: argparse.Namespace, exp_dir: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_path = exp_dir / "scored_candidates.jsonl"
    existing = read_jsonl(scored_path)
    if len(existing) == len(candidates) and all("source_score" in row and "target_score" in row for row in existing):
        return existing
    prompts = [str(row["prompt"]) for row in candidates]
    continuations = [str(row["generated_text"]) for row in candidates]
    tokenizer, model = load_model(args.source_model, args.device)
    source_scores = continuation_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        continuations=continuations,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.exp4_score_max_length,
        role="source",
    )
    del model, tokenizer
    clear_device(args.device)
    tokenizer, model = load_model(args.target_model, args.device)
    target_scores = continuation_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        continuations=continuations,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.exp4_score_max_length,
        role="target",
    )
    del model, tokenizer
    clear_device(args.device)
    scored = []
    for row, source_score, target_score in zip(candidates, source_scores, target_scores, strict=True):
        out = dict(row)
        out["source_score"] = float(source_score)
        out["target_score"] = float(target_score)
        scored.append(out)
    write_jsonl(scored_path, scored)
    return scored


def group_candidates(scored: list[dict[str, Any]], candidate_count: int) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_id[str(row["row_id"])].append(row)
    grouped = []
    for row_id, rows in sorted(by_id.items()):
        rows = sorted(rows, key=lambda row: int(row["candidate_index"]))
        if len(rows) < candidate_count:
            continue
        grouped.append(
            {
                "row_id": row_id,
                "split": rows[0]["split"],
                "source_scores": [float(row["source_score"]) for row in rows[:candidate_count]],
                "target_scores": [float(row["target_score"]) for row in rows[:candidate_count]],
                "correct": [bool(row["correct"]) for row in rows[:candidate_count]],
            }
        )
    return grouped


def run_exp4(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    exp_dir = out_dir / "exp4_l_a2_generated_solution_ceiling"
    exp_dir.mkdir(parents=True, exist_ok=True)
    examples = load_gsm8k_examples(ROOT / "data" / "gsm8k_100.jsonl", args.exp4_prompts)
    write_jsonl(
        exp_dir / "split_manifest.jsonl",
        [{"row_id": ex.row_id, "split": ex.split, "answer_text": ex.answer_text} for ex in examples],
    )
    candidates = generate_candidates(args, examples, exp_dir)
    scored = score_candidates(args, exp_dir, candidates)
    grouped = group_candidates(scored, args.exp4_candidates)
    dev = [row for row in grouped if row["split"] == "dev"]
    gate = [row for row in grouped if row["split"] == "gate"]
    best_alpha = 0.0
    best_acc = -1.0
    for step in range(-20, 41):
        alpha = step / 10.0
        hits = []
        for row in dev:
            target = zscore(np.asarray(row["target_scores"]))
            source = zscore(np.asarray(row["source_scores"]))
            pred = top1(target + alpha * source)
            hits.append(int(row["correct"][pred]))
        acc = float(np.mean(hits)) if hits else 0.0
        if acc > best_acc:
            best_alpha = alpha
            best_acc = acc
    target_hits: list[int] = []
    source_hits: list[int] = []
    combo_hits: list[int] = []
    raw_gate_rows = []
    for row in gate:
        target = zscore(np.asarray(row["target_scores"]))
        source = zscore(np.asarray(row["source_scores"]))
        target_pred = top1(target)
        source_pred = top1(source)
        combo_pred = top1(target + best_alpha * source)
        target_hits.append(int(row["correct"][target_pred]))
        source_hits.append(int(row["correct"][source_pred]))
        combo_hits.append(int(row["correct"][combo_pred]))
        raw_gate_rows.append(
            {
                "row_id": row["row_id"],
                "target_pred": target_pred,
                "source_pred": source_pred,
                "combo_pred": combo_pred,
                "target_correct": bool(row["correct"][target_pred]),
                "source_correct": bool(row["correct"][source_pred]),
                "combo_correct": bool(row["correct"][combo_pred]),
                "correct_indices": [i for i, ok in enumerate(row["correct"]) if ok],
            }
        )
    write_jsonl(exp_dir / "raw_gate_rows.jsonl", raw_gate_rows)
    gain = paired_ci(combo_hits, target_hits)
    source_sig = [score_signature(np.asarray(row["source_scores"])) for row in grouped]
    target_sig = [score_signature(np.asarray(row["target_scores"])) for row in grouped]
    source_top = [top1(np.asarray(row["source_scores"])) for row in grouped]
    correct_set = [tuple(i for i, ok in enumerate(row["correct"]) if ok) for row in grouped]
    cmi = conditional_mi(source_sig, correct_set, list(zip(source_top, target_sig, strict=True))) if grouped else 0.0
    generated_prompts = len({row["row_id"] for row in candidates})
    total_candidates = len(candidates)
    powered = (
        generated_prompts >= args.exp4_prompts
        and total_candidates >= args.exp4_prompts * args.exp4_candidates
        and gain["mde_half_width"] <= args.mde_target
    )
    if powered and cmi >= args.exp4_compare_bits and gain["ci95_low"] > 0:
        verdict = "L_A2_HAS_HEADROOM"
    elif powered:
        verdict = "L_A2_RECEIVER_LIMITED"
    else:
        verdict = "INCONCLUSIVE_UNDERPOWERED"
    summary = {
        "experiment": "EXP4_l_a2_generated_solution_ceiling",
        "created_utc": now_utc(),
        "wall_clock_seconds": time.time() - started,
        "status": verdict,
        "verdict": verdict,
        "powered": powered,
        "mde_target": args.mde_target,
        "achieved_n_dev": len(dev),
        "achieved_n_gate": len(gate),
        "achieved_prompts": generated_prompts,
        "achieved_candidates": total_candidates,
        "required_prompts": args.exp4_prompts,
        "required_candidates_per_prompt": args.exp4_candidates,
        "target_accuracy": float(np.mean(target_hits)) if target_hits else 0.0,
        "source_accuracy": float(np.mean(source_hits)) if source_hits else 0.0,
        "combo_accuracy": float(np.mean(combo_hits)) if combo_hits else 0.0,
        "combo_gain_vs_target": gain,
        "source_evidence_cmi_bits": cmi,
        "comparison_target_bits": args.exp4_compare_bits,
        "alpha": best_alpha,
        "confirm_rows_scored": 0,
        "gpu_command_if_underpowered": (
            "venv_arm64/bin/python scripts/overnight_v2_forced_probes.py --device cpu "
            "--run-exp exp4 --exp4-prompts 300 --exp4-candidates 16 --exp4-max-new-tokens 96 --no-confirm"
        ),
    }
    write_json(exp_dir / "summary.json", summary)
    print(
        f"[EXP4] complete verdict={verdict} prompts={generated_prompts} candidates={total_candidates} "
        f"gate={len(gate)} mde={gain['mde_half_width']:.4f} wall={summary['wall_clock_seconds']:.1f}s",
        flush=True,
    )
    return summary


def write_report(out_dir: Path, exp1: dict[str, Any] | None, exp4: dict[str, Any] | None) -> None:
    rows = []
    for label, summary in [("EXP1", exp1), ("EXP4", exp4)]:
        if summary is None:
            rows.append(f"| {label} | NOT_RUN | 0 | 0 | 0 | 0 | 0 | not run |")
            continue
        rows.append(
            "| {label} | `{status}` | {wall:.1f} | {dev} | {gate} | {n} | {mde:.6f} | {verdict} |".format(
                label=label,
                status=summary["status"],
                wall=float(summary["wall_clock_seconds"]),
                dev=int(summary["achieved_n_dev"]),
                gate=int(summary["achieved_n_gate"]),
                n=int(summary.get("achieved_n_dev_gate", summary.get("achieved_prompts", 0))),
                mde=float(
                    summary.get("feature_gain_receiver_source_vs_receiver", summary.get("combo_gain_vs_target"))[
                        "mde_half_width"
                    ]
                ),
                verdict=summary["verdict"],
            )
        )
    lines = [
        "# Overnight V2 Forced Probe Report",
        "",
        "- locality: CPU-only local run; no SSH, CUDA, 30B, or confirm access.",
        "- completion rule: powered verdict only when minimum n and MDE target are met; otherwise `INCONCLUSIVE_UNDERPOWERED`.",
        "",
        "## Wall-Clock / n / MDE",
        "",
        "| probe | status | wall_clock_seconds | achieved_n_dev | achieved_n_gate | achieved_total_n | achieved_MDE | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *rows,
        "",
        "## State Updates",
        "",
        "- KVComm/C2C packet smokes: `KILLED_AS_METHOD_EVIDENCE` where deterministic controls explain the signal.",
        "- C_F: `KILLED_CONTROL_CONTAMINATED`.",
        "- CacheWire deployable trace from v1: `INCONCLUSIVE_UNDERPOWERED`, not killed.",
        "- CacheWire oracle: `ORACLE_ALIVE_UNMEASURED_AT_POWER` until EXP1 v2 is powered.",
        "- L_A2: `NEEDS_SCORE_CACHE` unless EXP4 v2 reaches generated/scored candidate power.",
        "- C_A1: `NEXT_GPU_BACKFILL`; foreground remains empty.",
        "",
        "## Artifacts",
        "",
        f"- `{(out_dir / 'exp1_cachewire_powered_ceiling' / 'summary.json').relative_to(ROOT)}`",
        f"- `{(out_dir / 'exp4_l_a2_generated_solution_ceiling' / 'summary.json').relative_to(ROOT)}`",
        "",
    ]
    if exp1 is not None:
        lines.extend(
            [
                "## EXP1 CacheWire",
                "",
                f"- receiver feature accuracy: `{exp1['receiver_feature_accuracy']:.6f}`",
                f"- receiver+source feature accuracy: `{exp1['receiver_source_feature_accuracy']:.6f}`",
                f"- gain: `{exp1['feature_gain_receiver_source_vs_receiver']}`",
                f"- dense-fusion oracle gain: `{exp1['dense_fusion_oracle_gain_vs_target_scores']}`",
                f"- GPU command if underpowered: `{exp1['gpu_command_if_underpowered']}`",
                "",
            ]
        )
    if exp4 is not None:
        lines.extend(
            [
                "## EXP4 L-A2",
                "",
                f"- generated prompts: `{exp4['achieved_prompts']}`",
                f"- generated candidates: `{exp4['achieved_candidates']}`",
                f"- source-evidence CMI: `{exp4['source_evidence_cmi_bits']:.6f}` bits",
                f"- combo gain: `{exp4['combo_gain_vs_target']}`",
                f"- GPU command if underpowered: `{exp4['gpu_command_if_underpowered']}`",
                "",
            ]
        )
    (ROOT / "dashboard" / "overnight_v2.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-exp", choices=["all", "exp1", "exp4"], default="all")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--source-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mde-target", type=float, default=0.05)
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--exp1-rows", type=int, default=500)
    parser.add_argument("--exp1-min-rows", type=int, default=500)
    parser.add_argument("--exp1-max-scan-rows", type=int, default=900)
    parser.add_argument("--exp1-max-length", type=int, default=384)
    parser.add_argument("--exp4-prompts", type=int, default=100)
    parser.add_argument("--exp4-candidates", type=int, default=16)
    parser.add_argument("--exp4-generation-batch", type=int, default=4)
    parser.add_argument("--exp4-max-new-tokens", type=int, default=96)
    parser.add_argument("--exp4-temperature", type=float, default=0.8)
    parser.add_argument("--exp4-top-p", type=float, default=0.95)
    parser.add_argument("--exp4-prompt-max-length", type=int, default=512)
    parser.add_argument("--exp4-score-max-length", type=int, default=768)
    parser.add_argument("--exp4-compare-bits", type=float, default=0.281933)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_confirm:
        raise SystemExit("Refusing to run without --no-confirm")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    exp1 = None
    exp4 = None
    if args.run_exp in {"all", "exp1"}:
        exp1 = run_exp1(args, out_dir)
    else:
        p = out_dir / "exp1_cachewire_powered_ceiling" / "summary.json"
        exp1 = json.loads(p.read_text()) if p.exists() else None
    if args.run_exp in {"all", "exp4"}:
        exp4 = run_exp4(args, out_dir)
    else:
        p = out_dir / "exp4_l_a2_generated_solution_ceiling" / "summary.json"
        exp4 = json.loads(p.read_text()) if p.exists() else None
    write_report(out_dir, exp1, exp4)
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)), "exp1": exp1 and exp1["status"], "exp4": exp4 and exp4["status"]}, indent=2))


if __name__ == "__main__":
    main()
