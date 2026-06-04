#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmc.stage1_screens import write_jsonl

RUN_ID = "20260604_corrected_reprobe"
DEFAULT_OUT = ROOT / "results" / "overnight_v3" / RUN_ID
LETTERS = "ABCDEFGHIJ"
SEED = 20260604


@dataclass(frozen=True)
class McqaExample:
    task: str
    row_id: str
    question: str
    options: list[str]
    answer_index: int
    source: str
    split: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert_no_confirm_path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_no_confirm_path(path: Path | str) -> None:
    if "_confirm" in str(path):
        raise RuntimeError(f"confirm-like path forbidden in overnight_v3: {path}")


def split_dev_gate(key: str, gate_mod: int = 10) -> str:
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


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    if len(gold) == 0:
        return 0.0
    return float(np.mean(pred == gold))


def prompt_for_mcqa(example: McqaExample) -> str:
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(example.options))
    return (
        "Answer the multiple-choice question. Reply with only the option letter.\n\n"
        f"Question: {example.question}\n"
        f"Options:\n{options}\n\n"
        "Answer:"
    )


def load_local_mcqa(path: Path, task: str, target_rows: int) -> list[McqaExample]:
    rows = []
    for index, raw in enumerate(read_jsonl(path)):
        choices = [str(item) for item in raw.get("choices", [])]
        answer_index = int(raw.get("answer_index", raw.get("answer", -1)))
        if not 2 <= len(choices) <= len(LETTERS):
            continue
        if not 0 <= answer_index < len(choices):
            continue
        row_id = str(raw.get("id") or raw.get("content_id") or f"{task}_{index:04d}")
        rows.append(
            McqaExample(
                task=task,
                row_id=row_id,
                question=str(raw["question"]),
                options=choices,
                answer_index=answer_index,
                source=str(path.relative_to(ROOT)),
                split=split_dev_gate(f"v3|{task}|{row_id}"),
            )
        )
        if len(rows) >= target_rows:
            break
    return rows


def load_mmlu_redux(target_rows: int, configs: int) -> list[McqaExample]:
    rows: list[McqaExample] = []
    try:
        names = get_dataset_config_names("edinburgh-dawg/mmlu-redux")[:configs]
    except Exception as exc:
        print(f"[EXP1] MMLU-Redux unavailable: {exc}", flush=True)
        return []
    per_config = max(1, math.ceil(target_rows / max(1, len(names))))
    for config in names:
        try:
            dataset = load_dataset(
                "edinburgh-dawg/mmlu-redux",
                config,
                split=f"test[:{per_config}]",
                cache_dir=str(ROOT / ".hf_home" / "datasets"),
            )
        except Exception as exc:
            print(f"[EXP1] skip MMLU-Redux config {config}: {exc}", flush=True)
            continue
        for index, raw in enumerate(dataset):
            choices = [str(item) for item in raw.get("choices", [])]
            answer_index = int(raw.get("answer", -1))
            if not 2 <= len(choices) <= len(LETTERS):
                continue
            if not 0 <= answer_index < len(choices):
                continue
            row_id = f"mmlu_redux_{config}_{index:04d}"
            rows.append(
                McqaExample(
                    task="mmlu_redux",
                    row_id=row_id,
                    question=str(raw["question"]),
                    options=choices,
                    answer_index=answer_index,
                    source=f"edinburgh-dawg/mmlu-redux/{config}/test",
                    split=split_dev_gate(f"v3|mmlu_redux|{row_id}"),
                )
            )
            if len(rows) >= target_rows:
                return rows
    return rows[:target_rows]


def load_exp1_tasks(rows_per_task: int, mmlu_configs: int) -> dict[str, list[McqaExample]]:
    return {
        "openbookqa": load_local_mcqa(
            ROOT / "results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_validation.jsonl",
            "openbookqa",
            rows_per_task,
        ),
        "arc_challenge": load_local_mcqa(
            ROOT / "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl",
            "arc_challenge",
            rows_per_task,
        ),
        "mmlu_redux": load_mmlu_redux(rows_per_task, mmlu_configs),
    }


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
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def clear_device(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def token_ids_for_labels(tokenizer, labels: str) -> list[int]:
    ids = []
    for label in labels:
        encoded = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(encoded) != 1:
            encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"label {label!r} is not single-token for {tokenizer.name_or_path}")
        ids.append(int(encoded[0]))
    return ids


def extract_features(
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
    prompts = [prompt_for_mcqa(example) for example in examples]
    max_options = max(len(example.options) for example in examples) if examples else 0
    label_ids = token_ids_for_labels(tokenizer, LETTERS[:max_options])
    features: list[np.ndarray] = []
    label_scores: list[np.ndarray] = []
    row_ids: list[str] = []
    started = time.time()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_examples = examples[start : start + len(batch)]
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
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            lengths = attention_mask.sum(dim=1)
            last_pos = lengths - 1
            last = hidden[torch.arange(hidden.shape[0], device=device), last_pos, :]
            masked = hidden * attention_mask.unsqueeze(-1)
            mean = masked.sum(dim=1) / lengths.unsqueeze(-1).clamp_min(1)
            feat = torch.cat([last, mean], dim=1).detach().cpu().numpy().astype(np.float32)
            logits = out.logits[torch.arange(input_ids.shape[0], device=device), last_pos, :]
            scores_all = torch.log_softmax(logits[:, label_ids], dim=-1).detach().cpu().numpy().astype(np.float32)
        for local_index, example in enumerate(batch_examples):
            scores = np.full((max_options,), -1e9, dtype=np.float32)
            scores[: len(example.options)] = scores_all[local_index, : len(example.options)]
            label_scores.append(scores)
        features.extend(feat)
        row_ids.extend(example.row_id for example in batch_examples)
        print(f"[EXP1] {role} rows {len(features)}/{len(examples)} elapsed={time.time() - started:.1f}s", flush=True)
    del model, tokenizer
    clear_device(device)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        row_ids=np.asarray(row_ids),
        features=np.vstack(features) if features else np.zeros((0, 0), dtype=np.float32),
        label_scores=np.vstack(label_scores) if label_scores else np.zeros((0, 0), dtype=np.float32),
    )
    return {
        "path": str(output_npz.relative_to(ROOT)),
        "rows": len(row_ids),
        "feature_dim": int(features[0].shape[0]) if features else 0,
        "loaded_existing": False,
    }


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, classes: int, lam: float = 10.0) -> np.ndarray:
    if len(train_y) == 0 or len(test_x) == 0:
        return np.zeros((len(test_x),), dtype=np.int64)
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
        acc = accuracy(pred, dev_y)
        if acc > best_acc:
            best_alpha = alpha
            best_acc = acc
    return best_alpha


def load_npz(path: Path) -> dict[str, np.ndarray]:
    assert_no_confirm_path(path)
    return dict(np.load(path, allow_pickle=True))


def evaluate_exp1_task(args: argparse.Namespace, task: str, examples: list[McqaExample], exp_dir: Path) -> dict[str, Any]:
    started = time.time()
    task_dir = exp_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        task_dir / "split_manifest.jsonl",
        [
            {
                "task": example.task,
                "row_id": example.row_id,
                "split": example.split,
                "source": example.source,
                "answer_index": example.answer_index,
                "option_count": len(example.options),
            }
            for example in examples
        ],
    )
    if not examples:
        summary = {
            "task": task,
            "status": "PARKED_DATA_UNAVAILABLE",
            "verdict": "PARKED_DATA_UNAVAILABLE",
            "confirm_rows_scored": 0,
            "wall_clock_seconds": time.time() - started,
        }
        write_json(task_dir / "summary.json", summary)
        return summary
    source_artifact = extract_features(
        model_name=args.source_model,
        examples=examples,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.exp1_max_length,
        output_npz=task_dir / "source_features.npz",
        role=f"{task}/source",
    )
    target_artifact = extract_features(
        model_name=args.target_model,
        examples=examples,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.exp1_max_length,
        output_npz=task_dir / "receiver_target_features.npz",
        role=f"{task}/target",
    )
    source = load_npz(task_dir / "source_features.npz")
    target = load_npz(task_dir / "receiver_target_features.npz")
    features_source = source["features"].astype(np.float32)
    features_target = target["features"].astype(np.float32)
    scores_source = source["label_scores"].astype(np.float32)
    scores_target = target["label_scores"].astype(np.float32)
    y = np.asarray([example.answer_index for example in examples], dtype=np.int64)
    dev_indices = np.asarray([i for i, example in enumerate(examples) if example.split == "dev"], dtype=np.int64)
    gate_indices = np.asarray([i for i, example in enumerate(examples) if example.split == "gate"], dtype=np.int64)
    classes = int(max(len(example.options) for example in examples))
    if len(dev_indices) == 0 or len(gate_indices) == 0:
        summary = {
            "task": task,
            "status": "INCONCLUSIVE_UNDERPOWERED",
            "verdict": "INCONCLUSIVE_UNDERPOWERED",
            "reason": "missing dev or gate rows after deterministic split",
            "achieved_n_dev": int(len(dev_indices)),
            "achieved_n_gate": int(len(gate_indices)),
            "confirm_rows_scored": 0,
            "wall_clock_seconds": time.time() - started,
        }
        write_json(task_dir / "summary.json", summary)
        return summary
    receiver_probe_pred = ridge_predict(features_target[dev_indices], y[dev_indices], features_target[gate_indices], classes)
    receiver_source_probe_pred = ridge_predict(
        np.concatenate([features_target[dev_indices], features_source[dev_indices]], axis=1),
        y[dev_indices],
        np.concatenate([features_target[gate_indices], features_source[gate_indices]], axis=1),
        classes,
    )
    receiver_model_pred = np.argmax(scores_target[gate_indices], axis=1)
    source_model_pred = np.argmax(scores_source[gate_indices], axis=1)
    receiver_probe_acc = accuracy(receiver_probe_pred, y[gate_indices])
    receiver_model_acc = accuracy(receiver_model_pred, y[gate_indices])
    source_model_acc = accuracy(source_model_pred, y[gate_indices])
    baseline_gap = receiver_model_acc - receiver_probe_acc
    hidden_probe_sane = baseline_gap <= args.exp1_sanity_tolerance
    alpha = choose_alpha(scores_target[dev_indices], scores_source[dev_indices], y[dev_indices])
    dense_pred = np.argmax(
        np.apply_along_axis(zscore, 1, scores_target[gate_indices])
        + alpha * np.apply_along_axis(zscore, 1, scores_source[gate_indices]),
        axis=1,
    )
    dense_hits = [int(pred == gold) for pred, gold in zip(dense_pred, y[gate_indices], strict=True)]
    target_hits = [int(pred == gold) for pred, gold in zip(receiver_model_pred, y[gate_indices], strict=True)]
    dense_gain = paired_ci(dense_hits, target_hits)
    if hidden_probe_sane:
        baseline_pred = receiver_probe_pred
        source_aug_pred = receiver_source_probe_pred
        baseline_mode = "hidden_receiver_probe"
    else:
        baseline_pred = receiver_model_pred
        source_aug_pred = dense_pred
        baseline_mode = "receiver_label_scores_repair"
    baseline_hits = [int(pred == gold) for pred, gold in zip(baseline_pred, y[gate_indices], strict=True)]
    source_aug_hits = [int(pred == gold) for pred, gold in zip(source_aug_pred, y[gate_indices], strict=True)]
    gain = paired_ci(source_aug_hits, baseline_hits)
    if not hidden_probe_sane and baseline_mode != "receiver_label_scores_repair":
        status = "INCONCLUSIVE_BROKEN_BASELINE"
    elif gain["mde_half_width"] > args.mde_target:
        status = "INCONCLUSIVE_UNDERPOWERED"
    elif gain["ci95_low"] > 0:
        status = "HAS_HEADROOM"
    else:
        status = "RECEIVER_LIMITED"
    raw_rows = []
    for local, index in enumerate(gate_indices):
        raw_rows.append(
            {
                "task": task,
                "row_id": examples[int(index)].row_id,
                "split": "gate",
                "answer_index": int(y[index]),
                "receiver_model_pred": int(receiver_model_pred[local]),
                "source_model_pred": int(source_model_pred[local]),
                "receiver_probe_pred": int(receiver_probe_pred[local]),
                "receiver_source_probe_pred": int(receiver_source_probe_pred[local]),
                "receiver_baseline_pred": int(baseline_pred[local]),
                "source_augmented_pred": int(source_aug_pred[local]),
                "dense_fusion_pred": int(dense_pred[local]),
                "receiver_model_correct": bool(receiver_model_pred[local] == y[index]),
                "receiver_probe_correct": bool(receiver_probe_pred[local] == y[index]),
                "source_augmented_correct": bool(source_aug_pred[local] == y[index]),
                "dense_fusion_correct": bool(dense_pred[local] == y[index]),
            }
        )
    write_jsonl(task_dir / "raw_gate_rows.jsonl", raw_rows)
    source_score_std = np.std(scores_source, axis=1)
    summary = {
        "experiment": "EXP1_corrected_cache_ceiling",
        "task": task,
        "status": status,
        "verdict": status,
        "baseline_mode": baseline_mode,
        "hidden_probe_sane": hidden_probe_sane,
        "sanity_tolerance": args.exp1_sanity_tolerance,
        "achieved_n_total": int(len(examples)),
        "achieved_n_dev": int(len(dev_indices)),
        "achieved_n_gate": int(len(gate_indices)),
        "model_own_acc": receiver_model_acc,
        "source_model_acc": source_model_acc,
        "receiver_probe_acc": receiver_probe_acc,
        "receiver_plus_source_acc": accuracy(np.asarray(source_aug_pred), y[gate_indices]),
        "probe_gain_receiver_source_vs_receiver_only": gain,
        "dense_fusion_alpha": alpha,
        "dense_fusion_acc": accuracy(dense_pred, y[gate_indices]),
        "dense_oracle_gain_vs_receiver_label_scores": dense_gain,
        "source_label_score_degenerate_fraction": float(np.mean(source_score_std < 1e-6)),
        "source_feature_artifact": source_artifact,
        "target_feature_artifact": target_artifact,
        "confirm_rows_scored": 0,
        "mde_target": args.mde_target,
        "created_utc": now_utc(),
        "wall_clock_seconds": time.time() - started,
    }
    write_json(task_dir / "summary.json", summary)
    return summary


def run_exp1(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    exp_dir = out_dir / "exp1_corrected_cache_ceiling"
    exp_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_exp1_tasks(args.exp1_rows_per_task, args.exp1_mmlu_redux_configs)
    summaries = [evaluate_exp1_task(args, task, examples, exp_dir) for task, examples in tasks.items()]
    if any(item["status"] == "HAS_HEADROOM" for item in summaries):
        status = "HAS_HEADROOM_TASK_SLICE"
    elif any(item["status"] == "INCONCLUSIVE_BROKEN_BASELINE" for item in summaries):
        status = "INCONCLUSIVE_BROKEN_BASELINE"
    elif all(item["status"] == "RECEIVER_LIMITED" for item in summaries if item.get("achieved_n_gate", 0) > 0):
        status = "RECEIVER_LIMITED"
    else:
        status = "INCONCLUSIVE_UNDERPOWERED"
    summary = {
        "experiment": "EXP1_corrected_cache_ceiling",
        "status": status,
        "verdict": status,
        "tasks": summaries,
        "confirm_rows_scored": 0,
        "created_utc": now_utc(),
        "wall_clock_seconds": time.time() - started,
    }
    write_json(exp_dir / "summary.json", summary)
    return summary


def extract_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def group_candidates(scored: list[dict[str, Any]], candidate_count: int) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_id[str(row["row_id"])].append(row)
    grouped = []
    for row_id, rows in sorted(by_id.items()):
        rows = sorted(rows, key=lambda row: int(row["candidate_index"]))[:candidate_count]
        if len(rows) < candidate_count:
            continue
        grouped.append(
            {
                "row_id": row_id,
                "split": rows[0]["split"],
                "source_scores": [float(row["source_score"]) for row in rows],
                "target_scores": [float(row["target_score"]) for row in rows],
                "verifier_scores": [float(row.get("verifier_score", 0.0)) for row in rows],
                "correct": [bool(row["correct"]) for row in rows],
            }
        )
    return grouped


def verifier_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("prompt", ""))
    candidate = str(row.get("generated_text", ""))
    return (
        "You are checking a candidate solution to a math word problem.\n\n"
        f"Problem and instructions:\n{question}\n\n"
        f"Candidate solution:\n{candidate}\n\n"
        "Does the candidate solution give the correct final numeric answer? Reply Yes or No.\nAnswer:"
    )


def score_yes_no_verifier(args: argparse.Namespace, rows: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    if output_path.exists():
        return [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    tokenizer, model = load_model(args.target_model, args.device)
    yes_id = token_ids_for_labels(tokenizer, "Y")[0]
    no_id = token_ids_for_labels(tokenizer, "N")[0]
    scored = []
    started = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [verifier_prompt(row) for row in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.exp4_verifier_max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            lengths = attention_mask.sum(dim=1)
            last_pos = lengths - 1
            logits = out.logits[torch.arange(input_ids.shape[0], device=args.device), last_pos, :]
            logp = torch.log_softmax(logits[:, [yes_id, no_id]], dim=-1).detach().cpu().numpy()
        for row, pair in zip(batch, logp, strict=True):
            item = dict(row)
            item["verifier_score"] = float(pair[0] - pair[1])
            item["verifier_score_definition"] = "logP(Yes)-logP(No) for a target-model verifier prompt; gold answer not shown"
            scored.append(item)
        print(f"[EXP4] verifier scores {len(scored)}/{len(rows)} elapsed={time.time() - started:.1f}s", flush=True)
    del model, tokenizer
    clear_device(args.device)
    write_jsonl(output_path, scored)
    return scored


def choose_exp4_weights(dev: list[dict[str, Any]]) -> tuple[float, float]:
    best = (0.0, 0.0)
    best_acc = -1.0
    for beta_step in range(-10, 21):
        beta = beta_step / 10.0
        for alpha_step in range(-20, 41):
            alpha = alpha_step / 10.0
            hits = []
            for row in dev:
                target = zscore(np.asarray(row["target_scores"]))
                verifier = zscore(np.asarray(row["verifier_scores"]))
                source = zscore(np.asarray(row["source_scores"]))
                pred = top1(target + beta * verifier + alpha * source)
                hits.append(int(row["correct"][pred]))
            acc = float(np.mean(hits)) if hits else 0.0
            if acc > best_acc:
                best = (alpha, beta)
                best_acc = acc
    return best


def run_exp4(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    exp_dir = out_dir / "exp4_corrected_l_a2_rerank_ceiling"
    exp_dir.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / args.exp4_scored_candidates
    assert_no_confirm_path(source_path)
    if not source_path.exists():
        summary = {
            "experiment": "EXP4_corrected_l_a2_rerank_ceiling",
            "status": "PARKED_MISSING_CANDIDATE_CACHE",
            "verdict": "PARKED_MISSING_CANDIDATE_CACHE",
            "missing_path": str(source_path.relative_to(ROOT)),
            "gpu_command": args.exp4_gpu_command,
            "confirm_rows_scored": 0,
            "created_utc": now_utc(),
            "wall_clock_seconds": time.time() - started,
        }
        write_json(exp_dir / "summary.json", summary)
        return summary
    base_rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    if args.require_verifier_score:
        scored = score_yes_no_verifier(args, base_rows, exp_dir / "scored_candidates_with_verifier.jsonl")
    else:
        scored = base_rows
    grouped_all = group_candidates(scored, args.exp4_candidates)
    with_correct = [row for row in grouped_all if any(row["correct"])]
    write_jsonl(
        exp_dir / "correct_candidate_subset_manifest.jsonl",
        [{"row_id": row["row_id"], "split": row["split"], "correct_count": int(sum(row["correct"]))} for row in with_correct],
    )
    if len(with_correct) < args.exp4_min_correct_prompts:
        summary = {
            "experiment": "EXP4_corrected_l_a2_rerank_ceiling",
            "status": "PARKED_NEEDS_STRONGER_GENERATOR",
            "verdict": "PARKED_NEEDS_STRONGER_GENERATOR",
            "reason": "correct-candidate subset below threshold; no L_A2 gain verdict emitted",
            "achieved_prompts": len(grouped_all),
            "achieved_candidates": len(scored),
            "prompts_with_at_least_one_correct_candidate": len(with_correct),
            "required_correct_candidate_prompts": args.exp4_min_correct_prompts,
            "verifier_score_present": bool(scored and "verifier_score" in scored[0]),
            "verifier_score_definition": scored[0].get("verifier_score_definition") if scored else None,
            "gpu_command": args.exp4_gpu_command,
            "confirm_rows_scored": 0,
            "created_utc": now_utc(),
            "wall_clock_seconds": time.time() - started,
        }
        write_json(exp_dir / "summary.json", summary)
        return summary
    dev = [row for row in with_correct if row["split"] == "dev"]
    gate = [row for row in with_correct if row["split"] == "gate"]
    alpha, beta = choose_exp4_weights(dev)
    target_verifier_hits = []
    source_aug_hits = []
    raw_gate_rows = []
    for row in gate:
        target = zscore(np.asarray(row["target_scores"]))
        verifier = zscore(np.asarray(row["verifier_scores"]))
        source = zscore(np.asarray(row["source_scores"]))
        base_pred = top1(target + beta * verifier)
        aug_pred = top1(target + beta * verifier + alpha * source)
        target_verifier_hits.append(int(row["correct"][base_pred]))
        source_aug_hits.append(int(row["correct"][aug_pred]))
        raw_gate_rows.append(
            {
                "row_id": row["row_id"],
                "receiver_only_pred": base_pred,
                "receiver_plus_source_pred": aug_pred,
                "receiver_only_correct": bool(row["correct"][base_pred]),
                "receiver_plus_source_correct": bool(row["correct"][aug_pred]),
                "correct_indices": [i for i, ok in enumerate(row["correct"]) if ok],
            }
        )
    write_jsonl(exp_dir / "raw_gate_rows.jsonl", raw_gate_rows)
    gain = paired_ci(source_aug_hits, target_verifier_hits)
    if gain["mde_half_width"] > args.mde_target:
        status = "INCONCLUSIVE_UNDERPOWERED"
    elif gain["ci95_low"] > 0:
        status = "HAS_HEADROOM"
    else:
        status = "RECEIVER_LIMITED"
    summary = {
        "experiment": "EXP4_corrected_l_a2_rerank_ceiling",
        "status": status,
        "verdict": status,
        "achieved_prompts": len(grouped_all),
        "achieved_candidates": len(scored),
        "prompts_with_at_least_one_correct_candidate": len(with_correct),
        "achieved_n_dev": len(dev),
        "achieved_n_gate": len(gate),
        "alpha_source": alpha,
        "beta_verifier": beta,
        "receiver_plus_source_gain_vs_receiver_only": gain,
        "verifier_score_present": bool(scored and "verifier_score" in scored[0]),
        "confirm_rows_scored": 0,
        "created_utc": now_utc(),
        "wall_clock_seconds": time.time() - started,
    }
    write_json(exp_dir / "summary.json", summary)
    return summary


def write_report(out_dir: Path, exp1: dict[str, Any] | None, exp4: dict[str, Any] | None) -> None:
    lines = [
        "# Overnight V3 Corrected Re-Probe",
        "",
        "- locality: CPU-only local run; no CUDA, foreground GPU, SSH, or confirm access.",
        "- interpretation: broken or underpowered sanity gates are not negative method evidence.",
        "",
        "## Summary",
        "",
        "| probe | status | wall_clock_seconds | key n | sanity/gate note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    if exp1:
        task_status = ", ".join(f"{row['task']}={row['status']}" for row in exp1.get("tasks", []))
        key_n = sum(int(row.get("achieved_n_gate", 0)) for row in exp1.get("tasks", []))
        lines.append(f"| EXP1 corrected cache ceiling | `{exp1['status']}` | {exp1['wall_clock_seconds']:.1f} | {key_n} | {task_status} |")
    if exp4:
        key_n = int(exp4.get("prompts_with_at_least_one_correct_candidate", exp4.get("achieved_n_gate", 0)))
        lines.append(f"| EXP4 corrected L-A2 rerank | `{exp4['status']}` | {exp4['wall_clock_seconds']:.1f} | {key_n} | correct-candidate/verifier gate |")
    lines.extend(["", "## EXP1 Task Details", ""])
    if exp1:
        for row in exp1.get("tasks", []):
            lines.extend(
                [
                    f"### {row['task']}",
                    "",
                    f"- status: `{row['status']}`",
                    f"- n total/dev/gate: `{row.get('achieved_n_total')}` / `{row.get('achieved_n_dev')}` / `{row.get('achieved_n_gate')}`",
                    f"- model own acc: `{row.get('model_own_acc')}`",
                    f"- receiver probe acc: `{row.get('receiver_probe_acc')}`",
                    f"- baseline mode: `{row.get('baseline_mode')}`; hidden probe sane: `{row.get('hidden_probe_sane')}`",
                    f"- receiver+source acc: `{row.get('receiver_plus_source_acc')}`",
                    f"- gain+CI+MDE: `{row.get('probe_gain_receiver_source_vs_receiver_only')}`",
                    f"- dense fusion alpha/gain: `{row.get('dense_fusion_alpha')}` / `{row.get('dense_oracle_gain_vs_receiver_label_scores')}`",
                    "",
                ]
            )
    lines.extend(["## EXP4 Details", ""])
    if exp4:
        lines.extend(
            [
                f"- status: `{exp4['status']}`",
                f"- prompts/candidates: `{exp4.get('achieved_prompts')}` / `{exp4.get('achieved_candidates')}`",
                f"- prompts with >=1 correct candidate: `{exp4.get('prompts_with_at_least_one_correct_candidate')}`",
                f"- verifier score present: `{exp4.get('verifier_score_present')}`",
                f"- gain+CI+MDE: `{exp4.get('receiver_plus_source_gain_vs_receiver_only')}`",
                f"- GPU/stronger-generator command if parked: `{exp4.get('gpu_command')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{out_dir.relative_to(ROOT)}/exp1_corrected_cache_ceiling/summary.json`",
            f"- `{out_dir.relative_to(ROOT)}/exp4_corrected_l_a2_rerank_ceiling/summary.json`",
        ]
    )
    (ROOT / "dashboard/overnight_v3_corrected.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected overnight v3 CPU-only probes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--run-exp", choices=["all", "exp1", "exp4"], default="all")
    parser.add_argument("--source-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mde-target", type=float, default=0.05)
    parser.add_argument("--no-confirm", action="store_true", help="Required: fail unless set.")
    parser.add_argument("--exp1-rows-per-task", type=int, default=120)
    parser.add_argument("--exp1-mmlu-redux-configs", type=int, default=6)
    parser.add_argument("--exp1-max-length", type=int, default=384)
    parser.add_argument("--exp1-sanity-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--exp4-scored-candidates",
        default="results/overnight_v2/20260604_forced_powered_probes/exp4_l_a2_generated_solution_ceiling/scored_candidates.jsonl",
    )
    parser.add_argument("--exp4-prompts", type=int, default=300)
    parser.add_argument("--exp4-candidates", type=int, default=16)
    parser.add_argument("--exp4-max-new-tokens", type=int, default=96)
    parser.add_argument("--exp4-min-correct-prompts", type=int, default=80)
    parser.add_argument("--exp4-verifier-max-length", type=int, default=512)
    parser.add_argument("--require-verifier-score", action="store_true")
    parser.add_argument(
        "--exp4-gpu-command",
        default=(
            "GPU stronger-generator cache: generate >=300 dev/gate GSM8K/MATH prompts x16 candidates "
            "with a stronger generator, then score each candidate with source_score, target_score, and verifier_score; "
            "do not screen L_A2 until >=80 prompts have at least one correct candidate."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.no_confirm:
        raise SystemExit("--no-confirm is required")
    assert_no_confirm_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    exp1_summary = None
    exp4_summary = None
    if args.run_exp in {"all", "exp1"}:
        exp1_summary = run_exp1(args, args.out_dir)
    else:
        summary_path = args.out_dir / "exp1_corrected_cache_ceiling/summary.json"
        if summary_path.exists():
            exp1_summary = json.loads(summary_path.read_text())
    if args.run_exp in {"all", "exp4"}:
        exp4_summary = run_exp4(args, args.out_dir)
    else:
        summary_path = args.out_dir / "exp4_corrected_l_a2_rerank_ceiling/summary.json"
        if summary_path.exists():
            exp4_summary = json.loads(summary_path.read_text())
    write_report(args.out_dir, exp1_summary, exp4_summary)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.relative_to(ROOT)),
                "exp1": exp1_summary["status"] if exp1_summary else "NOT_RUN",
                "exp4": exp4_summary["status"] if exp4_summary else "NOT_RUN",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
