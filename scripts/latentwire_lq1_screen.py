#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmc.stage1_screens import write_jsonl

LADDER_PATH = ROOT / "scripts" / "latentwire_oracle_ladder.py"
spec = importlib.util.spec_from_file_location("latentwire_oracle_ladder", LADDER_PATH)
ladder = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["latentwire_oracle_ladder"] = ladder
spec.loader.exec_module(ladder)

BASE_DIR = ROOT / "results" / "mac_continue" / "latentwire_oracle_ladder"
LQ1_DIR = ROOT / "results" / "mac_continue" / "latentwire_query_packet"
CONFIRM_DIR = ROOT / "results" / "mac_continue" / "latentwire_one_way_confirm"
DEV_GATE_ROWS = BASE_DIR / "oracle_ladder_rows.jsonl"
MANIFEST = BASE_DIR / "fresh_split_manifest.jsonl"
LQ1_ROWS = LQ1_DIR / "query_packet_rows.jsonl"
LQ1_SUMMARY = LQ1_DIR / "summary.json"
CONFIRM_ROWS = CONFIRM_DIR / "confirm_rows.jsonl"
CONFIRM_SUMMARY = CONFIRM_DIR / "summary.json"
SEED = 20260604


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_examples_for_ids(row_ids: set[str], max_scan_rows: int) -> list[Any]:
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split=f"test[:{max_scan_rows}]",
        cache_dir=str(ROOT / ".hf_home" / "datasets"),
    )
    examples = []
    for row in dataset:
        row_id = f"mmlu_pro_{row['question_id']}"
        if row_id not in row_ids:
            continue
        options = [str(option) for option in row["options"]]
        if len(options) != 10:
            continue
        examples.append(
            ladder.Example(
                row_id=row_id,
                question=str(row["question"]),
                options=options,
                answer_index=int(row["answer_index"]),
                category=str(row.get("category", "")),
                split="confirm",
            )
        )
    missing = row_ids - {example.row_id for example in examples}
    if missing:
        raise RuntimeError(f"could not recover {len(missing)} manifest rows, e.g. {sorted(missing)[:3]}")
    return sorted(examples, key=lambda example: example.row_id)


def score_confirm_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    existing = {row["row_id"]: row for row in read_jsonl(CONFIRM_ROWS)}
    manifest = read_jsonl(MANIFEST)
    confirm_ids = {row["row_id"] for row in manifest if row["split"] == "confirm"}
    todo_ids = confirm_ids - set(existing)
    if todo_ids:
        examples = load_examples_for_ids(todo_ids, args.max_scan_rows)
        source_tokenizer, source_model = ladder.load_model(args.source_model, "cpu")
        source_scores = ladder.score_label_logits(
            source_model,
            source_tokenizer,
            examples,
            device="cpu",
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        del source_model, source_tokenizer

        target_tokenizer, target_model = ladder.load_model(args.target_model, "cpu")
        target_scores = ladder.score_label_logits(
            target_model,
            target_tokenizer,
            examples,
            device="cpu",
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        del target_model, target_tokenizer

        for example, source, target in zip(examples, source_scores, target_scores, strict=True):
            source_arr = np.asarray(source, dtype=np.float64)
            target_arr = np.asarray(target, dtype=np.float64)
            existing[example.row_id] = {
                "row_id": example.row_id,
                "split": "confirm",
                "category": example.category,
                "answer_index": example.answer_index,
                "option_count": 10,
                "score_surface": "next_token_label_logprob",
                "source_scores": source,
                "target_scores": target,
                "source_top1": ladder.top1(source_arr),
                "target_top1": ladder.top1(target_arr),
                "source_correct": ladder.top1(source_arr) == example.answer_index,
                "target_correct": ladder.top1(target_arr) == example.answer_index,
                "source_margin": ladder.margin(source_arr),
                "target_margin": ladder.margin(target_arr),
            }
        write_jsonl(CONFIRM_ROWS, sorted(existing.values(), key=lambda row: row["row_id"]))
    rows = [row for row in existing.values() if row["row_id"] in confirm_ids]
    return sorted(rows, key=lambda row: row["row_id"])


def apply_one_way_confirm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_summary = json.loads((BASE_DIR / "summary.json").read_text(encoding="utf-8"))
    alphas = base_summary["ladder"]["budget_alphas"]
    threshold = float(base_summary["ladder"]["switch_threshold"])
    rng = random.Random(SEED)
    vectors: dict[str, list[int]] = {
        "target_only": [],
        "source_index": [],
        "source_index_confidence": [],
        "current_packet_4bit": [],
        "deployable_wz": [],
        "full_source_score_fusion_oracle": [],
        "source_target_at_encoder_upper_bound": [],
        "random_same_byte": [],
    }
    for row in rows:
        answer = int(row["answer_index"])
        source = np.asarray(row["source_scores"], dtype=np.float64)
        target = np.asarray(row["target_scores"], dtype=np.float64)
        target_z = ladder.zscore(target)
        q4 = ladder.quantize(source, 4)
        full = ladder.zscore(source)
        source_conf = int(row["source_top1"] if (float(row["source_margin"]) - float(row["target_margin"])) >= threshold else row["target_top1"])
        current = ladder.top1(target_z + float(alphas["current_packet_4bit"]) * q4)
        full_pred = ladder.top1(target_z + float(alphas["full_source_score_fusion_oracle"]) * full)
        upper = answer if answer in {int(row["target_top1"]), int(row["source_top1"]), full_pred} else full_pred
        preds = {
            "target_only": int(row["target_top1"]),
            "source_index": int(row["source_top1"]),
            "source_index_confidence": source_conf,
            "current_packet_4bit": current,
            "deployable_wz": current,
            "full_source_score_fusion_oracle": full_pred,
            "source_target_at_encoder_upper_bound": upper,
            "random_same_byte": rng.randrange(int(row["option_count"])),
        }
        for name, pred in preds.items():
            vectors[name].append(int(pred == answer))
    acc = {name: sum(values) / len(values) if values else 0.0 for name, values in vectors.items()}
    best_non_oracle = max(
        ("target_only", "source_index", "source_index_confidence", "random_same_byte"),
        key=lambda name: acc[name],
    )
    return {
        "n": len(rows),
        "accuracy": acc,
        "best_non_oracle_baseline": best_non_oracle,
        "deployable_delta_vs_best": ladder.paired_delta_ci(vectors["deployable_wz"], vectors[best_non_oracle]),
        "full_source_delta_vs_best": ladder.paired_delta_ci(vectors["full_source_score_fusion_oracle"], vectors[best_non_oracle]),
        "upper_bound_delta_vs_best": ladder.paired_delta_ci(vectors["source_target_at_encoder_upper_bound"], vectors[best_non_oracle]),
    }


def top2_indices(scores: np.ndarray) -> tuple[int, int]:
    ordered = list(np.argsort(scores))
    return int(ordered[-1]), int(ordered[-2])


def choose_quantile_thresholds(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    return tuple(float(np.quantile(arr, q)) for q in (0.25, 0.50, 0.75))


def bucket(value: float, thresholds: tuple[float, float, float]) -> int:
    return int(value > thresholds[0]) + int(value > thresholds[1]) + int(value > thresholds[2])


def pair_reply(source_scores: np.ndarray, pair: tuple[int, int], thresholds: tuple[float, float, float]) -> tuple[int, float]:
    a, b = pair
    diff = float(ladder.zscore(source_scores)[a] - ladder.zscore(source_scores)[b])
    sign = 1 if diff >= 0 else -1
    strength = (bucket(abs(diff), thresholds) + 1) / 4.0
    return sign, strength


def decode_pair(target_scores: np.ndarray, pair: tuple[int, int], sign: int, strength: float, alpha: float) -> int:
    a, b = pair
    target_z = ladder.zscore(target_scores)
    a_score = float(target_z[a]) + alpha * sign * strength
    b_score = float(target_z[b]) - alpha * sign * strength
    return int(a if a_score >= b_score else b)


def choose_lq1_alpha(rows: list[dict[str, Any]], dev_indices: list[int], reply_thresholds: tuple[float, float, float]) -> float:
    best_alpha = 0.0
    best_acc = -1.0
    for step in range(0, 41):
        alpha = step / 10.0
        correct = 0
        for index in dev_indices:
            row = rows[index]
            pair = top2_indices(np.asarray(row["target_scores"], dtype=np.float64))
            sign, strength = pair_reply(np.asarray(row["source_scores"], dtype=np.float64), pair, reply_thresholds)
            pred = decode_pair(np.asarray(row["target_scores"], dtype=np.float64), pair, sign, strength, alpha)
            correct += int(pred == row["answer_index"])
        acc = correct / len(dev_indices) if dev_indices else 0.0
        if acc > best_acc:
            best_alpha, best_acc = alpha, acc
    return best_alpha


def apply_lq1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = [i for i, row in enumerate(rows) if row["split"] == "dev"]
    gate = [i for i, row in enumerate(rows) if row["split"] == "gate"]
    margin_thresholds = choose_quantile_thresholds([float(rows[i]["target_margin"]) for i in dev])
    reply_thresholds = choose_quantile_thresholds(
        [
            abs(float(ladder.zscore(np.asarray(rows[i]["source_scores"], dtype=np.float64))[top2_indices(np.asarray(rows[i]["target_scores"], dtype=np.float64))[0]])
                - float(ladder.zscore(np.asarray(rows[i]["source_scores"], dtype=np.float64))[top2_indices(np.asarray(rows[i]["target_scores"], dtype=np.float64))[1]]))
            for i in dev
        ]
    )
    alpha = choose_lq1_alpha(rows, dev, reply_thresholds)
    rng = random.Random(SEED)
    shuffled_indices = list(range(len(rows)))
    rng.shuffle(shuffled_indices)
    vectors: dict[str, list[int]] = {key: [] for key in [
        "target_only",
        "source_index",
        "source_index_confidence",
        "equal_total_byte_score_sketch",
        "equal_total_byte_text_proxy",
        "query_only",
        "reply_only",
        "l_q1",
        "wrong_row_query",
        "wrong_row_reply",
        "derangement",
        "coordinate_shuffle",
        "full_source_oracle",
        "source_target_at_encoder_upper_bound",
    ]}
    output_rows: list[dict[str, Any]] = []
    switch_threshold = float(json.loads((BASE_DIR / "summary.json").read_text(encoding="utf-8"))["ladder"]["switch_threshold"])
    two_bit_alpha = float(json.loads((BASE_DIR / "summary.json").read_text(encoding="utf-8"))["ladder"]["budget_alphas"]["wz_2bit"])
    full_alpha = float(json.loads((BASE_DIR / "summary.json").read_text(encoding="utf-8"))["ladder"]["budget_alphas"]["full_source_score_fusion_oracle"])
    for index, row in enumerate(rows):
        if row["split"] not in {"dev", "gate"}:
            continue
        answer = int(row["answer_index"])
        source = np.asarray(row["source_scores"], dtype=np.float64)
        target = np.asarray(row["target_scores"], dtype=np.float64)
        pair = top2_indices(target)
        query_bucket = bucket(float(row["target_margin"]), margin_thresholds)
        sign, strength = pair_reply(source, pair, reply_thresholds)
        lq1_pred = decode_pair(target, pair, sign, strength, alpha)
        reply_only_pred = int(pair[0] if sign >= 0 else pair[1])
        source_conf = int(row["source_top1"] if (float(row["source_margin"]) - float(row["target_margin"])) >= switch_threshold else row["target_top1"])
        equal_score = ladder.top1(ladder.zscore(target) + two_bit_alpha * ladder.quantize(source, 2))
        full_source = ladder.top1(ladder.zscore(target) + full_alpha * ladder.zscore(source))
        upper = answer if answer in {int(row["target_top1"]), int(row["source_top1"]), full_source} else full_source

        wrong_query_row = rows[shuffled_indices[index]]
        wrong_pair = top2_indices(np.asarray(wrong_query_row["target_scores"], dtype=np.float64))
        wrong_query_sign, wrong_query_strength = pair_reply(source, wrong_pair, reply_thresholds)
        wrong_query_pred = decode_pair(target, wrong_pair, wrong_query_sign, wrong_query_strength, alpha)

        wrong_reply_source = np.asarray(rows[shuffled_indices[index]]["source_scores"], dtype=np.float64)
        wrong_reply_sign, wrong_reply_strength = pair_reply(wrong_reply_source, pair, reply_thresholds)
        wrong_reply_pred = decode_pair(target, pair, wrong_reply_sign, wrong_reply_strength, alpha)

        deranged_source = np.roll(source, 1)
        deranged_sign, deranged_strength = pair_reply(deranged_source, pair, reply_thresholds)
        deranged_pred = decode_pair(target, pair, deranged_sign, deranged_strength, alpha)

        perm = list(range(int(row["option_count"])))
        rng.shuffle(perm)
        shuffled_source = source[perm]
        coord_sign, coord_strength = pair_reply(shuffled_source, pair, reply_thresholds)
        coord_pred = decode_pair(target, pair, coord_sign, coord_strength, alpha)

        preds = {
            "target_only": int(row["target_top1"]),
            "source_index": int(row["source_top1"]),
            "source_index_confidence": source_conf,
            "equal_total_byte_score_sketch": equal_score,
            "equal_total_byte_text_proxy": int(row["source_top1"]),
            "query_only": int(row["target_top1"]),
            "reply_only": reply_only_pred,
            "l_q1": lq1_pred,
            "wrong_row_query": wrong_query_pred,
            "wrong_row_reply": wrong_reply_pred,
            "derangement": deranged_pred,
            "coordinate_shuffle": coord_pred,
            "full_source_oracle": full_source,
            "source_target_at_encoder_upper_bound": upper,
        }
        if row["split"] == "gate":
            for name, pred in preds.items():
                vectors[name].append(int(pred == answer))
        output_rows.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "answer_index": answer,
                "target_pair": list(pair),
                "query_uncertainty_bucket": query_bucket,
                "query_bytes": 2,
                "reply_bytes": 1,
                "total_bytes": 3,
                **{f"{name}_pred": pred for name, pred in preds.items()},
            }
        )
    gate_acc = {name: sum(values) / len(values) if values else 0.0 for name, values in vectors.items()}
    equal_baselines = [
        "target_only",
        "source_index",
        "source_index_confidence",
        "equal_total_byte_score_sketch",
        "equal_total_byte_text_proxy",
    ]
    best_equal = max(equal_baselines, key=lambda name: gate_acc[name])
    lq1_delta = ladder.paired_delta_ci(vectors["l_q1"], vectors[best_equal])
    controls = ["wrong_row_query", "wrong_row_reply", "derangement", "coordinate_shuffle"]
    controls_collapse = all(gate_acc["l_q1"] > gate_acc[name] for name in controls)
    ablations_explain = gate_acc["query_only"] >= gate_acc["l_q1"] or gate_acc["reply_only"] >= gate_acc["l_q1"]
    source_top = [int(row["source_top1"]) for row in rows if row["split"] == "gate"]
    lq1_pred = [row["l_q1_pred"] for row in output_rows if row["split"] == "gate"]
    reply_pred = [row["reply_only_pred"] for row in output_rows if row["split"] == "gate"]
    passed = bool(lq1_delta["delta"] > 0 and controls_collapse and not ablations_explain)
    write_jsonl(LQ1_ROWS, output_rows)
    return {
        "n_gate": len(gate),
        "n_dev": len(dev),
        "alpha": alpha,
        "query_bytes": 2,
        "reply_bytes": 1,
        "total_bytes": 3,
        "gate_accuracy": gate_acc,
        "best_equal_total_byte_baseline": best_equal,
        "delta_vs_best_equal_total_byte_baseline": lq1_delta,
        "controls_collapse": controls_collapse,
        "query_only_or_reply_only_explains": ablations_explain,
        "source_copy_leakage_mi_bits": {
            "l_q1_pred_vs_source_top1": ladder.empirical_mi(lq1_pred, source_top),
            "reply_only_pred_vs_source_top1": ladder.empirical_mi(reply_pred, source_top),
        },
        "status": "PROVISIONAL_PROMOTE_TO_GPU" if passed else "KILLED",
    }


def write_dashboards(confirm: dict[str, Any], lq1: dict[str, Any]) -> None:
    q_lines = [
        "# LatentWire Query Packet",
        "",
        f"- status: `{lq1['status']}`",
        "- protocol: receiver-query-conditioned two-way packet, not a one-way source-private packet.",
        f"- dev rows: `{lq1['n_dev']}`",
        f"- gate rows: `{lq1['n_gate']}`",
        f"- byte accounting: query `{lq1['query_bytes']}` + reply `{lq1['reply_bytes']}` = `{lq1['total_bytes']}` total bytes",
        f"- selected alpha: `{lq1['alpha']}`",
        f"- best equal-total-byte baseline: `{lq1['best_equal_total_byte_baseline']}`",
        f"- gate accuracy: `{lq1['gate_accuracy']}`",
        f"- delta vs best equal-total-byte baseline: `{lq1['delta_vs_best_equal_total_byte_baseline']}`",
        f"- controls collapse: `{lq1['controls_collapse']}`",
        f"- query-only/reply-only explain: `{lq1['query_only_or_reply_only_explains']}`",
        f"- source-copy leakage MI: `{lq1['source_copy_leakage_mi_bits']}`",
        "",
        "Gate rule: point delta must be positive versus the best equal-total-byte baseline, controls must collapse, and query-only/reply-only ablations must not explain the result.",
    ]
    (ROOT / "dashboard" / "latentwire_query_packet.md").write_text("\n".join(q_lines).rstrip() + "\n", encoding="utf-8")
    c_lines = [
        "# LatentWire One-Way Confirm Closeout",
        "",
        "- scope: held-out confirm closeout for the one-way negative only; no positive method selection used confirm.",
        f"- confirm rows: `{confirm['n']}`",
        f"- best non-oracle baseline: `{confirm['best_non_oracle_baseline']}`",
        f"- confirm accuracy: `{confirm['accuracy']}`",
        f"- deployable delta vs best: `{confirm['deployable_delta_vs_best']}`",
        f"- full source-only oracle delta vs best: `{confirm['full_source_delta_vs_best']}`",
        f"- source+target-at-encoder upper bound delta vs best: `{confirm['upper_bound_delta_vs_best']}`",
        "",
        "The source+target-at-encoder row is an upper-bound diagnostic only and remains non-deployable.",
    ]
    (ROOT / "dashboard" / "latentwire_one_way_confirm_closeout.md").write_text("\n".join(c_lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run confirm closeout and L_Q1 query-packet screen")
    parser.add_argument("--max-scan-rows", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--source-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LQ1_DIR.mkdir(parents=True, exist_ok=True)
    CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    dev_gate_rows = read_jsonl(DEV_GATE_ROWS)
    confirm_rows = score_confirm_rows(args)
    confirm_summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": "BOUNDED_NEGATIVE",
        "confirm_policy": "negative_closeout_only_after_dev_gate_selection",
        **apply_one_way_confirm(confirm_rows),
    }
    lq1_summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scoop_verdict": "adjacent_not_exact_scoop_high_risk",
        **apply_lq1(dev_gate_rows),
    }
    write_json(CONFIRM_SUMMARY, confirm_summary)
    write_json(LQ1_SUMMARY, lq1_summary)
    write_dashboards(confirm_summary, lq1_summary)
    print(
        "lq1 screen complete:",
        f"confirm={confirm_summary['classification']}",
        f"lq1={lq1_summary['status']}",
        f"gate_n={lq1_summary['n_gate']}",
        f"delta={lq1_summary['delta_vs_best_equal_total_byte_baseline']['delta']:.6f}",
    )


if __name__ == "__main__":
    main()
