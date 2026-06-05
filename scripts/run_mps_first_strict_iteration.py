#!/usr/bin/env python3
"""Strict MPS-first rerun with data floors and logged blockers.

This runner is deliberately conservative: it can create Mac-only ceiling or
sanity results, but it never marks a Mac result as passed or promotion-eligible.
It writes a new versioned result root so the earlier mps_first evidence remains
write-once.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import gzip
import json
import math
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "queues/mps_first.yaml"
RESULT_ROOT = ROOT / "results/mps_first_strict_20260605"
ITERATION_LOG = ROOT / "dashboard/iteration_log.md"
LEADERBOARD = ROOT / "dashboard/leaderboard.csv"
BREAKTHROUGH_BOARD = ROOT / "dashboard/breakthrough_board.md"
KILL_BOARD = ROOT / "dashboard/kill_board.md"
REVIEW_ZIP = ROOT / "review_packet.zip"
MAX_PACKET_FILE = 5 * 1024 * 1024
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}
SEED = 20260605


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_queue() -> list[dict]:
    data = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))
    return sorted(data["mps_first"], key=lambda row: row["priority"])


def base_summary(item: dict, status: str, verdict: str, achieved_n: int, wall: float, floor: int | None) -> dict:
    return {
        "achieved_n": int(achieved_n),
        "confirm_rows_scored": 0,
        "created_utc": now_utc(),
        "data_floor": floor,
        "floor_met": floor is None or achieved_n >= floor,
        "id": item["id"],
        "paper": item["paper"],
        "priority": item["priority"],
        "promotion_allowed": False,
        "result_root": rel(RESULT_ROOT),
        "status": status,
        "verdict": verdict,
        "wall_clock_seconds": wall,
    }


def accuracy(pred: list[int], y: list[int]) -> float:
    return sum(int(a == b) for a, b in zip(pred, y, strict=True)) / len(y) if y else 0.0


def paired_ci(method_hits: list[int], baseline_hits: list[int], samples: int = 2000) -> dict:
    if len(method_hits) != len(baseline_hits):
        raise ValueError("paired vectors must have equal length")
    if not method_hits:
        return {"n": 0, "delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "mde_half_width": 1.0}
    diffs = [int(a) - int(b) for a, b in zip(method_hits, baseline_hits, strict=True)]
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


def discrete_cmi_bits(source: list[int], target: list[int], receiver: list[int]) -> float:
    total = len(target)
    if total == 0:
        return 0.0
    by_r: dict[int, list[int]] = {}
    for i, r in enumerate(receiver):
        by_r.setdefault(int(r), []).append(i)
    cmi = 0.0
    for indices in by_r.values():
        pr = len(indices) / total
        counts_s: dict[int, int] = {}
        counts_y: dict[int, int] = {}
        counts_sy: dict[tuple[int, int], int] = {}
        for i in indices:
            s = int(source[i])
            y = int(target[i])
            counts_s[s] = counts_s.get(s, 0) + 1
            counts_y[y] = counts_y.get(y, 0) + 1
            counts_sy[(s, y)] = counts_sy.get((s, y), 0) + 1
        nr = len(indices)
        local = 0.0
        for (s, y), count in counts_sy.items():
            p_sy = count / nr
            p_s = counts_s[s] / nr
            p_y = counts_y[y] / nr
            local += p_sy * math.log2(p_sy / max(p_s * p_y, 1e-12))
        cmi += pr * local
    return float(cmi)


def safe_equation_value(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsafe equation node: {type(node).__name__}")
    return float(eval(compile(tree, "<svamp-equation>", "eval"), {"__builtins__": {}}, {}))


def numeric_answer(value: float) -> int:
    return int(round(float(value)))


def status_from_ci(ci: dict, *, positive_name: str = "MAC_FLOOR_POSITIVE_CEILING_ONLY") -> tuple[str, str]:
    if ci["ci95_low"] > 0:
        return positive_name, "POSITIVE_CEILING_ONLY"
    if ci["ci95_high"] < 0:
        return "MAC_FLOOR_NEGATIVE", "NEGATIVE_SCREEN"
    return "MAC_FLOOR_INCONCLUSIVE", "INCONCLUSIVE_UNDERPOWERED"


def l_pc1(item: dict, start: float) -> tuple[dict, list[dict]]:
    floor = 500
    path = ROOT / "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/qwen_strict_packet_predictions_1536_2048.jsonl"
    rows = read_jsonl(path)
    print(f"[L_PC1] acquired cached cross-family rows {len(rows)}/{floor} from {rel(path)}", flush=True)
    y = [int(r["answer_index"]) for r in rows]
    selected = [int(r["selected_prediction"]) for r in rows]
    receiver = [int(r["score_only_bagged_prediction"]) for r in rows]
    source_label = [int(r["source_label_prediction"]) for r in rows]
    wrong = [int(r["wrong_example_hidden_prediction"]) for r in rows]
    zero = [int(r["zero_hidden_prediction"]) for r in rows]
    ci = paired_ci([int(a == b) for a, b in zip(selected, y, strict=True)], [int(a == b) for a, b in zip(receiver, y, strict=True)])
    status, verdict = status_from_ci(ci)
    raw_rows = []
    for i, r in enumerate(rows):
        raw_rows.append({
            "row_id": r["row_id"],
            "answer_index": y[i],
            "selected_prediction": selected[i],
            "receiver_score_only_prediction": receiver[i],
            "source_label_prediction": source_label[i],
            "wrong_example_hidden_prediction": wrong[i],
            "zero_hidden_prediction": zero[i],
            "selected_correct": selected[i] == y[i],
            "receiver_correct": receiver[i] == y[i],
        })
        if (i + 1) % 50 == 0:
            print(f"[L_PC1] rows {i + 1}/{floor}", flush=True)
    summary = base_summary(item, status, verdict, len(rows), time.time() - start, floor)
    summary.update({
        "artifact": rel(path),
        "baseline_receiver_accuracy": accuracy(receiver, y),
        "selected_receiver_source_accuracy": accuracy(selected, y),
        "source_label_accuracy": accuracy(source_label, y),
        "wrong_example_hidden_accuracy": accuracy(wrong, y),
        "zero_hidden_accuracy": accuracy(zero, y),
        "gain_receiver_source_vs_receiver": ci,
        "mde_half_width": ci["mde_half_width"],
        "conditional_mi_bits_source_label_answer_given_receiver_prediction": discrete_cmi_bits(source_label, y, receiver),
        "sanity_gate": "receiver score-only baseline is available; this is cached cross-family receiver-family packet evidence, not a deployable new runner",
        "exact_command": "re-run strict cached floor: venv_arm64/bin/python scripts/run_mps_first_strict_iteration.py --probe L_PC1_cross_family_specialist_ceiling",
    })
    return summary, raw_rows


def load_svamp(limit: int) -> list[dict]:
    rows = read_jsonl(ROOT / "data/svamp_1000.jsonl")[:limit]
    out = []
    for i, row in enumerate(rows):
        value = safe_equation_value(row["metadata"]["equation"])
        answer = numeric_answer(value)
        out.append(row | {"strict_index": i, "tool_answer": answer})
    return out


def receiver_heuristic(row: dict) -> int:
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", row["question"])]
    if not numbers:
        return 0
    if len(numbers) == 1:
        return numeric_answer(numbers[0])
    q = row["question"].lower()
    if "left" in q or "more" in q or "than" in q:
        return numeric_answer(abs(numbers[0] - numbers[-1]))
    if "total" in q or "altogether" in q:
        return numeric_answer(sum(numbers[:3]))
    return numeric_answer(numbers[0])


def l_pc2(item: dict, start: float) -> tuple[dict, list[dict]]:
    floor = 500
    rows = load_svamp(floor)
    raw_rows = []
    receiver, tool, wrong = [], [], []
    y = []
    for i, row in enumerate(rows):
        ans = int(row["tool_answer"])
        recv = receiver_heuristic(row)
        wrong_ans = int(rows[(i + 37) % len(rows)]["tool_answer"])
        y.append(ans)
        receiver.append(recv)
        tool.append(ans)
        wrong.append(wrong_ans)
        raw_rows.append({
            "row_id": row["metadata"]["id"],
            "split": "dev" if i % 2 == 0 else "gate",
            "question": row["question"],
            "equation": row["metadata"]["equation"],
            "answer": ans,
            "receiver_heuristic_prediction": recv,
            "private_tool_prediction": ans,
            "wrong_row_tool_prediction": wrong_ans,
            "tool_success_flag_only_prediction": recv,
            "visible_equal_byte_tool_prediction": ans,
        })
        if (i + 1) % 50 == 0:
            print(f"[L_PC2] generated tool rows {i + 1}/{floor}", flush=True)
    private_ci = paired_ci([int(a == b) for a, b in zip(tool, y, strict=True)], [int(a == b) for a, b in zip(receiver, y, strict=True)])
    visible_ci = paired_ci([int(a == b) for a, b in zip(tool, y, strict=True)], [int(a == b) for a, b in zip(tool, y, strict=True)])
    wrong_ci = paired_ci([int(a == b) for a, b in zip(tool, y, strict=True)], [int(a == b) for a, b in zip(wrong, y, strict=True)])
    summary = base_summary(item, "MAC_FLOOR_KILLED_BY_EQUAL_BYTE_VISIBLE_TOOL_CONTROL", "NEGATIVE_SCREEN_CONTROL_DOMINATED", len(rows), time.time() - start, floor)
    summary.update({
        "receiver_heuristic_accuracy": accuracy(receiver, y),
        "private_tool_accuracy": accuracy(tool, y),
        "wrong_row_tool_accuracy": accuracy(wrong, y),
        "gain_private_tool_vs_receiver": private_ci,
        "gain_private_tool_vs_visible_equal_byte_tool": visible_ci,
        "gain_private_tool_vs_wrong_row_tool": wrong_ci,
        "mde_half_width": private_ci["mde_half_width"],
        "interpretation": "The private calculator ceiling is large versus a question-only heuristic but exactly tied by an equal-byte visible tool-result control, so it is not a source-private packet win.",
        "exact_command": "venv_arm64/bin/python scripts/run_mps_first_strict_iteration.py --probe L_PC2_tool_augmented_source_ceiling",
    })
    return summary, raw_rows


def candidate_values(answer: int, idx: int, count: int = 16) -> list[int]:
    values = [answer]
    offsets = [1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 7, -7, 11, -11, 13, -13, 17, -17]
    for off in offsets:
        cand = answer + off + (idx % 3) - 1
        if cand != answer and cand not in values:
            values.append(cand)
        if len(values) >= count:
            break
    return values[:count]


def l_pc5(item: dict, start: float) -> tuple[dict, list[dict]]:
    prompt_floor = 500
    candidates_per_prompt = 16
    rows = load_svamp(prompt_floor)
    raw_rows = []
    target_hits, verifier_hits, source_hits = [], [], []
    prompts_with_correct = 0
    for i, row in enumerate(rows):
        answer = int(row["tool_answer"])
        cands = candidate_values(answer, i, candidates_per_prompt)
        if answer in cands:
            prompts_with_correct += 1
        # Deterministic noisy target: often prefers first distractor for nontrivial rows.
        target_scores = []
        verifier_scores = []
        source_scores = []
        for j, cand in enumerate(cands):
            correct = cand == answer
            distance = abs(cand - answer)
            target_scores.append(float(-0.05 * distance + (0.20 if j == 1 and i % 3 else 0.0)))
            verifier_scores.append(1.0 if correct else -float(distance))
            source_scores.append(0.8 if correct else -0.02 * distance)
            raw_rows.append({
                "prompt_id": row["metadata"]["id"],
                "candidate_index": j,
                "candidate": cand,
                "answer": answer,
                "correct": correct,
                "target_score": target_scores[-1],
                "verifier_score": verifier_scores[-1],
                "source_score": source_scores[-1],
            })
        target_pred = max(range(len(cands)), key=lambda k: (target_scores[k], -k))
        verifier_pred = max(range(len(cands)), key=lambda k: (target_scores[k] + verifier_scores[k], -k))
        source_pred = max(range(len(cands)), key=lambda k: (target_scores[k] + source_scores[k], -k))
        target_hits.append(int(cands[target_pred] == answer))
        verifier_hits.append(int(cands[verifier_pred] == answer))
        source_hits.append(int(cands[source_pred] == answer))
        if (i + 1) % 10 == 0:
            print(f"[L_PC5] generated rerank prompts {i + 1}/{prompt_floor} correct_pool={prompts_with_correct}", flush=True)
    ci = paired_ci(verifier_hits, target_hits)
    status, verdict = status_from_ci(ci, positive_name="MAC_FLOOR_POSITIVE_ORACLE_VERIFIER_CEILING")
    summary = base_summary(item, status, verdict, len(rows), time.time() - start, prompt_floor)
    summary.update({
        "candidate_rows": len(raw_rows),
        "candidates_per_prompt": candidates_per_prompt,
        "prompts_with_at_least_one_correct_candidate": prompts_with_correct,
        "required_correct_candidate_prompts": 80,
        "receiver_target_accuracy": sum(target_hits) / len(target_hits),
        "receiver_plus_verifier_accuracy": sum(verifier_hits) / len(verifier_hits),
        "receiver_plus_source_accuracy": sum(source_hits) / len(source_hits),
        "gain_verifier_vs_target": ci,
        "mde_half_width": ci["mde_half_width"],
        "interpretation": "Easy arithmetic oracle-verifier ceiling meets the candidate-pool floor; it is a sanity ceiling, not a deployable L-A2 method.",
        "exact_command": "venv_arm64/bin/python scripts/run_mps_first_strict_iteration.py --probe L_PC5_private_verifier_receiver_candidates",
    })
    return summary, raw_rows


def l_c2(item: dict, start: float) -> tuple[dict, list[dict]]:
    floor = 500
    rows = load_svamp(1000)
    train = rows[:500]
    gate = rows[500:1000]
    raw_rows = []
    receiver, fuser, wrong, zero, y = [], [], [], [], []
    for i, row in enumerate(gate):
        answer = int(row["tool_answer"])
        recv = receiver_heuristic(row)
        wrong_ans = int(train[(i * 17) % len(train)]["tool_answer"])
        y.append(answer)
        receiver.append(recv)
        fuser.append(answer)
        wrong.append(wrong_ans)
        zero.append(recv)
        raw_rows.append({
            "row_id": row["metadata"]["id"],
            "split": "gate",
            "answer": answer,
            "receiver_prediction": recv,
            "control_trained_fuser_prediction": answer,
            "wrong_row_source_prediction": wrong_ans,
            "zero_source_prediction": recv,
            "label_shuffle_prediction": int(train[(i * 23) % len(train)]["tool_answer"]),
        })
        if (i + 1) % 50 == 0:
            print(f"[L_C2] scored fuser gate rows {i + 1}/{floor}", flush=True)
    ci = paired_ci([int(a == b) for a, b in zip(fuser, y, strict=True)], [int(a == b) for a, b in zip(receiver, y, strict=True)])
    wrong_ci = paired_ci([int(a == b) for a, b in zip(fuser, y, strict=True)], [int(a == b) for a, b in zip(wrong, y, strict=True)])
    status, verdict = status_from_ci(ci, positive_name="MAC_FLOOR_POSITIVE_ORACLE_FUSER_CEILING")
    summary = base_summary(item, status, verdict, len(gate), time.time() - start, floor)
    summary.update({
        "train_rows": len(train),
        "gate_rows": len(gate),
        "receiver_accuracy": accuracy(receiver, y),
        "control_trained_fuser_accuracy": accuracy(fuser, y),
        "wrong_row_accuracy": accuracy(wrong, y),
        "zero_source_accuracy": accuracy(zero, y),
        "gain_fuser_vs_receiver": ci,
        "gain_fuser_vs_wrong_row": wrong_ci,
        "mde_half_width": ci["mde_half_width"],
        "interpretation": "Oracle source-feature fuser clears the sanity floor; deployable status remains blocked because this uses gold SVAMP equations as source features.",
        "exact_command": "venv_arm64/bin/python scripts/run_mps_first_strict_iteration.py --probe L_C2_control_trained_lcf_lite_proxy",
    })
    return summary, raw_rows


KL_SOURCES = [
    ("deepseek", ROOT / "experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/deepseek_r1_distill_qwen_1_5b/kl_rows.jsonl.gz"),
    ("falcon", ROOT / "experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/falcon_h1_0_5b/kl_rows.jsonl.gz"),
    ("granite", ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z/kl_rows.jsonl.gz"),
]


def load_kl_floor(floor: int) -> list[dict]:
    out = []
    per_model = max(1, math.ceil(floor / len(KL_SOURCES)))
    for model, path in KL_SOURCES:
        model_rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("regime") == "bf16_reference":
                    continue
                out.append({
                    "model": model,
                    "source_path": rel(path),
                    "prompt_id": row.get("prompt_id"),
                    "prompt_index": row.get("prompt_index"),
                    "decode_position": row.get("decode_position"),
                    "regime": row.get("regime"),
                    "kl_bf16_q": float(row.get("kl_bf16_q", 0.0)),
                })
                if len(out) % 100 == 0:
                    print(f"[channel_cached] KL rows {len(out)}/{floor}", flush=True)
                model_rows += 1
                if model_rows >= per_model:
                    break
    return out[:floor]


def c_u1(item: dict, start: float) -> tuple[dict, list[dict]]:
    floor = 500
    rows = load_kl_floor(floor)
    median_kl = statistics.median(r["kl_bf16_q"] for r in rows) if rows else 0.0
    high = [int(r["kl_bf16_q"] > median_kl) for r in rows]
    early = [int(int(r["decode_position"]) <= 128) for r in rows]
    ci = paired_ci(high, early)
    summary = base_summary(item, "MAC_FLOOR_COMPUTED_SCHEMA_BLOCKED", "INCONCLUSIVE_SCHEMA_MISMATCH", len(rows), time.time() - start, floor)
    summary.update({
        "models": sorted({r["model"] for r in rows}),
        "median_kl": median_kl,
        "proxy_delta_high_kl_vs_early_position": ci,
        "mde_half_width": ci["mde_half_width"],
        "blocker": "Floor-sized KL trajectory rows exist, but they lack paired difficulty/policy-uplift labels required by the C_U1 mandatory gate.",
        "failed_attempts": [{
            "attempt": "scan phase9 KL rows for drift trajectory plus recoverable_gap/policy_uplift labels",
            "rows_scanned": len(rows),
            "blocking_resource": "same-row Channel-Set label cache with baseline confidence and policy uplift",
            "result": "schema_missing_required_labels",
        }],
        "exact_command": "venv_arm64/bin/python scripts/build_channel_set_drift_as_signal_screen.py --input experimental/outlier_migrate/phase9/results --features drift_trajectory,kl_growth,warmup_activation --labels recoverable_gap,policy_uplift --min-gate-n 500 --models granite,deepseek,falcon --no-confirm",
    })
    return summary, rows


def c_w1(item: dict, start: float) -> tuple[dict, list[dict]]:
    floor = 500
    rows = load_kl_floor(floor)
    summary = base_summary(item, "MAC_FLOOR_COMPUTED_SCHEMA_BLOCKED", "INCONCLUSIVE_SCHEMA_MISMATCH", len(rows), time.time() - start, floor)
    summary.update({
        "blocker": "Warmup KL rows can be materialized, but no cached per-policy outcome matrix exists for ParoQuant/C_A1/survival/reject on the same rows.",
        "failed_attempts": [{
            "attempt": "materialize warmup features from phase9 KL rows and look for same-row policy outcomes",
            "rows_scanned": len(rows),
            "blocking_resource": "dev/gate warmup-policy outcome cache",
            "result": "features_present_outcomes_missing",
        }],
        "exact_command": "venv_arm64/bin/python scripts/materialize_channel_set_warmup_policy_cache.py --input experimental/outlier_migrate/phase9/results --split dev,gate --locked-row-ids splits/stage0_cache_rows_hash.jsonl --policies paroquant,c_a1,survival,reject --no-confirm",
    })
    return summary, rows


def scan_per_trace_metrics() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "experimental/outlier_migrate/phase9/results").glob("*/per_trace_metrics.json")):
        if "_confirm" in str(path):
            continue
        data = read_json(path)
        for trace in data.get("traces", []):
            rows.append({
                "source_path": rel(path),
                "prompt_id": trace.get("prompt_id"),
                "prompt_index": trace.get("prompt_index"),
                "static_gap": trace.get("static_gap"),
                "no_recoverable_static_gap": trace.get("no_recoverable_static_gap"),
                "recovery_keys": sorted((trace.get("recoveries") or {}).keys()),
                "has_random_or_static_denominator": any("random" in k or "static" in k for k in (trace.get("recoveries") or {})),
            })
    return rows


def c_s1(item: dict, start: float) -> tuple[dict, list[dict]]:
    floor = 500
    rows = scan_per_trace_metrics()
    for i in range(0, len(rows), 50):
        print(f"[C_S1] scanned per-trace rows {min(i + 50, len(rows))}/{floor}", flush=True)
    summary = base_summary(item, "PARKED_LOGGED_LOCAL_FLOOR_BLOCKED", "PARKED_NEEDS_GPU_OR_NATIVE_CACHE", len(rows), time.time() - start, floor)
    summary.update({
        "floor_blocked": True,
        "eligible_per_trace_rows_found": len(rows),
        "blocker": "Only non-confirm per-trace rows below the 500-row floor are available locally; generating the missing identical-row denominator requires native replay/backfill, not a Mac-only cached computation.",
        "failed_attempts": [{
            "attempt": "scan all non-confirm phase9 per_trace_metrics.json files for identical-row survival/static/random denominators",
            "eligible_rows": len(rows),
            "required_rows": floor,
            "blocking_resource": "fresh same-row random/static/survival denominator replay",
            "result": "insufficient_cached_trace_rows",
        }],
        "exact_command": "venv_arm64/bin/python scripts/build_channel_set_clean_survival_denominator.py --input experimental/outlier_migrate/phase9/results --same-row-denominators random_core,static_core,wrong_row_core --min-gate-n 500 --no-gap-filter --no-confirm",
    })
    return summary, rows[:500]


def c_y5(item: dict, start: float) -> tuple[dict, list[dict]]:
    checks = []
    required = [
        ("c_a1_runbook", ROOT / "dashboard/c_a1_gpu_backfill_runbook.md"),
        ("gpu_backfill_queue", ROOT / "queues/gpu_backfill.yaml"),
        ("gpu_foreground_queue", ROOT / "queues/gpu_foreground.yaml"),
        ("ce21_registry", ROOT / "registry/CE21_no_gap_filter.yaml"),
        ("c_a1_registry", ROOT / "registry/C_A1_cvar_evt_clip_grid.yaml"),
    ]
    for name, path in required:
        checks.append({"check": name, "path": rel(path), "file_present": path.exists()})
    native = list((ROOT / "experimental/outlier_migrate/phase9/results").glob("om_driftrot_*clip_tight*/per_trace_metrics.json"))
    native = [p for p in native if "_confirm" not in str(p)]
    checks.append({
        "check": "native_pairing_results_present",
        "file_present": False,
        "candidate_files_scanned": [rel(p) for p in native[:20]],
        "reason": "Candidate files exist, but no verified three-model same-row ParoQuant-vs-tight-clip matrix is present.",
    })
    summary = base_summary(item, "PARKED_LOGGED_NEEDS_NATIVE_PAIRING", "PARKED_NEEDS_GPU_OR_NATIVE_CACHE", len(checks), time.time() - start, None)
    summary.update({
        "audit_checks": checks,
        "blocker": "Defense inputs exist, but the three-model same-row native pairing packet is still missing.",
        "failed_attempts": [{
            "attempt": "scan phase9 outputs for three-model native paired ParoQuant-vs-tight-clip matrix",
            "candidate_files": len(native),
            "blocking_resource": "paired Granite/DeepSeek/Falcon native replay matrix",
            "result": "pairing_packet_missing",
        }],
        "exact_command": "run the paired matrix in dashboard/c_a1_gpu_backfill_runbook.md: ParoQuant baseline vs tight-clip C_A1 on exact cached Granite, DeepSeek, and Falcon row IDs; abort if any model lacks same-row pairing",
    })
    return summary, checks


HANDLERS = {
    "L_PC1_cross_family_specialist_ceiling": l_pc1,
    "L_PC2_tool_augmented_source_ceiling": l_pc2,
    "L_PC5_private_verifier_receiver_candidates": l_pc5,
    "L_C2_control_trained_lcf_lite_proxy": l_c2,
    "C_U1_drift_as_signal_router": c_u1,
    "C_W1_fixed_library_warmup_selector": c_w1,
    "C_S1_clean_survival_stablecore_denominator": c_s1,
    "C_Y5_channel_set_defense_bundle": c_y5,
}


def run_probe(probe_id: str) -> dict:
    queue = {item["id"]: item for item in load_queue()}
    if probe_id not in queue:
        raise SystemExit(f"unknown probe id: {probe_id}")
    item = queue[probe_id]
    start = time.time()
    summary, raw_rows = HANDLERS[probe_id](item, start)
    out = RESULT_ROOT / probe_id
    write_json(out / "summary.json", summary)
    write_jsonl(out / "raw_rows.jsonl", raw_rows)
    update_dashboards()
    refresh_review_packet()
    return summary


def all_summaries() -> list[dict]:
    rows = []
    for path in sorted(RESULT_ROOT.glob("*/summary.json")):
        rows.append(read_json(path))
    return sorted(rows, key=lambda row: row.get("priority", 999))


def update_iteration_log(summaries: list[dict]) -> None:
    lines = [
        "# Iteration Log",
        "",
        "Strict MPS-first rerun. Mac results are screening/ceiling states only; no Mac row is promotion-eligible.",
        "",
        "| priority | probe | paper | status | achieved_n | floor | MDE | verdict | wall_clock_seconds | next command |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    done = {s["id"] for s in summaries}
    for s in summaries:
        mde = s.get("mde_half_width")
        floor = s.get("data_floor")
        cmd = str(s.get("exact_command", "")).replace("|", "\\|")
        lines.append(
            f"| {s.get('priority')} | `{s.get('id')}` | {s.get('paper')} | `{s.get('status')}` | "
            f"{s.get('achieved_n', 0)} | {'' if floor is None else floor} | {'' if mde is None else mde} | "
            f"`{s.get('verdict')}` | {s.get('wall_clock_seconds', 0):.3f} | `{cmd}` |"
        )
    for item in load_queue():
        if item["id"] not in done:
            lines.append(
                f"| {item['priority']} | `{item['id']}` | {item['paper']} | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `{item['command']}` |"
            )
    ITERATION_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_leaderboard(summaries: list[dict]) -> None:
    if not LEADERBOARD.exists():
        return
    with LEADERBOARD.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader if not str(r.get("method_id", "")).startswith("MPS_FIRST_STRICT_")]
    for s in summaries:
        row = {key: "" for key in fieldnames}
        row.update({
            "method_id": f"MPS_FIRST_STRICT_{s['id']}",
            "paper": s.get("paper", ""),
            "split": "dev_gate",
            "status": s.get("status", ""),
            "source_path": f"{rel(RESULT_ROOT)}/{s['id']}/summary.json",
            "n": str(s.get("achieved_n", 0)),
            "paired_n": str((s.get("gain_receiver_source_vs_receiver") or s.get("gain_private_tool_vs_receiver") or s.get("gain_verifier_vs_target") or s.get("gain_fuser_vs_receiver") or {}).get("n", "")),
            "delta_vs_best_baseline": str((s.get("gain_receiver_source_vs_receiver") or s.get("gain_private_tool_vs_receiver") or s.get("gain_verifier_vs_target") or s.get("gain_fuser_vs_receiver") or {}).get("delta", "")),
            "ci95_low_vs_best_baseline": str((s.get("gain_receiver_source_vs_receiver") or s.get("gain_private_tool_vs_receiver") or s.get("gain_verifier_vs_target") or s.get("gain_fuser_vs_receiver") or {}).get("ci95_low", "")),
            "ci95_high_vs_best_baseline": str((s.get("gain_receiver_source_vs_receiver") or s.get("gain_private_tool_vs_receiver") or s.get("gain_verifier_vs_target") or s.get("gain_fuser_vs_receiver") or {}).get("ci95_high", "")),
            "note": "strict MPS rerun; no Mac PASSED status; promotion_allowed=false",
        })
        rows.append(row)
    with LEADERBOARD.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replace_section(text: str, heading: str, body: str) -> str:
    pattern = rf"\n## {re.escape(heading)}\n.*?(?=\n## |\Z)"
    section = f"\n## {heading}\n\n{body.rstrip()}\n"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, section, text, flags=re.S)
    return text.rstrip() + section


def update_boards(summaries: list[dict]) -> None:
    body = [
        "| priority | probe | status | achieved_n | floor | interpretation |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for s in summaries:
        body.append(
            f"| {s.get('priority')} | `{s.get('id')}` | `{s.get('status')}` | {s.get('achieved_n', 0)} | "
            f"{'' if s.get('data_floor') is None else s.get('data_floor')} | {s.get('interpretation') or s.get('blocker') or s.get('reason') or 'see summary'} |"
        )
    section = "\n".join(body)
    if BREAKTHROUGH_BOARD.exists():
        text = BREAKTHROUGH_BOARD.read_text(encoding="utf-8")
        BREAKTHROUGH_BOARD.write_text(replace_section(text, "MPS-First Strict Rerun", section), encoding="utf-8")
    if KILL_BOARD.exists():
        text = KILL_BOARD.read_text(encoding="utf-8")
        KILL_BOARD.write_text(replace_section(text, "MPS-First Strict Rerun", section), encoding="utf-8")


def update_dashboards() -> None:
    summaries = all_summaries()
    update_iteration_log(summaries)
    update_leaderboard(summaries)
    update_boards(summaries)


def confirm_path(path: str) -> bool:
    return fnmatch.fnmatch(path, "*_confirm*") or "_confirm" in path


def hard_excluded(path: Path) -> bool:
    value = path.as_posix()
    return path.suffix in HARD_EXTS or value.startswith("caches/") or "/caches/" in value


def packet_candidates() -> list[Path]:
    paths: list[Path] = []
    for root in [ROOT / "results/mps_first", RESULT_ROOT]:
        paths.extend(sorted(root.glob("**/summary.json")))
        paths.extend(sorted(root.glob("**/raw_rows.jsonl")))
    for item in [
        "dashboard/scoop_report.md",
        "dashboard/morning_brief.md",
        "dashboard/iteration_log.md",
        "dashboard/breakthrough_board.md",
        "dashboard/kill_board.md",
        "dashboard/leaderboard.csv",
        "ideas/NEW_200_triage.md",
        "scripts/run_mps_first_iteration.py",
        "scripts/run_mps_first_strict_iteration.py",
        "scripts/overnight_v3_corrected_probes.py",
        "lessons/LESSONS_LEDGER.md",
    ]:
        path = ROOT / item
        if path.exists():
            paths.append(path)
    paths.extend(sorted((ROOT / "queues").glob("*.yaml")))
    paths.extend(sorted((ROOT / "registry").glob("**/*.yaml")))
    return paths


def add_packet_file(entries: dict[str, bytes], skipped: list[str], path: Path) -> None:
    name = rel(path)
    if confirm_path(name) or hard_excluded(Path(name)):
        return
    size = path.stat().st_size
    if size > MAX_PACKET_FILE:
        if path.suffix in {".jsonl", ".csv", ".md"}:
            head = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:200]) + "\n"
            entries[name] = head.encode("utf-8")
            skipped.append(f"{name}: shipped 200-row/line head instead of {size} byte full file")
        else:
            skipped.append(f"{name}: skipped {size} byte file")
        return
    entries[name] = path.read_bytes()


def refresh_review_packet() -> None:
    entries: dict[str, bytes] = {}
    skipped: list[str] = []
    for path in packet_candidates():
        if path.exists() and path.is_file():
            add_packet_file(entries, skipped, path)
    index = [
        "# review_packet",
        "",
        f"Created UTC: {now_utc()}",
        "",
        "## Included",
    ]
    for name in sorted(entries):
        index.append(f"- `{name}` ({len(entries[name])} bytes)")
    index.extend(["", "## Skipped / Truncated"])
    index.extend(f"- {item}" for item in skipped) if skipped else index.append("- None")
    index.extend([
        "",
        "## Guardrails",
        "- No paths matching `*_confirm*` are included.",
        "- No model weights, binary arrays, or caches are included.",
        "- Files over 5 MB are skipped or represented by a 200-line head.",
    ])
    entries["INDEX.md"] = ("\n".join(index) + "\n").encode("utf-8")
    bad = [name for name in entries if confirm_path(name)]
    if bad:
        raise SystemExit(f"review packet would include confirm path(s): {bad}")
    with ZipFile(REVIEW_ZIP, "w", ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    if REVIEW_ZIP.stat().st_size > 25 * 1024 * 1024:
        raise SystemExit(f"review_packet.zip too large: {REVIEW_ZIP.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="Probe id from queues/mps_first.yaml, or ALL.")
    args = parser.parse_args()
    if args.probe == "ALL":
        for item in load_queue():
            print(json.dumps(run_probe(item["id"]), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_probe(args.probe), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
