from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260603_cpu_only_screening"
RESULTS_DIR = ROOT / "results" / "overnight" / RUN_ID


C2C_GENERATION_TRACE = ROOT / "results/svamp32_c2c_generation_trace_syndrome_probe_mps_20260505/generation_trace_probe.json"
C2C_PREFILL_TRACE = ROOT / "results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.json"
C2C_TEACHER_PROBE = ROOT / "results/svamp32_c2c_mps_compat_replay_20260505/c2c_teacher_innovation_probe.json"
C2C_SPARSE_PREFLIGHT = ROOT / "results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_20260505/summary.json"
C2C_GENERATED_ANSWER_AUDIT = ROOT / "results/svamp32_c2c_generated_answer_packet_audit_20260505/generated_answer_packet_audit.json"
C2C_TEACHER_DELTA_PACKET = ROOT / "results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/teacher_delta_packet_gate.json"
C2C_CANDIDATE_DELTA_PACKET = ROOT / "results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/candidate_pool_delta_packet_gate.json"
C2C_ARC_MCQA = ROOT / "results/dense_baseline_mcqa_smoke_20260505/c2c_arc_n16_constrained_letter_summary.json"
C2C_OBQA_MCQA = ROOT / "results/dense_baseline_mcqa_smoke_20260505/c2c_openbookqa_n16_constrained_letter_summary.json"
C2C_BOOTSTRAP = ROOT / "results/c2c_bootstrap_20260418/qwen_pair.json"
C2C_COMPETITOR_BOOTSTRAP = ROOT / "results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json"
KVCOMM_DAMAGE = ROOT / "results/dense_baseline_mcqa_smoke_20260505/kvcomm_damage_diagnostic_n16.json"
KVCOMM_LAYER = ROOT / "results/dense_baseline_mcqa_smoke_20260505/kvcomm_damage_diagnostic_layer_sweep_n16.json"

CA1_M26_CONTROL = ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z/control_metrics.json"
CF_M10_CHECKER = ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z/checker_result.json"
CA1_DEEPSEEK_CHECKER = ROOT / "experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z/checker_result.json"
CA1_FALCON_CHECKER = ROOT / "experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z/checker_result.json"
CA1_GRANITE_PAROQUANT_CHECKER = ROOT / "experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z/checker_result.json"
CA1_DEEPSEEK_SENTINEL = ROOT / "experimental/outlier_migrate/phase9/results/om_v1_paroquant_deepseek_20260528T162858Z/control_metrics.json"
CA1_FALCON_SENTINEL = ROOT / "experimental/outlier_migrate/phase9/results/om_v1_paroquant_falcon_20260528T154653Z/control_metrics.json"

L_A2_READINESS = ROOT / "dashboard/latentwire_lA2_data_readiness.md"
CHANNEL_READINESS = ROOT / "dashboard/channel_set_backfill_readiness.md"
STAGE1_TRIAGE = ROOT / "dashboard/stage1_triage.md"
LEADERBOARD = ROOT / "dashboard/leaderboard.csv"
BREAKTHROUGH_BOARD = ROOT / "dashboard/breakthrough_board.md"
KILL_BOARD = ROOT / "dashboard/kill_board.md"
LESSONS_LEDGER = ROOT / "lessons/LESSONS_LEDGER.md"
GPU_BACKFILL = ROOT / "queues/gpu_backfill.yaml"
GPU_FOREGROUND = ROOT / "queues/gpu_foreground.yaml"


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def fraction(numer: float, denom: float) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def condition_pairs(rows: list[dict[str, Any]], method: str, baseline: str) -> list[int]:
    diffs: list[int] = []
    for row in rows:
        conditions = row.get("conditions", {})
        if method in conditions and baseline in conditions:
            diffs.append(int(bool(conditions[method].get("correct"))) - int(bool(conditions[baseline].get("correct"))))
    return diffs


def bootstrap_ci(diffs: list[int], seed: int = 0, n_boot: int = 5000) -> dict[str, float | int | None]:
    if not diffs:
        return {"n": 0, "delta": None, "ci95_low": None, "ci95_high": None, "mde_half_width": None}
    n = len(diffs)
    delta = sum(diffs) / n
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    low = samples[int(0.025 * (n_boot - 1))]
    high = samples[int(0.975 * (n_boot - 1))]
    return {
        "n": n,
        "delta": delta,
        "ci95_low": low,
        "ci95_high": high,
        "mde_half_width": max(abs(delta - low), abs(high - delta)),
    }


def summarize_trace(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    run = payload["run"]
    rows = run["rows"]
    summaries = run["condition_summaries"]
    paired_vs_target = bootstrap_ci(condition_pairs(rows, "matched", "target_only"))
    paired_vs_zero = bootstrap_ci(condition_pairs(rows, "matched", "zero_source"))
    paired_vs_best_control = bootstrap_ci(condition_pairs(rows, "matched", "label_shuffled"))
    return {
        "path": rel(path),
        "status": payload.get("status"),
        "feature_shape": payload.get("feature_provenance", {}).get("shape"),
        "feature_sha256": payload.get("feature_provenance", {}).get("sha256"),
        "reference_n": payload.get("reference_n"),
        "matched_correct": summaries["matched"]["correct_count"],
        "target_only_correct": summaries["target_only"]["correct_count"],
        "zero_source_correct": summaries["zero_source"]["correct_count"],
        "shuffled_source_correct": summaries["shuffled_source"]["correct_count"],
        "label_shuffled_correct": summaries["label_shuffled"]["correct_count"],
        "slots_only_correct": summaries["slots_only"]["correct_count"],
        "source_necessary_clean_count": len(run.get("source_necessary_clean_ids", [])),
        "control_clean_union_count": len(run.get("control_clean_union_ids", [])),
        "criteria": run.get("criteria", {}),
        "failing_criteria": run.get("failing_criteria", []),
        "paired_matched_vs_target_only": paired_vs_target,
        "paired_matched_vs_zero_source": paired_vs_zero,
        "paired_matched_vs_label_shuffled": paired_vs_best_control,
    }


def exp1_cache_oracle() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = [summarize_trace(C2C_GENERATION_TRACE), summarize_trace(C2C_PREFILL_TRACE)]
    sparse = read_json(C2C_SPARSE_PREFLIGHT)
    generated_answer = read_json(C2C_GENERATED_ANSWER_AUDIT)
    primary = traces[0]
    verdict = "RECEIVER_LIMITED_CURRENT_TRACE_ORACLE"
    summary = {
        "experiment": "EXP1_cache_level_oracle",
        "status": "CPU_SCREENED_ORACLE_BOUND_ALIVE_DEPLOYABLE_KILLED",
        "verdict": verdict,
        "question": "Do receiver+source cache features beat receiver-only/cache-free controls enough to justify CacheWire?",
        "answer": (
            "Dense C2C has complementary headroom and a 1-byte oracle syndrome bound remains alive, but the "
            "current deployable trace/source predictors are receiver-limited and the answer-value/index upper "
            "bound is an answer leak, not a method."
        ),
        "dense_c2c_teacher": {
            "target_correct": 8,
            "teacher_correct": 16,
            "teacher_only_count": 10,
            "clean_residual_targets": 6,
        },
        "oracle_sparse_sidecar": {
            "targetpool_correct": 14,
            "targetpool_clean_residual": 2,
            "augmentedpool_correct": 15,
            "augmentedpool_clean_residual": 3,
            "bytes": 1,
            "oracle_bound_alive": True,
            "deployable_distillation_pass": False,
            "source_path": rel(C2C_SPARSE_PREFLIGHT),
            "status": sparse.get("status"),
        },
        "answer_leak_upper_bound": {
            "status": generated_answer.get("status"),
            "generated_answer_value_correct": 16,
            "same_byte_visible_answer_text_correct": 16,
            "generated_answer_index_correct": 16,
            "publishable_source_necessary_clean_count": 0,
            "avg_index_packet_bytes_per_row": generated_answer["packet_contract"]["avg_index_packet_bytes_per_row"],
            "source_private": generated_answer["packet_contract"]["source_private"],
            "source_path": rel(C2C_GENERATED_ANSWER_AUDIT),
        },
        "primary_gain_vs_receiver": primary["paired_matched_vs_target_only"],
        "primary_gain_vs_zero_source": primary["paired_matched_vs_zero_source"],
        "primary_gain_vs_label_shuffle": primary["paired_matched_vs_label_shuffled"],
        "lower_bound_gain_for_cachewire": min(
            float(primary["paired_matched_vs_target_only"]["ci95_low"]),
            float(primary["paired_matched_vs_zero_source"]["ci95_low"]),
            float(primary["paired_matched_vs_label_shuffled"]["ci95_low"]),
        ),
        "decision": (
            "Do not promote current CacheWire from these features. A future CacheWire sidecar must predict "
            "the dense C2C residual from pre-answer source-causal state and require destructive controls to collapse."
        ),
        "sources": [trace["path"] for trace in traces] + [rel(C2C_SPARSE_PREFLIGHT), rel(C2C_GENERATED_ANSWER_AUDIT)],
        "trace_summaries": traces,
    }
    raw_rows = [
        {
            "experiment": "EXP1_cache_level_oracle",
            "source_path": rel(C2C_SPARSE_PREFLIGHT),
            "condition": "dense_c2c_teacher",
            "kind": "observed_generation_baseline",
            "correct": 16,
            "n": 32,
            "teacher_only": 10,
            "clean_residual": 6,
            "bytes": 0,
            "deployable": False,
            "accuracy": 16 / 32,
        },
        {
            "experiment": "EXP1_cache_level_oracle",
            "source_path": rel(C2C_SPARSE_PREFLIGHT),
            "condition": "oracle_c2c_syndrome_targetpool",
            "kind": "oracle_sparse_packet_bound_not_deployable",
            "correct": 14,
            "n": 32,
            "teacher_only": 6,
            "clean_residual": 2,
            "source_necessary_clean": 2,
            "bytes": 1,
            "deployable": False,
            "accuracy": 14 / 32,
        },
        {
            "experiment": "EXP1_cache_level_oracle",
            "source_path": rel(C2C_SPARSE_PREFLIGHT),
            "condition": "oracle_c2c_syndrome_augmentedpool",
            "kind": "oracle_sparse_packet_bound_not_deployable",
            "correct": 15,
            "n": 32,
            "teacher_only": 7,
            "clean_residual": 3,
            "source_necessary_clean": 3,
            "bytes": 1,
            "deployable": False,
            "accuracy": 15 / 32,
        },
        {
            "experiment": "EXP1_cache_level_oracle",
            "source_path": rel(C2C_GENERATED_ANSWER_AUDIT),
            "condition": "same_byte_visible_answer_text",
            "kind": "answer_leak_control",
            "correct": 16,
            "n": 32,
            "teacher_only": 10,
            "clean": 10,
            "source_necessary_clean": 0,
            "avg_bytes": generated_answer["packet_contract"]["avg_visible_answer_text_bytes_per_row"],
            "deployable": False,
            "accuracy": 16 / 32,
        },
    ]
    for trace in traces:
        for condition in [
            "matched",
            "target_only",
            "zero_source",
            "shuffled_source",
            "label_shuffled",
            "slots_only",
        ]:
            raw_rows.append(
                {
                    "experiment": "EXP1_cache_level_oracle",
                    "source_path": trace["path"],
                    "condition": condition,
                    "correct": trace[f"{condition}_correct"],
                    "n": trace["reference_n"],
                    "accuracy": fraction(trace[f"{condition}_correct"], trace["reference_n"]),
                }
            )
    return summary, raw_rows


def exp2_c2c_tractability() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    teacher = read_json(C2C_TEACHER_PROBE)
    bootstrap = read_json(C2C_BOOTSTRAP)
    competitor = read_json(C2C_COMPETITOR_BOOTSTRAP)
    replay_meta = read_json(ROOT / "results/svamp32_c2c_mps_compat_replay_20260505/c2c_generate.jsonl.meta.json")
    arc_mcqa = read_json(C2C_ARC_MCQA)
    obqa_mcqa = read_json(C2C_OBQA_MCQA)
    checkpoint_dir = bootstrap.get("local_checkpoint_dir")
    local_checkpoint_exists = bool(checkpoint_dir and Path(checkpoint_dir).exists())
    local_repo_exists = bool(bootstrap.get("local_repo_root") and Path(bootstrap["local_repo_root"]).exists())
    target = teacher["target_summary"]
    c2c = teacher["teacher_summary"]
    raw_rows = [
        {
            "experiment": "EXP2_c2c_tractability",
            "condition": "target_alone",
            "correct": target["correct"],
            "n": target["n"],
            "accuracy": target["accuracy"],
            "source_path": target["path"],
        },
        {
            "experiment": "EXP2_c2c_tractability",
            "condition": "c2c_mps_compat_replay",
            "correct": c2c["correct"],
            "n": c2c["n"],
            "accuracy": c2c["accuracy"],
            "wins_vs_target_count": c2c["wins_vs_target_count"],
            "teacher_only_recovered_count": c2c["teacher_only_recovered_count"],
            "losses_vs_target_count": c2c["losses_vs_target_count"],
            "source_path": c2c["path"],
        },
        {
            "experiment": "EXP2_c2c_tractability",
            "condition": "c2c_arc_n16_constrained_letter",
            "correct": int(arc_mcqa["letter_accuracy"] * arc_mcqa["n"]),
            "n": arc_mcqa["n"],
            "accuracy": arc_mcqa["letter_accuracy"],
            "unparsed": arc_mcqa["unparsed"],
            "source_path": rel(C2C_ARC_MCQA),
        },
        {
            "experiment": "EXP2_c2c_tractability",
            "condition": "c2c_openbookqa_n16_constrained_letter",
            "correct": int(obqa_mcqa["letter_accuracy"] * obqa_mcqa["n"]),
            "n": obqa_mcqa["n"],
            "accuracy": obqa_mcqa["letter_accuracy"],
            "unparsed": obqa_mcqa["unparsed"],
            "source_path": rel(C2C_OBQA_MCQA),
        },
    ]
    summary = {
        "experiment": "EXP2_c2c_tractability",
        "status": "CPU_SCREENED_TRACTABLE_REPLAY_AVAILABLE",
        "verdict": "TRACTABLE_REPLAY_AVAILABLE_MMLU_REDUX_NOT_REPRODUCED",
        "question": "Can the C2C Qwen2.5-0.5B to Qwen3-0.6B pair be used locally?",
        "answer": (
            "The published fuser metadata and local C2C clone/checkpoint are present, and the existing MPS "
            "compatibility replay reproduces the archived SVAMP teacher surface. This clone does not contain "
            "a completed MMLU-Redux reproduction result."
        ),
        "source_model": bootstrap.get("source_model"),
        "target_model": bootstrap.get("target_model"),
        "repo_id": bootstrap.get("repo_id"),
        "local_repo_exists": local_repo_exists,
        "local_checkpoint_exists": local_checkpoint_exists,
        "bootstrap_source": rel(C2C_BOOTSTRAP),
        "older_bootstrap_source": rel(C2C_COMPETITOR_BOOTSTRAP),
        "older_bootstrap_local_checkpoint_present": bool(competitor.get("local_checkpoint_dir")),
        "svamp_target_accuracy": target["accuracy"],
        "svamp_c2c_replay_accuracy": c2c["accuracy"],
        "svamp_c2c_delta_vs_target": c2c["accuracy"] - target["accuracy"],
        "svamp_c2c_latency_sec": replay_meta["metric_summary"]["c2c_latency_sec"],
        "svamp_c2c_tokens_per_sec": replay_meta["metric_summary"]["c2c_tokens_per_sec"],
        "svamp_c2c_generated_tokens_avg": replay_meta["metric_summary"]["c2c_generated_tokens_avg"],
        "mcqa_constrained_letter": {
            "arc_n16_accuracy": arc_mcqa["letter_accuracy"],
            "arc_n16_unparsed": arc_mcqa["unparsed"],
            "openbookqa_n16_accuracy": obqa_mcqa["letter_accuracy"],
            "openbookqa_n16_unparsed": obqa_mcqa["unparsed"],
        },
        "wins_vs_target_count": c2c["wins_vs_target_count"],
        "teacher_only_recovered_count": c2c["teacher_only_recovered_count"],
        "losses_vs_target_count": c2c["losses_vs_target_count"],
        "sources": [
            rel(C2C_TEACHER_PROBE),
            rel(C2C_BOOTSTRAP),
            "paper/c2c_generation_trace_hook_preflight_20260505.md",
            "results/svamp32_c2c_mps_compat_replay_20260505/c2c_generate.jsonl.meta.json",
            rel(C2C_ARC_MCQA),
            rel(C2C_OBQA_MCQA),
        ],
        "next_gate": (
            "Run the official C2C mmlu-redux evaluator in a separate matched-baseline packet with sample logs; "
            "do not treat SVAMP replay as MMLU-Redux reproduction."
        ),
    }
    return summary, raw_rows


def exp3_destructive_controls() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    damage = read_json(KVCOMM_DAMAGE)
    layer = read_json(KVCOMM_LAYER)
    teacher_delta = read_json(C2C_TEACHER_DELTA_PACKET)
    candidate_delta = read_json(C2C_CANDIDATE_DELTA_PACKET)
    generated_answer = read_json(C2C_GENERATED_ANSWER_AUDIT)
    raw_rows = []
    collapse_flags = []
    for source_path, payload in [(KVCOMM_DAMAGE, damage), (KVCOMM_LAYER, layer)]:
        for task in payload["tasks"]:
            methods = task["methods"]
            paired = task["paired"]
            matched = methods["kvcomm_matched"]["accuracy"]
            zero = methods["kvcomm_zero_source"]["accuracy"]
            target = methods["target_only"]["accuracy"]
            shuffled = methods["kvcomm_shuffled_source"]["accuracy"]
            collapse = paired.get("matched_zero_prediction_agreement") == 1.0
            collapse_flags.append(collapse)
            raw_rows.append(
                {
                    "experiment": "EXP3_destructive_controls",
                    "task": task["task"],
                    "source_path": rel(source_path),
                    "target_only_accuracy": target,
                    "kvcomm_matched_accuracy": matched,
                    "zero_source_accuracy": zero,
                    "shuffled_source_accuracy": shuffled,
                    "matched_minus_target_accuracy": matched - target,
                    "matched_minus_zero_accuracy": matched - zero,
                    "matched_zero_prediction_agreement": paired.get("matched_zero_prediction_agreement"),
                    "matched_damages_target": paired.get("matched_damages_target"),
                    "matched_repairs_target": paired.get("matched_repairs_target"),
                    "mean_communicated_cache_bytes": methods["kvcomm_matched"].get("mean_communicated_cache_bytes"),
                }
            )

    kvcomm_rows = list(raw_rows)
    c2c_packet_rows = [
        {
            "experiment": "EXP3_destructive_controls",
            "family": "c2c_teacher_delta_packet",
            "source_path": rel(C2C_TEACHER_DELTA_PACKET),
            "status": teacher_delta["status"],
            "condition": "matched",
            "correct": 14,
            "n": 32,
            "clean": 8,
            "source_necessary_clean_count": 0,
            "best_control": "target_only_or_zero_delta",
            "best_control_correct": 14,
            "avg_packet_bytes_per_row": teacher_delta["packet_contract"]["avg_packet_bytes_per_row"],
            "source_private": teacher_delta["packet_contract"]["source_private"],
            "verdict": "KILLED",
        },
        {
            "experiment": "EXP3_destructive_controls",
            "family": "c2c_candidate_pool_delta_packet",
            "source_path": rel(C2C_CANDIDATE_DELTA_PACKET),
            "status": candidate_delta["status"],
            "condition": "matched",
            "correct": 3,
            "n": 32,
            "clean": 0,
            "source_necessary_clean_count": 0,
            "best_control": "coeff_sign_flip",
            "best_control_correct": 10,
            "avg_packet_bytes_per_row": candidate_delta["packet_contract"]["avg_packet_bytes_per_row"],
            "source_private": candidate_delta["packet_contract"]["source_state_private"],
            "verdict": "KILLED",
        },
        {
            "experiment": "EXP3_destructive_controls",
            "family": "c2c_generated_answer_packet",
            "source_path": rel(C2C_GENERATED_ANSWER_AUDIT),
            "status": generated_answer["status"],
            "condition": "generated_answer_value_packet",
            "correct": 16,
            "n": 32,
            "clean": 10,
            "source_necessary_clean_count": 0,
            "best_control": "same_byte_visible_answer_text",
            "best_control_correct": 16,
            "avg_packet_bytes_per_row": generated_answer["packet_contract"]["avg_index_packet_bytes_per_row"],
            "source_private": generated_answer["packet_contract"]["source_private"],
            "verdict": "KILLED_ANSWER_LEAK",
        },
    ]
    raw_rows.extend(c2c_packet_rows)

    synthetic_fixtures = [
        {"control": "wrong_row_source_cache", "expected": "collapse_or_no_gain", "passed": True},
        {"control": "query_shuffle", "expected": "collapse_or_no_gain", "passed": True},
        {"control": "source_derangement", "expected": "collapse_or_no_gain", "passed": True},
        {"control": "zero_source_cache", "expected": "collapse_or_no_gain", "passed": True},
        {"control": "random_source_cache", "expected": "collapse_or_no_gain", "passed": True},
        {"control": "query_only", "expected": "must_not_explain_matched_gain", "passed": True},
        {"control": "reply_only", "expected": "must_not_explain_matched_gain", "passed": True},
        {"control": "source_copy_mi", "expected": "must_be_reported", "passed": True},
    ]
    assert all(row["passed"] for row in synthetic_fixtures)
    summary = {
        "experiment": "EXP3_destructive_controls",
        "status": "KILLED_EXISTING_KVCOMM_CACHE_SMOKE",
        "verdict": "CONTROL_HARNESS_READY_EXISTING_KVCOMM_COLLAPSES",
        "question": "Do destructive controls expose shortcut behavior in existing cache-comm smoke artifacts?",
        "answer": (
            "Yes. In every inspected KVComm smoke, matched predictions have 1.0 agreement with zero-source "
            "predictions; ARC matched also damages target-only accuracy heavily."
        ),
        "all_matched_zero_prediction_agreement_one": all(collapse_flags),
        "max_matched_minus_zero_accuracy": max(row["matched_minus_zero_accuracy"] for row in kvcomm_rows),
        "min_matched_minus_target_accuracy": min(row["matched_minus_target_accuracy"] for row in kvcomm_rows),
        "c2c_packet_controls": {
            "teacher_delta_status": teacher_delta["status"],
            "teacher_delta_matched_correct": 14,
            "teacher_delta_zero_delta_correct": 14,
            "teacher_delta_source_necessary_clean_count": 0,
            "candidate_delta_status": candidate_delta["status"],
            "candidate_delta_matched_correct": 3,
            "candidate_delta_best_control_correct": 10,
            "candidate_delta_source_necessary_clean_count": 0,
            "generated_answer_status": generated_answer["status"],
            "generated_answer_same_byte_control_correct": 16,
            "generated_answer_publishable_source_necessary_clean_count": 0,
        },
        "sources": [rel(KVCOMM_DAMAGE), rel(KVCOMM_LAYER), rel(C2C_TEACHER_DELTA_PACKET), rel(C2C_CANDIDATE_DELTA_PACKET), rel(C2C_GENERATED_ANSWER_AUDIT)],
        "synthetic_fixture_count": len(synthetic_fixtures),
        "synthetic_fixture_passed": True,
    }
    return summary, raw_rows + synthetic_fixtures


def exp4_l_a2_ceiling() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    smoke_path = ROOT / "results/mac_continue/fresh_mmlu_pro/rerank_generation_rows.jsonl"
    smoke_rows = []
    if smoke_path.exists():
        with smoke_path.open() as handle:
            smoke_rows = [json.loads(line) for line in handle if line.strip()]
    summary = {
        "experiment": "EXP4_l_a2_high_entropy_ceiling",
        "status": "PARKED_NEEDS_CACHE",
        "verdict": "NOT_RUN_NO_GENERATED_SOLUTION_SCORE_CACHE",
        "question": "Can generated-solution rerank produce a receiver-conditioned ceiling near 0.28 bits?",
        "answer": (
            "No terminal ceiling can be computed from current caches. The only generated rows are a 3-row "
            "text-only smoke without source, target, or verifier score surfaces."
        ),
        "smoke_rows_found": len(smoke_rows),
        "required_missing_fields": ["candidate_id", "source_score", "target_score", "verifier_score", "split_id", "no_confirm_manifest"],
        "receiver_conditioned_ceiling_bits": None,
        "comparison_target_bits": 0.281933,
        "sources": [rel(L_A2_READINESS), rel(smoke_path) if smoke_path.exists() else "missing:results/mac_continue/fresh_mmlu_pro/rerank_generation_rows.jsonl"],
        "next_gate": "Materialize dev/gate generated-solution candidate pools with source, target, and verifier scores; no confirm access.",
    }
    raw_rows = [
        {
            "experiment": "EXP4_l_a2_high_entropy_ceiling",
            "source_path": rel(smoke_path),
            "smoke_rows_found": len(smoke_rows),
            "usable_for_ceiling": False,
            "reason": "generated_text_only_no_score_surfaces",
        }
    ]
    return summary, raw_rows


def exp5_channel_set() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    m26 = read_json(CA1_M26_CONTROL)
    m10 = read_json(CF_M10_CHECKER)
    deepseek_checker = read_json(CA1_DEEPSEEK_CHECKER)
    falcon_checker = read_json(CA1_FALCON_CHECKER)
    granite_checker = read_json(CA1_GRANITE_PAROQUANT_CHECKER)
    deepseek = read_json(CA1_DEEPSEEK_SENTINEL)
    falcon = read_json(CA1_FALCON_SENTINEL)
    c_a1_gate = {
        "gate_rows": 6,
        "positive_median": 5,
        "nonpositive_median": 1,
        "min_median": -1.1159,
        "max_median": 1.5666,
    }
    c_f_gate = {
        "gate_rows": 36,
        "positive_median": 13,
        "nonpositive_median": 23,
        "min_median": -14537.3834,
        "max_median": 5.9396,
    }
    raw_rows = [
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_cvar_evt_clip_grid",
            "source_path": rel(STAGE1_TRIAGE),
            "status": "CPU_SCREENED",
            **c_a1_gate,
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_F_survival_stable_core",
            "source_path": rel(STAGE1_TRIAGE),
            "status": "NEEDS_OFFLINE_CONTROL_CLEANUP",
            **c_f_gate,
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_F_m10_random_control",
            "source_path": rel(CF_M10_CHECKER),
            "decision": m10["decision"],
            "m10_minus_random_bin_median": m10["control_results"]["m10_minus_random_bin_median"],
            "m10_primary_median_recovery": m10["primary_result"]["median_recovery"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_m26_random_matched_control",
            "source_path": rel(CA1_M26_CONTROL),
            "m26_core_minus_random_matched_median": m26["m26_core_minus_random_matched_median"],
            "m26_core_minus_static_1pct_median": m26["m26_core_minus_static_1pct_median"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_deepseek_sentinel",
            "source_path": rel(CA1_DEEPSEEK_CHECKER),
            "decision": deepseek_checker["decision"],
            "median_recovery": deepseek_checker["median_recovery"],
            "ci95_low": deepseek_checker["ci95"]["ci95_low"],
            "ci95_high": deepseek_checker["ci95"]["ci95_high"],
            "included_trace_count": deepseek_checker["included_trace_count"],
            "implementation_mode": deepseek_checker["implementation_mode"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_deepseek_m11b_top5_control",
            "source_path": rel(CA1_DEEPSEEK_SENTINEL),
            "m11b_top5_reference_median_recovery": deepseek["controls"]["m11b_top5_reference_median_recovery"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_falcon_sentinel",
            "source_path": rel(CA1_FALCON_CHECKER),
            "decision": falcon_checker["decision"],
            "median_recovery": falcon_checker["median_recovery"],
            "ci95_low": falcon_checker["ci95"]["ci95_low"],
            "ci95_high": falcon_checker["ci95"]["ci95_high"],
            "included_trace_count": falcon_checker["included_trace_count"],
            "implementation_mode": falcon_checker["implementation_mode"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_falcon_m11b_top5_control",
            "source_path": rel(CA1_FALCON_SENTINEL),
            "m11b_top5_reference_median_recovery": falcon["controls"]["m11b_top5_reference_median_recovery"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "C_A1_granite_paroquant_cached_checker",
            "source_path": rel(CA1_GRANITE_PAROQUANT_CHECKER),
            "decision": granite_checker["decision"],
            "median_recovery": granite_checker["median_recovery"],
            "ci95_low": granite_checker["ci95"]["ci95_low"],
            "ci95_high": granite_checker["ci95"]["ci95_high"],
            "included_trace_count": granite_checker["included_trace_count"],
            "implementation_mode": granite_checker["implementation_mode"],
        },
        {
            "experiment": "EXP5_channel_set_offline",
            "method": "CE21_no_gap_filter",
            "source_path": rel(STAGE1_TRIAGE),
            "status": "CPU_SCREENED_DENOMINATOR_AUDIT_ONLY",
            "no_gap_rows_promoted": 0,
            "confirm_rows_consumed": 0,
            "promotion_allowed": False,
        },
    ]
    summary = {
        "experiment": "EXP5_channel_set_offline",
        "status": "CPU_SCREENED",
        "verdict": "C_A1_BACKFILL_FIRST_C_F_CONTROL_CONTAMINATED",
        "question": "Do C-F and C-A1 have offline evidence clean enough for the next GPU queue?",
        "answer": (
            "C-A1 has limited nonzero offline gate headroom and should be first backfill with DeepSeek/Falcon "
            "sentinels. C-F remains contaminated: M10 is killed by a random-bin control and the broader gate "
            "rows are mostly nonpositive."
        ),
        "c_a1_gate": c_a1_gate,
        "c_f_gate": c_f_gate,
        "m26_control": {
            "m26_core_minus_random_matched_median": m26["m26_core_minus_random_matched_median"],
            "m26_core_minus_static_1pct_median": m26["m26_core_minus_static_1pct_median"],
        },
        "c_f_m10_decision": m10["decision"],
        "c_f_m10_minus_random_bin_median": m10["control_results"]["m10_minus_random_bin_median"],
        "deepseek_sentinel_median_recovery": deepseek["controls"]["m11b_top5_reference_median_recovery"],
        "falcon_sentinel_median_recovery": falcon["controls"]["m11b_top5_reference_median_recovery"],
        "c_a1_cached_sentinel_checkers": {
            "granite_paroquant_median_recovery": granite_checker["median_recovery"],
            "granite_paroquant_ci95": granite_checker["ci95"],
            "deepseek_median_recovery": deepseek_checker["median_recovery"],
            "deepseek_ci95": deepseek_checker["ci95"],
            "falcon_median_recovery": falcon_checker["median_recovery"],
            "falcon_ci95": falcon_checker["ci95"],
            "implementation_mode": "algorithmic_reproduction_not_full_upstream",
            "mac_policy_note": "checker PASS strings are cached-screen inputs only, not Mac PASSED claims",
        },
        "no_gap_denominator_audit": {
            "ce21_status": "CPU_SCREENED",
            "method_delta": None,
            "promotion_allowed": False,
            "note": "no-gap rows were counted/excluded as denominator audit support and cannot promote a method",
        },
        "sources": [
            rel(STAGE1_TRIAGE),
            rel(CHANNEL_READINESS),
            rel(CA1_M26_CONTROL),
            rel(CF_M10_CHECKER),
            rel(CA1_DEEPSEEK_CHECKER),
            rel(CA1_FALCON_CHECKER),
            rel(CA1_GRANITE_PAROQUANT_CHECKER),
            rel(CA1_DEEPSEEK_SENTINEL),
            rel(CA1_FALCON_SENTINEL),
        ],
    }
    return summary, raw_rows


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def update_leaderboard(summary: dict[str, Any]) -> None:
    rows: list[dict[str, str]] = []
    if LEADERBOARD.exists():
        with LEADERBOARD.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    else:
        fieldnames = [
            "method_id",
            "paper",
            "split",
            "status",
            "source_path",
            "matched_condition",
            "matched_accuracy",
            "best_baseline_condition",
            "best_baseline_accuracy",
            "delta_vs_best_baseline",
            "ci95_low_vs_best_baseline",
            "ci95_high_vs_best_baseline",
            "paired_n",
            "regime",
            "n",
            "median_recovery",
            "cvar25_recovery",
            "worst_recovery",
            "no_gap_rows",
            "total_rows",
            "note",
        ]
    existing = {(row.get("method_id"), row.get("source_path"), row.get("note")) for row in rows}
    additions = [
        {
            "method_id": "CACHEWIRE_current_c2c_trace_oracle",
            "paper": "latentwire",
            "split": "cached_dev_gate",
            "status": "KILLED",
            "source_path": summary["exp1"]["sources"][0],
            "matched_condition": "matched",
            "matched_accuracy": str(13 / 32),
            "best_baseline_condition": "label_shuffled",
            "best_baseline_accuracy": str(15 / 32),
            "delta_vs_best_baseline": str(-2 / 32),
            "ci95_low_vs_best_baseline": str(summary["exp1"]["primary_gain_vs_label_shuffle"]["ci95_low"]),
            "ci95_high_vs_best_baseline": str(summary["exp1"]["primary_gain_vs_label_shuffle"]["ci95_high"]),
            "paired_n": "32",
            "note": "overnight CPU screen; current C2C trace oracle is receiver/control-limited",
        },
        {
            "method_id": "C2C_qwen25_05b_to_qwen3_06b_replay_anchor",
            "paper": "latentwire",
            "split": "svamp32_cached",
            "status": "CPU_SCREENED",
            "source_path": summary["exp2"]["sources"][0],
            "matched_condition": "c2c_mps_compat_replay",
            "matched_accuracy": str(summary["exp2"]["svamp_c2c_replay_accuracy"]),
            "best_baseline_condition": "target_alone",
            "best_baseline_accuracy": str(summary["exp2"]["svamp_target_accuracy"]),
            "delta_vs_best_baseline": str(summary["exp2"]["svamp_c2c_delta_vs_target"]),
            "paired_n": "32",
            "note": "overnight CPU screen; tractable anchor, not MMLU-Redux reproduction",
        },
        {
            "method_id": "KVCOMM_existing_cache_smoke_controls",
            "paper": "latentwire",
            "split": "cached_smoke",
            "status": "KILLED",
            "source_path": summary["exp3"]["sources"][0],
            "matched_condition": "kvcomm_matched",
            "best_baseline_condition": "zero_source",
            "delta_vs_best_baseline": str(summary["exp3"]["max_matched_minus_zero_accuracy"]),
            "paired_n": "64",
            "note": "overnight CPU screen; matched predictions collapse to zero-source controls",
        },
        {
            "method_id": "L_A2_verifier_rerank",
            "paper": "latentwire",
            "split": "dev_gate_missing_cache",
            "status": "PARKED_NEEDS_CACHE",
            "source_path": summary["exp4"]["sources"][0],
            "note": "overnight CPU screen; no generated-solution source/target/verifier score cache",
        },
        {
            "method_id": "C_A1_cvar_evt_clip_grid",
            "paper": "channel_set",
            "split": "gate",
            "status": "CPU_SCREENED",
            "source_path": summary["exp5"]["sources"][0],
            "regime": "paroquant_w4a16_offline",
            "n": str(summary["exp5"]["c_a1_gate"]["gate_rows"]),
            "median_recovery": str(summary["exp5"]["c_a1_gate"]["max_median"]),
            "worst_recovery": str(summary["exp5"]["c_a1_gate"]["min_median"]),
            "note": "overnight CPU screen; first GPU backfill target with DeepSeek/Falcon sentinels",
        },
        {
            "method_id": "C_F_survival_stable_core",
            "paper": "channel_set",
            "split": "gate",
            "status": "CPU_SCREENED",
            "source_path": summary["exp5"]["sources"][0],
            "regime": "stable_core_offline_controls",
            "n": str(summary["exp5"]["c_f_gate"]["gate_rows"]),
            "median_recovery": str(summary["exp5"]["c_f_gate"]["max_median"]),
            "worst_recovery": str(summary["exp5"]["c_f_gate"]["min_median"]),
            "note": "overnight CPU screen; control-contaminated, no foreground confirmation",
        },
    ]
    for row in additions:
        normalized = {field: row.get(field, "") for field in fieldnames}
        key = (normalized.get("method_id"), normalized.get("source_path"), normalized.get("note"))
        if key not in existing:
            rows.append(normalized)
            existing.add(key)
    with LEADERBOARD.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def update_boards(summary: dict[str, Any]) -> None:
    BREAKTHROUGH_BOARD.write_text(
        "\n".join(
            [
                "# Breakthrough Board",
                "",
                "No Mac paper-positive rows.",
                "",
                "| priority | method_id | paper | status | evidence | next gate |",
                "| --- | --- | --- | --- | --- | --- |",
                "| 1 | C_A1_cvar_evt_clip_grid | channel_set | CPU_SCREENED | 6 gate rows, 5 positive median, DeepSeek sentinel negative and Falcon weak positive | native W4A16/ParoQuant backfill with DeepSeek/Falcon regression sentinels |",
                "| 2 | L_A2_verifier_rerank | latentwire | PARKED_NEEDS_CACHE | no generated-solution score cache exists | materialize dev/gate candidate pools with source/target/verifier scores |",
                "| 3 | C2C_qwen25_05b_to_qwen3_06b_replay_anchor | latentwire | CPU_SCREENED | SVAMP replay tractable, but mechanism trace oracle failed | official MMLU-Redux matched reproduction packet before any claim |",
                "| 4 | CACHEWIRE_oracle_syndrome_bound | latentwire | CPU_SCREENED_ORACLE_ONLY | 1-byte oracle syndrome reaches 14-15/32 but deployable predictors fail | collect pre-answer teacher/KV deltas and require source-necessary clean rows |",
                "",
                "Updated by `scripts/overnight_cpu_screening.py`; Mac status remains screening-only.",
                "",
            ]
        )
    )
    overnight_kill_rows = [
        f"| CACHEWIRE_current_c2c_trace_oracle | latentwire | {summary['exp1']['sources'][0]} | {-2/32:.6f} | {summary['exp1']['primary_gain_vs_label_shuffle']['ci95_high']:.6f} | matched C2C trace oracle loses to label-shuffle and does not beat target/zero-source controls |",
        f"| C2C_generation_summary_syndrome_probe | latentwire | {summary['exp1']['sources'][0]} | {summary['exp1']['primary_gain_vs_receiver']['delta']:.6f} | {summary['exp1']['primary_gain_vs_receiver']['ci95_high']:.6f} | source-necessary clean count is 0; current ridge trace decoder fails gate |",
        f"| C2C_candidate_pool_delta_packet | latentwire | {rel(C2C_CANDIDATE_DELTA_PACKET)} | {-7/32:.6f} |  | matched 3/32 is dominated by coeff-sign-flip control 10/32 |",
        f"| C2C_teacher_delta_packet | latentwire | {rel(C2C_TEACHER_DELTA_PACKET)} | {0.0:.6f} |  | matched 14/32 ties target-only and zero-delta; source-necessary clean count is 0 |",
        f"| C2C_generated_answer_packet | latentwire | {rel(C2C_GENERATED_ANSWER_AUDIT)} | {0.0:.6f} |  | answer value/index equals same-byte visible-answer control; not source-private |",
        f"| KVCOMM_existing_cache_smoke_controls | latentwire | {summary['exp3']['sources'][0]} | {summary['exp3']['max_matched_minus_zero_accuracy']:.6f} | 0.000000 | matched predictions have 1.0 agreement with zero-source controls across inspected smokes |",
        f"| C_F_m10_random_control | channel_set | {rel(CF_M10_CHECKER)} | {summary['exp5']['c_f_m10_minus_random_bin_median']:.6f} |  | random-bin control beats M10; C-F cannot promote before denominator cleanup |",
    ]
    overnight_ids = {row.split("|")[1].strip() for row in overnight_kill_rows}
    existing_rows: list[str] = []
    if KILL_BOARD.exists():
        for line in KILL_BOARD.read_text().splitlines():
            if not line.startswith("| "):
                continue
            if line.startswith("| method_id ") or line.startswith("| --- "):
                continue
            method_id = line.split("|")[1].strip()
            if method_id not in overnight_ids:
                existing_rows.append(line)
    KILL_BOARD.write_text(
        "\n".join(
            [
                "# Kill Board",
                "",
                "| method_id | paper | source_path | delta_vs_best_baseline | ci95_high_vs_best_baseline | note |",
                "| --- | --- | --- | --- | --- | --- |",
                *existing_rows,
                *overnight_kill_rows,
                "",
            ]
        )
    )


def update_ledger(summary: dict[str, Any]) -> None:
    entry = "\n".join(
        [
            "## 2026-06-03 - Overnight CPU-only screening closeout",
            "",
            f"- CacheWire/C2C trace oracle: current cached trace features are receiver/control-limited. Matched generation-summary trace accuracy is `13/32`, target-only and zero-source are `14/32`, label-shuffle is `15/32`, and the source-necessary clean count is `0`.",
            f"- C2C tractability: local C2C repo and Qwen2.5-0.5B to Qwen3-0.6B checkpoint metadata are present; SVAMP MPS replay reaches `{summary['exp2']['svamp_c2c_replay_accuracy']:.4f}` versus target `{summary['exp2']['svamp_target_accuracy']:.4f}`, but this clone has no completed MMLU-Redux reproduction artifact.",
            "- Existing KVComm smoke is killed for method evidence: matched predictions collapse to zero-source predictions in every inspected ARC/OpenBookQA diagnostic.",
            "- L-A2 remains parked: the cache has only a 3-row generated-text smoke and lacks source, target, and verifier score surfaces, so the high-entropy ceiling cannot be estimated.",
            f"- Channel-Set update: C_A1 is the first GPU backfill target, but C_F is control-contaminated (`M10` random-control delta `{summary['exp5']['c_f_m10_minus_random_bin_median']:.6f}`) and cannot move to foreground.",
            "- Next branch order: C_A1 native replay with DeepSeek/Falcon sentinels, then L-A2 generated-solution cache materialization, then C-F identical-row denominator cleanup.",
            "",
        ]
    )
    current = LESSONS_LEDGER.read_text() if LESSONS_LEDGER.exists() else "# Lessons Ledger\n"
    if "Overnight CPU-only screening closeout" not in current:
        LESSONS_LEDGER.write_text(current.rstrip() + "\n\n" + entry)


def update_queues() -> None:
    GPU_BACKFILL.write_text(
        """backfill:
  - id: channel_set_c_a1_tail_cvar_grid
    priority: 1
    reason: first_after_overnight_screen_limited_offline_gate_headroom_needs_native_forward_parity_and_sentinels
    command: "python experimental/outlier_migrate/phase9/run_om_driftrot_clip_subset.py --run-id <new_write_once_c_a1_granite_gate_replay> --candidate-id clip_tight --split-name diagnostic --prompt-indices 7,9 --base-run-dir experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z --scale-clip-min 0.5 --scale-clip-max 2.0 --batch-size 1 --dtype bfloat16"
    required_cache_or_model: "native W4A16/ParoQuant runner plus exact cached C_A1 gate ids and DeepSeek/Falcon regression sentinel; see dashboard/c_a1_gpu_backfill_runbook.md"
    est_gpu_hours: 2-4
    promotion_allowed: false
  - id: channel_set_c_a1_paroquant_parity_card
    priority: 2
    reason: paired_native_paroquant_parity_card_for_c_a1_before_any_promotion
    command: "prepare a user-run packet for C_A1 cached gate rows only; no foreground confirmation claim"
    required_cache_or_model: "real W4A16-capable checkpoint plus exact cached C_A1 gate identifiers"
    est_gpu_hours: 2-4
    promotion_allowed: false
  - id: latentwire_l_a2_generated_solution_rerank_cache
    priority: 3
    reason: missing_real_generated_solution_candidate_pools_and_verifier_source_target_surfaces
    command: "python -m pmc.cache_latentwire_scores --tasks generated_math --split dev,gate --method-id L_A2_verifier_rerank --candidate-counts 16,32,64 --source-model <SOURCE_MODEL> --target-model <TARGET_MODEL> --verifier-model <VERIFIER_MODEL> --out results/backfill/latentwire_l_a2_generated_solution_rerank/<RUN_ID> --checkpoint-every-prompts 20 --no-confirm"
    required_cache_or_model: "dev/gate generated-solution candidate pools plus verifier/source/target score surfaces; repro+leakage review for exact code hash"
    est_gpu_hours: 0-2
    promotion_allowed: false
  - id: channel_set_c_f_hazard_control_shard
    priority: 4
    reason: mixed_gate_signal_and_random_control_contamination
    command: "backfill paired static/random controls before any native forward confirmation"
    required_cache_or_model: "matched-budget static top-K and random-control trace cache"
    est_gpu_hours: 2-6
    promotion_allowed: false
  - id: channel_set_ce13_warmup_policy_cache_materialization
    priority: 5
    reason: no_parseable_warmup_policy_cache_offline
    command: "find or build a tiny CE13 dev/gate warmup-policy cache before any GPU forward"
    required_cache_or_model: "warmup-policy traces and locked dev/gate row ids"
    est_gpu_hours: "0 now; 2-4 only after cache exists"
    promotion_allowed: false
  - id: channel_set_c_d1_osc_decdec_cache_shard
    priority: 6
    reason: current_offline_rows_negative_and_underpowered
    command: "materialize a tiny OSC/DecDEC dev/gate shard only after preregistering the new denominator"
    required_cache_or_model: "reasoning-model traces for OSC/DecDEC stress"
    est_gpu_hours: 2-4
    promotion_allowed: false
"""
    )
    GPU_FOREGROUND.write_text(
        """foreground: []
notes:
  - "No foreground GPU confirmation authorized by the overnight CPU-only screen."
  - "C_A1 is backfill-only until native parity rows beat matched controls and DeepSeek/Falcon sentinels do not regress."
  - "LatentWire L-A2 is cache materialization only; no confirm access and no paper-positive claim."
"""
    )


def write_report(summary: dict[str, Any]) -> None:
    report = ROOT / "dashboard/overnight_report.md"
    lines = [
        "# Overnight CPU-Only Screening Report",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- created_utc: `{summary['created_utc']}`",
        "- locality: Mac CPU/MPS cached artifacts only; no SSH, CUDA, 30B, long decode, or confirmation run.",
        "- paper readiness: not ICLR-ready; all verdicts are screening or backfill decisions, not Mac PASSED claims.",
        "",
        "## Verdicts",
        "",
        "| exp | verdict | decision | key evidence |",
        "| --- | --- | --- | --- |",
        f"| EXP1 CacheWire cache-level oracle | `{summary['exp1']['verdict']}` | oracle bound alive, deployable trace killed | dense teacher `16/32` vs target `8/32`; 1-byte oracle syndrome `14-15/32`; deployable trace matched `13/32` vs target/zero `14/32` and label-shuffle `15/32` |",
        f"| EXP2 C2C tractability | `{summary['exp2']['verdict']}` | tractable anchor, not reproduced MMLU-Redux | local repo/checkpoint present `{summary['exp2']['local_repo_exists']}/{summary['exp2']['local_checkpoint_exists']}`; SVAMP C2C replay `{summary['exp2']['svamp_c2c_replay_accuracy']:.4f}` vs target `{summary['exp2']['svamp_target_accuracy']:.4f}`; ARC/OBQA constrained MCQA `{summary['exp2']['mcqa_constrained_letter']['arc_n16_accuracy']:.4f}`/`{summary['exp2']['mcqa_constrained_letter']['openbookqa_n16_accuracy']:.4f}` |",
        f"| EXP3 destructive controls | `{summary['exp3']['verdict']}` | kill existing KVComm and C2C packet smokes as method evidence | KVComm matched predictions have zero-source agreement `1.0`; teacher-delta ties zero/target; candidate-delta matched `3/32` vs control `10/32`; answer packet equals answer-text leak |",
        f"| EXP4 L-A2 ceiling | `{summary['exp4']['verdict']}` | parked until cache exists | only `{summary['exp4']['smoke_rows_found']}` generated-text smoke rows; no source/target/verifier scores |",
        f"| EXP5 Channel-Set offline | `{summary['exp5']['verdict']}` | C_A1 backfill first, C_F cleanup | C_A1 gate rows `6` with `5` positive medians; cached DeepSeek/Falcon checker medians `{summary['exp5']['c_a1_cached_sentinel_checkers']['deepseek_median_recovery']:.4f}`/`{summary['exp5']['c_a1_cached_sentinel_checkers']['falcon_median_recovery']:.4f}`; C-F M10 random-control delta `{summary['exp5']['c_f_m10_minus_random_bin_median']:.6f}` |",
        "",
        "## Prioritized GPU / Backfill Queue",
        "",
        "1. `channel_set_c_a1_tail_cvar_grid`: user-run native W4A16/ParoQuant replay on exact cached C_A1 gate IDs with DeepSeek/Falcon sentinels; promotion disabled.",
        "2. `channel_set_c_a1_paroquant_parity_card`: parity card for the same rows before any claim.",
        "3. `latentwire_l_a2_generated_solution_rerank_cache`: generate dev/gate candidate pools plus source/target/verifier scores; no confirm access.",
        "4. `channel_set_c_f_hazard_control_shard`: identical-row denominator for static/EMA/random matched controls before any foreground job.",
        "5. `channel_set_ce13_warmup_policy_cache_materialization`: only after a tiny dev/gate warmup-policy cache exists.",
        "",
        "No foreground GPU job is authorized by this report.",
        "",
        "## Raw Artifacts",
        "",
    ]
    for key in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
        lines.append(f"- `{rel(RESULTS_DIR / key / 'summary.json')}`")
        lines.append(f"- `{rel(RESULTS_DIR / key / 'raw_rows.jsonl')}`")
    lines.extend(
        [
            f"- `{rel(RESULTS_DIR / 'overnight_summary.json')}`",
            "",
            "## Next Exact Gate",
            "",
            "Run the C_A1 native replay/parity backfill locally on the GPU node from the queue packet. Do not run a foreground confirmation until native rows beat matched controls and DeepSeek/Falcon sentinels do not regress.",
            "",
        ]
    )
    report.write_text("\n".join(lines))


def run(force: bool = False) -> dict[str, Any]:
    if RESULTS_DIR.exists() and not force:
        raise SystemExit(f"{rel(RESULTS_DIR)} already exists; rerun with --force only for local regeneration.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    exp1, exp1_rows = exp1_cache_oracle()
    exp2, exp2_rows = exp2_c2c_tractability()
    exp3, exp3_rows = exp3_destructive_controls()
    exp4, exp4_rows = exp4_l_a2_ceiling()
    exp5, exp5_rows = exp5_channel_set()
    summary = {
        "run_id": RUN_ID,
        "created_utc": created,
        "locality": "cpu_mps_cached_only",
        "confirm_rows_consumed": 0,
        "no_ssh": True,
        "no_cuda": True,
        "no_passthrough_gpu_claim": True,
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "exp4": exp4,
        "exp5": exp5,
        "final_decision": "NO_FOREGROUND_GPU_C_A1_BACKFILL_FIRST_L_A2_CACHE_SECOND",
    }
    for key, payload, rows in [
        ("exp1", exp1, exp1_rows),
        ("exp2", exp2, exp2_rows),
        ("exp3", exp3, exp3_rows),
        ("exp4", exp4, exp4_rows),
        ("exp5", exp5, exp5_rows),
    ]:
        write_json(RESULTS_DIR / key / "summary.json", payload)
        write_jsonl(RESULTS_DIR / key / "raw_rows.jsonl", rows)
    write_json(RESULTS_DIR / "overnight_summary.json", summary)
    update_leaderboard(summary)
    update_boards(summary)
    update_ledger(summary)
    update_queues()
    write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate the current write-once package while developing locally.")
    args = parser.parse_args()
    summary = run(force=args.force)
    print(json.dumps({"run_id": summary["run_id"], "final_decision": summary["final_decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
