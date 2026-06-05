#!/usr/bin/env python3
"""Drain MPS-first queue items into explicit result or parked states.

This script intentionally does not launch CUDA, foreground GPU, confirmation
access, or large training. It materializes the current evidence state for one
queue item at a time, updates dashboards, and refreshes review_packet.zip.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "queues/mps_first.yaml"
RESULT_ROOT = ROOT / "results/mps_first"
ITERATION_LOG = ROOT / "dashboard/iteration_log.md"
LEADERBOARD = ROOT / "dashboard/leaderboard.csv"
BREAKTHROUGH_BOARD = ROOT / "dashboard/breakthrough_board.md"
KILL_BOARD = ROOT / "dashboard/kill_board.md"
REVIEW_ZIP = ROOT / "review_packet.zip"
MAX_PACKET_FILE = 5 * 1024 * 1024
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_queue() -> list[dict]:
    data = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))
    return sorted(data["mps_first"], key=lambda row: row["priority"])


def base_summary(item: dict, status: str, verdict: str, achieved_n: int, wall: float) -> dict:
    return {
        "achieved_n": achieved_n,
        "confirm_rows_scored": 0,
        "created_utc": now_utc(),
        "id": item["id"],
        "paper": item["paper"],
        "priority": item["priority"],
        "promotion_allowed": False,
        "status": status,
        "verdict": verdict,
        "wall_clock_seconds": wall,
    }


def l_pc1(item: dict, start: float) -> tuple[dict, list[dict]]:
    path = ROOT / "results/source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate_20260503_validation1024_2048/hellaswag_qwen_hybrid_to_phi_cross_family_gate.json"
    rows = []
    if path.exists():
        data = read_json(path)
        h = data["headline"]
        rows.append({
            "artifact": str(path.relative_to(ROOT)),
            "artifact_heldout_eval_rows": h.get("heldout_eval_rows"),
            "candidate_only_accuracy": h.get("candidate_only_accuracy"),
            "hybrid_accuracy": h.get("hybrid_accuracy"),
            "hybrid_delta_vs_candidate_only": h.get("hybrid_delta_vs_candidate_only"),
            "hybrid_ci95_low_vs_candidate_only": h.get("hybrid_ci95_low_vs_candidate_only"),
            "hybrid_ci95_high_vs_candidate_only": h.get("hybrid_ci95_high_vs_candidate_only"),
            "mandatory_gate_evaluable": False,
            "reason": "existing cross-family artifact is a receiver-visible one-candidate hint, not a source-private receiver-conditioned ceiling probe",
        })
    summary = base_summary(
        item,
        "PARKED_NEEDS_RECEIVER_CONDITIONED_CEILING_RUNNER",
        "PARKED",
        rows[0]["artifact_heldout_eval_rows"] if rows else 0,
        time.time() - start,
    )
    summary.update({
        "blocker": "No eligible existing artifact measures I(source_signal;Y|receiver_state) for a complementary source-private signal.",
        "existing_artifact_checked": str(path.relative_to(ROOT)) if path.exists() else None,
        "mandatory_gate": item["mandatory_gate"],
        "exact_command": "venv_arm64/bin/python scripts/run_mps_first_iteration.py --probe L_PC1_cross_family_specialist_ceiling after authoring a reviewed CPU/MPS scorer that uses Qwen2.5-Math-1.5B or DeepSeek-R1-Distill-Qwen-1.5B as source and Qwen3-0.6B as receiver on frozen dev/gate rows, writes source/receiver label scores, computes conditional MI, and enforces --no-confirm.",
    })
    return summary, rows or [{"reason": summary["blocker"], "mandatory_gate": item["mandatory_gate"]}]


def l_pc2(item: dict, start: float) -> tuple[dict, list[dict]]:
    candidates = list((ROOT / "results").glob("*tool*"))
    rows = [{
        "local_tool_artifact_candidates": [str(p.relative_to(ROOT)) for p in candidates[:20]],
        "eligible_tool_private_score_cache_found": False,
        "reason": "No dev/gate cache found where the source privately ran calculator/code-exec and receiver did not see the tool result.",
    }]
    summary = base_summary(item, "PARKED_NEEDS_TOOL_PRIVATE_CACHE_AND_RUNNER", "PARKED", 0, time.time() - start)
    summary.update({
        "blocker": rows[0]["reason"],
        "mandatory_gate": item["mandatory_gate"],
        "exact_command": "venv_arm64/bin/python scripts/build_tool_private_source_ceiling.py --dataset data/gsm8k_100.jsonl --source-tool calculator --source-model Qwen/Qwen2.5-Math-1.5B-Instruct --receiver-model Qwen/Qwen3-0.6B --split dev,gate --min-gate-n 400 --mde-target 0.05 --no-confirm",
    })
    return summary, rows


def l_pc5(item: dict, start: float) -> tuple[dict, list[dict]]:
    summary_path = ROOT / "results/overnight_v3/20260604_corrected_reprobe/exp4_corrected_l_a2_rerank_ceiling/summary.json"
    candidate_path = ROOT / "results/overnight_v3/20260604_corrected_reprobe/exp4_corrected_l_a2_rerank_ceiling/scored_candidates_with_verifier.jsonl"
    data = read_json(summary_path)
    rows_by_prompt: dict[str, dict] = {}
    correct_candidates = 0
    for row in read_jsonl(candidate_path):
        rid = str(row.get("row_id"))
        rec = rows_by_prompt.setdefault(rid, {"row_id": rid, "candidate_count": 0, "correct_count": 0, "split": row.get("split")})
        rec["candidate_count"] += 1
        if row.get("correct"):
            rec["correct_count"] += 1
            correct_candidates += 1
    raw_rows = list(rows_by_prompt.values())
    achieved = int(data.get("prompts_with_at_least_one_correct_candidate", 0))
    required = int(data.get("required_correct_candidate_prompts", 80))
    summary = base_summary(item, "PARKED_POOL_TOO_WEAK", "PARKED", achieved, time.time() - start)
    summary.update({
        "achieved_prompts": data.get("achieved_prompts"),
        "achieved_candidates": data.get("achieved_candidates"),
        "correct_candidate_count": correct_candidates,
        "prompts_with_at_least_one_correct_candidate": achieved,
        "required_correct_candidate_prompts": required,
        "verifier_score_present": data.get("verifier_score_present"),
        "mde_target": 0.05,
        "mde_half_width": None,
        "reason": "candidate pool is nondegenerate for only 36 prompts, below the 80-prompt rerank gate; no gain verdict emitted",
        "source_summary": str(summary_path.relative_to(ROOT)),
        "exact_command": data.get("gpu_command"),
    })
    return summary, raw_rows


def l_c2(item: dict, start: float) -> tuple[dict, list[dict]]:
    fuser_paths = sorted((ROOT / ".hf_home").glob("hub/models--nics-efc--C2C_Fuser/snapshots/*/*/config.json"))
    anchor = ROOT / "registry/L_C2_c2c_kv_lcf_anchor.yaml"
    rows = [{
        "local_c2c_fuser_configs": [str(p.relative_to(ROOT)) for p in fuser_paths],
        "anchor_registry": str(anchor.relative_to(ROOT)),
        "eligible_control_trained_lcf_lite_found": False,
        "reason": "C2C fusers are local, but no reviewed byte-limited control-trained LCF-lite runner with wrong-row/zero-source objective exists.",
    }]
    summary = base_summary(item, "PARKED_NEEDS_CONTROL_TRAINED_FUSER_PRELAUNCH", "PARKED", 0, time.time() - start)
    summary.update({
        "blocker": rows[0]["reason"],
        "mandatory_gate": item["mandatory_gate"],
        "exact_command": "venv_arm64/bin/python scripts/build_control_trained_lcf_lite_proxy.py --source-cache <dev_gate_source_features> --receiver-cache <dev_gate_receiver_features> --wrong-row --zero-source --label-shuffle --min-gate-n 400 --mde-target 0.05 --no-confirm",
    })
    return summary, rows


def c_u1(item: dict, start: float) -> tuple[dict, list[dict]]:
    path = ROOT / "results/stage1/channel_set_raw_rows.jsonl"
    rows = read_jsonl(path)
    usable = []
    for row in rows:
        usable.append({
            "row_id": row.get("row_id"),
            "split": row.get("split"),
            "source_path": row.get("path"),
            "static_gap": row.get("static_gap"),
            "recoveries": row.get("recoveries"),
            "has_drift_trajectory_features": False,
            "reason": "row has recovery/static_gap but no drift trajectory or baseline confidence feature needed for C_U1",
        })
    summary = base_summary(item, "PARKED_NEEDS_DRIFT_FEATURE_CACHE", "PARKED", len(usable), time.time() - start)
    summary.update({
        "source_rows": str(path.relative_to(ROOT)),
        "mandatory_gate": item["mandatory_gate"],
        "blocker": "Stage-1 rows contain recovery/static_gap but not drift trajectory features paired with difficulty/uplift labels across >=2 models.",
        "models_observed_from_paths": sorted({p for p in ("deepseek" if "deepseek" in str(r.get("path")) else "falcon" if "falcon" in str(r.get("path")) else "granite" if "granite" in str(r.get("path")) else "unknown" for r in rows)}),
        "exact_command": "venv_arm64/bin/python scripts/build_channel_set_drift_as_signal_screen.py --input experimental/outlier_migrate/phase9/results --rows results/stage1/channel_set_raw_rows.jsonl --features drift_trajectory,kl_growth,warmup_activation --labels recoverable_gap,policy_uplift --min-gate-n 200 --models granite,deepseek,falcon --no-confirm",
    })
    return summary, usable[:500]


def c_w1(item: dict, start: float) -> tuple[dict, list[dict]]:
    rows = [{
        "dashboard": "dashboard/channel_set_backfill_readiness.md",
        "eligible_warmup_policy_cache_found": False,
        "reason": "Existing dashboard states no parseable warmup-policy cache exists.",
    }]
    summary = base_summary(item, "PARKED_NEEDS_WARMUP_POLICY_CACHE", "PARKED", 0, time.time() - start)
    summary.update({
        "blocker": rows[0]["reason"],
        "mandatory_gate": item["mandatory_gate"],
        "exact_command": "venv_arm64/bin/python scripts/materialize_channel_set_warmup_policy_cache.py --input experimental/outlier_migrate/phase9/results --split dev,gate --locked-row-ids splits/stage0_cache_rows_hash.jsonl --policies paroquant,c_a1,survival,reject --no-confirm",
    })
    return summary, rows


def c_s1(item: dict, start: float) -> tuple[dict, list[dict]]:
    stage_rows = read_jsonl(ROOT / "results/stage1/channel_set_raw_rows.jsonl")
    cf_rows = [r for r in stage_rows if "m10" in str(r.get("path", "")).lower() or "mpred" in str(r.get("path", "")).lower() or "m2_" in str(r.get("path", "")).lower()]
    rows = [{
        "inspected_stage1_rows": len(stage_rows),
        "candidate_survival_related_rows": len(cf_rows),
        "eligible_identical_row_denominator_found": False,
        "reason": "Available C-F/survival-like evidence is contaminated or lacks a fresh identical-row random/static denominator.",
    }]
    summary = base_summary(item, "PARKED_NEEDS_IDENTICAL_ROW_DENOMINATOR", "PARKED", 0, time.time() - start)
    summary.update({
        "blocker": rows[0]["reason"],
        "mandatory_gate": item["mandatory_gate"],
        "exact_command": "venv_arm64/bin/python scripts/build_channel_set_clean_survival_denominator.py --input experimental/outlier_migrate/phase9/results --same-row-denominators random_core,static_core,wrong_row_core --min-gate-n 200 --no-gap-filter --no-confirm",
    })
    return summary, rows


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
        checks.append({"check": name, "path": str(path.relative_to(ROOT)), "file_present": path.exists()})
    runbook = (ROOT / "dashboard/c_a1_gpu_backfill_runbook.md").read_text(encoding="utf-8")
    checks.extend([
        {"check": "runbook_mentions_granite", "file_present": "Granite" in runbook},
        {"check": "runbook_mentions_deepseek", "file_present": "DeepSeek" in runbook},
        {"check": "runbook_mentions_falcon", "file_present": "Falcon" in runbook},
        {"check": "native_pairing_results_present", "file_present": False, "reason": "C_A1 is still NEXT_GPU_BACKFILL; no native paired replay result exists yet."},
    ])
    summary = base_summary(item, "PARKED_NEEDS_NATIVE_PAIRING", "PARKED", len(checks), time.time() - start)
    summary.update({
        "audit_checks": checks,
        "blocker": "Defense bundle inputs exist, but claim-bearing C_A1 native paired ParoQuant-vs-tight-clip rows are still missing.",
        "mandatory_gate": item["mandatory_gate"],
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
    out.mkdir(parents=True, exist_ok=True)
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
        "MPS-first queue execution log. Statuses are screening/parked states only unless a summary explicitly records a powered verdict.",
        "",
        "| priority | probe | paper | status | achieved_n | MDE | verdict | wall_clock_seconds | next command |",
        "| ---: | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    done = {s["id"] for s in summaries}
    for s in summaries:
        mde = s.get("mde_half_width")
        cmd = str(s.get("exact_command", "")).replace("|", "\\|")
        lines.append(
            f"| {s.get('priority')} | `{s.get('id')}` | {s.get('paper')} | `{s.get('status')}` | "
            f"{s.get('achieved_n', 0)} | {'' if mde is None else mde} | `{s.get('verdict')}` | "
            f"{s.get('wall_clock_seconds', 0):.3f} | `{cmd}` |"
        )
    queue = load_queue()
    for item in queue:
        if item["id"] not in done:
            lines.append(
                f"| {item['priority']} | `{item['id']}` | {item['paper']} | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `{item['command']}` |"
            )
    ITERATION_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_leaderboard(summaries: list[dict]) -> None:
    if not LEADERBOARD.exists():
        return
    with LEADERBOARD.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader if not str(r.get("method_id", "")).startswith("MPS_FIRST_")]
    for s in summaries:
        row = {k: "" for k in fieldnames}
        row.update({
            "method_id": f"MPS_FIRST_{s['id']}",
            "paper": s.get("paper", ""),
            "split": "dev_gate",
            "status": s.get("status", ""),
            "source_path": f"results/mps_first/{s['id']}/summary.json",
            "n": str(s.get("achieved_n", 0)),
            "note": "mps_first iteration state; no Mac PASSED status",
        })
        rows.append(row)
    with LEADERBOARD.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replace_section(text: str, heading: str, body: str) -> str:
    pattern = rf"\n## {re.escape(heading)}\n.*?(?=\n## |\Z)"
    section = f"\n## {heading}\n\n{body.rstrip()}\n"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, section, text, flags=re.S)
    return text.rstrip() + section


def update_boards(summaries: list[dict]) -> None:
    body_lines = [
        "| priority | probe | status | achieved_n | interpretation |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for s in summaries:
        body_lines.append(
            f"| {s.get('priority')} | `{s.get('id')}` | `{s.get('status')}` | {s.get('achieved_n', 0)} | {s.get('blocker') or s.get('reason') or 'see summary'} |"
        )
    if BREAKTHROUGH_BOARD.exists():
        txt = BREAKTHROUGH_BOARD.read_text(encoding="utf-8")
        BREAKTHROUGH_BOARD.write_text(replace_section(txt, "MPS-First Iteration", "\n".join(body_lines)), encoding="utf-8")
    if KILL_BOARD.exists():
        txt = KILL_BOARD.read_text(encoding="utf-8")
        KILL_BOARD.write_text(replace_section(txt, "MPS-First Parked/Negative State", "\n".join(body_lines)), encoding="utf-8")


def update_dashboards() -> None:
    summaries = all_summaries()
    update_iteration_log(summaries)
    update_leaderboard(summaries)
    update_boards(summaries)


def confirm_path(path: str) -> bool:
    return fnmatch.fnmatch(path, "*_confirm*") or "_confirm" in path


def hard_excluded(path: Path) -> bool:
    rel = path.as_posix()
    return path.suffix in HARD_EXTS or "caches/" in rel or rel.startswith("caches/")


def packet_candidates() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "results/mps_first").glob("**/summary.json")))
    paths.extend(sorted((ROOT / "results/mps_first").glob("**/raw_rows.jsonl")))
    for p in [
        "dashboard/scoop_report.md",
        "dashboard/morning_brief.md",
        "dashboard/iteration_log.md",
        "dashboard/breakthrough_board.md",
        "dashboard/kill_board.md",
        "dashboard/leaderboard.csv",
        "ideas/NEW_200_triage.md",
        "scripts/run_mps_first_iteration.py",
        "scripts/overnight_v3_corrected_probes.py",
        "lessons/LESSONS_LEDGER.md",
    ]:
        path = ROOT / p
        if path.exists():
            paths.append(path)
    paths.extend(sorted((ROOT / "queues").glob("*.yaml")))
    paths.extend(sorted((ROOT / "registry").glob("**/*.yaml")))
    return paths


def add_packet_file(entries: dict[str, bytes], skipped: list[str], path: Path) -> None:
    rel = path.relative_to(ROOT).as_posix()
    if confirm_path(rel) or hard_excluded(Path(rel)):
        return
    size = path.stat().st_size
    if size > MAX_PACKET_FILE:
        if path.suffix in {".jsonl", ".csv", ".md"}:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            head = "\n".join(lines[:200]) + "\n"
            entries[rel] = head.encode("utf-8")
            skipped.append(f"{rel}: shipped 200-row/line head instead of {size} byte full file")
        else:
            skipped.append(f"{rel}: skipped {size} byte file")
        return
    entries[rel] = path.read_bytes()


def refresh_review_packet() -> None:
    entries: dict[str, bytes] = {}
    skipped: list[str] = []
    for path in packet_candidates():
        if path.exists() and path.is_file():
            add_packet_file(entries, skipped, path)
    index_lines = [
        "# review_packet",
        "",
        f"Created UTC: {now_utc()}",
        "",
        "## Included",
    ]
    for name in sorted(entries):
        index_lines.append(f"- `{name}` ({len(entries[name])} bytes)")
    index_lines.extend(["", "## Skipped / Truncated"])
    if skipped:
        index_lines.extend(f"- {item}" for item in skipped)
    else:
        index_lines.append("- None")
    index_lines.extend([
        "",
        "## Guardrails",
        "- No paths matching `*_confirm*` are included.",
        "- No model weights, binary arrays, or caches are included.",
        "- Files over 5 MB are skipped or represented by a 200-line head.",
    ])
    entries["INDEX.md"] = ("\n".join(index_lines) + "\n").encode("utf-8")
    bad = [name for name in entries if confirm_path(name)]
    if bad:
        raise SystemExit(f"review packet would include confirm path(s): {bad}")
    with ZipFile(REVIEW_ZIP, "w", ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    if REVIEW_ZIP.stat().st_size > 25 * 1024 * 1024:
        raise SystemExit(f"review_packet.zip too large: {REVIEW_ZIP.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="Probe id from queues/mps_first.yaml, or ALL.")
    args = parser.parse_args()
    if args.probe == "ALL":
        for item in load_queue():
            run_probe(item["id"])
    else:
        summary = run_probe(args.probe)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
