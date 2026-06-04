#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmc.stage1_screens import read_jsonl_rows  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "results" / "stage1" / "leaderboard.csv"
PLANTED = ROOT / "results" / "stage1" / "planted_sentinel_checks.json"


def fnum(value: str | int | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inum(value: str | int | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def method_rows(rows: list[dict[str, str]], method_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row["method_id"] == method_id]


def split_n(rows: list[dict[str, str]], split: str) -> int:
    total = 0
    for row in rows:
        if row["split"] != split:
            continue
        total += inum(row.get("paired_n")) or inum(row.get("n"))
    return total


def status_counts(rows: list[dict[str, str]]) -> str:
    counts = Counter(row["status"] for row in rows)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def path_summary(rows: list[dict[str, str]], limit: int = 6) -> str:
    paths = sorted({row["source_path"] for row in rows if row.get("source_path")})
    shown = ", ".join(f"`{path}`" for path in paths[:limit])
    if len(paths) > limit:
        shown += f", ... +{len(paths) - limit} more"
    return shown or "none"


def delta_summary(rows: list[dict[str, str]]) -> str:
    gate = [row for row in rows if row["split"] == "gate"]
    deltas = [fnum(row.get("delta_vs_best_baseline")) for row in gate]
    deltas = [value for value in deltas if value is not None]
    lows = [fnum(row.get("ci95_low_vs_best_baseline")) for row in gate]
    lows = [value for value in lows if value is not None]
    highs = [fnum(row.get("ci95_high_vs_best_baseline")) for row in gate]
    highs = [value for value in highs if value is not None]
    if not deltas:
        return "not applicable"
    return (
        f"gate delta min={min(deltas):.4f}, max={max(deltas):.4f}; "
        f"CI-low max={max(lows):.4f}; CI-high max={max(highs):.4f}"
    )


def strongest_baseline(rows: list[dict[str, str]]) -> str:
    best: tuple[float, str] | None = None
    for row in rows:
        baseline = row.get("best_baseline_condition")
        acc = fnum(row.get("best_baseline_accuracy"))
        if not baseline or acc is None:
            continue
        if best is None or acc > best[0]:
            best = (acc, baseline)
    if best is None:
        return "not available"
    return f"{best[1]} (accuracy={best[0]:.4f})"


def fixed_packet_condition_accuracy(raw_path: Path) -> dict[str, float]:
    counts: dict[str, list[int]] = defaultdict(list)
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "gate" or "fixed_packet_gate" not in row["path"]:
                continue
            counts[str(row["condition"])].append(1 if row["correct"] else 0)
    return {key: sum(values) / len(values) for key, values in sorted(counts.items()) if values}


def l_b1_cache_metrics(leaderboard: list[dict[str, str]], split: str) -> dict[str, float | int | None]:
    paths = sorted(
        {
            Path(row["source_path"])
            for row in leaderboard
            if row["method_id"] == "latentwire_cached_fixed_packet_baseline"
        }
    )
    agg = Counter()
    for path in paths:
        rows = [row for row in read_jsonl_rows(path) if row.split == split]
        by_row: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            by_row[row.row_id][str(row.payload.get("condition"))] = row.payload
        for conditions in by_row.values():
            target = conditions.get("target_only")
            matched = conditions.get("matched_source_private_packet")
            if not target or not matched:
                continue
            source_index = matched.get("metadata", {}).get("source_selected_index")
            answer_index = matched.get("answer_index")
            if source_index is None or answer_index is None:
                continue
            source_correct = source_index == answer_index
            target_correct = bool(target.get("correct"))
            matched_correct = bool(matched.get("correct"))
            matched_source = matched.get("prediction_index") == source_index
            agg["n"] += 1
            agg["source_correct"] += int(source_correct)
            agg["target_correct"] += int(target_correct)
            agg["matched_correct"] += int(matched_correct)
            agg["matched_equals_source"] += int(matched_source)
            if target_correct and not source_correct:
                agg["target_correct_source_wrong"] += 1
                agg["damage"] += int(not matched_correct)
            if (not target_correct) and source_correct:
                agg["target_wrong_source_correct"] += 1
                agg["repair"] += int(matched_correct)

    n = agg["n"]
    damage_den = agg["target_correct_source_wrong"]
    repair_den = agg["target_wrong_source_correct"]
    return {
        "n": n,
        "source_accuracy": agg["source_correct"] / n if n else None,
        "target_accuracy": agg["target_correct"] / n if n else None,
        "matched_accuracy": agg["matched_correct"] / n if n else None,
        "matched_equals_source_rate": agg["matched_equals_source"] / n if n else None,
        "damage_n": agg["damage"],
        "damage_den": damage_den,
        "damage_rate": agg["damage"] / damage_den if damage_den else None,
        "repair_n": agg["repair"],
        "repair_den": repair_den,
        "repair_rate": agg["repair"] / repair_den if repair_den else None,
    }


def metric_line(metrics: dict[str, float | int | None]) -> str:
    return (
        f"n={metrics['n']}; target_acc={metrics['target_accuracy']:.4f}; "
        f"source_acc={metrics['source_accuracy']:.4f}; matched_acc={metrics['matched_accuracy']:.4f}; "
        f"matched_equals_source={metrics['matched_equals_source_rate']:.4f}; "
        f"damage={metrics['damage_n']}/{metrics['damage_den']} ({metrics['damage_rate']:.4f}); "
        f"repair={metrics['repair_n']}/{metrics['repair_den']} ({metrics['repair_rate']:.4f})"
    )


def channel_gate_signal(rows: list[dict[str, str]]) -> str:
    gate = [row for row in rows if row["split"] == "gate"]
    values = [fnum(row.get("median_recovery")) for row in gate]
    values = [value for value in values if value is not None]
    if not values:
        return "no numeric gate recovery"
    return (
        f"gate rows={len(gate)}, positive_median={sum(value > 0 for value in values)}, "
        f"nonpositive_median={sum(value <= 0 for value in values)}, "
        f"min_median={min(values):.4f}, max_median={max(values):.4f}"
    )


def ambiguous_clusters(rows: list[dict[str, str]]) -> Counter[str]:
    clusters: Counter[str] = Counter()
    for row in rows:
        if row["status"] != "AMBIGUOUS":
            continue
        if row["split"] == "dev":
            clusters["missing-metric/dev-only-no-gate-decision"] += 1
        elif row["paper"] == "channel_set" and inum(row.get("n")) <= 1:
            clusters["underpower/gate-n-le-1"] += 1
        elif row["paper"] == "channel_set":
            clusters["missing-baseline-or-gpu-forward"] += 1
        else:
            clusters["true-near-tie"] += 1
    return clusters


def method_section(
    title: str,
    status: str,
    rows: list[dict[str, str]],
    strongest: str,
    delta: str,
    controls: str,
    leakage: str,
    next_falsifier: str,
    eligibility: str,
    extra: list[str] | None = None,
) -> str:
    lines = [
        f"## {title}",
        f"- status: `{status}`",
        f"- result paths: {path_summary(rows)}",
        f"- n_dev: `{split_n(rows, 'dev')}`",
        f"- n_gate: `{split_n(rows, 'gate')}`",
        "- n_confirm: `0`",
        f"- strongest baseline: {strongest}",
        f"- method delta vs baseline: {delta}",
        f"- control outcomes: {controls}",
        f"- leakage outcomes: {leakage}",
        f"- next smallest falsifier: {next_falsifier}",
        f"- eligibility: `{eligibility}`",
    ]
    if extra:
        lines.extend(extra)
    return "\n".join(lines) + "\n"


def build_stage1_triage(rows: list[dict[str, str]]) -> str:
    fixed_rows = method_rows(rows, "latentwire_cached_fixed_packet_baseline")
    wz_rows = method_rows(rows, "L_SCORECOMP_wz_bins_deployable")
    c_a1 = method_rows(rows, "C_A1_cvar_evt_clip_grid")
    c_f = method_rows(rows, "C_F_survival_stable_core")
    ce21 = method_rows(rows, "CE21_no_gap_filter")
    c_d1 = method_rows(rows, "C_D1_osc_static_cluster_stress")
    ce13 = method_rows(rows, "CE13_warmup_policy_selector")
    c_a2 = method_rows(rows, "C_A2_horizon_rotation")
    l_b1_dev = l_b1_cache_metrics(rows, "dev")
    l_b1_gate = l_b1_cache_metrics(rows, "gate")
    cond_acc = fixed_packet_condition_accuracy(ROOT / "results" / "stage1" / "latentwire_raw_rows.jsonl")
    clusters = ambiguous_clusters(rows)
    planted = json.loads(PLANTED.read_text(encoding="utf-8"))

    parts = [
        "# Stage-1 Triage",
        "",
        "Triage-only pass from commit `1231a08a`; no new experiment data was generated. "
        f"Confirm rows consumed: `{planted.get('stage1_confirm_rows_consumed', 0)}`. "
        f"Parsed confirm rows counted but excluded: `{planted['parsed_confirm_rows_counted_not_consumed']}`.",
        "",
        "## Non-Killed Row Buckets",
        f"- status counts: `{dict(Counter(row['status'] for row in rows))}`",
        "- AMBIGUOUS clusters: "
        + ", ".join(f"{key}={value}" for key, value in sorted(clusters.items())),
        "- PROVISIONAL_PROMOTE_TO_GPU rows: `0`",
        "",
        method_section(
            "L_SCORECOMP_deployable_WZ",
            "KILLED",
            wz_rows,
            strongest_baseline(wz_rows),
            delta_summary(wz_rows),
            "wrong-row/derangement controls are not present in the parsed WZ rows; equal-byte score/label baselines dominate all killed gate rows",
            "row confirm guard passed; deployability remains constrained because these cached rows are old WZ packet artifacts, not fresh high-entropy source-minus-target WZ",
            "fresh high-entropy small-model WZ on Mac, with delta_beyond_score positive vs all equal-byte baselines before any confirm access",
            "kill",
        ),
        method_section(
            "L_SCORECOMP_equal_byte_baselines",
            "CPU_SCREENED",
            fixed_rows,
            strongest_baseline(fixed_rows),
            delta_summary(fixed_rows),
            "baseline-only rows; fixed packet sometimes beats target/shuffle, but it is source-copy context, not a candidate-method promotion",
            "row confirm guard passed; baseline rows are prior test/validation artifacts and cannot confirm a paper claim",
            "reuse as locked baselines in fresh high-entropy LatentWire screens",
            "Mac_continue",
            [
                "- gate condition accuracy: "
                + ", ".join(f"{key}={value:.4f}" for key, value in cond_acc.items())
            ],
        ),
        method_section(
            "L_B1_damage_avoidance",
            "KILLED",
            fixed_rows,
            "source-index/source-selected metadata",
            "cached fixed packet has repair but no damage avoidance on gate",
            "matched packet equals source-selected answer on nearly all gate rows; this is source-copy behavior",
            "AURC, risk@coverage, confidence leakage, packet-source-top1 MI, wrong-row, and derangement are missing-row-fields in current cache",
            "fresh Mac small-model L-B1 with explicit confidence/risk fields and wrong-row/derangement controls",
            "Mac_continue",
            [
                f"- dev cached L-B1 proxy: {metric_line(l_b1_dev)}",
                f"- gate cached L-B1 proxy: {metric_line(l_b1_gate)}",
            ],
        ),
        method_section(
            "L_A1_MMLU_ceiling",
            "AMBIGUOUS",
            [],
            "not available",
            "no cached MMLU-Pro ceiling rows in this Stage-1 surface",
            "not run",
            "no confirm leakage; no rows consumed",
            "Mac MMLU-Pro info-ceiling probe with target-only, source-index, and equal-byte baselines",
            "Mac_continue",
        ),
        method_section(
            "L_A2_rerank",
            "AMBIGUOUS",
            [],
            "not available",
            "no generated-solution candidate pools found in bounded cache pass",
            "not run",
            "no confirm leakage; no rows consumed",
            "materialize a tiny dev/gate candidate-pool shard on Mac before any confirmation plan",
            "Mac_continue",
        ),
        method_section(
            "C_A1",
            "CPU_SCREENED",
            c_a1,
            "local ParoQuant cached parity / static recovery denominator",
            channel_gate_signal(c_a1),
            "offline per-trace recovery only; no native W4A16 forward confirmation",
            "confirm rows excluded; native quantized-forward leakage cannot be assessed on Mac",
            "GPU backfill parity card only if the same gate packet is replayed with real quantized forwards",
            "GPU_backfill",
        ),
        method_section(
            "C_F",
            "CPU_SCREENED",
            c_f,
            "static top-K / random-matched controls",
            channel_gate_signal(c_f),
            "mixed gate medians; some random/control regimes are positive, so there is no foreground confirmation target",
            "confirm rows excluded; forward-pass evidence is absent",
            "tighten offline paired baseline and hazard denominator before GPU foreground",
            "GPU_backfill",
        ),
        method_section(
            "CE13",
            "AMBIGUOUS",
            ce13,
            "not available",
            "no parseable warmup-policy cache found",
            "not run",
            "no offline evidence justifying foreground GPU",
            "find or build a tiny dev/gate warmup-policy cache before any GPU forward",
            "Mac_continue",
        ),
        method_section(
            "C_D1",
            "KILLED",
            c_d1,
            "static_top10 / random_reactive_top1",
            channel_gate_signal(c_d1),
            "all parsed gate medians are negative and underpowered",
            "confirm rows excluded; no positive offline headroom",
            "do not allocate GPU unless a new preregistered OSC surface appears",
            "kill",
        ),
        method_section(
            "CE21",
            "CPU_SCREENED",
            ce21,
            "denominator audit only",
            "no method delta; denominator audit rows only",
            "no-gap rows were counted, not promoted",
            "confirm rows excluded",
            "carry as audit support for future Channel-Set screens",
            "Mac_continue",
        ),
        method_section(
            "C_A2_tests",
            "AMBIGUOUS",
            c_a2,
            "not available",
            "tests-only row; no method screen",
            "orthogonality/full-precision/KV-cache-basis tests not proven by current leaderboard",
            "no offline evidence justifying foreground GPU",
            "run local correctness tests or keep parked; no GPU foreground",
            "Mac_continue",
        ),
    ]
    return "\n".join(parts)


def build_gpu_plan(rows: list[dict[str, str]]) -> str:
    return """# GPU Handoff Plan

No foreground GPU job is authorized from this triage.

- PROVISIONAL_PROMOTE_TO_GPU rows: `0`
- Foreground confirmation queue: empty
- Channel-Set state: C_A1 has limited positive offline gate medians, C_F is mixed and control-contaminated, C_D1 is negative, CE13 has no parseable cache, and C_A2 is tests-only.
- LatentWire state: keep on Mac. WZ/source-copy cached family is killed on this surface; L-B1 needs fresh risk/confidence rows before it is a live lead.

## Allowed GPU Backfill Only

Backfill may prepare Channel-Set parity/cache evidence, not confirmation claims:
- ParoQuant parity replay for the exact C_A1 cached gate packets.
- OSC/DecDEC cache shard only to replace the current underpowered negative row.
- Hazard/static-control shard for C_F only if paired denominator and random-control collapse are defined first.

## Hard Stop

Do not run foreground GPU confirmation until a future dashboard contains at least one `PROVISIONAL_PROMOTE_TO_GPU` row or a Channel-Set row with positive offline gate headroom, matched-budget baselines, and control collapse.
"""


def write_queues() -> None:
    write_text(
        ROOT / "queues" / "gpu_foreground.yaml",
        """foreground: []
notes:
  - "No PROVISIONAL_PROMOTE_TO_GPU row exists after Stage-1 triage."
  - "Do not start a foreground GPU confirmation run from current evidence."
""",
    )
    write_text(
        ROOT / "queues" / "gpu_backfill.yaml",
        """backfill:
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
  - id: channel_set_c_f_hazard_control_shard
    reason: mixed_gate_signal_and_random_control_contamination
    command: "backfill paired static/random controls before any native forward confirmation"
    required_cache_or_model: "matched-budget static top-K and random-control trace cache"
    est_gpu_hours: 2-6
    promotion_allowed: false
""",
    )
    write_text(
        ROOT / "queues" / "mac_continue.yaml",
        """mac_continue:
  - id: latentwire_mmlu_pro_info_ceiling
    reason: high_entropy_wz_ceiling_is_mac_doable_and_not_in_cache
    command: "run a tiny dev/gate MMLU-Pro info-ceiling probe with sub-1B models; no confirm rows"
    required_cache_or_model: "small Qwen-class CPU/MPS model and fresh dev/gate rows"
  - id: latentwire_fresh_wz_high_entropy
    reason: cached_4way_wz_family_is_killed_but_high_entropy_surface_not_tested
    command: "run deployable source-score-only WZ on fresh high-entropy dev/gate rows"
    required_cache_or_model: "source scores only at encoder; target scores as decoder side information"
  - id: latentwire_l_b1_damage_avoidance
    reason: cached_fixed_packet_is_source_copy_and_lacks_risk_fields
    command: "generate dev/gate rows with confidence, coverage, wrong-row, derangement, and leakage controls"
    required_cache_or_model: "small-model candidate/risk rows with source-index+confidence baseline"
  - id: latentwire_l_a2_rerank_pool
    reason: candidate_pools_missing_from_stage1_cache
    command: "materialize a tiny generated-solution candidate pool on Mac before rerank screening"
    required_cache_or_model: "small-model generated-solution candidate pools"
""",
    )


def update_lessons() -> None:
    path = ROOT / "lessons" / "LESSONS_LEDGER.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Lessons Ledger\n"
    marker = "2026-06-03 - Stage-1 cached WZ/source-copy family"
    entry = """## 2026-06-03 - Stage-1 cached WZ/source-copy family

- Hypothesis update: cached `L_SCORECOMP_wz_bins_deployable` packet artifacts are killed on the row-safe dev/gate screen; 12 gate rows have negative CI-high versus equal-byte baselines and 0 rows promote.
- L-B1 cache update: fixed-packet matched rows behave as source copy on the admitted Stage-1 cache. Gate matched-equals-source is `0.9969`, damage on target-correct/source-wrong rows is `192/192`, and repair is source-driven (`380/382` on target-wrong/source-correct rows).
- Ruled out: treating old 4-way source-copy packet wins as a deployable positive method or GPU foreground trigger.
- Still alive: fresh high-entropy LatentWire WZ/L-B1 on Mac with explicit risk/confidence/leakage controls; Channel-Set C_A1/C_F only as GPU backfill candidates until offline controls are stronger.
"""
    if marker not in existing:
        write_text(path, existing.rstrip() + "\n\n" + entry)


def main() -> int:
    os.chdir(ROOT)
    rows = read_csv(LEADERBOARD)
    write_text(ROOT / "dashboard" / "stage1_triage.md", build_stage1_triage(rows))
    write_text(ROOT / "dashboard" / "gpu_handoff_plan.md", build_gpu_plan(rows))
    write_queues()
    update_lessons()
    print("stage1 triage complete: foreground_gpu=0, mac_continue=4, gpu_backfill=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
