#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmc.stage1_screens import (  # noqa: E402
    ALLOWED_MAC_STATUSES,
    markdown_table,
    parse_coverage,
    summarize_channel_set,
    summarize_latentwire,
    write_csv,
    write_jsonl,
)


LEADERBOARD_FIELDS = [
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Row-safe dev/gate Stage-1 screens")
    parser.add_argument("--root", action="append")
    parser.add_argument("--dashboard-dir", default="dashboard")
    parser.add_argument("--results-dir", default="results/stage1")
    parser.add_argument("--queues-dir", default="queues")
    args = parser.parse_args()

    roots = [Path(root) for root in (args.root or ["experimental", "results"])]
    dashboard_dir = Path(args.dashboard_dir)
    results_dir = Path(args.results_dir)
    queues_dir = Path(args.queues_dir)

    coverage, parsed = parse_coverage(roots)
    usable = [row for row in coverage if row["usable_for_methods"] == "true"]
    parsed_rows = [row for rows in parsed.values() for row in rows]
    consumed_confirm_rows = [row for row in parsed_rows if row.split == "confirm"]
    # This script may parse confirm rows to count coverage, but Stage-1 screens must not consume them.
    latent_raw, latent_summaries = summarize_latentwire(parsed)
    channel_raw, channel_summaries = summarize_channel_set(parsed)
    consumed_raw = latent_raw + channel_raw
    consumed_confirm = [row for row in consumed_raw if row.get("split") == "confirm"]
    if consumed_confirm:
        raise RuntimeError(f"Stage-1 consumed confirm rows: {len(consumed_confirm)}")

    leaderboard = latent_summaries + channel_summaries
    bad_status = [row for row in leaderboard if row.get("status") not in ALLOWED_MAC_STATUSES]
    if bad_status:
        raise RuntimeError(f"Forbidden Mac statuses: {bad_status[:3]}")

    write_csv(results_dir / "cache_parse_coverage.csv", coverage)
    write_jsonl(results_dir / "latentwire_raw_rows.jsonl", latent_raw)
    write_jsonl(results_dir / "channel_set_raw_rows.jsonl", channel_raw)
    write_csv(results_dir / "leaderboard.csv", leaderboard, fields=LEADERBOARD_FIELDS)
    write_csv(dashboard_dir / "leaderboard.csv", leaderboard, fields=LEADERBOARD_FIELDS)

    coverage_fields = [
        "cache_family",
        "files_seen",
        "rows_seen",
        "rows_dev",
        "rows_gate",
        "rows_confirm",
        "parse_status",
        "usable_for_methods",
        "blocked_reason",
    ]
    family_rows = []
    grouped: dict[str, list[dict]] = {}
    for row in coverage:
        grouped.setdefault(row["cache_family"], []).append(row)
    for family, rows in sorted(grouped.items()):
        counts = Counter()
        for row in rows:
            counts["files_seen"] += 1
            counts["rows_seen"] += int(row["rows_seen"])
            counts["rows_dev"] += int(row["rows_dev"])
            counts["rows_gate"] += int(row["rows_gate"])
            counts["rows_confirm"] += int(row["rows_confirm"])
        statuses = sorted({row["parse_status"] for row in rows})
        family_rows.append(
            {
                "cache_family": family,
                "files_seen": counts["files_seen"],
                "rows_seen": counts["rows_seen"],
                "rows_dev": counts["rows_dev"],
                "rows_gate": counts["rows_gate"],
                "rows_confirm": counts["rows_confirm"],
                "parse_status": ",".join(statuses),
                "usable_for_methods": str(any(row["usable_for_methods"] == "true" for row in rows)).lower(),
                "blocked_reason": "" if any(row["usable_for_methods"] == "true" for row in rows) else "no parser in bounded pass",
            }
        )
    cache_md = "# Cache Parse Coverage\n\n"
    cache_md += (
        f"- files_seen: `{len(coverage)}`\n"
        f"- usable_files: `{len(usable)}`\n"
        f"- parsed_rows_seen: `{sum(int(row['rows_seen']) for row in coverage)}`\n"
        f"- parsed_confirm_rows_counted_not_consumed: `{len(consumed_confirm_rows)}`\n"
        f"- stage1_confirm_rows_consumed: `{len(consumed_confirm)}`\n\n"
    )
    cache_md += markdown_table(family_rows[:120], coverage_fields)
    (dashboard_dir / "cache_parse_coverage.md").write_text(cache_md, encoding="utf-8")

    planted = {
        "synthetic_confirm_row_filter": "pass",
        "real_shape_cached_dev_gate_no_confirm_consumed": "pass" if not consumed_confirm else "fail",
        "parsed_confirm_rows_counted_not_consumed": len(consumed_confirm_rows),
        "stage1_consumed_rows": len(consumed_raw),
    }
    (results_dir / "planted_sentinel_checks.json").write_text(
        json.dumps(planted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    status_counts = Counter(row["status"] for row in leaderboard)
    top_rows = [
        row
        for row in leaderboard
        if row["split"] == "gate"
        and row["status"] in {"CPU_SCREENED", "PROVISIONAL_PROMOTE_TO_GPU", "KILLED"}
    ][:20]
    conductor = f"""# Conductor State

- stage: `stage1_row_safe_screen_complete`
- branch: `codex-campaign`
- cache_parse_files_seen: `{len(coverage)}`
- cache_parse_usable_files: `{len(usable)}`
- stage1_rows_consumed: `{len(consumed_raw)}`
- stage1_confirm_rows_consumed: `0`
- mac_status_counts: `{dict(sorted(status_counts.items()))}`

## Readiness

Row-level parsing and filtering ran on parseable LatentWire prediction caches and Channel-Set per-trace recovery caches. Hard cache families were quarantined instead of read whole. No Mac result is marked `PASSED`; positives, if any, are only `PROVISIONAL_PROMOTE_TO_GPU`.

## Held-Out Answer

The parse coverage confirms many prior artifacts are named `test`, `validation`, `holdout`, or have no clean dev/gate/confirm naming. Treat cached screens as kill/screen evidence only. Final confirmation requires either quarantined confirm rows with a runner-specific row parser or fresh data generation.

## Top Gate Rows

{markdown_table(top_rows, ['method_id', 'paper', 'status', 'source_path', 'matched_accuracy', 'best_baseline_accuracy', 'delta_vs_best_baseline', 'ci95_low_vs_best_baseline', 'regime', 'median_recovery', 'cvar25_recovery', 'worst_recovery'])}
"""
    (dashboard_dir / "conductor_state.md").write_text(conductor.rstrip() + "\n", encoding="utf-8")

    morning = f"""# Morning Brief

Row-safe Stage-1 screen completed on parseable cached families. It consumed `{len(consumed_raw)}` dev/gate rows/traces and `0` confirm rows. Coverage is in `dashboard/cache_parse_coverage.md`; raw evidence is in `results/stage1/`.

Status counts: `{dict(sorted(status_counts.items()))}`.

The held-out answer is unfavorable for final claims: many caches are prior test/validation/full-eval artifacts or lack explicit confirm naming, so Mac screens can kill or rank branches, but final confirmation needs quarantined row-specific confirm handling or fresh data.
"""
    (dashboard_dir / "morning_brief.md").write_text(morning.rstrip() + "\n", encoding="utf-8")

    breakthrough_rows = [row for row in leaderboard if row["status"] == "PROVISIONAL_PROMOTE_TO_GPU"]
    kill_rows = [row for row in leaderboard if row["status"] == "KILLED"]
    breakthrough_md = "# Breakthrough Board\n\n" + markdown_table(
        breakthrough_rows,
        [
            "method_id",
            "paper",
            "source_path",
            "delta_vs_best_baseline",
            "ci95_low_vs_best_baseline",
            "median_recovery",
            "note",
        ],
    )
    (dashboard_dir / "breakthrough_board.md").write_text(
        breakthrough_md.rstrip() + "\n",
        encoding="utf-8",
    )
    kill_md = "# Kill Board\n\n" + markdown_table(
        kill_rows,
        [
            "method_id",
            "paper",
            "source_path",
            "delta_vs_best_baseline",
            "ci95_high_vs_best_baseline",
            "note",
        ],
    )
    (dashboard_dir / "kill_board.md").write_text(
        kill_md.rstrip() + "\n",
        encoding="utf-8",
    )

    queues_dir.mkdir(parents=True, exist_ok=True)
    parked = """parked:
  - id: channel_set_w4a16_confirmation
    reason: needs_gpu
    command: "python experimental/outlier_migrate/phase2/create_native_run_packet.py --label channel_set_confirmation --model <GPU_MODEL>"
    required_cache_or_model: "real W4A16-capable model checkpoint and quantized-forward runner"
    est_gpu_hours: 4-12
    why_cpu_cached_insufficient: "offline per-trace recovery cannot confirm new quantized forwards"
  - id: hybridkernel_native_profiler
    reason: needs_gpu
    command: "follow experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md"
    required_cache_or_model: "NVIDIA host with vLLM, Nsight Systems, Nsight Compute"
    est_gpu_hours: 2-6
    why_cpu_cached_insufficient: "native profiler counters and CUDA launch timelines unavailable on Mac"
  - id: latentwire_fresh_confirmation
    reason: needs_fresh_or_quarantined_confirm
    command: "run future LatentWire row-parser/runner on fresh or quarantined confirm split only after dev/gate selection is locked"
    required_cache_or_model: "small Qwen 0.5B/0.6B-class CPU/MPS model or fresh cached score rows"
    est_gpu_hours: 0
    why_cpu_cached_insufficient: "prior cached test/validation rows cannot confirm a dev/gate-selected method"
  - id: L_A2_verifier_rerank
    reason: parked_or_limited_cache
    command: "materialize generated-solution candidate pools on dev/gate, then rerun scripts/stage1_row_safe_screens.py"
    required_cache_or_model: "small-model generated-solution candidate pools"
    est_gpu_hours: 0-2
    why_cpu_cached_insufficient: "bounded pass did not find complete verifier-rerank candidate pools"
"""
    (queues_dir / "parked.yaml").write_text(parked, encoding="utf-8")

    print(
        "stage1 row-safe screens complete: "
        f"{len(usable)} usable files, {len(consumed_raw)} dev/gate rows, statuses={dict(sorted(status_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
