#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmc.cache_split import build_inventory, confirm_open_guard, path_hash


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_inventory_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unit_id",
        "root",
        "path",
        "file_count",
        "row_count",
        "partition_status",
        "scan_statuses",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-0 cache inventory and split freeze")
    parser.add_argument("--root", action="append")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--dashboard-dir", default="dashboard")
    parser.add_argument("--queues-dir", default="queues")
    parser.add_argument("--max-scan-bytes", type=int, default=50_000_000)
    args = parser.parse_args()

    roots = [Path(root) for root in (args.root or ["experimental", "results"])]
    splits_dir = Path(args.splits_dir)
    dashboard_dir = Path(args.dashboard_dir)
    queues_dir = Path(args.queues_dir)

    units, cache_rows = build_inventory(roots, max_scan_bytes=args.max_scan_bytes)
    unit_payload = [
        {
            "unit_id": unit.unit_id,
            "root": unit.root,
            "path": unit.path,
            "file_count": unit.file_count,
            "row_count": unit.row_count,
            "scan_statuses": list(unit.scan_statuses),
            "partition_status": unit.partition_status,
        }
        for unit in units
    ]

    row_counts = Counter(row.split for row in cache_rows)
    status_counts = Counter(unit.partition_status for unit in units)
    scan_counts = Counter()
    for unit in units:
        scan_counts.update(unit.scan_statuses)

    write_json(
        splits_dir / "stage0_cache_inventory.json",
        {
            "schema": "stage0_cache_inventory_v1",
            "roots": [root.as_posix() for root in roots],
            "max_scan_bytes": args.max_scan_bytes,
            "unit_count": len(units),
            "row_entity_count": len(cache_rows),
            "split_counts": dict(sorted(row_counts.items())),
            "partition_status_counts": dict(sorted(status_counts.items())),
            "scan_status_counts": dict(sorted(scan_counts.items())),
            "units": unit_payload,
        },
    )
    write_inventory_csv(splits_dir / "stage0_cache_inventory.csv", unit_payload)

    splits_dir.mkdir(parents=True, exist_ok=True)
    with (splits_dir / "stage0_cache_rows_hash.jsonl").open("w", encoding="utf-8") as handle:
        for row in cache_rows:
            handle.write(
                json.dumps(
                    {
                        "unit_id": row.unit_id,
                        "path": row.path,
                        "row_key": row.row_key,
                        "split": row.split,
                        "row_hash": row.row_hash,
                        "scan_status": row.scan_status,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    confirm_paths = sorted({row.path for row in cache_rows if row.split == "confirm"})
    with (splits_dir / "stage0_confirm_quarantine_hashes.txt").open("w", encoding="utf-8") as handle:
        handle.write("# sha256(abs_path) path\n")
        for source_path in confirm_paths:
            handle.write(f"{path_hash(Path(source_path))} {source_path}\n")

    planted_status: dict[str, object] = {
        "planted_confirm_path_guard": "skipped_no_confirm_paths",
        "planted_dev_path_allowed": "skipped_no_dev_only_paths",
    }
    if confirm_paths:
        try:
            with confirm_open_guard(splits_dir / "stage0_confirm_quarantine_hashes.txt"):
                Path(confirm_paths[0]).open("rb").close()
        except RuntimeError:
            planted_status["planted_confirm_path_guard"] = "pass"
        else:
            planted_status["planted_confirm_path_guard"] = "fail"

    dev_only_paths = sorted(
        {
            row.path
            for row in cache_rows
            if row.split == "dev"
        }
        - set(confirm_paths)
    )
    if dev_only_paths:
        try:
            with confirm_open_guard(splits_dir / "stage0_confirm_quarantine_hashes.txt"):
                Path(dev_only_paths[0]).open("rb").close()
        except RuntimeError:
            planted_status["planted_dev_path_allowed"] = "fail"
        else:
            planted_status["planted_dev_path_allowed"] = "pass"

    write_json(splits_dir / "stage0_planted_guard_checks.json", planted_status)

    full_eval_like = sum(
        count
        for status, count in status_counts.items()
        if status in {"prior_eval_rows_present_no_clean_confirm", "no_named_partition_detected"}
    )
    scan_mode = (
        "metadata_file_level" if args.max_scan_bytes <= 0 else f"row_scan_up_to_{args.max_scan_bytes}_bytes"
    )
    conductor = f"""# Conductor State

- stage: `stage0_cache_freeze_complete`
- branch: `codex-campaign`
- cache_units: `{len(units)}`
- row_entities: `{len(cache_rows)}`
- scan_mode: `{scan_mode}`
- split_counts: `{dict(sorted(row_counts.items()))}`
- partition_status_counts: `{dict(sorted(status_counts.items()))}`
- planted_confirm_path_guard: `{planted_status["planted_confirm_path_guard"]}`
- planted_dev_path_allowed: `{planted_status["planted_dev_path_allowed"]}`

## Readiness

Stage 0 is complete for local cache handling. Existing cache rows/entities are frozen into dev/gate/confirm assignments in `splits/stage0_cache_rows_hash.jsonl`, and confirm-bearing source paths are listed in `splits/stage0_confirm_quarantine_hashes.txt`.

This run used `{scan_mode}`. With `metadata_file_level`, each discovered cache/result file is treated as the split entity; method-specific Stage-1 runners may do finer row parsing later, but they must still consume this manifest and preserve the confirm quarantine.

All Stage-1/2 screens must fit only on dev/gate entities. Any positive from these historical caches remains **PROVISIONAL** until held-out confirmation or fresh data exists.

## Saturation

- Alive locally: cache-only LatentWire and Channel-Set screening that consumes `splits/stage0_cache_rows_hash.jsonl`.
- Parked: CUDA, W4A16, 30B, long 20K-token decode, native vLLM/Nsight profiler, fresh confirmation data.
- Highest-priority next branch: implement the Stage-1 method runners against the frozen dev/gate manifests, beginning with LatentWire L-ScoreComp and Channel-Set C-A1/C-F offline recovery.
"""
    write_text(dashboard_dir / "conductor_state.md", conductor)

    morning = f"""# Morning Brief

Stage 0 cache freeze ran locally on the Mac in `{scan_mode}` mode. It found `{len(units)}` cache units and `{len(cache_rows)}` row/file entities. `{full_eval_like}` units lack a clean named dev/gate/confirm partition or look like prior eval/full-test caches, so any screen using them is provisional.

Next: build/run Stage-1 screens only against dev/gate split manifests, then park all confirmation/GPU work in `queues/parked.yaml`.
"""
    write_text(dashboard_dir / "morning_brief.md", morning)
    write_text(
        dashboard_dir / "breakthrough_board.md",
        "# Breakthrough Board\n\nNo positive method promoted in this pass. Stage 0 only froze the cache/leakage surface.\n",
    )
    write_text(
        dashboard_dir / "kill_board.md",
        "# Kill Board\n\nNo method killed in this pass. Stage 0 did not execute method screens.\n",
    )
    write_text(
        queues_dir / "parked.yaml",
        """parked:
  - id: channel_set_w4a16_confirmation
    reason: needs_gpu
    detail: Real W4A16 quantized-forward recovery confirmation is not runnable on Apple Silicon.
  - id: hybridkernel_native_profiler
    reason: needs_gpu
    detail: User-operated NVIDIA/vLLM/Nsight profiler packet remains the next systems gate.
  - id: long_reasoning_decode_and_30b
    reason: needs_gpu
    detail: 30B / long 20K-token decode / ParoQuant-on-real-model work is parked for the GPU node.
  - id: heldout_confirmation
    reason: needs_fresh_or_quarantined_confirm
    detail: Mac cache screens may not access confirm-split entities during Stage 1/2.
""",
    )

    print(
        "stage0 cache freeze complete: "
        f"{len(units)} units, {len(cache_rows)} row/file entities, splits={dict(sorted(row_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
