#!/usr/bin/env python3
"""Handoff integrity check. Run as Step 0 of Hours 0-6 (CODEX_NEXT_72H.md §3).
Stalls the campaign if the bundle is misplaced or oversized, so the Conductor never
starts from missing/stale instructions."""
from pathlib import Path
import sys

REQUIRED = [
    "AGENTS.md",
    "plans/CODEX_NEXT_72H.md",
    "plans/TEAM_OPERATING_SYSTEM.md",
    "plans/EXPERIMENT_SELECTION_SPEC.md",
    "plans/PLANTED_TESTS_SPEC.md",
    "plans/DATA_RETENTION_SPEC.md",
    "plans/CAMPAIGN_BACKGROUND.md",
]
AGENTS_MAX_BYTES = 32 * 1024  # Codex project_doc_max_bytes default; larger => silent truncation

def main() -> int:
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        print(f"FAIL: missing required handoff files at exact paths: {missing}")
        return 1
    n = Path("AGENTS.md").stat().st_size
    if n >= AGENTS_MAX_BYTES:
        print(f"FAIL: AGENTS.md is {n} bytes >= 32 KiB cap; Codex will silently truncate. Move detail into plans/.")
        return 1
    bg = Path("plans/CAMPAIGN_BACKGROUND.md").read_text(errors="ignore")[:600]
    if "BACKGROUND" not in bg.upper():
        print("WARN: plans/CAMPAIGN_BACKGROUND.md is not clearly marked rationale-only at the top.")
    print(f"handoff files OK; AGENTS.md = {n} bytes (< 32 KiB).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
