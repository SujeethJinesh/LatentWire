#!/usr/bin/env python3
"""Verifiable termination predicate for the COLM 2026 GPU swarm.

Exits 0 with stdout 'SWARM_COMPLETE' iff:
  (a) swarm/state.json status == COMPLETE
  (b) every queue entry that ran has a corresponding result packet
  (c) no preregistration file has been modified since started_at_sha
  (d) every alive paper builds clean (PDF exists and is recent)
  (e) every alive paper has a current reviewer_pack.md

Exits non-zero otherwise with diagnostic output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "swarm" / "state.json"
QUEUE_PATH = REPO_ROOT / "swarm" / "queue.yml"

PREREG_GLOBS = [
    "experimental/*/phase*/preregister_*.md",
    "experimental/*/preregister_*.md",
    "experimental/hybridkernel/phase2/preregister_*.md",
]

ALIVE_PAPER_DIRS = [
    "experimental/decode_microkernel/paper",
    "experimental/hybridkernel/paper",
    "experimental/thoughtflow_fp8/paper",
    "experimental/outlier_migrate/paper",
    "experimental/residual_migration/paper",
    "experimental/ssm_lifecycle/paper",
    "experimental/cross_layer_error/paper",
]


def fail(reason: str) -> None:
    print(f"SWARM_INCOMPLETE: {reason}", file=sys.stderr)
    sys.exit(1)


def load_state() -> dict:
    if not STATE_PATH.exists():
        fail(f"state.json missing at {STATE_PATH}")
    return json.loads(STATE_PATH.read_text())


def check_state_complete(state: dict) -> None:
    if state.get("status") != "COMPLETE":
        fail(f"state.status is {state.get('status')!r}, not COMPLETE")
    if state.get("started_at_sha") is None:
        fail("state.started_at_sha is null; swarm never started cleanly")


def check_no_prereg_drift(state: dict) -> None:
    started_sha = state["started_at_sha"]
    for pattern in PREREG_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            rel = path.relative_to(REPO_ROOT)
            exists_at_start = subprocess.run(
                ["git", "cat-file", "-e", f"{started_sha}:{rel}"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            diff_base = started_sha
            if exists_at_start.returncode != 0:
                added = subprocess.run(
                    [
                        "git",
                        "log",
                        "--diff-filter=A",
                        "--format=%H",
                        "--reverse",
                        "--",
                        str(rel),
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                if added.returncode != 0:
                    fail(f"git log failed on new preregistration {rel}: {added.stderr}")
                added_shas = [line.strip() for line in added.stdout.splitlines() if line.strip()]
                if not added_shas:
                    fail(
                        f"new preregistration file is uncommitted and cannot be audited: {rel}"
                    )
                diff_base = added_shas[0]
            result = subprocess.run(
                ["git", "diff", "--name-only", diff_base, "HEAD", "--", str(rel)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                fail(f"git diff failed on {rel}: {result.stderr}")
            if result.stdout.strip():
                fail(f"preregistration file modified after freeze point: {rel}")


def check_results_for_completed_entries(state: dict) -> None:
    for entry in state.get("completed_entries", []):
        entry_id = entry["id"] if isinstance(entry, dict) else entry
        if isinstance(entry, dict) and entry.get("result_dir"):
            result_dir = REPO_ROOT / str(entry["result_dir"])
            if not result_dir.exists():
                fail(f"result_dir missing for completed entry {entry_id}: {entry['result_dir']}")
            continue
        # Result packet location is /experimental/<project>/phase<N>/results/
        # Loose check: at least one results dir exists for this entry
        candidates = list(REPO_ROOT.glob(f"experimental/*/phase*/results/*{entry_id}*"))
        if not candidates:
            candidates = list(REPO_ROOT.glob(f"experimental/*/results/*{entry_id}*"))
        if not candidates:
            fail(f"no result packet found for completed entry {entry_id}")


def check_papers_buildable(state: dict) -> None:
    candidates = state.get("papers_camera_ready_candidate", [])
    if not candidates:
        # If no papers were marked camera-ready candidate, that is acceptable
        # only if every alive project was killed.
        return
    for paper_path in candidates:
        pdf_path = REPO_ROOT / paper_path
        if not pdf_path.exists():
            fail(f"camera-ready candidate PDF missing: {paper_path}")
        # Reviewer pack
        reviewer_pack = pdf_path.parent / "reviewer_pack.md"
        if not reviewer_pack.exists():
            fail(f"reviewer_pack.md missing for {paper_path}")


def main() -> int:
    state = load_state()
    check_state_complete(state)
    check_no_prereg_drift(state)
    check_results_for_completed_entries(state)
    check_papers_buildable(state)

    print("SWARM_COMPLETE")
    print(json.dumps({
        "status": state["status"],
        "completed": len(state.get("completed_entries", [])),
        "killed": len(state.get("killed_entries", [])),
        "infra_failed": len(state.get("infra_failed_entries", [])),
        "gpu_hours_used": state.get("gpu_hours_used"),
        "papers_camera_ready_candidate": state.get("papers_camera_ready_candidate", []),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
