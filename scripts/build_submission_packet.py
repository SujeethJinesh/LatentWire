#!/usr/bin/env python3
"""Build submission_packet.zip with final paper sources and review evidence."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ZIP_BYTES = 25 * 1024 * 1024
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def forbidden_path(name: str) -> bool:
    return fnmatch.fnmatch(name, "*_confirm*")


def hard_excluded(name: str) -> bool:
    path = Path(name)
    return path.suffix in HARD_EXTS or name.startswith("caches/") or "/caches/" in name


def support_entries() -> dict[str, bytes]:
    sha = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--short", "--branch"])
    diffstat = run(["git", "diff", "--stat", "HEAD"])
    return {
        "COMMIT.txt": (
            f"branch: {branch}\n"
            f"commit: {sha}\n"
            f"created_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
            f"{status}\n"
        ).encode("utf-8"),
        "DIFFSTAT.txt": ((diffstat or "No uncommitted tracked diff.\n") + "\n").encode("utf-8"),
    }


def candidate_paths() -> list[Path]:
    patterns = [
        "paper/latentwire/draft.md",
        "paper/latentwire/main.tex",
        "paper/latentwire/main.pdf",
        "paper/latentwire/figures/*",
        "paper/latentwire/tables/*",
        "paper/channel_set/draft.md",
        "paper/channel_set/main.tex",
        "paper/channel_set/main.pdf",
        "paper/channel_set/figures/*",
        "paper/channel_set/tables/*",
        "paper/references.bib",
        "paper/response_plan.md",
        "reviews/mock_colm_board_iter*.json",
        "dashboard/paper_review_trajectory.md",
        "queues/gpu_foreground.yaml",
        "TESTS_RUN.txt",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(set(paths))


def add_file(entries: dict[str, bytes], skipped: list[str], path: Path) -> None:
    name = path.relative_to(ROOT).as_posix()
    if forbidden_path(name) or hard_excluded(name):
        skipped.append(f"{name}: hard-excluded")
        return
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        skipped.append(f"{name}: skipped {size} byte file")
        return
    entries[name] = path.read_bytes()


def build(out: Path) -> Path:
    entries = support_entries()
    skipped: list[str] = []
    for path in candidate_paths():
        add_file(entries, skipped, path)

    index = [
        "# Submission Packet Index",
        "",
        "Final paper sources, PDFs, figures, provenance tables, response plan, and mock-COLM board records.",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Commit: `{run(['git', 'rev-parse', 'HEAD'])}`",
        "",
        "## Included",
    ]
    descriptions = {
        "paper/latentwire/main.tex": "LatentWire LaTeX source.",
        "paper/latentwire/main.pdf": "LatentWire built PDF.",
        "paper/channel_set/main.tex": "Channel-Set LaTeX source.",
        "paper/channel_set/main.pdf": "Channel-Set built PDF.",
        "paper/references.bib": "Shared bibliography.",
        "paper/response_plan.md": "Reviewer-objection response plan.",
        "dashboard/paper_review_trajectory.md": "Mock-COLM review trajectory.",
        "queues/gpu_foreground.yaml": "Foreground GPU queue state; kept empty for finalization.",
        "TESTS_RUN.txt": "Validation command log.",
    }
    for name in sorted(entries):
        desc = descriptions.get(name, "Included paper, figure, table, provenance, or support artifact.")
        index.append(f"- `{name}` ({len(entries[name])} bytes): {desc}")
    index.extend(["", "## SKIPPED-too-large"])
    index.extend(f"- {item}" for item in skipped) if skipped else index.append("- None")
    index.extend(
        [
            "",
            "## Guardrails",
            "- No forbidden split-entry path is included.",
            "- No model weights, binary arrays, or caches are included.",
            "- C_A1 is parked and optional; the foreground GPU queue is empty.",
        ]
    )
    entries["README_INDEX.md"] = ("\n".join(index) + "\n").encode("utf-8")

    bad = [name for name in entries if forbidden_path(name)]
    if bad:
        raise SystemExit(f"would include forbidden paths: {bad}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    if out.stat().st_size > MAX_ZIP_BYTES:
        raise SystemExit(f"package too large: {out.stat().st_size}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "submission_packet.zip")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    build(out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
