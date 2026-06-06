#!/usr/bin/env python3
"""Build papers_review.zip for external paper review."""

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
        "paper/latentwire/draft.pdf",
        "paper/latentwire/figures/*",
        "paper/latentwire/tables/*",
        "paper/channel_set/draft.md",
        "paper/channel_set/draft.pdf",
        "paper/channel_set/figures/*",
        "paper/channel_set/tables/*",
        "paper/response_plan.md",
        "reviews/mock_colm_board_iter*.json",
        "dashboard/paper_review_trajectory.md",
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
        if path.suffix in {".md", ".txt", ".json", ".jsonl", ".csv", ".yaml"}:
            data = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:200]).encode("utf-8")
            entries[name] = data
            skipped.append(f"{name}: included 200-line head instead of full {size} byte file")
        else:
            skipped.append(f"{name}: skipped {size} byte file")
        return
    entries[name] = path.read_bytes()


def build(out: Path) -> Path:
    entries = support_entries()
    skipped: list[str] = []
    for path in candidate_paths():
        add_file(entries, skipped, path)

    index = [
        "# Papers Review Package Index",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Commit: `{run(['git', 'rev-parse', 'HEAD'])}`",
        "",
        "## Included",
    ]
    for name in sorted(entries):
        index.append(f"- `{name}` ({len(entries[name])} bytes)")
    index.extend(["", "## SKIPPED-too-large / Excluded"])
    index.extend(f"- {item}" for item in skipped) if skipped else index.append("- None")
    index.extend(
        [
            "",
            "## Guardrails",
            "- No forbidden split-entry path is included.",
            "- Model weights, binary arrays, and caches are excluded.",
            "- Papers preserve the LatentWire bounded-negative and Channel-Set measurement/regime claim boundaries.",
        ]
    )
    entries["INDEX.md"] = ("\n".join(index) + "\n").encode("utf-8")

    bad = [name for name in entries if forbidden_path(name)]
    if bad:
        raise SystemExit(f"would include forbidden paths: {bad}")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    if out.stat().st_size > MAX_ZIP_BYTES:
        raise SystemExit(f"package too large: {out.stat().st_size}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "papers_review.zip")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    build(out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
