#!/usr/bin/env python3
"""Build the reviewer packet and a timestamped immutable copy."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def confirm_path(name: str) -> bool:
    return fnmatch.fnmatch(name, "*_confirm*")


def hard_excluded(path: Path) -> bool:
    rel = path.as_posix()
    return path.suffix in HARD_EXTS or rel.startswith("caches/") or "/caches/" in rel


def add_if_exists(paths: list[Path], rel: str) -> None:
    path = ROOT / rel
    if path.exists() and path.is_file():
        paths.append(path)


def glob_paths(pattern: str) -> list[Path]:
    return sorted(ROOT.glob(pattern))


def packet_candidates() -> list[Path]:
    paths: list[Path] = []
    for pattern in [
        "dashboard/*.md",
        "dashboard/leaderboard.csv",
        "paper/latentwire/claim_boundary.md",
        "paper/channel_set/figures_todo.md",
        "queues/*.yaml",
        "registry/**/*.yaml",
        "lessons/LESSONS_LEDGER.md",
        "lessons/*.md",
        "results/mps_first*/**/summary.json",
        "results/mps_first*/**/raw_rows.jsonl",
    ]:
        paths.extend(path for path in glob_paths(pattern) if path.is_file())
    for rel in [
        "COMMIT.txt",
        "DIFFSTAT.txt",
        "OMITTED_ARTIFACTS.md",
        "NEXT_6_COMMANDS.md",
        "TESTS_RUN.txt",
        "scripts/audit_confirm_paths.py",
        "scripts/build_review_packet.py",
        "scripts/check_review_packet.py",
        "scripts/check_handoff.py",
        "scripts/run_mps_first_iteration.py",
        "scripts/run_mps_first_strict_iteration.py",
        "scripts/stage1_row_safe_screens.py",
        "scripts/overnight_v3_corrected_probes.py",
    ]:
        add_if_exists(paths, rel)
    return sorted(set(paths))


def write_support_files() -> list[Path]:
    sha = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--short", "--branch"])
    diffstat = run(["git", "diff", "--stat", "HEAD"])
    (ROOT / "COMMIT.txt").write_text(
        f"branch: {branch}\ncommit: {sha}\ncreated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n{status}\n",
        encoding="utf-8",
    )
    (ROOT / "DIFFSTAT.txt").write_text((diffstat or "No uncommitted tracked diff.\n") + "\n", encoding="utf-8")
    (ROOT / "OMITTED_ARTIFACTS.md").write_text(
        "# Omitted Artifacts\n\n"
        "- Model weights and binary arrays are excluded.\n"
        "- Caches are excluded.\n"
        "- Files over 5 MB are skipped or represented by a 200-line head when text-like.\n",
        encoding="utf-8",
    )
    (ROOT / "NEXT_6_COMMANDS.md").write_text(
        "# Next 6 Commands\n\n"
        "1. `venv_arm64/bin/python scripts/check_review_packet.py review_packet.zip`\n"
        "2. `sed -n '1,240p' dashboard/confirm_path_audit.md`\n"
        "3. `sed -n '1,220p' dashboard/l_c2_oracle_decomposition.md`\n"
        "4. `sed -n '1,220p' dashboard/l_pc5_deployable_verifier_plan.md`\n"
        "5. `sed -n '1,220p' dashboard/l_pc1_reconciliation.md`\n"
        "6. `sed -n '1,260p' dashboard/c_a1_gpu_backfill_runbook.md`\n",
        encoding="utf-8",
    )
    return [ROOT / name for name in ["COMMIT.txt", "DIFFSTAT.txt", "OMITTED_ARTIFACTS.md", "NEXT_6_COMMANDS.md"]]


def add_packet_file(entries: dict[str, bytes], omitted: list[str], path: Path) -> None:
    name = path.relative_to(ROOT).as_posix()
    if confirm_path(name) or hard_excluded(Path(name)):
        omitted.append(f"{name}: hard-excluded by path or extension")
        return
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        if path.suffix in {".jsonl", ".csv", ".md", ".txt", ".yaml", ".json"}:
            head = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:200]) + "\n"
            entries[name] = head.encode("utf-8")
            omitted.append(f"{name}: included 200-line head instead of full {size} byte file")
        else:
            omitted.append(f"{name}: skipped {size} byte file")
        return
    entries[name] = path.read_bytes()


def build(out: Path) -> Path:
    support = write_support_files()
    entries: dict[str, bytes] = {}
    omitted: list[str] = []
    for path in sorted(set(packet_candidates() + support)):
        if path.exists() and path.is_file():
            add_packet_file(entries, omitted, path)
    index_lines = [
        "# Review Packet Index",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Commit: `{run(['git', 'rev-parse', 'HEAD'])}`",
        "",
        "## Included",
    ]
    for name in sorted(entries):
        index_lines.append(f"- `{name}` ({len(entries[name])} bytes)")
    index_lines.extend(["", "## SKIPPED-too-large / Truncated"])
    index_lines.extend(f"- {item}" for item in omitted) if omitted else index_lines.append("- None")
    index_lines.extend([
        "",
        "## Guardrails",
        "- No raw paths matching `*_confirm*` are included.",
        "- Confirm-looking embedded `source_path` values must be audited in `dashboard/confirm_path_audit.md`.",
        "- No model weights, binary arrays, or caches are included.",
    ])
    entries["INDEX.md"] = ("\n".join(index_lines) + "\n").encode("utf-8")
    bad = [name for name in entries if confirm_path(name)]
    if bad:
        raise SystemExit(f"would include forbidden confirm paths: {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    sha = run(["git", "rev-parse", "--short", "HEAD"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned = ROOT / "review_packets" / f"review_packet_{stamp}_{sha}.zip"
    versioned.parent.mkdir(parents=True, exist_ok=True)
    versioned.write_bytes(out.read_bytes())
    return versioned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "review_packet.zip")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    versioned = build(out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")
    print(f"wrote {versioned.relative_to(ROOT)} ({versioned.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
