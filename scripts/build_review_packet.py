#!/usr/bin/env python3
"""Build the reviewer packet and a timestamped immutable copy."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}
EMBEDDED_CONFIRM_RE = re.compile(r"confirm|confirmation", re.I)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def confirm_path(name: str) -> bool:
    return fnmatch.fnmatch(name, "*_confirm*")


def hard_excluded(path: Path) -> bool:
    rel = path.as_posix()
    return path.suffix in HARD_EXTS or rel.startswith("caches/") or "/caches/" in rel


def embedded_confirm_findings(name: str, data: bytes) -> list[str]:
    path = Path(name)
    if path.suffix not in {".json", ".jsonl"}:
        return []
    text = data.decode("utf-8", errors="replace")
    findings: list[str] = []

    def check_value(value, pointer: str) -> None:
        if isinstance(value, str):
            if EMBEDDED_CONFIRM_RE.search(value):
                findings.append(f"{name}:{pointer}={value}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                check_value(item, f"{pointer}[{idx}]")
        elif isinstance(value, dict):
            walk(value, pointer)

    def walk(obj, pointer: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = f"{pointer}/{key}" if pointer else str(key)
                if key in {"source_path", "access_manifest", "file_access_manifest"}:
                    check_value(value, child)
                elif isinstance(value, (dict, list)):
                    check_value(value, child)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_value(item, f"{pointer}[{idx}]")

    try:
        if path.suffix == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    walk(json.loads(line), f"line{line_no}")
        else:
            walk(json.loads(text), "")
    except Exception:
        for match in re.finditer(r'"(?:source_path|access_manifest|file_access_manifest)"\s*:\s*"([^"]*)"', text):
            if EMBEDDED_CONFIRM_RE.search(match.group(1)):
                findings.append(f"{name}:regex={match.group(1)}")
    return findings


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
        "results/overnight_mps/**/summary.json",
        "results/overnight_mps/**/raw_rows.jsonl",
        "results/overnight_mps/**/raw_rows.checkpoint.jsonl",
        "results/overnight_mps/**/run_events.jsonl",
    ]:
        paths.extend(path for path in glob_paths(pattern) if path.is_file())
    cheap_dirs = sorted(path for path in (ROOT / "results/cheap_exhaustion").glob("*") if path.is_dir())
    if cheap_dirs:
        latest = cheap_dirs[-1]
        add_if_exists(paths, (latest / "summary.json").relative_to(ROOT).as_posix())
        add_if_exists(paths, (latest / "cache_inventory.json").relative_to(ROOT).as_posix())
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
        "scripts/overnight_mps_live.py",
        "scripts/cheap_exhaustion_scan.py",
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
        "2. `sed -n '1,260p' dashboard/cheap_exhaustion_report.md`\n"
        "3. `sed -n '1,260p' dashboard/c_a1_gpu_backfill_runbook.md`\n"
        "4. `sed -n '1,220p' queues/gpu_backfill.yaml`\n"
        "5. `sed -n '1,80p' queues/gpu_foreground.yaml`\n"
        "6. `local_runner enqueue channel_set_c_a1_pair_materialization --models granite,deepseek,falcon --split dev,gate --prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl --policies paroquant_baseline,tight_clip_c_a1 --scale-clip-min 0.5 --scale-clip-max 2.0 --require-same-row --write-access-manifest --fail-on-confirm --out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}`\n",
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
            data = head.encode("utf-8")
            findings = embedded_confirm_findings(name, data)
            if findings:
                omitted.append(f"{name}: hard-excluded embedded confirm-looking source/access paths {findings[:5]}")
                return
            entries[name] = data
            omitted.append(f"{name}: included 200-line head instead of full {size} byte file")
        else:
            omitted.append(f"{name}: skipped {size} byte file")
        return
    data = path.read_bytes()
    findings = embedded_confirm_findings(name, data)
    if findings:
        omitted.append(f"{name}: hard-excluded embedded confirm-looking source/access paths {findings[:5]}")
        return
    entries[name] = data


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
        "- Files with confirm-looking embedded `source_path` or access-manifest values are omitted and listed above.",
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
