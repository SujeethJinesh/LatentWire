#!/usr/bin/env python3
"""Validate review packets for leakage and handoff hazards."""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}
PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")
CONFIRM_SOURCE_RE = re.compile(r'"source_path"\s*:\s*"[^"]*(?:confirm|confirmation)[^"]*"', re.I)
QUEUE_REF_RE = re.compile(
    r"(?<![\w./-])((?:dashboard|scripts|queues|registry|paper|lessons)/[A-Za-z0-9._/@+=:-]+)"
)
REQUIRED_PACKET_FILES = {
    "COMMIT.txt",
    "DIFFSTAT.txt",
    "INDEX.md",
    "NEXT_6_COMMANDS.md",
    "OMITTED_ARTIFACTS.md",
    "TESTS_RUN.txt",
    "dashboard/c_a1_gpu_backfill_runbook.md",
    "dashboard/cheap_exhaustion_report.md",
    "dashboard/confirm_path_audit.md",
    "lessons/LESSONS_LEDGER.md",
    "paper/channel_set/figures_todo.md",
    "paper/latentwire/claim_boundary.md",
    "queues/gpu_backfill.yaml",
    "queues/gpu_foreground.yaml",
}


def fail(message: str) -> None:
    raise SystemExit(f"review packet validation failed: {message}")


def read_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def validate_yaml_payload(name: str, text: str) -> None:
    try:
        payload = yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - exact parser message is enough
        fail(f"{name} is not valid YAML: {exc}")
    if payload is None:
        return
    if PLACEHOLDER_RE.search(text):
        fail(f"{name} contains unresolved placeholder text")
    if name == "queues/gpu_backfill.yaml" and ("venv_arm64" in text or "--device cpu" in text):
        lowered = text.lower()
        if "mac_only: true" not in lowered and "mac-only" not in lowered:
            fail(f"{name} contains Mac/CPU command without explicit Mac-only marking")


def validate_queue_references(name: str, text: str, names: set[str]) -> None:
    if not name.startswith("queues/"):
        return
    missing = []
    for raw in QUEUE_REF_RE.findall(text):
        ref = raw.rstrip(".,;:)'\"]")
        if "*" in ref or ref.endswith("/"):
            continue
        if ref not in names:
            missing.append(ref)
    if missing:
        fail(f"{name} references file(s) absent from packet: {sorted(set(missing))[:10]}")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def validate_commit_head(zf: zipfile.ZipFile) -> None:
    commit_text = read_text(zf, "COMMIT.txt")
    index_text = read_text(zf, "INDEX.md")
    head = git_head()
    commit_match = re.search(r"^commit:\s*([0-9a-f]{40})\s*$", commit_text, re.M)
    if not commit_match:
        fail("COMMIT.txt does not contain a full `commit: <sha>` line")
    packet_sha = commit_match.group(1)
    if packet_sha != head:
        fail(f"COMMIT.txt commit {packet_sha} does not match current HEAD {head}")
    if f"Commit: `{head}`" not in index_text:
        fail("INDEX.md Commit line does not match current HEAD")

    dirty_tracked = []
    bad_untracked = []
    for line in commit_text.splitlines():
        if not line or line.startswith(("branch:", "commit:", "created_utc:", "## ")):
            continue
        if line.startswith("?? "):
            path = line[3:].strip()
            if "/" in path or not path.endswith(".zip"):
                bad_untracked.append(path)
        else:
            dirty_tracked.append(line)
    if dirty_tracked:
        fail(f"COMMIT.txt records tracked worktree changes: {dirty_tracked[:10]}")
    if bad_untracked:
        fail(f"COMMIT.txt records disallowed untracked paths: {bad_untracked[:10]}")


def validate_required_files(names: set[str]) -> None:
    missing = sorted(REQUIRED_PACKET_FILES - names)
    if missing:
        fail(f"required packet file(s) missing: {missing}")


def validate_queue_state(zf: zipfile.ZipFile) -> None:
    foreground = yaml.safe_load(read_text(zf, "queues/gpu_foreground.yaml")) or {}
    if foreground.get("foreground") != []:
        fail("queues/gpu_foreground.yaml is not empty")
    backfill = yaml.safe_load(read_text(zf, "queues/gpu_backfill.yaml")) or {}
    jobs = backfill.get("backfill") or []
    if not jobs:
        fail("queues/gpu_backfill.yaml has no backfill jobs")
    first = jobs[0]
    if first.get("id") != "channel_set_c_a1_tail_cvar_grid":
        fail("first gpu_backfill job is not channel_set_c_a1_tail_cvar_grid")
    if first.get("promotion_allowed") is not False:
        fail("C_A1 gpu_backfill job must have promotion_allowed=false")
    if "channel_set_c_a1_pair_materialization" not in str(first.get("command", "")):
        fail("first C_A1 backfill command is not the fresh-row materialization command")


def validate_runbook_and_next_commands(zf: zipfile.ZipFile) -> None:
    runbook = read_text(zf, "dashboard/c_a1_gpu_backfill_runbook.md")
    if "RUNBOOK_STATUS: BLOCKED_CONFIRM_CONTAMINATED" not in runbook:
        fail("C_A1 runbook does not declare BLOCKED_CONFIRM_CONTAMINATED")
    if "Fresh Row-Materialization Command" not in runbook:
        fail("C_A1 runbook does not include the fresh materialization command section")
    command = "local_runner enqueue channel_set_c_a1_pair_materialization"
    if command not in runbook:
        fail("C_A1 runbook missing exact materialization command")
    if command not in read_text(zf, "NEXT_6_COMMANDS.md"):
        fail("NEXT_6_COMMANDS.md missing exact C_A1 materialization command")


def validate_embedded_confirm_sources(zf: zipfile.ZipFile, names: list[str]) -> None:
    audit_name = "dashboard/confirm_path_audit.md"
    if audit_name not in names:
        fail("dashboard/confirm_path_audit.md is missing")
    audit = read_text(zf, audit_name)
    if "AUDIT_STATUS: COMPLETE" not in audit:
        fail("confirm_path_audit.md does not declare AUDIT_STATUS: COMPLETE")
    if "C_A1_GATE_STATUS: INVALID_CONFIRM_CONTAMINATED" not in audit:
        fail("confirm_path_audit.md does not explicitly gate C_A1 contaminated cached evidence")
    unaudited = []
    for name in names:
        if name == "INDEX.md":
            continue
        if not name.endswith((".json", ".jsonl", ".csv", ".md")):
            continue
        text = read_text(zf, name)
        if "confirm" not in text.lower() and "confirmation" not in text.lower():
            continue
        has_confirm_source_line = any(
            "source_path" in line
            and re.search(r"confirm|confirmation", line, re.I)
            and re.search(r"(?:experimental|results|dashboard|paper)/", line)
            for line in text.splitlines()
        )
        if CONFIRM_SOURCE_RE.search(text) or has_confirm_source_line:
            if name != audit_name and name not in audit:
                unaudited.append(name)
    if unaudited:
        fail(f"embedded confirm-looking source_path values are not audited for: {unaudited[:10]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    zip_path = args.zip_path if args.zip_path.is_absolute() else ROOT / args.zip_path
    if not zip_path.exists():
        fail(f"packet not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        name_set = set(names)
        validate_required_files(name_set)
        validate_commit_head(zf)
        validate_queue_state(zf)
        validate_runbook_and_next_commands(zf)
        raw_confirm = [name for name in names if re.search(r"_confirm", name, re.I)]
        if raw_confirm:
            fail(f"raw confirm path(s) included: {raw_confirm[:10]}")
        hard = [name for name in names if Path(name).suffix in HARD_EXTS]
        if hard:
            fail(f"hard binary artifacts included: {hard[:10]}")
        caches = [name for name in names if name.startswith("caches/") or "/caches/" in name]
        if caches:
            fail(f"cache paths included: {caches[:10]}")
        for name in names:
            if name.endswith(".yaml"):
                text = read_text(zf, name)
                validate_yaml_payload(name, text)
                validate_queue_references(name, text, name_set)
            elif name.endswith((".md", ".txt", ".csv", ".json", ".jsonl", ".py")):
                text = read_text(zf, name)
                if PLACEHOLDER_RE.search(text) and name in {
                    "dashboard/c_a1_gpu_backfill_runbook.md",
                    "dashboard/gpu_handoff_plan.md",
                    "dashboard/channel_set_next_gpu.md",
                }:
                    fail(f"{name} contains unresolved placeholder text")
        validate_embedded_confirm_sources(zf, names)
        if zip_path.stat().st_size > 25 * 1024 * 1024:
            fail(f"packet too large: {zip_path.stat().st_size}")
    print(f"review packet OK: {zip_path} ({zip_path.stat().st_size} bytes, {len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
