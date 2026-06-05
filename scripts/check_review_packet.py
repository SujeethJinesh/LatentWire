#!/usr/bin/env python3
"""Validate review packets for leakage and handoff hazards."""

from __future__ import annotations

import argparse
import re
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
        if "INDEX.md" not in names:
            fail("INDEX.md missing")
        if "COMMIT.txt" not in names:
            fail("COMMIT.txt missing")
        if "DIFFSTAT.txt" not in names:
            fail("DIFFSTAT.txt missing")
        if "OMITTED_ARTIFACTS.md" not in names:
            fail("OMITTED_ARTIFACTS.md missing")
        if "NEXT_6_COMMANDS.md" not in names:
            fail("NEXT_6_COMMANDS.md missing")
        if "TESTS_RUN.txt" not in names:
            fail("TESTS_RUN.txt missing")
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
