#!/usr/bin/env python3
"""Validate anonymized COLM OpenReview PDFs and final packet."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import yaml
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDFS = [ROOT / "openreview_latentwire.pdf", ROOT / "openreview_channel_set.pdf"]
FORBIDDEN_TEXT = [
    r"/Users/",
    r"sujeeth",
    r"jinesh",
    r"LatentWire\.git",
    r"\b[0-9a-f]{40}\b",
    r"\bC[-_]A1\b",
    r"\bL[-_]IB1\b",
    r"\bL[-_]Q1\b",
    r"\bL[-_]B1\b",
    r"\bL[-_]PC\d+\b",
    r"\bL[-_]C2\b",
    r"\bC[-_]U1\b",
    r"\bC[-_]S1\b",
    r"\bC[-_]Y5\b",
    r"KILL_UTILITY",
    r"\bresults/",
    r"\bdashboard/",
    r"\bexperimental/",
    r"\bregistry/",
    r"\btables/",
]


def fail(message: str) -> None:
    raise SystemExit(f"openreview hygiene failed: {message}")


def pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main_pages(path: Path) -> tuple[int, int]:
    reader = PdfReader(path)
    total = len(reader.pages)
    ref_start = None
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if re.search(r"(^|\n)\s*References\s*(\n|$)", text):
            ref_start = idx
            break
    main = total if ref_start is None else ref_start - 1
    return main, total


def check_pdf(path: Path) -> None:
    if not path.exists():
        fail(f"missing PDF: {path}")
    main, total = main_pages(path)
    if not 4 <= main <= 10:
        fail(f"{path.name} main-text page count {main} outside 4-10")
    text = pdf_text(path)
    if "??" in text:
        fail(f"{path.name} contains unresolved reference marker '??'")
    for pattern in FORBIDDEN_TEXT:
        if re.search(pattern, text, re.I):
            fail(f"{path.name} contains forbidden deanonymizing/internal text matching {pattern}")
    metadata = PdfReader(path).metadata or {}
    meta_text = "\n".join(str(value) for value in metadata.values() if value)
    for pattern in [r"/Users/", r"sujeeth", r"jinesh", r"\b[0-9a-f]{40}\b"]:
        if re.search(pattern, meta_text, re.I):
            fail(f"{path.name} metadata contains forbidden text matching {pattern}")
    print(f"{path.relative_to(ROOT)}: main_pages={main}, total_pages={total}, bytes={path.stat().st_size}")


def check_logs() -> None:
    for path in [ROOT / "paper" / "latentwire" / "main.log", ROOT / "paper" / "channel_set" / "main.log"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        bad = [
            "Undefined control sequence",
            "LaTeX Error",
            "Emergency stop",
            "Fatal error occurred",
        ]
        hits = [needle for needle in bad if needle in text]
        if hits:
            fail(f"{path.relative_to(ROOT)} has build warning/error markers: {hits}")


def check_queue() -> None:
    foreground = yaml.safe_load((ROOT / "queues" / "gpu_foreground.yaml").read_text()) or {}
    if foreground.get("foreground") != []:
        fail("gpu_foreground.yaml is not empty")


def check_packet(path: Path) -> None:
    if not path.exists():
        fail(f"missing packet: {path}")
    if path.stat().st_size > 25 * 1024 * 1024:
        fail(f"packet too large: {path.stat().st_size}")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    required = {
        "README.md",
        "openreview_latentwire.pdf",
        "openreview_channel_set.pdf",
        "paper/latentwire/main.tex",
        "paper/channel_set/main.tex",
        "paper/references.bib",
        "paper/colm2026_conference.sty",
        "paper/colm2026_conference.bst",
    }
    missing = sorted(required - set(names))
    if missing:
        fail(f"final packet missing required files: {missing}")
    bad = [name for name in names if re.search(r"_confirm", name, re.I)]
    if bad:
        fail(f"final packet contains forbidden split-looking paths: {bad[:10]}")
    banned_zips = [name for name in names if name.endswith(".zip")]
    if banned_zips:
        fail(f"final packet contains nested zip(s): {banned_zips}")
    print(f"{path.relative_to(ROOT)}: files={len(names)}, bytes={path.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=ROOT / "final_submission_packet.zip")
    args = parser.parse_args()
    for path in PDFS:
        check_pdf(path)
    check_logs()
    check_queue()
    packet = args.packet if args.packet.is_absolute() else ROOT / args.packet
    check_packet(packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
