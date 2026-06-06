#!/usr/bin/env python3
"""Build the internal final submission packet with anonymized COLM assets."""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 25 * 1024 * 1024


def forbidden_path(name: str) -> bool:
    return fnmatch.fnmatch(name, "*_confirm*")


def candidates() -> list[Path]:
    patterns = [
        "openreview_latentwire.pdf",
        "openreview_channel_set.pdf",
        "paper/latentwire/main.tex",
        "paper/latentwire/main.pdf",
        "paper/latentwire/figures/*",
        "paper/channel_set/main.tex",
        "paper/channel_set/main.pdf",
        "paper/channel_set/figures/*",
        "paper/references.bib",
        "paper/colm2026_conference.sty",
        "paper/colm2026_conference.bst",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(set(paths))


def build(out: Path) -> Path:
    entries: dict[str, bytes] = {}
    skipped: list[str] = []
    for path in candidates():
        name = path.relative_to(ROOT).as_posix()
        if forbidden_path(name):
            skipped.append(f"{name}: forbidden split-looking path")
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            skipped.append(f"{name}: skipped {path.stat().st_size} byte file")
            continue
        entries[name] = path.read_bytes()
    readme = [
        "# Final Submission Packet",
        "",
        "Internal convenience packet for OpenReview upload preparation.",
        "Upload only the two anonymized single PDFs to OpenReview, not this zip.",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Included",
    ]
    for name in sorted(entries):
        readme.append(f"- `{name}`")
    readme.extend(["", "## SKIPPED-too-large"])
    readme.extend(f"- {item}" for item in skipped) if skipped else readme.append("- None")
    readme.extend(
        [
            "",
            "## Human Checklist",
            "- Open both PDFs and inspect equations, tables, and figures.",
            "- Upload only the anonymized single PDFs.",
            "- Do not upload internal review or submission zips as supplementary material.",
        ]
    )
    entries["README.md"] = ("\n".join(readme) + "\n").encode("utf-8")
    bad = [name for name in entries if forbidden_path(name)]
    if bad:
        raise SystemExit(f"would include forbidden paths: {bad}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    if out.stat().st_size > MAX_BYTES:
        raise SystemExit(f"package too large: {out.stat().st_size}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "final_submission_packet.zip")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    build(out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
