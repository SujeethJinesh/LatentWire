#!/usr/bin/env python3
"""Audit embedded confirm-looking source paths in screening artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard/confirm_path_audit.md"
CONFIRM_RE = re.compile(r"confirm|confirmation", re.I)
SCAN_ROOTS = [ROOT / "results", ROOT / "dashboard"]
TEXT_EXTS = {".json", ".jsonl", ".csv", ".md"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def iter_text_files() -> list[Path]:
    paths = []
    for root in SCAN_ROOTS:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in TEXT_EXTS and "_confirm" not in rel(path):
                    paths.append(path)
    return sorted(paths)


def record_hit(hits: list[dict], file_path: Path, line_no: int | str, source_path: str, context: str) -> None:
    if CONFIRM_RE.search(source_path):
        hits.append(
            {
                "file": rel(file_path),
                "line": str(line_no),
                "source_path": source_path,
                "context": context[:240],
            }
        )


def scan_json_line(hits: list[dict], path: Path, line_no: int, payload: object) -> None:
    if isinstance(payload, dict):
        value = payload.get("source_path")
        if isinstance(value, str):
            record_hit(hits, path, line_no, value, json.dumps(payload, sort_keys=True))
        for nested in payload.values():
            scan_json_line(hits, path, line_no, nested)
    elif isinstance(payload, list):
        for item in payload:
            scan_json_line(hits, path, line_no, item)


def scan_file(path: Path) -> list[dict]:
    hits: list[dict] = []
    if path.suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader, start=2):
                    for key, value in row.items():
                        if key == "source_path" and isinstance(value, str):
                            record_hit(hits, path, index, value, str(row))
        except UnicodeDecodeError:
            return hits
        return hits
    if path.suffix in {".json", ".jsonl"}:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if "source_path" not in line or not CONFIRM_RE.search(line):
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        record_hit(hits, path, line_no, line.strip(), line.strip())
                    else:
                        scan_json_line(hits, path, line_no, payload)
        except UnicodeDecodeError:
            return hits
        return hits
    if path.suffix == ".md":
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if "source_path" in line and CONFIRM_RE.search(line):
                    record_hit(hits, path, line_no, line.strip(), line.strip())
        except OSError:
            return hits
    return hits


def classify_hits(hits: list[dict]) -> tuple[list[dict], list[dict]]:
    contaminated = []
    historical = []
    for hit in hits:
        source = hit["source_path"]
        if "experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_" in source:
            contaminated.append(hit | {"verdict": "CONTAMINATED_C_A1_GATE_SOURCE"})
        elif "experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_" in source:
            contaminated.append(hit | {"verdict": "CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE"})
        else:
            historical.append(hit | {"verdict": "HISTORICAL_TEXT_OR_NON_CLAIM_CONTEXT"})
    return contaminated, historical


def main() -> int:
    hits: list[dict] = []
    for path in iter_text_files():
        hits.extend(scan_file(path))
    contaminated, historical = classify_hits(hits)
    c_a1_contaminated = [
        hit for hit in contaminated
        if "om_driftrot_granite_clip_tight_confirmation_" in hit["source_path"]
        and "leaderboard.csv" in hit["file"]
    ]
    lines = [
        "# Confirm Path Audit",
        "",
        "AUDIT_STATUS: COMPLETE",
        f"C_A1_GATE_STATUS: {'INVALID_CONFIRM_CONTAMINATED' if c_a1_contaminated else 'NO_CONFIRM_SOURCE_FOUND'}",
        "",
        "## Summary",
        "",
        f"- Embedded confirm-looking `source_path` hits: `{len(hits)}`",
        f"- Contaminated screening sources: `{len(contaminated)}`",
        f"- Historical/text/non-claim contexts: `{len(historical)}`",
        "- Raw `*_confirm*` paths remain excluded from review packets.",
        "",
        "## C_A1 Gate Verdict",
        "",
    ]
    if c_a1_contaminated:
        lines.extend(
            [
                "C_A1 cached gate evidence is **invalid for GPU handoff selection** because at least one gate/source row references `om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`.",
                "",
                "Required action: rebuild the C_A1 backfill packet on fresh non-confirm dev/gate row IDs before any GPU spend. Do not replay the contaminated cached Granite gate IDs.",
            ]
        )
    else:
        lines.append("No C_A1 confirmation-derived source rows were found.")
    lines.extend(["", "## Contaminated Hits", ""])
    if contaminated:
        lines.append("| file | line | verdict | source_path |")
        lines.append("| --- | ---: | --- | --- |")
        for hit in contaminated:
            lines.append(f"| `{hit['file']}` | {hit['line']} | `{hit['verdict']}` | `{hit['source_path']}` |")
    else:
        lines.append("- None")
    lines.extend(["", "## Historical / Non-Claim Hits", ""])
    if historical:
        lines.append("| file | line | verdict | source_path |")
        lines.append("| --- | ---: | --- | --- |")
        for hit in historical[:80]:
            lines.append(f"| `{hit['file']}` | {hit['line']} | `{hit['verdict']}` | `{hit['source_path']}` |")
        if len(historical) > 80:
            lines.append(f"| ... | ... | ... | `{len(historical) - 80} additional hits omitted from markdown table` |")
    else:
        lines.append("- None")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(OUT)} with {len(hits)} hit(s)")
    if c_a1_contaminated:
        print("C_A1 gate status: INVALID_CONFIRM_CONTAMINATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
