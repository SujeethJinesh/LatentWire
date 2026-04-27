#!/usr/bin/env python3
"""Convert local reference PDFs into markdown text extracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _slug(path: pathlib.Path) -> str:
    rel = path.relative_to(ROOT / "references")
    stem = "__".join(rel.with_suffix("").parts)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return f"{stem}.md"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pdf(path: pathlib.Path) -> tuple[str, int, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised manually.
        raise SystemExit(
            "pypdf is required. Install into the repo venv with "
            "`./venv_arm64/bin/python -m pip install pypdf`."
        ) from exc

    reader = PdfReader(str(path))
    chunks: list[str] = []
    errors: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - keep conversion resilient.
            text = ""
            errors.append(f"page {index}: {exc}")
        chunks.append(f"\n\n<!-- page {index} -->\n\n{text}")
    return "".join(chunks).strip(), len(reader.pages), errors


def _clean_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.replace("\t", "    ").rstrip() for line in lines).strip()


def convert(*, references_dir: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    pdfs = sorted(
        path
        for path in references_dir.rglob("*.pdf")
        if output_dir not in path.parents
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for pdf in pdfs:
        out = output_dir / _slug(pdf)
        try:
            text, pages, errors = _extract_pdf(pdf)
            status = "ok"
        except UnicodeEncodeError:
            text, pages, errors = _extract_pdf(pdf)
            text = text.encode("utf-8", "replace").decode("utf-8")
            status = "ok_replaced_surrogates"
        except Exception as exc:  # noqa: BLE001 - record failed PDFs and continue.
            text, pages, errors = "", 0, [str(exc)]
            status = "failed"
        text = _clean_text(text)
        if status != "failed":
            out.write_text(
                f"# {_display_path(pdf)}\n\n{text}\n",
                encoding="utf-8",
                errors="replace",
            )
        entries.append(
            {
                "source_pdf": _display_path(pdf),
                "source_sha256": _sha256(pdf),
                "output_md": _display_path(out),
                "pages": pages,
                "chars": len(text),
                "errors": errors,
                "status": status,
            }
        )
    manifest = {"files": entries}
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references-dir", default="references")
    parser.add_argument("--output-dir", default="references/pdf_markdown")
    args = parser.parse_args()

    manifest = convert(
        references_dir=(ROOT / args.references_dir).resolve(),
        output_dir=(ROOT / args.output_dir).resolve(),
    )
    statuses: dict[str, int] = {}
    for entry in manifest["files"]:
        statuses[str(entry["status"])] = statuses.get(str(entry["status"]), 0) + 1
    print(
        json.dumps(
            {
                "converted": len(manifest["files"]),
                "statuses": dict(sorted(statuses.items())),
                "manifest": args.output_dir.rstrip("/") + "/conversion_manifest.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
