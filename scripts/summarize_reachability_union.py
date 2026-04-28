#!/usr/bin/env python3
"""Summarize the union of candidate-pool reachability audits."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_spec(raw: str) -> tuple[str, pathlib.Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected label=path: {raw!r}")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"Expected non-empty label and path: {raw!r}")
    return label, _resolve(path)


def _ids(payload: dict[str, Any], key: str) -> set[str]:
    return {str(item) for item in payload.get(key, [])}


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Reachability Union Summary",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- reference rows: `{payload['reference_n']}`",
        "",
        "## Inputs",
        "",
    ]
    for item in payload["inputs"]:
        lines.append(f"- `{item['label']}`: `{item['path']}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- union oracle: `{payload['sample_oracle_correct']}/{payload['reference_n']}`",
            f"- C2C clean residual in union: `{payload['c2c_clean_residual_in_pool']}/{payload['c2c_clean_residual_total']}`",
            f"- C2C teacher-only in union: `{payload['c2c_teacher_only_in_pool']}/{payload['c2c_teacher_only_total']}`",
            f"- source-contrastive clean in union: `{payload['source_contrastive_clean_in_pool']}/{payload['source_contrastive_clean_total']}`",
            "",
            "## IDs",
            "",
            "- oracle IDs: " + (", ".join(f"`{item}`" for item in payload["sample_oracle_ids"]) or "none"),
            "- C2C clean residual IDs: "
            + (", ".join(f"`{item}`" for item in payload["c2c_clean_residual_in_pool_ids"]) or "none"),
            "- C2C teacher-only IDs: "
            + (", ".join(f"`{item}`" for item in payload["c2c_teacher_only_in_pool_ids"]) or "none"),
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(specs: Sequence[tuple[str, pathlib.Path]], *, run_date: str, command: str) -> dict[str, Any]:
    payloads = [(label, path, _read_json(path)) for label, path in specs]
    if not payloads:
        raise ValueError("At least one reachability input is required")
    reference_n = int(payloads[0][2].get("reference_n", 0))
    c2c_total = int(payloads[0][2].get("c2c_clean_residual_total", 0))
    teacher_total = int(payloads[0][2].get("c2c_teacher_only_total", 0))
    source_clean_total = int(payloads[0][2].get("source_contrastive_clean_total", 0))
    oracle = set().union(*(_ids(payload, "sample_oracle_ids") for _, _, payload in payloads))
    c2c_clean = set().union(*(_ids(payload, "c2c_clean_residual_in_pool_ids") for _, _, payload in payloads))
    teacher = set().union(*(_ids(payload, "c2c_teacher_only_in_pool_ids") for _, _, payload in payloads))
    source_clean = set().union(*(_ids(payload, "source_contrastive_clean_in_pool_ids") for _, _, payload in payloads))
    return {
        "date": run_date,
        "status": "reachability_union_summarized",
        "command": command,
        "git_commit": _git_commit(),
        "inputs": [
            {
                "label": label,
                "path": _display_path(path),
                "sample_oracle_correct": int(payload.get("sample_oracle_correct", 0)),
                "c2c_clean_residual_in_pool": int(payload.get("c2c_clean_residual_in_pool", 0)),
            }
            for label, path, payload in payloads
        ],
        "reference_n": reference_n,
        "sample_oracle_correct": len(oracle),
        "sample_oracle_ids": sorted(oracle),
        "c2c_clean_residual_in_pool": len(c2c_clean),
        "c2c_clean_residual_total": c2c_total,
        "c2c_clean_residual_in_pool_ids": sorted(c2c_clean),
        "c2c_teacher_only_in_pool": len(teacher),
        "c2c_teacher_only_total": teacher_total,
        "c2c_teacher_only_in_pool_ids": sorted(teacher),
        "source_contrastive_clean_in_pool": len(source_clean),
        "source_contrastive_clean_total": source_clean_total,
        "source_contrastive_clean_in_pool_ids": sorted(source_clean),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachability", action="append", type=_parse_spec, required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_argv = sys.argv if argv is None else ["scripts/summarize_reachability_union.py", *argv]
    payload = summarize(args.reachability, run_date=str(args.date), command=shlex.join(raw_argv))
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "oracle": payload["sample_oracle_correct"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
