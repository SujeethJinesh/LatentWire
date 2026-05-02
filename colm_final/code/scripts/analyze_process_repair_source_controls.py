#!/usr/bin/env python3
"""Check whether process-repair gains survive source-destroying controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ControlSpec:
    label: str
    path: pathlib.Path


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else ROOT / p


def _display(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _method_rows(path: pathlib.Path, method: str) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(path) if str(row.get("method")) == method]


def _by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id"))
        if example_id in out:
            raise ValueError(f"duplicate example_id={example_id}")
        out[example_id] = dict(row)
    return out


def _correct_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in rows if bool(row.get("correct"))}


def _parse_control(text: str) -> ControlSpec:
    if "=" not in text:
        raise argparse.ArgumentTypeError("control must be label=path")
    label, path = text.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("control label is empty")
    return ControlSpec(label=label, path=_resolve(path))


def _subset_exact(reference_ids: Sequence[str], rows: Sequence[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    rows_by_id = _by_id(rows)
    missing = [example_id for example_id in reference_ids if example_id not in rows_by_id]
    extra = sorted(set(rows_by_id) - set(reference_ids))
    if missing or extra:
        raise ValueError(
            f"{label} does not match reference IDs: missing={missing[:5]} extra={extra[:5]}"
        )
    return [rows_by_id[example_id] for example_id in reference_ids]


def _summarize_rows(rows: Sequence[dict[str, Any]], *, reference_ids: Sequence[str]) -> dict[str, Any]:
    exact_rows = _subset_exact(reference_ids, rows, label="summary")
    correct = _correct_ids(exact_rows)
    return {
        "n": len(exact_rows),
        "correct_count": len(correct),
        "accuracy": len(correct) / max(len(exact_rows), 1),
        "correct_ids": sorted(correct),
    }


def analyze(
    *,
    matched_path: pathlib.Path,
    controls: Sequence[ControlSpec],
    method: str,
    target_method: str,
    target_self_method: str,
    run_date: str,
) -> dict[str, Any]:
    matched_method_rows = _method_rows(matched_path, method)
    target_rows = _method_rows(matched_path, target_method)
    target_self_rows = _method_rows(matched_path, target_self_method)
    if not matched_method_rows:
        raise ValueError(f"No matched rows for method={method!r}")
    if not target_rows:
        raise ValueError(f"No target rows for method={target_method!r}")
    if not target_self_rows:
        raise ValueError(f"No target-self rows for method={target_self_method!r}")

    reference_ids = [str(row["example_id"]) for row in matched_method_rows]
    target_rows = _subset_exact(reference_ids, target_rows, label=target_method)
    target_self_rows = _subset_exact(reference_ids, target_self_rows, label=target_self_method)

    matched_correct = _correct_ids(matched_method_rows)
    target_correct = _correct_ids(target_rows)
    target_self_correct = _correct_ids(target_self_rows)
    matched_only_vs_target = matched_correct - target_correct
    matched_only_vs_target_self = matched_correct - target_self_correct

    control_summaries: list[dict[str, Any]] = []
    control_correct_union: set[str] = set()
    control_target_only_union: set[str] = set()
    control_target_self_only_union: set[str] = set()
    for spec in controls:
        control_rows = _method_rows(spec.path, method)
        if not control_rows:
            raise ValueError(f"No control rows for method={method!r} in {spec.path}")
        control_rows = _subset_exact(reference_ids, control_rows, label=spec.label)
        control_correct = _correct_ids(control_rows)
        control_correct_union |= control_correct
        control_target_only = control_correct - target_correct
        control_target_self_only = control_correct - target_self_correct
        control_target_only_union |= control_target_only
        control_target_self_only_union |= control_target_self_only
        control_summaries.append(
            {
                "label": spec.label,
                "path": _display(spec.path),
                "sha256": _sha256(spec.path),
                "correct_count": len(control_correct),
                "accuracy": len(control_correct) / max(len(reference_ids), 1),
                "matched_only_vs_target_overlap_ids": sorted(matched_only_vs_target & control_correct),
                "matched_only_vs_target_self_overlap_ids": sorted(
                    matched_only_vs_target_self & control_correct
                ),
                "control_only_vs_target_ids": sorted(control_target_only),
                "control_only_vs_target_self_ids": sorted(control_target_self_only),
            }
        )

    source_specific_vs_target = matched_only_vs_target - control_target_only_union
    source_specific_vs_target_self = matched_only_vs_target_self - control_target_self_only_union
    gate_pass = bool(matched_only_vs_target_self) and len(source_specific_vs_target_self) == len(
        matched_only_vs_target_self
    )

    return {
        "date": run_date,
        "status": (
            "process_repair_source_controls_support_matched_source"
            if gate_pass
            else "process_repair_source_controls_do_not_clear_gate"
        ),
        "config": {
            "method": method,
            "target_method": target_method,
            "target_self_method": target_self_method,
        },
        "artifacts": {
            "matched": {"path": _display(matched_path), "sha256": _sha256(matched_path)},
            "controls": [
                {"label": spec.label, "path": _display(spec.path), "sha256": _sha256(spec.path)}
                for spec in controls
            ],
        },
        "reference_ids": reference_ids,
        "summaries": {
            "matched": _summarize_rows(matched_method_rows, reference_ids=reference_ids),
            "target": _summarize_rows(target_rows, reference_ids=reference_ids),
            "target_self": _summarize_rows(target_self_rows, reference_ids=reference_ids),
        },
        "matched_only_vs_target_ids": sorted(matched_only_vs_target),
        "matched_only_vs_target_self_ids": sorted(matched_only_vs_target_self),
        "control_correct_union_ids": sorted(control_correct_union),
        "control_only_vs_target_union_ids": sorted(control_target_only_union),
        "control_only_vs_target_self_union_ids": sorted(control_target_self_only_union),
        "source_specific_vs_target_ids": sorted(source_specific_vs_target),
        "source_specific_vs_target_self_ids": sorted(source_specific_vs_target_self),
        "control_summaries": control_summaries,
        "gate": {
            "matched_only_vs_target_self_count": len(matched_only_vs_target_self),
            "source_specific_vs_target_self_count": len(source_specific_vs_target_self),
            "required": "all matched-only-vs-target-self IDs must be absent from source-destroying controls",
            "passed": gate_pass,
        },
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summaries = payload["summaries"]
    lines = [
        "# Process Repair Source-Control Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- method: `{payload['config']['method']}`",
        f"- matched correct: `{summaries['matched']['correct_count']}/{summaries['matched']['n']}`",
        f"- target correct: `{summaries['target']['correct_count']}/{summaries['target']['n']}`",
        f"- target-self repair correct: `{summaries['target_self']['correct_count']}/{summaries['target_self']['n']}`",
        f"- matched-only vs target IDs: `{len(payload['matched_only_vs_target_ids'])}`",
        f"- matched-only vs target-self IDs: `{len(payload['matched_only_vs_target_self_ids'])}`",
        f"- source-specific vs target-self after controls: `{len(payload['source_specific_vs_target_self_ids'])}`",
        "",
        "## Controls",
        "",
        "| Control | Correct | Overlap With Matched-Only vs Target-Self | Control-Only vs Target-Self |",
        "|---|---:|---:|---:|",
    ]
    n = summaries["matched"]["n"]
    for row in payload["control_summaries"]:
        lines.append(
            "| {label} | {correct}/{n} | {overlap} | {control_only} |".format(
                label=row["label"],
                correct=row["correct_count"],
                n=n,
                overlap=len(row["matched_only_vs_target_self_overlap_ids"]),
                control_only=len(row["control_only_vs_target_self_ids"]),
            )
        )
    lines.extend(
        [
            "",
            "## Source-Specific IDs",
            "",
            "- matched-only vs target-self: "
            + (", ".join(f"`{x}`" for x in payload["matched_only_vs_target_self_ids"]) or "none"),
            "- retained after controls: "
            + (", ".join(f"`{x}`" for x in payload["source_specific_vs_target_self_ids"]) or "none"),
            "",
            "## Decision",
            "",
            str(payload["gate"]["required"]) + f"; passed: `{payload['gate']['passed']}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", required=True)
    parser.add_argument("--control", action="append", type=_parse_control, required=True)
    parser.add_argument("--method", default="process_repair_selected_route")
    parser.add_argument("--target-method", default="target_alone")
    parser.add_argument("--target-self-method", default="target_self_repair")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = analyze(
        matched_path=_resolve(args.matched),
        controls=list(args.control),
        method=str(args.method),
        target_method=str(args.target_method),
        target_self_method=str(args.target_self_method),
        run_date=str(args.date),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(output_md, payload)
    print(json.dumps({"status": payload["status"], "output_json": _display(output_json)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
