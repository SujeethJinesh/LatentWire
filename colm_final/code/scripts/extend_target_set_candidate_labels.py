#!/usr/bin/env python3
"""Append candidate labels to a target-set JSON without recomputing clean IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_contrastive_target_set as target_set


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_spec(raw: str) -> target_set.RowSpec:
    parsed = target_set._parse_spec(raw)
    return target_set.RowSpec(parsed.label, parsed.path, parsed.method)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _selected_ids(payload: dict[str, Any], fields: Sequence[str]) -> list[str]:
    values: list[str] = []
    ids = payload.get("ids", {})
    for field in fields:
        values.extend(str(value) for value in ids.get(field, []))
    return _ordered_unique(values)


def _ids_from_file(path: pathlib.Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values.append(stripped)
    return _ordered_unique(values)


def _filter_payload(payload: dict[str, Any], selected_ids: Sequence[str]) -> dict[str, Any]:
    selected = set(selected_ids)
    out = json.loads(json.dumps(payload))
    out["reference_ids"] = [example_id for example_id in payload["reference_ids"] if example_id in selected]
    out["reference_n"] = len(out["reference_ids"])
    out["rows"] = [
        row for row in payload.get("rows", []) if str(row.get("example_id")) in selected
    ]
    out["ids"] = {
        key: [str(value) for value in values if str(value) in selected]
        for key, values in payload.get("ids", {}).items()
    }
    out["extension_preserves_clean_ids"] = True
    return out


def _candidate_summary(spec: target_set.RowSpec, reference_ids: list[str]) -> dict[str, Any]:
    rows = target_set._ordered_subset(target_set._records_for_method(spec), reference_ids)
    return {
        "label": spec.label,
        "path": _display_path(spec.path),
        "method": spec.method,
        "n": len(rows),
        "correct": len(target_set._correct_ids(rows)),
        "numeric_coverage": target_set._numeric_coverage(rows),
        "sha256": _sha256(spec.path),
    }


def _write_md(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Extended Target Candidate Set",
        "",
        f"- date: `{manifest['date']}`",
        f"- status: `{manifest['status']}`",
        f"- git commit: `{manifest.get('git_commit') or 'unknown'}`",
        f"- base target set: `{manifest['base_target_set']}`",
        f"- output target set: `{manifest['output_json']}`",
        f"- selected IDs: `{manifest['reference_n']}`",
        "",
        "| Label | Correct | Numeric Coverage | Path |",
        "|---|---:|---:|---|",
    ]
    for row in manifest["candidate_summaries"]:
        lines.append(
            f"| `{row['label']}` | {row['correct']}/{row['n']} | "
            f"{row['numeric_coverage']}/{row['n']} | `{row['path']}` |"
        )
    lines.extend(["", "## Command", "", "```bash", manifest["command"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-target-set", required=True)
    parser.add_argument("--id-fields", nargs="+")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--ids-file")
    parser.add_argument(
        "--override-clean-residual-ids",
        nargs="+",
        help="Replace ids.clean_residual_targets after filtering; values must be in the selected ID set.",
    )
    parser.add_argument("--candidate", action="append", type=_parse_spec, default=[])
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--manifest-json")
    args = parser.parse_args(argv)
    id_sources = [bool(args.id_fields), bool(args.ids), bool(args.ids_file)]
    if sum(int(source) for source in id_sources) != 1:
        parser.error("provide exactly one of --id-fields, --ids, or --ids-file")
    raw_argv = sys.argv if argv is None else ["scripts/extend_target_set_candidate_labels.py", *argv]

    base_path = _resolve(args.base_target_set)
    payload = _read_json(base_path)
    if args.id_fields:
        selected = _selected_ids(payload, args.id_fields)
        id_source: dict[str, Any] = {"type": "id_fields", "id_fields": list(args.id_fields)}
    elif args.ids_file:
        ids_file = _resolve(args.ids_file)
        selected = _ids_from_file(ids_file)
        id_source = {
            "type": "ids_file",
            "ids_file": _display_path(ids_file),
            "ids_file_sha256": _sha256(ids_file),
        }
    else:
        selected = _ordered_unique([str(value) for value in args.ids])
        id_source = {"type": "ids"}
    if not selected:
        raise ValueError(f"No IDs selected from {base_path} via {id_source['type']}")
    out_payload = _filter_payload(payload, selected)
    if args.override_clean_residual_ids:
        override_clean = _ordered_unique([str(value) for value in args.override_clean_residual_ids])
        selected_set = set(out_payload["reference_ids"])
        missing_override = [example_id for example_id in override_clean if example_id not in selected_set]
        if missing_override:
            raise ValueError(f"override clean residual IDs are not selected: {missing_override}")
        out_payload.setdefault("ids", {})["clean_residual_targets"] = override_clean
        for row in out_payload.get("rows", []):
            labels = list(row.get("labels", []))
            if str(row.get("example_id")) in override_clean and "clean_residual_target_override" not in labels:
                labels.append("clean_residual_target_override")
            row["labels"] = labels
    baselines = list(out_payload.setdefault("artifacts", {}).setdefault("baselines", []))
    candidate_summaries: list[dict[str, Any]] = []
    for spec in args.candidate:
        candidate_summaries.append(_candidate_summary(spec, out_payload["reference_ids"]))
        baselines.append({"label": spec.label, "path": _display_path(spec.path), "method": spec.method})
    out_payload["artifacts"]["baselines"] = baselines
    out_payload["candidate_extension"] = {
        "base_target_set": _display_path(base_path),
        "id_source": id_source,
        "override_clean_residual_ids": list(args.override_clean_residual_ids or []),
        "candidate_labels": [spec.label for spec in args.candidate],
    }

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "date": str(args.date),
        "status": "target_set_candidate_labels_extended",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "base_target_set": _display_path(base_path),
        "base_target_set_sha256": _sha256(base_path),
        "id_source": id_source,
        "output_json": _display_path(output_json),
        "output_json_sha256": _sha256(output_json),
        "output_md": _display_path(output_md),
        "reference_n": len(out_payload["reference_ids"]),
        "selected_ids": out_payload["reference_ids"],
        "candidate_summaries": candidate_summaries,
    }
    _write_md(output_md, manifest)
    manifest_path = _resolve(args.manifest_json) if args.manifest_json else output_json.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_json": manifest["output_json"]}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
