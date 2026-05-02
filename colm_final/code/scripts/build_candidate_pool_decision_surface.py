#!/usr/bin/env python3
"""Build a candidate-pool decision surface from an existing target set."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from copy import deepcopy
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


def _parse_candidate_spec(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in raw.split(","):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value in {raw!r}: {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    required = {"label", "path", "method"}
    missing = sorted(required - set(fields))
    if missing:
        raise argparse.ArgumentTypeError(f"Missing {missing} in extra candidate spec {raw!r}")
    return {
        "label": fields["label"],
        "path": _display_path(_resolve(fields["path"])),
        "method": fields["method"],
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    ids = payload["ids"]["clean_source_only"]
    lines = [
        "# Candidate Pool Decision Surface",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- base target set: `{payload['base_target_set']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- clean decision IDs: `{len(ids)}`",
        "",
        "## Clean IDs",
        "",
        *(f"- `{example_id}`" for example_id in ids),
        "",
        "## Extra Candidate Labels",
        "",
        *(
            f"- `{item['label']}` from `{item['path']}` method `{item['method']}`"
            for item in payload["candidate_pool_decision"]["extra_candidates"]
        ),
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_surface(
    *,
    base_target_set: pathlib.Path,
    extra_candidates: Sequence[dict[str, str]],
    clean_ids: Sequence[str],
    run_date: str,
    command: str,
) -> dict[str, Any]:
    base = _read_json(base_target_set)
    payload = deepcopy(base)
    clean = sorted({str(example_id) for example_id in clean_ids})
    reference_ids = {str(example_id) for example_id in payload.get("reference_ids", [])}
    missing = sorted(set(clean) - reference_ids)
    if missing:
        raise ValueError(f"Clean IDs are not in the base reference set: {missing}")

    artifacts = payload.setdefault("artifacts", {})
    baselines = list(artifacts.get("baselines", []))
    seen_labels = {str(item.get("label")) for item in baselines}
    for candidate in extra_candidates:
        label = str(candidate["label"])
        if label in seen_labels:
            raise ValueError(f"Duplicate baseline/extra candidate label: {label}")
        baselines.append(dict(candidate))
        seen_labels.add(label)
    artifacts["baselines"] = baselines

    ids = payload.setdefault("ids", {})
    ids["clean_source_only"] = clean
    ids["clean_residual_targets"] = clean
    ids["source_only"] = sorted(set(ids.get("source_only", [])) | set(clean))
    rows_by_id = {str(row.get("example_id")): row for row in payload.get("rows", [])}
    for example_id in clean:
        row = rows_by_id.get(example_id)
        if row is None:
            continue
        labels = set(row.get("labels", []))
        labels.add("source_only")
        labels.add("clean_source_only")
        row["labels"] = sorted(labels)

    counts = payload.setdefault("counts", {})
    counts["clean_source_only"] = len(clean)
    payload["date"] = run_date
    payload["status"] = "candidate_pool_decision_surface_ready"
    payload["base_target_set"] = _display_path(base_target_set)
    payload["git_commit"] = _git_commit()
    payload["command"] = command
    payload["candidate_pool_decision"] = {
        "clean_id_source": "explicit",
        "candidate_pool": "base_target_set_plus_extra_candidate_rows",
        "extra_candidates": list(extra_candidates),
    }
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-target-set", required=True)
    parser.add_argument("--extra-candidate", action="append", type=_parse_candidate_spec, default=[])
    parser.add_argument("--clean-id", action="append", default=[])
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_argv = sys.argv if argv is None else ["scripts/build_candidate_pool_decision_surface.py", *argv]

    payload = build_surface(
        base_target_set=_resolve(args.base_target_set),
        extra_candidates=args.extra_candidate,
        clean_ids=args.clean_id,
        run_date=str(args.date),
        command=shlex.join(raw_argv),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "clean_ids": payload["ids"]["clean_source_only"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
