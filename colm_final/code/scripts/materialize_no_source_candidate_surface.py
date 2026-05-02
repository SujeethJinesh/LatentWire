#!/usr/bin/env python3
"""Materialize a no-source candidate surface from existing repair artifacts.

The emitted target-set keeps the real source row only for source/headroom
accounting. Candidate labels are target-side or source-destroyed artifacts:
target, target self-repair, selected-route controls, process repair controls,
and optionally expanded `candidate_scores` entries from a no-source pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_contrastive_target_set as target_set


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    path: pathlib.Path
    method: str


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
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


def _parse_spec(raw: str) -> CandidateSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected label=path=...,method=...: {raw!r}")
    label, rest = raw.split("=", 1)
    fields: dict[str, str] = {}
    for item in rest.split(","):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value in {raw!r}: {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    if not label or not fields.get("path") or not fields.get("method"):
        raise argparse.ArgumentTypeError(f"Spec needs label, path, and method: {raw!r}")
    return CandidateSpec(label=label, path=_resolve(fields["path"]), method=fields["method"])


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _candidate_value(item: dict[str, Any]) -> str:
    for key in ("normalized_prediction", "tail_numeric_mention", "value", "prediction"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _copy_candidate_rows(spec: CandidateSpec, reference_ids: Sequence[str]) -> tuple[pathlib.Path, dict[str, Any]]:
    rows = target_set._records_for_method(
        target_set.RowSpec(spec.label, spec.path, spec.method)
    )
    ordered = target_set._ordered_subset(rows, list(reference_ids))
    return spec.path, {
        "label": spec.label,
        "path": _display_path(spec.path),
        "method": spec.method,
        "rows": len(ordered),
        "correct": len(target_set._correct_ids(ordered)),
        "sha256": _sha256_file(spec.path),
        "expanded_from_candidate_scores": False,
    }


def _expand_candidate_score_rows(
    *,
    spec: CandidateSpec,
    reference_ids: Sequence[str],
    output_dir: pathlib.Path,
    include_sources: set[str] | None,
    exclude_sources: set[str],
) -> tuple[list[CandidateSpec], list[dict[str, Any]]]:
    rows = target_set._records_for_method(
        target_set.RowSpec(spec.label, spec.path, spec.method)
    )
    ordered = target_set._ordered_subset(rows, list(reference_ids))
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        example_id = str(row["example_id"])
        for item in row.get("candidate_scores") or []:
            source = str(item.get("source") or item.get("label") or "")
            if not source:
                continue
            if include_sources is not None and source not in include_sources:
                continue
            if source in exclude_sources:
                continue
            label = f"{spec.label}_{source}".replace("-", "_")
            value = _candidate_value(item)
            buckets.setdefault(label, []).append(
                {
                    "example_id": example_id,
                    "index": row.get("index"),
                    "method": label,
                    "prediction": f"candidate answer: {value}",
                    "normalized_prediction": value,
                    "correct": bool(item.get("correct")),
                    "candidate_source": source,
                    "candidate_surface_source": spec.label,
                    "candidate_answer_agreement": item.get("answer_agreement"),
                    "candidate_format_score": item.get("format_score"),
                    "candidate_numeric_consistency_score": item.get("numeric_consistency_score"),
                    "candidate_completion_score": item.get("completion_score"),
                    "candidate_tail_numeric_mention": item.get("tail_numeric_mention"),
                }
            )

    specs: list[CandidateSpec] = []
    summaries: list[dict[str, Any]] = []
    for label, bucket_rows in sorted(buckets.items()):
        if len(bucket_rows) != len(reference_ids):
            raise ValueError(
                f"Expanded label {label!r} has {len(bucket_rows)} rows, expected {len(reference_ids)}"
            )
        path = output_dir / f"{label}.jsonl"
        _write_jsonl(path, bucket_rows)
        specs.append(CandidateSpec(label=label, path=path, method=label))
        summaries.append(
            {
                "label": label,
                "path": _display_path(path),
                "method": label,
                "rows": len(bucket_rows),
                "correct": len(target_set._correct_ids(bucket_rows)),
                "sha256": _sha256_file(path),
                "expanded_from_candidate_scores": True,
                "source_artifact": _display_path(spec.path),
                "source_method": spec.method,
            }
        )
    return specs, summaries


def _write_manifest(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# No-Source Candidate Surface",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- target set: `{payload['outputs']['target_set_json']}`",
        f"- target-set SHA256: `{payload['outputs']['target_set_sha256']}`",
        "",
        "## Candidate Rows",
        "",
        "| Label | Rows | Correct | Expanded | Path | SHA256 |",
        "|---|---:|---:|---|---|---|",
    ]
    for item in payload["candidate_summaries"]:
        lines.append(
            "| `{label}` | {rows} | {correct} | `{expanded}` | `{path}` | `{sha}` |".format(
                label=item["label"],
                rows=item["rows"],
                correct=item["correct"],
                expanded=item["expanded_from_candidate_scores"],
                path=item["path"],
                sha=item["sha256"],
            )
        )
    lines.extend(
        [
            "",
            "## Surface Counts",
            "",
            f"- target correct: `{payload['target_set_counts']['target_correct']}/{payload['reference_n']}`",
            f"- source correct: `{payload['target_set_counts']['source_correct']}/{payload['reference_n']}`",
            f"- clean source-only after no-source baselines: `{payload['target_set_counts']['clean_source_only']}`",
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


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-target-set", required=True)
    parser.add_argument("--candidate", action="append", type=_parse_spec, default=[])
    parser.add_argument("--expand-candidate-scores", action="append", type=_parse_spec, default=[])
    parser.add_argument("--include-candidate-source", action="append", default=[])
    parser.add_argument("--exclude-candidate-source", action="append", default=["target"])
    parser.add_argument("--min-source-only", type=int, default=0)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_argv = sys.argv if argv is None else ["scripts/materialize_no_source_candidate_surface.py", *argv]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_target_set = _read_json(_resolve(args.base_target_set))
    reference_ids = [str(example_id) for example_id in base_target_set["reference_ids"]]
    artifacts = base_target_set["artifacts"]
    target_spec = target_set.RowSpec(
        str(artifacts["target"]["label"]),
        _resolve(str(artifacts["target"]["path"])),
        str(artifacts["target"]["method"]),
    )
    source_spec = target_set.RowSpec(
        str(artifacts["source"]["label"]),
        _resolve(str(artifacts["source"]["path"])),
        str(artifacts["source"]["method"]),
    )
    baseline_specs = [
        target_set.RowSpec(str(item["label"]), _resolve(str(item["path"])), str(item["method"]))
        for item in artifacts.get("baselines", [])
    ]

    candidate_summaries: list[dict[str, Any]] = []
    for spec in args.candidate:
        _, summary = _copy_candidate_rows(spec, reference_ids)
        baseline_specs.append(target_set.RowSpec(spec.label, spec.path, spec.method))
        candidate_summaries.append(summary)

    include_sources = set(args.include_candidate_source) if args.include_candidate_source else None
    exclude_sources = set(args.exclude_candidate_source)
    for spec in args.expand_candidate_scores:
        expanded_specs, summaries = _expand_candidate_score_rows(
            spec=spec,
            reference_ids=reference_ids,
            output_dir=output_dir,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
        )
        baseline_specs.extend(
            target_set.RowSpec(item.label, item.path, item.method)
            for item in expanded_specs
        )
        candidate_summaries.extend(summaries)

    payload = target_set.build_target_set(
        target_spec=target_spec,
        source_spec=source_spec,
        control_specs=[],
        baseline_specs=baseline_specs,
        min_source_only=int(args.min_source_only),
        run_date=str(args.date),
    )
    payload["surface_kind"] = "no_source_candidate_surface"
    payload["base_target_set"] = _display_path(_resolve(args.base_target_set))
    target_set_json = output_dir / "source_contrastive_target_set.json"
    target_set_md = output_dir / "source_contrastive_target_set.md"
    target_set_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target_set._write_markdown(target_set_md, payload)

    manifest = {
        "date": str(args.date),
        "status": "no_source_candidate_surface_materialized",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "base_target_set": _display_path(_resolve(args.base_target_set)),
        "reference_n": len(reference_ids),
        "candidate_summaries": candidate_summaries,
        "target_set_counts": payload["counts"],
        "outputs": {
            "target_set_json": _display_path(target_set_json),
            "target_set_md": _display_path(target_set_md),
            "target_set_sha256": _sha256_file(target_set_json),
        },
    }
    manifest_json = output_dir / "manifest.json"
    manifest_md = output_dir / "manifest.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest(manifest_md, manifest)
    print(json.dumps({"status": manifest["status"], "output_json": _display_path(target_set_json)}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
