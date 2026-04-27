#!/usr/bin/env python3
"""Rank existing target-set surfaces by answer-masking risk."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


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


def _candidate_paths(results_root: pathlib.Path) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in results_root.rglob("*.json"):
        if ".debug" in path.parts:
            continue
        name = path.name
        if "manifest" in name:
            continue
        if "target_set" in name or name == "source_contrastive_target_set.json":
            out.append(path)
    return sorted(set(out))


def _clean_ids(surface: decoder.Surface) -> tuple[str, set[str]]:
    for key in ("clean_source_only", "clean_residual_targets", "source_only"):
        ids = decoder._source_ids(surface, key)
        if ids:
            return key, ids
    return "none", set()


def _source_answer_values(profile: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    if profile["final"] is not None:
        values.add(str(profile["final"]))
    values |= {str(value) for value in profile["verified"]}
    return values


def audit_surface(path: pathlib.Path) -> dict[str, Any]:
    surface = decoder._load_surface(path.stem, path)
    clean_field, clean_ids = _clean_ids(surface)
    target_correct = {
        example_id
        for example_id in surface.reference_ids
        if bool(surface.records_by_label["target"][example_id].get("correct"))
    }
    rows: list[dict[str, Any]] = []
    clean_in_pool: list[str] = []
    answer_explained: list[str] = []
    answer_unexplained: list[str] = []
    for example_id in sorted(clean_ids):
        gold = decoder._gold_numeric(surface, example_id)
        candidates = decoder._candidate_pool(surface, example_id)
        pool_values = {candidate.value for candidate in candidates}
        gold_in_pool = gold in pool_values
        profile = decoder._source_profile(surface.records_by_label.get("source", {}).get(example_id))
        answer_values = _source_answer_values(profile)
        explained_by_source_answer = gold in answer_values
        if gold_in_pool:
            clean_in_pool.append(example_id)
            if explained_by_source_answer:
                answer_explained.append(example_id)
            else:
                answer_unexplained.append(example_id)
        rows.append(
            {
                "example_id": example_id,
                "gold": gold,
                "target_correct": example_id in target_correct,
                "gold_in_target_side_pool": gold_in_pool,
                "source_final": profile["final"],
                "source_verified": sorted(profile["verified"]),
                "explained_by_source_answer": explained_by_source_answer,
                "candidate_values": sorted(pool_values),
            }
        )
    return {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "n": len(surface.reference_ids),
        "clean_field": clean_field,
        "clean_n": len(clean_ids),
        "clean_in_pool": len(clean_in_pool),
        "clean_in_pool_ids": clean_in_pool,
        "answer_explained_clean_in_pool": len(answer_explained),
        "answer_explained_clean_in_pool_ids": answer_explained,
        "answer_unexplained_clean_in_pool": len(answer_unexplained),
        "answer_unexplained_clean_in_pool_ids": answer_unexplained,
        "rows": rows,
    }


def run_audit(results_root: pathlib.Path, *, limit: int | None = None) -> dict[str, Any]:
    loaded: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    paths = _candidate_paths(results_root)
    if limit is not None:
        paths = paths[: int(limit)]
    for path in paths:
        try:
            row = audit_surface(path)
        except Exception as exc:  # noqa: BLE001 - audit should report bad artifacts.
            skipped.append({"path": _display_path(path), "error": str(exc)})
            continue
        if row["clean_n"] > 0:
            loaded.append(row)
    loaded.sort(
        key=lambda item: (
            -int(item["answer_unexplained_clean_in_pool"]),
            -int(item["clean_in_pool"]),
            str(item["path"]),
        )
    )
    return {
        "results_root": _display_path(results_root),
        "candidate_paths": len(paths),
        "loaded_with_clean_ids": len(loaded),
        "skipped": skipped,
        "surfaces": loaded,
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    result = payload["result"]
    lines = [
        "# Source-Surface Answer-Masking Audit",
        "",
        f"- date: `{payload['date']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- candidate paths: `{result['candidate_paths']}`",
        f"- loaded surfaces with clean IDs: `{result['loaded_with_clean_ids']}`",
        f"- skipped: `{len(result['skipped'])}`",
        "",
        "| Rank | Surface | N | Clean Field | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(result["surfaces"], start=1):
        lines.append(
            f"| {rank} | `{row['path']}` | {row['n']} | `{row['clean_field']}` | "
            f"{row['clean_n']} | {row['clean_in_pool']} | {row['answer_unexplained_clean_in_pool']} |"
        )
    lines.extend(["", "## Top Surface Details", ""])
    for row in result["surfaces"][:10]:
        lines.extend(
            [
                f"### `{row['path']}`",
                "",
                f"- clean in pool IDs: {', '.join(f'`{x}`' for x in row['clean_in_pool_ids']) or 'none'}",
                (
                    "- answer-unexplained clean in pool IDs: "
                    f"{', '.join(f'`{x}`' for x in row['answer_unexplained_clean_in_pool_ids']) or 'none'}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Decision Rule",
            "",
            "A surface is immediately useful for the next source-sidecar gate only if "
            "`answer_unexplained_clean_in_pool` is nonzero. If all reachable clean "
            "IDs are explained by source final or verified numeric answers, the "
            "surface is a candidate-headroom diagnostic rather than positive-method "
            "evidence.",
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


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/audit_source_surface_answer_masking.py", *argv]
    result = run_audit(_resolve(args.results_root), limit=args.limit)
    payload = {
        "date": str(args.date),
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "result": result,
    }
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(output_md, payload)
    top = result["surfaces"][0] if result["surfaces"] else {}
    print(
        json.dumps(
            {
                "surfaces": result["loaded_with_clean_ids"],
                "top_surface": top.get("path"),
                "top_answer_unexplained_clean_in_pool": top.get("answer_unexplained_clean_in_pool", 0),
            },
            indent=2,
        )
    )
    return payload


if __name__ == "__main__":
    main()
