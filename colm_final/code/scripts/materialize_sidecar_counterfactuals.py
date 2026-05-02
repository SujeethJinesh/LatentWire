#!/usr/bin/env python3
"""Materialize counterfactual candidate-score sidecars for control gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import date
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


MODES = ("source_answer_masked", "source_final_only")


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


def _source_answer_values(profile: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    if profile["final"] is not None:
        values.add(str(profile["final"]))
    values |= {str(value) for value in profile["verified"]}
    return values


def transform_scores(
    scores: Iterable[dict[str, Any]],
    *,
    profile: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    answer_values = _source_answer_values(profile)
    final = str(profile["final"]) if profile["final"] is not None else None
    transformed: list[dict[str, Any]] = []
    for item in scores:
        out = dict(item)
        value = str(out.get("value"))
        if mode == "source_answer_masked":
            if value in answer_values:
                out["score"] = 0.0
        elif mode == "source_final_only":
            out["score"] = 3.0 if final is not None and value == final else 0.0
        transformed.append(out)
    transformed.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("label", "")), str(item.get("value", ""))))
    return transformed


def materialize(
    *,
    target_set_path: pathlib.Path,
    sidecar_path: pathlib.Path,
    mode: str,
    output_jsonl: pathlib.Path,
) -> dict[str, Any]:
    surface = decoder._load_surface("surface", target_set_path)
    profiles = {
        example_id: decoder._source_profile(surface.records_by_label.get("source", {}).get(example_id))
        for example_id in surface.reference_ids
    }
    expected = set(surface.reference_ids)
    observed: set[str] = set()
    masked_answer_hits = 0
    final_only_hits = 0
    rows = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as out_handle:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            observed.add(example_id)
            if example_id not in profiles:
                raise ValueError(f"Sidecar example ID not in target set: {example_id}")
            profile = profiles[example_id]
            transformed = transform_scores(row.get("candidate_scores") or [], profile=profile, mode=mode)
            answer_values = _source_answer_values(profile)
            if mode == "source_answer_masked" and any(str(item.get("value")) in answer_values for item in row.get("candidate_scores") or []):
                masked_answer_hits += 1
            if mode == "source_final_only" and profile["final"] is not None and any(
                str(item.get("value")) == str(profile["final"]) for item in row.get("candidate_scores") or []
            ):
                final_only_hits += 1
            top_score = float(transformed[0].get("score", 0.0)) if transformed else 0.0
            second_score = float(transformed[1].get("score", top_score)) if len(transformed) > 1 else top_score
            out_row = dict(row)
            out_row["candidate_scores"] = transformed
            out_row["confidence"] = float(top_score - second_score)
            out_row["counterfactual_mode"] = mode
            out_handle.write(json.dumps(out_row, sort_keys=True) + "\n")
            rows += 1
    if observed != expected:
        raise ValueError(f"Sidecar IDs mismatch: missing={sorted(expected - observed)} extra={sorted(observed - expected)}")
    return {
        "mode": mode,
        "n": rows,
        "target_set": _display_path(target_set_path),
        "target_set_sha256": _sha256(target_set_path),
        "input_sidecar_jsonl": _display_path(sidecar_path),
        "input_sidecar_jsonl_sha256": _sha256(sidecar_path),
        "output_jsonl": _display_path(output_jsonl),
        "output_jsonl_sha256": _sha256(output_jsonl),
        "masked_answer_hits": masked_answer_hits,
        "final_only_hits": final_only_hits,
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    result = payload["result"]
    lines = [
        "# Sidecar Counterfactual",
        "",
        f"- date: `{payload['date']}`",
        f"- mode: `{result['mode']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- n: `{result['n']}`",
        f"- output: `{result['output_jsonl']}`",
        f"- output sha256: `{result['output_jsonl_sha256']}`",
        f"- masked answer hits: `{result['masked_answer_hits']}`",
        f"- final-only hits: `{result['final_only_hits']}`",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-set", required=True)
    parser.add_argument("--sidecar-jsonl", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/materialize_sidecar_counterfactuals.py", *argv]
    result = materialize(
        target_set_path=_resolve(args.target_set),
        sidecar_path=_resolve(args.sidecar_jsonl),
        mode=str(args.mode),
        output_jsonl=_resolve(args.output_jsonl),
    )
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
    print(json.dumps({"mode": result["mode"], "n": result["n"], "output": result["output_jsonl"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
