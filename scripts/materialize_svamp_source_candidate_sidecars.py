#!/usr/bin/env python3
"""Materialize source-derived candidate-score sidecars for SVAMP target sets.

The output is intentionally compatible with
`scripts/analyze_svamp_source_semantic_predicate_decoder.py --*-sidecar-jsonl`.
It scores only target-side candidate values; source-only values are never added
to the candidate pool by this producer.
"""

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

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


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


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _candidate_score(
    *,
    candidate: decoder.Candidate,
    profile: dict[str, Any],
    fallback: str | None,
    label_prior: float,
) -> float:
    score = float(label_prior)
    value = candidate.value
    if fallback is not None and value == fallback:
        score += 0.25
    if profile["final"] is not None and value == profile["final"]:
        score += 3.0
    if value in profile["verified"]:
        score += 2.0
    if value in profile["mention_set"]:
        score += 1.0
    if profile["last"] and value == profile["last"][-1]:
        score += 1.0
    if value in profile["last"][-2:]:
        score += 0.5
    if value in profile["pair"]:
        score += 0.5
    return float(score)


def _materialize(
    *,
    target_set_path: pathlib.Path,
    sidecar_bits: int,
    label_prior: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    surface = decoder._load_surface("surface", target_set_path)
    rows: list[dict[str, Any]] = []
    missing_source = 0
    empty_pool = 0
    source_final_in_pool = 0
    source_mentioned_pool_hits = 0
    top_label_counts: dict[str, int] = {}

    for example_id in surface.reference_ids:
        candidates = decoder._candidate_pool(surface, example_id)
        if not candidates:
            empty_pool += 1
        source_row = surface.records_by_label.get("source", {}).get(example_id)
        if source_row is None:
            missing_source += 1
        profile = decoder._source_profile(source_row)
        fallback = decoder._prediction_numeric(surface.records_by_label["target"].get(example_id))
        candidate_scores: list[dict[str, Any]] = []
        for candidate in candidates:
            labels = tuple(label for label in candidate.labels if label != "source")
            label = labels[0] if labels else "target"
            score = _candidate_score(
                candidate=candidate,
                profile=profile,
                fallback=fallback,
                label_prior=label_prior,
            )
            candidate_scores.append(
                {
                    "label": label,
                    "value": candidate.value,
                    "score": score,
                }
            )
        candidate_scores.sort(key=lambda item: (-float(item["score"]), item["label"], item["value"]))
        if profile["final"] is not None and any(item["value"] == profile["final"] for item in candidate_scores):
            source_final_in_pool += 1
        mentioned = set(profile["mention_set"])
        if any(item["value"] in mentioned for item in candidate_scores):
            source_mentioned_pool_hits += 1
        top_label = str(candidate_scores[0]["label"]) if candidate_scores else None
        if top_label is not None:
            top_label_counts[top_label] = top_label_counts.get(top_label, 0) + 1
        top_score = float(candidate_scores[0]["score"]) if candidate_scores else 0.0
        second_score = float(candidate_scores[1]["score"]) if len(candidate_scores) > 1 else top_score
        margin = top_score - second_score
        rows.append(
            {
                "example_id": example_id,
                "candidate_scores": candidate_scores,
                "confidence": float(margin),
                "sidecar_bits": int(sidecar_bits),
                "source_final_in_target_pool": bool(
                    profile["final"] is not None
                    and any(item["value"] == profile["final"] for item in candidate_scores)
                ),
                "source_mentioned_target_pool_hit": bool(
                    any(item["value"] in mentioned for item in candidate_scores)
                ),
            }
        )

    summary = {
        "target_set": _display_path(target_set_path),
        "n": len(surface.reference_ids),
        "sidecar_bits": int(sidecar_bits),
        "sidecar_bytes": max(1, (int(sidecar_bits) + 7) // 8),
        "missing_source": int(missing_source),
        "empty_pool": int(empty_pool),
        "source_final_in_pool": int(source_final_in_pool),
        "source_mentioned_pool_hits": int(source_mentioned_pool_hits),
        "top_label_counts": dict(sorted(top_label_counts.items())),
    }
    return rows, summary


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SVAMP Source Candidate Sidecars",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        "",
        "This CPU-only materializer emits source-derived candidate-score sidecars",
        "over target-side candidate values only. It does not add source-only",
        "answers to the decoder pool.",
        "",
        "## Surfaces",
        "",
    ]
    for label, summary in payload["summaries"].items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- target set: `{summary['target_set']}`",
                f"- n: `{summary['n']}`",
                f"- sidecar bytes: `{summary['sidecar_bytes']}`",
                f"- source final in target pool: `{summary['source_final_in_pool']}`",
                f"- source-mentioned target-pool hits: `{summary['source_mentioned_pool_hits']}`",
                f"- top label counts: `{summary['top_label_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- live sidecar: `{payload['outputs'].get('live_sidecar', '')}`",
            f"- holdout sidecar: `{payload['outputs'].get('holdout_sidecar', '')}`",
            "",
            "## Command",
            "",
            "```bash",
            str(payload.get("command") or "unknown"),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-target-set", required=True)
    parser.add_argument("--holdout-target-set")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sidecar-bits", type=int, default=8)
    parser.add_argument("--label-prior", type=float, default=0.0)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/materialize_svamp_source_candidate_sidecars.py", *argv]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "date": args.date,
        "status": "source_candidate_sidecars_materialized",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "config": {
            "sidecar_bits": int(args.sidecar_bits),
            "label_prior": float(args.label_prior),
            "live_target_set": _display_path(_resolve(args.live_target_set)),
            "holdout_target_set": _display_path(_resolve(args.holdout_target_set)) if args.holdout_target_set else None,
            "candidate_pool": "target_side_only",
        },
        "summaries": {},
        "outputs": {},
        "hashes": {},
    }

    live_rows, live_summary = _materialize(
        target_set_path=_resolve(args.live_target_set),
        sidecar_bits=int(args.sidecar_bits),
        label_prior=float(args.label_prior),
    )
    live_path = output_dir / "live_candidate_sidecars.jsonl"
    _write_jsonl(live_path, live_rows)
    payload["summaries"]["live"] = live_summary
    payload["outputs"]["live_sidecar"] = _display_path(live_path)
    payload["hashes"]["live_sidecar_sha256"] = _sha256_file(live_path)

    if args.holdout_target_set:
        holdout_rows, holdout_summary = _materialize(
            target_set_path=_resolve(args.holdout_target_set),
            sidecar_bits=int(args.sidecar_bits),
            label_prior=float(args.label_prior),
        )
        holdout_path = output_dir / "holdout_candidate_sidecars.jsonl"
        _write_jsonl(holdout_path, holdout_rows)
        payload["summaries"]["holdout"] = holdout_summary
        payload["outputs"]["holdout_sidecar"] = _display_path(holdout_path)
        payload["hashes"]["holdout_sidecar_sha256"] = _sha256_file(holdout_path)

    json_path = output_dir / "manifest.json"
    md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(md_path, payload)
    payload["outputs"]["manifest_json"] = _display_path(json_path)
    payload["outputs"]["manifest_md"] = _display_path(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(md_path, payload)
    print(json.dumps({"status": payload["status"], "manifest": _display_path(json_path)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
