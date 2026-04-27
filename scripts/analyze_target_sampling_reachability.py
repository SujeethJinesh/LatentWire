#!/usr/bin/env python3
"""Audit raw target-only sampling reachability against frozen SVAMP metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _records_for_artifact(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spec = target_set.RowSpec(
        str(artifact["label"]),
        _resolve(str(artifact["path"])),
        str(artifact["method"]),
    )
    return {str(row["example_id"]): row for row in target_set._records_for_method(spec)}


def _correct_ids(records: dict[str, dict[str, Any]], ids: Sequence[str]) -> set[str]:
    return {example_id for example_id in ids if bool(records.get(example_id, {}).get("correct"))}


def _answer_values(row: dict[str, Any]) -> set[str]:
    answer = row.get("answer")
    if isinstance(answer, list):
        return {str(item) for item in answer}
    if answer is None:
        return set()
    return {str(answer)}


def _prediction_value(row: dict[str, Any]) -> str:
    for key in ("normalized_prediction", "tail_numeric_mention", "value", "prediction"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _load_id_groups(path: pathlib.Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    payload = _read_json(path)
    groups = payload.get("ids") or {}
    return {str(name): {str(item) for item in values} for name, values in groups.items()}


def _summarize_diversity(grouped_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    unique_counts: dict[str, int] = {}
    duplicate_rows = 0
    total_rows = 0
    for example_id, rows in grouped_rows.items():
        values = [_prediction_value(row) for row in rows]
        nonempty = [value for value in values if value]
        unique = set(nonempty)
        unique_counts[example_id] = len(unique)
        duplicate_rows += max(0, len(nonempty) - len(unique))
        total_rows += len(rows)
    counts = list(unique_counts.values())
    return {
        "unique_answer_counts_by_id": unique_counts,
        "mean_unique_answers_per_id": statistics.fmean(counts) if counts else 0.0,
        "median_unique_answers_per_id": statistics.median(counts) if counts else 0.0,
        "min_unique_answers_per_id": min(counts) if counts else 0,
        "max_unique_answers_per_id": max(counts) if counts else 0,
        "duplicate_nonempty_rows": duplicate_rows,
        "duplicate_nonempty_fraction": duplicate_rows / total_rows if total_rows else 0.0,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Target Sampling Reachability Audit",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- samples: `{payload['sample_rows']}` rows over `{payload['covered_ids']}/{payload['reference_n']}` IDs",
        f"- sample rows SHA256: `{payload['sample_rows_sha256']}`",
        "",
        "## Summary",
        "",
        f"- target baseline correct: `{payload['target_correct']}/{payload['reference_n']}`",
        f"- sample candidate oracle: `{payload['sample_oracle_correct']}/{payload['reference_n']}`",
        f"- sample oracle gain vs target: `{payload['sample_oracle_gain_vs_target']}`",
        f"- source-contrastive clean in pool: `{payload['source_contrastive_clean_in_pool']}/{payload['source_contrastive_clean_total']}`",
        f"- C2C clean residual in pool: `{payload['c2c_clean_residual_in_pool']}/{payload['c2c_clean_residual_total']}`",
        f"- C2C teacher-only in pool: `{payload['c2c_teacher_only_in_pool']}/{payload['c2c_teacher_only_total']}`",
        f"- mean unique sampled answers per ID: `{payload['diversity']['mean_unique_answers_per_id']:.3f}`",
        f"- duplicate nonempty row fraction: `{payload['diversity']['duplicate_nonempty_fraction']:.3f}`",
        "",
        "## Reachable IDs",
        "",
        "- sample oracle IDs: " + (", ".join(f"`{item}`" for item in payload["sample_oracle_ids"]) or "none"),
        "- oracle gain IDs: " + (", ".join(f"`{item}`" for item in payload["sample_oracle_gain_ids"]) or "none"),
        "- C2C clean residual IDs in pool: "
        + (", ".join(f"`{item}`" for item in payload["c2c_clean_residual_in_pool_ids"]) or "none"),
        "- source-contrastive clean IDs in pool: "
        + (", ".join(f"`{item}`" for item in payload["source_contrastive_clean_in_pool_ids"]) or "none"),
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--base-target-set", required=True)
    parser.add_argument("--c2c-headroom-json")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_argv = sys.argv if argv is None else ["scripts/analyze_target_sampling_reachability.py", *argv]

    samples_path = _resolve(args.samples_jsonl)
    base_path = _resolve(args.base_target_set)
    c2c_path = _resolve(args.c2c_headroom_json) if args.c2c_headroom_json else None
    sample_rows = _read_jsonl(samples_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["example_id"])].append(row)

    base = _read_json(base_path)
    reference_ids = [str(example_id) for example_id in base["reference_ids"]]
    artifacts = base["artifacts"]
    target_records = _records_for_artifact(artifacts["target"])
    source_records = _records_for_artifact(artifacts["source"])
    target_correct = _correct_ids(target_records, reference_ids)
    source_correct = _correct_ids(source_records, reference_ids)
    source_contrastive_clean = {str(item) for item in (base.get("ids", {}).get("clean_source_only") or [])}
    c2c_groups = _load_id_groups(c2c_path)
    c2c_clean = c2c_groups.get("clean_residual_targets", set())
    c2c_teacher_only = c2c_groups.get("teacher_only", set())

    oracle_ids: list[str] = []
    oracle_gain_ids: list[str] = []
    numeric_covered = 0
    rows_by_id: list[dict[str, Any]] = []
    for example_id in reference_ids:
        rows = grouped.get(example_id, [])
        has_prediction = any(_prediction_value(row) for row in rows)
        numeric_covered += int(has_prediction)
        sample_correct = any(bool(row.get("correct")) for row in rows)
        if not sample_correct:
            sample_correct = any(_prediction_value(row) in _answer_values(row) for row in rows)
        if sample_correct:
            oracle_ids.append(example_id)
            if example_id not in target_correct:
                oracle_gain_ids.append(example_id)
        rows_by_id.append(
            {
                "example_id": example_id,
                "target_correct": example_id in target_correct,
                "source_correct": example_id in source_correct,
                "sample_rows": len(rows),
                "sample_oracle": sample_correct,
                "unique_sample_values": sorted({_prediction_value(row) for row in rows if _prediction_value(row)}),
                "source_contrastive_clean": example_id in source_contrastive_clean,
                "c2c_clean_residual": example_id in c2c_clean,
                "c2c_teacher_only": example_id in c2c_teacher_only,
            }
        )

    oracle_set = set(oracle_ids)
    source_contrastive_clean_in_pool = sorted(source_contrastive_clean & oracle_set)
    c2c_clean_in_pool = sorted(c2c_clean & oracle_set)
    c2c_teacher_only_in_pool = sorted(c2c_teacher_only & oracle_set)
    c2c_clean_count = len(c2c_clean)
    c2c_clean_hit = len(c2c_clean_in_pool)
    oracle_gain = len([example_id for example_id in oracle_ids if example_id not in target_correct])
    if c2c_clean_count and c2c_clean_hit >= 3:
        decision = "Strong pass: C2C clean residual target reachability is large enough to justify a strict source-derived selector or connector gate."
    elif c2c_clean_count and c2c_clean_hit >= 2:
        decision = "Pass: C2C clean residual target reachability matches the clean6 floor; next gate should test a source-derived selector only on reachable clean IDs."
    elif c2c_clean_count and c2c_clean_hit == 1:
        decision = "Borderline: top up sampling before training a selector, unless the single reachable ID is qualitatively new and source-necessary."
    else:
        decision = "Fail: target/no-source sampling did not expose enough C2C clean residual reachability; switch candidate-surface generator instead of training a selector."

    payload = {
        "date": args.date,
        "status": "target_sampling_reachability_audited",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "samples_jsonl": _display_path(samples_path),
        "sample_rows_sha256": _sha256_file(samples_path),
        "base_target_set": _display_path(base_path),
        "base_target_set_sha256": _sha256_file(base_path),
        "c2c_headroom_json": _display_path(c2c_path) if c2c_path else None,
        "c2c_headroom_sha256": _sha256_file(c2c_path) if c2c_path else None,
        "reference_n": len(reference_ids),
        "sample_rows": len(sample_rows),
        "covered_ids": len(grouped),
        "numeric_covered_ids": numeric_covered,
        "target_correct": len(target_correct),
        "source_correct": len(source_correct),
        "sample_oracle_correct": len(oracle_ids),
        "sample_oracle_gain_vs_target": oracle_gain,
        "sample_oracle_ids": sorted(oracle_ids),
        "sample_oracle_gain_ids": sorted(oracle_gain_ids),
        "source_contrastive_clean_total": len(source_contrastive_clean),
        "source_contrastive_clean_in_pool": len(source_contrastive_clean_in_pool),
        "source_contrastive_clean_in_pool_ids": source_contrastive_clean_in_pool,
        "c2c_clean_residual_total": len(c2c_clean),
        "c2c_clean_residual_in_pool": len(c2c_clean_in_pool),
        "c2c_clean_residual_in_pool_ids": c2c_clean_in_pool,
        "c2c_teacher_only_total": len(c2c_teacher_only),
        "c2c_teacher_only_in_pool": len(c2c_teacher_only_in_pool),
        "c2c_teacher_only_in_pool_ids": c2c_teacher_only_in_pool,
        "diversity": _summarize_diversity(grouped),
        "rows": rows_by_id,
        "decision": decision,
    }
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "sample_oracle": len(oracle_ids)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
