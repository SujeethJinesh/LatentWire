#!/usr/bin/env python3
"""Evaluate a deterministic top-candidate sidecar selector with controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import shlex
import subprocess
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


CONDITIONS = ("matched", "zero_source", "shuffled_source", "random_sidecar", "target_only", "slots_only")


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


def _load_sidecars(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in out:
                duplicates.add(example_id)
            out[example_id] = row
    if duplicates:
        raise ValueError(f"Duplicate sidecar IDs: {sorted(duplicates)}")
    return out


def _top_from_sidecar(sidecar: dict[str, Any] | None) -> tuple[str | None, float]:
    if not sidecar:
        return None, 0.0
    scores = list(sidecar.get("candidate_scores") or [])
    if not scores:
        return None, 0.0
    scores.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("label", "")), str(item.get("value", ""))))
    top = scores[0]
    second = float(scores[1].get("score", top.get("score", 0.0))) if len(scores) > 1 else float(top.get("score", 0.0))
    return str(top.get("value")), float(top.get("score", 0.0)) - second


def _condition_sidecar(
    *,
    condition: str,
    sidecars: dict[str, dict[str, Any]],
    surface: decoder.Surface,
    index: int,
) -> dict[str, Any] | None:
    example_id = surface.reference_ids[index]
    if condition == "matched":
        return sidecars.get(example_id)
    if condition == "zero_source":
        return None
    if condition == "shuffled_source":
        other_index = decoder._hash_nonself_index(surface, index, salt="top_select_shuffle")
        return sidecars.get(surface.reference_ids[other_index])
    if condition == "random_sidecar":
        matched = sidecars.get(example_id)
        if not matched:
            return None
        scores = list(matched.get("candidate_scores") or [])
        rng = random.Random(hashlib.sha256(f"random_sidecar:{example_id}".encode("utf-8")).hexdigest())
        rng.shuffle(scores)
        score_values = sorted((float(item.get("score", 0.0)) for item in scores), reverse=True)
        scores = [{**item, "score": score_values[offset]} for offset, item in enumerate(scores)]
        return {**matched, "candidate_scores": scores}
    return None


def _decode_condition(
    *,
    condition: str,
    sidecars: dict[str, dict[str, Any]],
    surface: decoder.Surface,
    index: int,
    min_confidence: float,
) -> dict[str, Any]:
    example_id = surface.reference_ids[index]
    fallback = decoder._prediction_numeric(surface.records_by_label["target"].get(example_id))
    gold = decoder._gold_numeric(surface, example_id)
    candidates = {candidate.value for candidate in decoder._candidate_pool(surface, example_id)}
    selected = fallback
    accepted = False
    top_value: str | None = None
    confidence = 0.0
    if condition in {"target_only", "slots_only"}:
        accepted = False
    else:
        top_value, confidence = _top_from_sidecar(
            _condition_sidecar(
                condition=condition,
                sidecars=sidecars,
                surface=surface,
                index=index,
            )
        )
        if (
            top_value is not None
            and top_value in candidates
            and top_value != fallback
            and confidence >= min_confidence
        ):
            selected = top_value
            accepted = True
    return {
        "condition": condition,
        "prediction": selected,
        "fallback_prediction": fallback,
        "gold": gold,
        "correct": selected == gold,
        "accepted": bool(accepted),
        "top_value": top_value,
        "confidence": float(confidence),
    }


def _summarize(rows: Sequence[dict[str, Any]], clean_ids: set[str], target_correct_ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        correct = {
            row["example_id"]
            for row in rows
            if row["conditions"].get(condition, {}).get("correct", False)
        }
        accepted = {
            row["example_id"]
            for row in rows
            if row["conditions"].get(condition, {}).get("accepted", False)
        }
        harms = {
            row["example_id"]
            for row in rows
            if row["conditions"].get(condition, {}).get("accepted", False)
            and not row["conditions"].get(condition, {}).get("correct", False)
            and row["example_id"] in target_correct_ids
        }
        out[condition] = {
            "correct": len(correct),
            "correct_ids": sorted(correct),
            "accepted": len(accepted),
            "accepted_ids": sorted(accepted),
            "clean_correct": len(correct & clean_ids),
            "clean_correct_ids": sorted(correct & clean_ids),
            "accepted_harm": len(harms),
            "accepted_harm_ids": sorted(harms),
        }
    control_clean = set().union(
        *(set(out[condition]["clean_correct_ids"]) for condition in CONDITIONS if condition != "matched")
    )
    out["control_clean_union_ids"] = sorted(control_clean)
    out["source_necessary_clean_ids"] = sorted(set(out["matched"]["clean_correct_ids"]) - control_clean)
    return out


def analyze(
    *,
    target_set_path: pathlib.Path,
    sidecar_path: pathlib.Path,
    min_confidence: float,
) -> dict[str, Any]:
    surface = decoder._load_surface("surface", target_set_path)
    sidecars = _load_sidecars(sidecar_path)
    expected = set(surface.reference_ids)
    observed = set(sidecars)
    if observed != expected:
        raise ValueError(f"Sidecar IDs mismatch: missing={sorted(expected - observed)} extra={sorted(observed - expected)}")
    clean_ids = decoder._source_ids(surface, "clean_source_only") or decoder._source_ids(surface, "clean_residual_targets")
    target_correct_ids = {
        example_id
        for example_id in surface.reference_ids
        if bool(surface.records_by_label["target"][example_id].get("correct"))
    }
    rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(surface.reference_ids):
        rows.append(
            {
                "example_id": example_id,
                "conditions": {
                    condition: _decode_condition(
                        condition=condition,
                        sidecars=sidecars,
                        surface=surface,
                        index=index,
                        min_confidence=float(min_confidence),
                    )
                    for condition in CONDITIONS
                },
            }
        )
    return {
        "target_set": _display_path(target_set_path),
        "target_set_sha256": _sha256(target_set_path),
        "sidecar_jsonl": _display_path(sidecar_path),
        "sidecar_jsonl_sha256": _sha256(sidecar_path),
        "n": len(surface.reference_ids),
        "min_confidence": float(min_confidence),
        "clean_ids": sorted(clean_ids),
        "summaries": _summarize(rows, clean_ids, target_correct_ids),
        "rows": rows,
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summaries = payload["summaries"]
    lines = [
        "# Candidate-Score Sidecar Top Selector",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- min confidence: `{payload['result']['min_confidence']}`",
        "",
        "| Condition | Correct | Accepted | Clean Correct | Accepted Harm |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summaries[condition]
        lines.append(
            f"| `{condition}` | {row['correct']}/{payload['result']['n']} | "
            f"{row['accepted']} | {row['clean_correct']} | {row['accepted_harm']} |"
        )
    lines.extend(
        [
            "",
            f"- source-necessary clean IDs: {', '.join(f'`{x}`' for x in summaries['source_necessary_clean_ids']) or 'none'}",
            f"- control clean union IDs: {', '.join(f'`{x}`' for x in summaries['control_clean_union_ids']) or 'none'}",
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
    parser.add_argument("--target-set", required=True)
    parser.add_argument("--sidecar-jsonl", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-source-necessary-clean", type=int, default=1)
    parser.add_argument("--max-control-clean-union", type=int, default=0)
    parser.add_argument("--max-accepted-harm", type=int, default=0)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/analyze_candidate_score_sidecar_top_select.py", *argv]
    result = analyze(
        target_set_path=_resolve(args.target_set),
        sidecar_path=_resolve(args.sidecar_jsonl),
        min_confidence=float(args.min_confidence),
    )
    summaries = result["summaries"]
    pass_rule = {
        "min_source_necessary_clean": len(summaries["source_necessary_clean_ids"]) >= int(args.min_source_necessary_clean),
        "max_control_clean_union": len(summaries["control_clean_union_ids"]) <= int(args.max_control_clean_union),
        "max_accepted_harm": summaries["matched"]["accepted_harm"] <= int(args.max_accepted_harm),
    }
    payload = {
        "date": str(args.date),
        "status": "top_sidecar_selector_passes_smoke" if all(pass_rule.values()) else "top_sidecar_selector_fails_smoke",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "pass_rule": pass_rule,
        "result": result,
        "summaries": summaries,
    }
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(output_md, payload)
    print(json.dumps({"status": payload["status"], "source_necessary_clean": summaries["source_necessary_clean_ids"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
