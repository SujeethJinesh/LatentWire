#!/usr/bin/env python3
"""Evaluate answer-null source predicate syndromes against target candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import shlex
import subprocess
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder
from scripts import materialize_masked_process_verifier_sidecars as process_sidecar


CONDITIONS = ("matched", "shuffled_source", "random_syndrome", "target_only", "slots_only")
RELATION_PATTERNS = {
    "rel_difference": ("more than", "less than", "fewer than", "difference", "remain", "left"),
    "rel_total": ("total", "altogether", "in all", "combined"),
    "rel_each": ("each", "per", "every"),
    "rel_ratio": ("twice", "half", "ratio", "times as"),
}
UNIT_WORDS = {
    "bags",
    "books",
    "cans",
    "cups",
    "days",
    "flour",
    "flowers",
    "frog",
    "grasshopper",
    "inches",
    "kids",
    "movies",
    "orchids",
    "roses",
    "sugar",
    "visitors",
}


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


def predicate_syndrome(text: str, *, answer_values: set[str] | None = None) -> set[str]:
    masked = process_sidecar.mask_answer_values(text, answer_values or set()).lower()
    predicates: set[str] = set(decoder._operation_names(masked))
    for name, needles in RELATION_PATTERNS.items():
        if any(needle in masked for needle in needles):
            predicates.add(name)
    for unit in UNIT_WORDS:
        if re.search(rf"\b{re.escape(unit)}\b", masked):
            predicates.add(f"unit:{unit}")
    if "+" in masked:
        predicates.add("op_add")
    if "-" in masked:
        predicates.add("op_sub")
    if "*" in masked or "×" in masked:
        predicates.add("op_mul")
    if "/" in masked:
        predicates.add("op_div")
    for equation in process_sidecar.EQUATION_RE.findall(masked):
        shape = process_sidecar.NUMERIC_RE.sub("N", equation.lower()).replace("×", "*").replace("x", "*")
        compact_shape = re.sub(r"\s+", "", shape)
        predicates.add(f"eq:{compact_shape}")
    return predicates


def _candidate_text(surface: decoder.Surface, example_id: str, candidate: decoder.Candidate) -> str:
    parts: list[str] = []
    for label in candidate.labels:
        row = surface.records_by_label.get(label, {}).get(example_id)
        if row is not None:
            parts.append(str(row.get("prediction", "") or ""))
    return "\n".join(parts)


def _condition_predicates(
    *,
    surface: decoder.Surface,
    syndromes: dict[str, set[str]],
    index: int,
    condition: str,
) -> set[str]:
    example_id = surface.reference_ids[index]
    if condition == "matched":
        return syndromes.get(example_id, set())
    if condition == "shuffled_source":
        other_index = decoder._hash_nonself_index(surface, index, salt="answer_null_predicate_shuffle")
        return syndromes.get(surface.reference_ids[other_index], set())
    if condition == "random_syndrome":
        universe = sorted(set().union(*syndromes.values())) if syndromes else []
        size = len(syndromes.get(example_id, set()))
        rng = random.Random(hashlib.sha256(f"answer_null_random:{example_id}".encode("utf-8")).hexdigest())
        return set(rng.sample(universe, min(size, len(universe))))
    return set()


def _score_candidate(predicates: set[str], candidate_predicates: set[str]) -> float:
    overlap = predicates & candidate_predicates
    if not overlap:
        return 0.0
    weighted = 0.0
    for item in overlap:
        if item.startswith("eq:"):
            weighted += 2.0
        elif item.startswith("op_") or item.startswith("rel_"):
            weighted += 1.0
        elif item.startswith("unit:"):
            weighted += 0.4
    weighted += len(overlap) / max(len(predicates | candidate_predicates), 1)
    return float(weighted)


def _decode_condition(
    *,
    surface: decoder.Surface,
    syndromes: dict[str, set[str]],
    index: int,
    condition: str,
    min_confidence: float,
) -> dict[str, Any]:
    example_id = surface.reference_ids[index]
    fallback = decoder._prediction_numeric(surface.records_by_label["target"].get(example_id))
    gold = decoder._gold_numeric(surface, example_id)
    selected = fallback
    accepted = False
    top_value: str | None = None
    confidence = 0.0
    if condition in {"target_only", "slots_only"}:
        predicates = set()
    else:
        predicates = _condition_predicates(surface=surface, syndromes=syndromes, index=index, condition=condition)
    scored: list[tuple[float, str, set[str]]] = []
    for candidate in decoder._candidate_pool(surface, example_id):
        cand_predicates = predicate_syndrome(_candidate_text(surface, example_id, candidate), answer_values=set())
        scored.append((_score_candidate(predicates, cand_predicates), candidate.value, cand_predicates))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored:
        top_score, top_value, _ = scored[0]
        second = scored[1][0] if len(scored) > 1 else top_score
        confidence = float(top_score - second)
        if top_value != fallback and confidence >= min_confidence:
            selected = top_value
            accepted = True
    return {
        "condition": condition,
        "prediction": selected,
        "fallback_prediction": fallback,
        "gold": gold,
        "correct": selected == gold,
        "accepted": accepted,
        "top_value": top_value,
        "confidence": confidence,
        "predicate_count": len(predicates),
    }


def _summarize(rows: Sequence[dict[str, Any]], clean_ids: set[str], target_correct_ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        correct = {row["example_id"] for row in rows if row["conditions"][condition]["correct"]}
        accepted = {row["example_id"] for row in rows if row["conditions"][condition]["accepted"]}
        harms = {
            row["example_id"]
            for row in rows
            if row["conditions"][condition]["accepted"]
            and not row["conditions"][condition]["correct"]
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


def analyze(target_set_path: pathlib.Path, *, min_confidence: float) -> dict[str, Any]:
    surface = decoder._load_surface("surface", target_set_path)
    syndromes: dict[str, set[str]] = {}
    for example_id in surface.reference_ids:
        source_row = surface.records_by_label.get("source", {}).get(example_id)
        profile = decoder._source_profile(source_row)
        syndromes[example_id] = predicate_syndrome(str((source_row or {}).get("prediction", "") or ""), answer_values=_source_answer_values(profile))
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
                        surface=surface,
                        syndromes=syndromes,
                        index=index,
                        condition=condition,
                        min_confidence=min_confidence,
                    )
                    for condition in CONDITIONS
                },
            }
        )
    return {
        "target_set": _display_path(target_set_path),
        "target_set_sha256": _sha256(target_set_path),
        "n": len(surface.reference_ids),
        "min_confidence": float(min_confidence),
        "clean_ids": sorted(clean_ids),
        "summaries": _summarize(rows, clean_ids, target_correct_ids),
        "rows": rows,
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summaries = payload["result"]["summaries"]
    lines = [
        "# Answer-Null Predicate Syndrome",
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
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-source-necessary-clean", type=int, default=1)
    parser.add_argument("--max-control-clean-union", type=int, default=0)
    parser.add_argument("--max-accepted-harm", type=int, default=0)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/analyze_answer_null_predicate_syndrome.py", *argv]
    result = analyze(_resolve(args.target_set), min_confidence=float(args.min_confidence))
    summaries = result["summaries"]
    pass_rule = {
        "min_source_necessary_clean": len(summaries["source_necessary_clean_ids"]) >= int(args.min_source_necessary_clean),
        "max_control_clean_union": len(summaries["control_clean_union_ids"]) <= int(args.max_control_clean_union),
        "max_accepted_harm": summaries["matched"]["accepted_harm"] <= int(args.max_accepted_harm),
    }
    payload = {
        "date": str(args.date),
        "status": "answer_null_predicate_syndrome_passes_smoke" if all(pass_rule.values()) else "answer_null_predicate_syndrome_fails_smoke",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "pass_rule": pass_rule,
        "result": result,
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
