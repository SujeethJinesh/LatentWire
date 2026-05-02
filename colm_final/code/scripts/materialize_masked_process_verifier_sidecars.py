#!/usr/bin/env python3
"""Materialize answer-masked process-overlap sidecars for target candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
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


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "let",
    "more",
    "of",
    "on",
    "or",
    "she",
    "so",
    "step",
    "than",
    "that",
    "the",
    "then",
    "there",
    "to",
    "we",
    "with",
}

NUMERIC_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")
WORD_RE = re.compile(r"[a-z][a-z_]{2,}")
EQUATION_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*[+\-*/x×]\s*[-+]?\d+(?:\.\d+)?(?:\s*=\s*[-+]?\d+(?:\.\d+)?)?")


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


def mask_answer_values(text: str, answer_values: set[str]) -> str:
    masked = str(text or "")
    for value in sorted(answer_values, key=len, reverse=True):
        escaped = re.escape(value)
        masked = re.sub(rf"(?<![\w.]){escaped}(?![\w.])", "<ANS>", masked)
    return masked


def process_features(text: str, *, answer_values: set[str] | None = None) -> set[str]:
    masked = mask_answer_values(text, answer_values or set()).lower()
    features: set[str] = set(decoder._operation_names(masked))
    if "+" in masked:
        features.add("op_add")
    if "-" in masked:
        features.add("op_sub")
    if "*" in masked or "×" in masked:
        features.add("op_mul")
    if "/" in masked:
        features.add("op_div")
    for word in WORD_RE.findall(masked):
        if word not in STOPWORDS:
            features.add(f"w:{word}")
    for value in NUMERIC_RE.findall(masked):
        normalized = decoder._normal(value)
        if normalized is not None:
            features.add(f"num:{normalized}")
    for equation in EQUATION_RE.findall(masked):
        shape = NUMERIC_RE.sub("N", equation.lower()).replace("×", "*").replace("x", "*")
        compact_shape = re.sub(r"\s+", "", shape)
        features.add(f"eq:{compact_shape}")
        if "+" in compact_shape:
            features.add("op_add")
        if "-" in compact_shape:
            features.add("op_sub")
        if "*" in compact_shape:
            features.add("op_mul")
        if "/" in compact_shape:
            features.add("op_div")
    return features


def _candidate_text(surface: decoder.Surface, example_id: str, candidate: decoder.Candidate) -> str:
    parts: list[str] = []
    for label in candidate.labels:
        row = surface.records_by_label.get(label, {}).get(example_id)
        if row is not None:
            parts.append(str(row.get("prediction", "") or ""))
    return "\n".join(parts)


def _score_candidate(source_features: set[str], candidate_features: set[str]) -> float:
    if not source_features or not candidate_features:
        return 0.0
    overlap = source_features & candidate_features
    op_overlap = {item for item in overlap if item.startswith("op_")}
    equation_overlap = {item for item in overlap if item.startswith("eq:")}
    numeric_overlap = {item for item in overlap if item.startswith("num:")}
    lexical_overlap = {item for item in overlap if item.startswith("w:")}
    score = 0.0
    score += 2.0 * len(op_overlap)
    score += 1.5 * len(equation_overlap)
    score += 0.5 * min(len(numeric_overlap), 6)
    score += 0.1 * min(len(lexical_overlap), 20)
    score += len(overlap) / max(len(source_features | candidate_features), 1)
    return float(score)


def _materialize_one(target_set_path: pathlib.Path, output_jsonl: pathlib.Path) -> dict[str, Any]:
    surface = decoder._load_surface("surface", target_set_path)
    rows: list[dict[str, Any]] = []
    empty_pool = 0
    clean_ids = decoder._source_ids(surface, "clean_source_only") or decoder._source_ids(surface, "clean_residual_targets")
    answer_excluded_top = 0
    for example_id in surface.reference_ids:
        source_row = surface.records_by_label.get("source", {}).get(example_id)
        profile = decoder._source_profile(source_row)
        answer_values = _source_answer_values(profile)
        source_text = str((source_row or {}).get("prediction", "") or "")
        source_features = process_features(source_text, answer_values=answer_values)
        candidates = decoder._candidate_pool(surface, example_id)
        if not candidates:
            empty_pool += 1
        candidate_scores: list[dict[str, Any]] = []
        for candidate in candidates:
            labels = tuple(label for label in candidate.labels if label != "source")
            label = labels[0] if labels else "target"
            candidate_features = process_features(_candidate_text(surface, example_id, candidate), answer_values=set())
            score = _score_candidate(source_features, candidate_features)
            candidate_scores.append(
                {
                    "label": label,
                    "value": candidate.value,
                    "score": score,
                    "feature_overlap": sorted((source_features & candidate_features))[:64],
                }
            )
        candidate_scores.sort(key=lambda item: (-float(item["score"]), str(item["label"]), str(item["value"])))
        if candidate_scores and candidate_scores[0]["value"] not in answer_values:
            answer_excluded_top += 1
        top_score = float(candidate_scores[0]["score"]) if candidate_scores else 0.0
        second_score = float(candidate_scores[1]["score"]) if len(candidate_scores) > 1 else top_score
        rows.append(
            {
                "example_id": example_id,
                "candidate_scores": candidate_scores,
                "confidence": float(top_score - second_score),
                "sidecar_bits": 8,
                "source_answer_values_masked": sorted(answer_values),
                "source_feature_count": len(source_features),
                "counterfactual_mode": "answer_masked_process_verifier",
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "target_set": _display_path(target_set_path),
        "target_set_sha256": _sha256(target_set_path),
        "output_jsonl": _display_path(output_jsonl),
        "output_jsonl_sha256": _sha256(output_jsonl),
        "n": len(surface.reference_ids),
        "clean_n": len(clean_ids),
        "empty_pool": empty_pool,
        "answer_excluded_top": answer_excluded_top,
    }


def materialize(
    *,
    live_target_set: pathlib.Path,
    output_dir: pathlib.Path,
    holdout_target_set: pathlib.Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces = {
        "live": _materialize_one(live_target_set, output_dir / "live_masked_process_sidecars.jsonl")
    }
    if holdout_target_set is not None:
        surfaces["holdout"] = _materialize_one(holdout_target_set, output_dir / "holdout_masked_process_sidecars.jsonl")
    return {"surfaces": surfaces}


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Masked Process-Verifier Sidecars",
        "",
        f"- date: `{payload['date']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        "",
        "| Surface | N | Clean | Empty Pool | Answer-Excluded Top | Output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, row in payload["result"]["surfaces"].items():
        lines.append(
            f"| `{label}` | {row['n']} | {row['clean_n']} | {row['empty_pool']} | "
            f"{row['answer_excluded_top']} | `{row['output_jsonl']}` |"
        )
    lines.extend(["", "## Command", "", "```bash", payload["command"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-target-set", required=True)
    parser.add_argument("--holdout-target-set")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/materialize_masked_process_verifier_sidecars.py", *argv]
    result = materialize(
        live_target_set=_resolve(args.live_target_set),
        holdout_target_set=_resolve(args.holdout_target_set) if args.holdout_target_set else None,
        output_dir=_resolve(args.output_dir),
    )
    payload = {
        "date": str(args.date),
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "result": result,
    }
    output_dir = _resolve(args.output_dir)
    manifest_json = output_dir / "manifest.json"
    manifest_md = output_dir / "manifest.md"
    manifest_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(manifest_md, payload)
    print(json.dumps({"surfaces": sorted(result["surfaces"])}, indent=2))
    return payload


if __name__ == "__main__":
    main()
