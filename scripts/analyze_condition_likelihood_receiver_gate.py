#!/usr/bin/env python3
"""Gate condition-specific receiver likelihood sketches.

Unlike ``analyze_svamp70_source_likelihood_sketch_gate.py``, this analyzer does
not synthesize controls by shuffling a matched sketch or forcing target
fallback. Each condition is read from its own receiver-scored JSONL, so controls
can mutate the candidate pool before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp70_source_likelihood_sketch_gate as base
from scripts import harness_common as harness


CONDITIONS = ("matched", "zero_source", "shuffled_source", "label_shuffle", "target_only", "slots_only")
FEATURES = base.FEATURES


@dataclass(frozen=True)
class GateConfig:
    outer_folds: int = 5
    accept_penalty: float = 0.10
    max_sidecar_bits: int = 8
    min_live_correct: int = 25
    min_live_clean_source_necessary: int = 3
    min_holdout_correct: int = 10
    min_holdout_clean_source_necessary: int = 1
    max_clean_control_union: int = 0
    max_accepted_harm: int = 1


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_condition_spec(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise ValueError(f"Condition sketch spec must be condition=path, got {spec!r}")
    condition, raw_path = spec.split("=", 1)
    condition = condition.strip()
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}; expected one of {CONDITIONS}")
    return condition, _resolve(raw_path.strip())


def _condition_paths(specs: Sequence[str]) -> dict[str, pathlib.Path]:
    parsed: dict[str, pathlib.Path] = {}
    for spec in specs:
        condition, path = _parse_condition_spec(spec)
        if condition in parsed:
            raise ValueError(f"Duplicate condition sketch for {condition!r}")
        parsed[condition] = path
    if "matched" not in parsed:
        raise ValueError("A matched condition sketch is required")
    return parsed


def _candidate_items(sketch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): dict(item) for item in sketch.get("candidate_scores", [])}


def _candidate_correct(item: dict[str, Any]) -> bool:
    if "candidate_correct" in item:
        return bool(item["candidate_correct"])
    return bool(item.get("correct"))


def _candidate_prediction(item: dict[str, Any]) -> str:
    for key in ("candidate_raw_text", "candidate_text", "prediction"):
        if item.get(key) is not None:
            return str(item[key])
    return ""


def _canonical_answer_value(item: dict[str, Any]) -> str:
    for key in ("candidate_raw_text", "prediction", "candidate_text"):
        value = item.get(key)
        if value is None:
            continue
        numeric = harness._extract_prediction_numeric_answer(str(value))
        if numeric is not None:
            return f"num:{numeric}"
        text = " ".join(str(value).strip().lower().split())
        if text:
            return f"text:{text}"
    return ""


def _duplicates_non_source_answer(sketch: dict[str, Any] | None, label: str, item: dict[str, Any]) -> bool:
    if sketch is None:
        return False
    selected = _canonical_answer_value(item)
    if not selected:
        return False
    for other_label, other_item in _candidate_items(sketch).items():
        if other_label == label:
            continue
        if _canonical_answer_value(other_item) == selected:
            return True
    return False


def _sketches_by_id(path: pathlib.Path, *, max_sidecar_bits: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in _read_jsonl(path):
        example_id = str(row["example_id"])
        if example_id in by_id:
            duplicates.add(example_id)
        ordered_ids.append(example_id)
        by_id[example_id] = base._sketch_from_scores(row, max_sidecar_bits=max_sidecar_bits)
    if duplicates:
        raise ValueError(f"Duplicate example IDs in {path}: {sorted(duplicates)}")
    return ordered_ids, by_id


def _load_condition_sketches(
    paths: dict[str, pathlib.Path],
    *,
    max_sidecar_bits: int,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    matched_ids, matched = _sketches_by_id(paths["matched"], max_sidecar_bits=max_sidecar_bits)
    expected = set(matched_ids)
    sketches: dict[str, dict[str, dict[str, Any]]] = {"matched": matched}
    for condition, path in paths.items():
        if condition == "matched":
            continue
        ordered, by_id = _sketches_by_id(path, max_sidecar_bits=max_sidecar_bits)
        if ordered != matched_ids:
            missing = sorted(expected - set(ordered))
            extra = sorted(set(ordered) - expected)
            raise ValueError(
                f"Condition {condition!r} does not match matched IDs; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        sketches[condition] = by_id
    return matched_ids, sketches


def _fallback_from_row(row: dict[str, Any], fallback_label: str) -> dict[str, Any]:
    items = _candidate_items(row["condition_sketches"]["matched"])
    if fallback_label not in items:
        raise ValueError(f"Fallback label {fallback_label!r} missing for {row['example_id']}")
    return items[fallback_label]


def _thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values if math.isfinite(float(value))))
    thresholds = list(unique)
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(thresholds))


def _accepts(value: Any, *, threshold: float | None, direction: str) -> bool:
    if threshold is None or value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if direction == "ge":
        return numeric >= threshold
    if direction == "le":
        return numeric <= threshold
    raise ValueError(f"Unsupported direction: {direction!r}")


def _apply_rule(row: dict[str, Any], condition: str, rule: dict[str, Any]) -> dict[str, Any]:
    sketch = row["condition_sketches"].get(condition)
    fallback = row["fallback_item"]
    accepted = False
    item = fallback
    label = row["fallback_label"]
    if sketch is not None and rule.get("feature"):
        accepted = _accepts(
            sketch.get(str(rule["feature"])),
            threshold=float(rule["threshold"]),
            direction=str(rule["direction"]),
        )
        if accepted:
            top_label = str(sketch.get("top_label"))
            items = _candidate_items(sketch)
            if top_label in items:
                label = top_label
                item = items[top_label]
            else:
                accepted = False
    return {
        "label": label,
        "prediction": _candidate_prediction(item),
        "correct": _candidate_correct(item),
        "accepted_sidecar": bool(accepted),
        "sidecar_bits": int(sketch.get("sidecar_bits", 0)) if sketch else 0,
        "duplicates_non_source_answer": _duplicates_non_source_answer(sketch, label, item),
    }


def _fit_stump(rows: Sequence[dict[str, Any]], *, train_folds: set[int] | None, accept_penalty: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in FEATURES:
        values = [
            float(row["condition_sketches"]["matched"].get(feature))
            for row in rows
            if (train_folds is None or int(row["fold"]) in train_folds)
            and row["condition_sketches"]["matched"].get(feature) is not None
        ]
        for threshold in _thresholds(values):
            for direction in ("ge", "le"):
                help_count = harm_count = accept_count = 0
                for row in rows:
                    if train_folds is not None and int(row["fold"]) not in train_folds:
                        continue
                    routed = _apply_rule(row, "matched", {"feature": feature, "threshold": threshold, "direction": direction})
                    if not routed["accepted_sidecar"]:
                        continue
                    accept_count += 1
                    help_count += int(routed["correct"] and not row["fallback_correct"])
                    harm_count += int((not routed["correct"]) and row["fallback_correct"])
                score = help_count - harm_count - float(accept_penalty) * accept_count
                candidate = {
                    "feature": feature,
                    "threshold": float(threshold),
                    "direction": direction,
                    "train_help": int(help_count),
                    "train_harm": int(harm_count),
                    "train_accept": int(accept_count),
                    "train_score": float(score),
                }
                if best is None or (
                    candidate["train_score"],
                    -candidate["train_harm"],
                    candidate["train_help"],
                    -candidate["train_accept"],
                    candidate["feature"],
                ) > (
                    best["train_score"],
                    -best["train_harm"],
                    best["train_help"],
                    -best["train_accept"],
                    best["feature"],
                ):
                    best = candidate
    return best or {
        "feature": None,
        "threshold": None,
        "direction": "none",
        "train_help": 0,
        "train_harm": 0,
        "train_accept": 0,
        "train_score": 0.0,
    }


def _summarize_condition(rows: Sequence[dict[str, Any]], condition: str, ids: dict[str, set[str]]) -> dict[str, Any]:
    correct_ids = {str(row["example_id"]) for row in rows if bool(row["conditions"][condition]["correct"])}
    accepted_ids = {str(row["example_id"]) for row in rows if bool(row["conditions"][condition]["accepted_sidecar"])}
    duplicate_answer_ids = {
        str(row["example_id"])
        for row in rows
        if row["conditions"][condition]["correct"]
        and row["conditions"][condition]["accepted_sidecar"]
        and row["conditions"][condition].get("duplicates_non_source_answer")
    }
    accepted_harm_ids = {
        str(row["example_id"])
        for row in rows
        if row["conditions"][condition]["accepted_sidecar"]
        and not row["conditions"][condition]["correct"]
        and row["fallback_correct"]
    }
    return {
        "condition": condition,
        "correct_count": len(correct_ids),
        "correct_ids": sorted(correct_ids),
        "accepted_count": len(accepted_ids),
        "accepted_ids": sorted(accepted_ids),
        "clean_correct_ids": sorted(correct_ids & ids["clean_residual_targets"]),
        "target_self_correct_ids": sorted(correct_ids & ids["target_self_repair"]),
        "accepted_harm_ids": sorted(accepted_harm_ids),
        "accepted_harm_count": len(accepted_harm_ids),
        "duplicate_answer_ids": sorted(duplicate_answer_ids),
        "duplicate_answer_count": len(duplicate_answer_ids),
    }


def _evaluate(
    *,
    rows: Sequence[dict[str, Any]],
    target_ids: dict[str, set[str]],
    conditions: Sequence[str],
    rules_by_fold: dict[int, dict[str, Any]] | None = None,
    global_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    routed: list[dict[str, Any]] = []
    for row in rows:
        rule = global_rule if global_rule is not None else rules_by_fold[int(row["fold"])]  # type: ignore[index]
        routed.append(
            {
                "index": row["index"],
                "example_id": row["example_id"],
                "fallback_label": row["fallback_label"],
                "fallback_correct": row["fallback_correct"],
                "router_rule": rule,
                "conditions": {condition: _apply_rule(row, condition, rule) for condition in conditions},
            }
        )
    summaries = {condition: _summarize_condition(routed, condition, target_ids) for condition in conditions}
    control_union = set().union(
        *[set(summaries[condition]["clean_correct_ids"]) for condition in conditions if condition != "matched"]
    )
    matched_clean = set(summaries["matched"]["clean_correct_ids"])
    duplicate_clean = set(summaries["matched"]["duplicate_answer_ids"]) & target_ids["clean_residual_targets"]
    accepted_harm = summaries["matched"]["accepted_harm_count"]
    mean_sidecar_bits = sum(row["conditions"]["matched"]["sidecar_bits"] for row in routed) / max(len(routed), 1)
    return {
        "condition_summaries": summaries,
        "control_clean_union_ids": sorted(control_union),
        "duplicate_answer_clean_ids": sorted(duplicate_clean),
        "source_necessary_clean_ids": sorted(matched_clean - control_union - duplicate_clean),
        "accepted_harm": int(accepted_harm),
        "mean_sidecar_bits": float(mean_sidecar_bits),
        "rows": routed,
    }


def _prepare_rows(
    *,
    paths: dict[str, pathlib.Path],
    target_set_path: pathlib.Path,
    fallback_label: str,
    outer_folds: int,
    max_sidecar_bits: int,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], list[str]]:
    reference_ids, sketches = _load_condition_sketches(paths, max_sidecar_bits=max_sidecar_bits)
    rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(reference_ids):
        condition_sketches = {condition: by_id[example_id] for condition, by_id in sketches.items()}
        row = {
            "index": index,
            "example_id": example_id,
            "fold": base._fold_for_id(example_id, outer_folds),
            "fallback_label": fallback_label,
            "condition_sketches": condition_sketches,
        }
        fallback = _fallback_from_row(row, fallback_label)
        row["fallback_item"] = fallback
        row["fallback_correct"] = _candidate_correct(fallback)
        rows.append(row)
    return rows, base._load_target_ids(target_set_path), list(paths)


def _live_cv(rows: Sequence[dict[str, Any]], target_ids: dict[str, set[str]], config: GateConfig, conditions: Sequence[str]) -> dict[str, Any]:
    rules: dict[int, dict[str, Any]] = {}
    for fold in range(config.outer_folds):
        rule = _fit_stump(
            rows,
            train_folds=set(range(config.outer_folds)) - {fold},
            accept_penalty=config.accept_penalty,
        )
        rule["fold"] = fold
        rules[fold] = rule
    result = _evaluate(rows=rows, target_ids=target_ids, conditions=conditions, rules_by_fold=rules)
    result["fold_rules"] = [rules[idx] for idx in sorted(rules)]
    return result


def _status(result: dict[str, Any], *, config: GateConfig, holdout: bool) -> tuple[str, list[str]]:
    matched = result["condition_summaries"]["matched"]
    min_correct = config.min_holdout_correct if holdout else config.min_live_correct
    min_clean = config.min_holdout_clean_source_necessary if holdout else config.min_live_clean_source_necessary
    criteria = {
        "min_correct": matched["correct_count"] >= min_correct,
        "min_clean_source_necessary": len(result["source_necessary_clean_ids"]) >= min_clean,
        "max_clean_control_union": len(result["control_clean_union_ids"]) <= config.max_clean_control_union,
        "max_accepted_harm": result["accepted_harm"] <= config.max_accepted_harm,
    }
    failing = [name for name, passed in criteria.items() if not passed]
    return ("passes" if not failing else "fails", failing)


def _predictions_from_result(result: dict[str, Any], *, method: str) -> list[dict[str, Any]]:
    return [
        {
            "index": row["index"],
            "example_id": row["example_id"],
            "method": method,
            "prediction": row["conditions"]["matched"]["prediction"],
            "correct": bool(row["conditions"]["matched"]["correct"]),
            "accepted_sidecar": bool(row["conditions"]["matched"]["accepted_sidecar"]),
            "selected_label": row["conditions"]["matched"]["label"],
        }
        for row in result["rows"]
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = GateConfig(
        outer_folds=int(args.outer_folds),
        accept_penalty=float(args.accept_penalty),
        max_sidecar_bits=int(args.max_sidecar_bits),
        min_live_correct=int(args.min_live_correct),
        min_live_clean_source_necessary=int(args.min_live_clean_source_necessary),
        min_holdout_correct=int(args.min_holdout_correct),
        min_holdout_clean_source_necessary=int(args.min_holdout_clean_source_necessary),
        max_clean_control_union=int(args.max_clean_control_union),
        max_accepted_harm=int(args.max_accepted_harm),
    )
    live_paths = _condition_paths(args.live_condition_sketch)
    holdout_paths = _condition_paths(args.holdout_condition_sketch)
    if set(live_paths) != set(holdout_paths):
        raise ValueError("Live and holdout condition sets must match")
    conditions = [condition for condition in CONDITIONS if condition in live_paths]
    live_rows, live_ids, _ = _prepare_rows(
        paths=live_paths,
        target_set_path=_resolve(args.live_target_set_json),
        fallback_label=args.fallback_label,
        outer_folds=config.outer_folds,
        max_sidecar_bits=config.max_sidecar_bits,
    )
    holdout_rows, holdout_ids, _ = _prepare_rows(
        paths=holdout_paths,
        target_set_path=_resolve(args.holdout_target_set_json),
        fallback_label=args.fallback_label,
        outer_folds=config.outer_folds,
        max_sidecar_bits=config.max_sidecar_bits,
    )
    live_cv = _live_cv(live_rows, live_ids, config, conditions)
    global_rule = _fit_stump(live_rows, train_folds=None, accept_penalty=config.accept_penalty)
    holdout = _evaluate(rows=holdout_rows, target_ids=holdout_ids, conditions=conditions, global_rule=global_rule)
    live_status, live_failing = _status(live_cv, config=config, holdout=False)
    holdout_status, holdout_failing = _status(holdout, config=config, holdout=True)
    return {
        "date": str(args.date),
        "status": (
            "condition_likelihood_receiver_passes_live_and_holdout"
            if live_status == "passes" and holdout_status == "passes"
            else "condition_likelihood_receiver_fails_gate"
        ),
        "config": {
            "conditions": conditions,
            "features": list(FEATURES),
            "outer_folds": config.outer_folds,
            "accept_penalty": config.accept_penalty,
            "max_sidecar_bits": config.max_sidecar_bits,
            "fallback_label": args.fallback_label,
            "min_live_correct": config.min_live_correct,
            "min_live_clean_source_necessary": config.min_live_clean_source_necessary,
            "min_holdout_correct": config.min_holdout_correct,
            "min_holdout_clean_source_necessary": config.min_holdout_clean_source_necessary,
            "max_clean_control_union": config.max_clean_control_union,
            "max_accepted_harm": config.max_accepted_harm,
        },
        "artifacts": {
            "live_condition_sketches": {condition: _display_path(path) for condition, path in live_paths.items()},
            "live_condition_sketch_sha256": {condition: _sha256_file(path) for condition, path in live_paths.items()},
            "holdout_condition_sketches": {condition: _display_path(path) for condition, path in holdout_paths.items()},
            "holdout_condition_sketch_sha256": {condition: _sha256_file(path) for condition, path in holdout_paths.items()},
        },
        "live_cv": {**live_cv, "status": live_status, "failing_criteria": live_failing},
        "frozen_rule": global_rule,
        "holdout_frozen": {**holdout, "status": holdout_status, "failing_criteria": holdout_failing},
        "predictions": _predictions_from_result(holdout, method="condition_likelihood_receiver"),
    }


def _summary_lines(label: str, result: dict[str, Any]) -> list[str]:
    matched = result["condition_summaries"]["matched"]
    return [
        f"### {label}",
        "",
        f"- status: `{result['status']}`",
        f"- failing criteria: `{', '.join(result['failing_criteria']) or 'none'}`",
        f"- matched correct: `{matched['correct_count']}`",
        f"- matched accepted: `{matched['accepted_count']}`",
        f"- clean source-necessary: `{len(result['source_necessary_clean_ids'])}`",
        f"- clean control union: `{len(result['control_clean_union_ids'])}`",
        f"- duplicate-answer clean IDs: `{len(result.get('duplicate_answer_clean_ids', []))}`",
        f"- accepted harm: `{result['accepted_harm']}`",
        f"- mean sidecar bits: `{result['mean_sidecar_bits']:.3f}`",
        f"- source-necessary IDs: {', '.join(f'`{item}`' for item in result['source_necessary_clean_ids']) or 'none'}",
        "",
    ]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Condition-Specific Likelihood Receiver Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- frozen feature: `{payload['frozen_rule'].get('feature')}`",
        f"- frozen direction: `{payload['frozen_rule'].get('direction')}`",
        f"- frozen threshold: `{payload['frozen_rule'].get('threshold')}`",
        f"- conditions: `{', '.join(payload['config']['conditions'])}`",
        "",
    ]
    lines.extend(_summary_lines("Live CV", payload["live_cv"]))
    lines.extend(_summary_lines("Holdout Frozen", payload["holdout_frozen"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-condition-sketch", action="append", required=True)
    parser.add_argument("--live-target-set-json", required=True)
    parser.add_argument("--holdout-condition-sketch", action="append", required=True)
    parser.add_argument("--holdout-target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--accept-penalty", type=float, default=0.10)
    parser.add_argument("--max-sidecar-bits", type=int, default=8)
    parser.add_argument("--min-live-correct", type=int, default=25)
    parser.add_argument("--min-live-clean-source-necessary", type=int, default=3)
    parser.add_argument("--min-holdout-correct", type=int, default=10)
    parser.add_argument("--min-holdout-clean-source-necessary", type=int, default=1)
    parser.add_argument("--max-clean-control-union", type=int, default=0)
    parser.add_argument("--max-accepted-harm", type=int, default=1)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-predictions-jsonl", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = analyze(args)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    predictions_path = _resolve(args.output_predictions_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    _write_jsonl(predictions_path, payload["predictions"])
    print(json.dumps({"status": payload["status"], "output_json": _display_path(output_json)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
