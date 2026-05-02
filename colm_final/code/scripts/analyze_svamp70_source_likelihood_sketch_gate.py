#!/usr/bin/env python3
"""Gate a rate-capped source likelihood sketch on SVAMP70 live/holdout.

The sketch is a compact source-derived preference over a target-side candidate
pool. It is intentionally separate from the model scorer: this analyzer takes
precomputed candidate likelihood scores, quantizes the margin/confidence, and
tests whether the resulting sidecar can select a helpful candidate while
source-destroyed sketches cannot.
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

from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


CONDITIONS = ("matched", "zero_source", "shuffled_source", "label_shuffle", "target_only", "slots_only")
FEATURES = (
    "top_score",
    "margin",
    "quantized_margin",
    "confidence",
    "top_is_source",
    "top_is_text",
    "top_is_target",
)


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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _records_for_method(spec: syndrome.RowSpec) -> list[dict[str, Any]]:
    return syndrome._records_for_method(spec)


def _records_by_id(spec: syndrome.RowSpec) -> dict[str, dict[str, Any]]:
    return _by_id(_records_for_method(spec))


def _load_target_ids(path: pathlib.Path) -> dict[str, set[str]]:
    ids = _read_json(path).get("ids", {})
    return {
        "clean_residual_targets": {str(value) for value in ids.get("clean_residual_targets", [])},
        "target_self_repair": {str(value) for value in ids.get("target_self_repair", [])},
        "clean_source_only": {str(value) for value in ids.get("clean_source_only", [])},
    }


def _fold_for_id(example_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % max(int(folds), 1)


def _answer_list(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    if isinstance(answer, list):
        return [str(value) for value in answer]
    return [str(answer)]


def _correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def _prediction_numeric_or_text(row: dict[str, Any]) -> str:
    numeric = syndrome._prediction_numeric(row)
    return numeric if numeric is not None else str(row.get("prediction", ""))


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values if math.isfinite(float(value))))
    thresholds = list(unique)
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(thresholds))


def _quantize_unit(value: float, *, bits: int) -> int:
    levels = max(2**max(int(bits), 1) - 1, 1)
    clipped = max(0.0, min(1.0, float(value)))
    return int(round(clipped * levels))


def _score_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("candidate_scores", row.get("scores"))
    if isinstance(raw, dict):
        return [
            {"label": str(label), "score": float(score)}
            for label, score in raw.items()
            if score is not None
        ]
    if not isinstance(raw, list):
        raise ValueError(f"Sketch row {row.get('example_id')} lacks candidate_scores")
    items: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("score") is None:
            continue
        items.append({**item, "label": str(item["label"]), "score": float(item["score"])})
    return items


def _sketch_from_scores(row: dict[str, Any], *, max_sidecar_bits: int) -> dict[str, Any]:
    items = sorted(_score_items(row), key=lambda item: (-float(item["score"]), str(item["label"])))
    if not items:
        return {
            "example_id": str(row.get("example_id")),
            "top_label": None,
            "top_score": None,
            "margin": 0.0,
            "quantized_margin": 0,
            "confidence": 0.0,
            "top_is_source": 0.0,
            "top_is_text": 0.0,
            "top_is_target": 0.0,
            "sidecar_bits": 0,
        }
    top = items[0]
    second_score = float(items[1]["score"]) if len(items) > 1 else float(top["score"])
    margin = float(top["score"]) - second_score
    confidence = float(row.get("confidence", margin))
    label_bits = max(1, math.ceil(math.log2(max(len(items), 2))))
    confidence_bits = max(0, int(max_sidecar_bits) - label_bits)
    quantized_margin = _quantize_unit(1.0 / (1.0 + math.exp(-margin)), bits=confidence_bits)
    top_label = str(top["label"])
    return {
        "example_id": str(row.get("example_id")),
        "top_label": top_label,
        "top_score": float(top["score"]),
        "margin": float(margin),
        "quantized_margin": float(quantized_margin),
        "confidence": float(confidence),
        "top_is_source": float(top_label == "source"),
        "top_is_text": float(top_label == "text"),
        "top_is_target": float(top_label == "target"),
        "sidecar_bits": int(label_bits + confidence_bits),
        "candidate_scores": items,
    }


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


def _fit_stump(
    rows: Sequence[dict[str, Any]],
    *,
    train_folds: set[int] | None,
    accept_penalty: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in FEATURES:
        values = [
            float(row["sketch_conditions"]["matched"].get(feature))
            for row in rows
            if (train_folds is None or int(row["fold"]) in train_folds)
            and row["sketch_conditions"]["matched"].get(feature) is not None
        ]
        for threshold in _candidate_thresholds(values):
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


def _apply_rule(row: dict[str, Any], condition: str, rule: dict[str, Any]) -> dict[str, Any]:
    sketch = row["sketch_conditions"][condition]
    accepted = False
    if rule.get("feature") and sketch is not None:
        accepted = _accepts(
            sketch.get(str(rule["feature"])),
            threshold=float(rule["threshold"]),
            direction=str(rule["direction"]),
        )
    label = sketch.get("top_label") if accepted and sketch is not None else row["fallback_label"]
    if label not in row["candidate_rows"]:
        label = row["fallback_label"]
        accepted = False
    candidate = row["candidate_rows"][label]
    return {
        "label": label,
        "prediction": _prediction_numeric_or_text(candidate),
        "correct": _correct(candidate),
        "accepted_sidecar": bool(accepted),
        "sidecar_bits": int(sketch.get("sidecar_bits", 0)) if sketch else 0,
    }


def _summarize_condition(rows: Sequence[dict[str, Any]], condition: str, ids: dict[str, set[str]]) -> dict[str, Any]:
    correct_ids = {
        str(row["example_id"])
        for row in rows
        if bool(row["conditions"][condition]["correct"])
    }
    accepted_ids = {
        str(row["example_id"])
        for row in rows
        if bool(row["conditions"][condition]["accepted_sidecar"])
    }
    return {
        "condition": condition,
        "correct_count": len(correct_ids),
        "correct_ids": sorted(correct_ids),
        "accepted_count": len(accepted_ids),
        "accepted_ids": sorted(accepted_ids),
        "clean_correct_ids": sorted(correct_ids & ids["clean_residual_targets"]),
        "target_self_correct_ids": sorted(correct_ids & ids["target_self_repair"]),
    }


def _evaluate(
    *,
    rows: Sequence[dict[str, Any]],
    target_ids: dict[str, set[str]],
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
                "conditions": {
                    condition: _apply_rule(row, condition, rule)
                    for condition in CONDITIONS
                },
            }
        )
    summaries = {
        condition: _summarize_condition(routed, condition, target_ids)
        for condition in CONDITIONS
    }
    control_union = set().union(
        *[
            set(summaries[condition]["clean_correct_ids"])
            for condition in CONDITIONS
            if condition != "matched"
        ]
    )
    matched_clean = set(summaries["matched"]["clean_correct_ids"])
    source_necessary = matched_clean - control_union
    accepted_harm = sum(
        int(
            row["conditions"]["matched"]["accepted_sidecar"]
            and not row["conditions"]["matched"]["correct"]
            and row["fallback_correct"]
        )
        for row in routed
    )
    mean_sidecar_bits = (
        sum(row["conditions"]["matched"]["sidecar_bits"] for row in routed) / max(len(routed), 1)
    )
    return {
        "condition_summaries": summaries,
        "control_clean_union_ids": sorted(control_union),
        "source_necessary_clean_ids": sorted(source_necessary),
        "accepted_harm": int(accepted_harm),
        "mean_sidecar_bits": float(mean_sidecar_bits),
        "rows": routed,
    }


def _live_cv(rows: Sequence[dict[str, Any]], target_ids: dict[str, set[str]], config: GateConfig) -> dict[str, Any]:
    rules: dict[int, dict[str, Any]] = {}
    for fold in range(config.outer_folds):
        rule = _fit_stump(
            rows,
            train_folds=set(range(config.outer_folds)) - {fold},
            accept_penalty=config.accept_penalty,
        )
        rule["fold"] = fold
        rules[fold] = rule
    result = _evaluate(rows=rows, target_ids=target_ids, rules_by_fold=rules)
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


def _sketch_by_id(path: pathlib.Path, *, max_sidecar_bits: int) -> dict[str, dict[str, Any]]:
    return {
        str(row["example_id"]): _sketch_from_scores(row, max_sidecar_bits=max_sidecar_bits)
        for row in _read_jsonl(path)
    }


def _condition_sketches(
    *,
    example_id: str,
    index: int,
    reference_ids: Sequence[str],
    matched_by_id: dict[str, dict[str, Any]],
    shuffle_offset: int,
    label_shuffle_offset: int,
) -> dict[str, dict[str, Any] | None]:
    shuffled_id = reference_ids[(index + int(shuffle_offset)) % len(reference_ids)]
    label_id = reference_ids[(index + int(label_shuffle_offset)) % len(reference_ids)]
    return {
        "matched": matched_by_id[example_id],
        "zero_source": None,
        "shuffled_source": matched_by_id[shuffled_id],
        "label_shuffle": matched_by_id[label_id],
        "target_only": None,
        "slots_only": None,
    }


def _prepare_rows(
    *,
    sketch_path: pathlib.Path,
    target_set_path: pathlib.Path,
    candidate_specs: Sequence[syndrome.RowSpec],
    fallback_label: str,
    shuffle_offset: int,
    label_shuffle_offset: int,
    outer_folds: int,
    max_sidecar_bits: int,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    candidate_by_label = {spec.label: _records_by_id(spec) for spec in candidate_specs}
    if fallback_label not in candidate_by_label:
        raise ValueError(f"fallback label {fallback_label!r} missing from candidate labels")
    reference_ids = [str(row["example_id"]) for row in _records_for_method(candidate_specs[0])]
    for label, records in candidate_by_label.items():
        missing = [example_id for example_id in reference_ids if example_id not in records]
        if missing:
            raise ValueError(f"candidate {label!r} missing IDs: {missing[:5]}")
    sketches = _sketch_by_id(sketch_path, max_sidecar_bits=max_sidecar_bits)
    missing_sketches = [example_id for example_id in reference_ids if example_id not in sketches]
    if missing_sketches:
        raise ValueError(f"sketch file missing IDs: {missing_sketches[:5]}")
    rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(reference_ids):
        candidate_rows = {label: records[example_id] for label, records in candidate_by_label.items()}
        fallback = candidate_rows[fallback_label]
        rows.append(
            {
                "index": index,
                "example_id": example_id,
                "fold": _fold_for_id(example_id, outer_folds),
                "fallback_label": fallback_label,
                "fallback_correct": _correct(fallback),
                "candidate_rows": candidate_rows,
                "sketch_conditions": _condition_sketches(
                    example_id=example_id,
                    index=index,
                    reference_ids=reference_ids,
                    matched_by_id=sketches,
                    shuffle_offset=shuffle_offset,
                    label_shuffle_offset=label_shuffle_offset,
                ),
            }
        )
    return rows, _load_target_ids(target_set_path)


def _predictions_from_result(result: dict[str, Any], *, method: str, candidate_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["example_id"]): row for row in candidate_rows}
    records: list[dict[str, Any]] = []
    for row in result["rows"]:
        ref = by_id[str(row["example_id"])]
        condition = row["conditions"]["matched"]
        records.append(
            {
                "index": row["index"],
                "example_id": row["example_id"],
                "method": method,
                "prediction": condition["prediction"],
                "answer": ref.get("answer"),
                "correct": bool(condition["correct"]),
                "accepted_sidecar": bool(condition["accepted_sidecar"]),
                "selected_label": condition["label"],
            }
        )
    return records


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
    live_sketch = _resolve(args.live_sketch_jsonl)
    holdout_sketch = _resolve(args.holdout_sketch_jsonl)
    live_candidates = [syndrome._parse_spec(spec) for spec in args.live_candidate]
    holdout_candidates = [syndrome._parse_spec(spec) for spec in args.holdout_candidate]
    live_rows, live_ids = _prepare_rows(
        sketch_path=live_sketch,
        target_set_path=_resolve(args.live_target_set_json),
        candidate_specs=live_candidates,
        fallback_label=args.fallback_label,
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        outer_folds=config.outer_folds,
        max_sidecar_bits=config.max_sidecar_bits,
    )
    holdout_rows, holdout_ids = _prepare_rows(
        sketch_path=holdout_sketch,
        target_set_path=_resolve(args.holdout_target_set_json),
        candidate_specs=holdout_candidates,
        fallback_label=args.fallback_label,
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        outer_folds=config.outer_folds,
        max_sidecar_bits=config.max_sidecar_bits,
    )
    live_cv = _live_cv(live_rows, live_ids, config)
    global_rule = _fit_stump(live_rows, train_folds=None, accept_penalty=config.accept_penalty)
    holdout = _evaluate(rows=holdout_rows, target_ids=holdout_ids, global_rule=global_rule)
    live_status, live_failing = _status(live_cv, config=config, holdout=False)
    holdout_status, holdout_failing = _status(holdout, config=config, holdout=True)
    return {
        "date": str(args.date),
        "status": (
            "source_likelihood_sketch_passes_live_and_holdout"
            if live_status == "passes" and holdout_status == "passes"
            else "source_likelihood_sketch_fails_gate"
        ),
        "config": {
            "features": list(FEATURES),
            "conditions": list(CONDITIONS),
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
            "live_sketch_jsonl": _display_path(live_sketch),
            "live_sketch_sha256": _sha256_file(live_sketch),
            "holdout_sketch_jsonl": _display_path(holdout_sketch),
            "holdout_sketch_sha256": _sha256_file(holdout_sketch),
        },
        "live_cv": {**live_cv, "status": live_status, "failing_criteria": live_failing},
        "frozen_rule": global_rule,
        "holdout_frozen": {**holdout, "status": holdout_status, "failing_criteria": holdout_failing},
        "predictions": _predictions_from_result(
            holdout,
            method="source_likelihood_sketch",
            candidate_rows=_records_for_method(holdout_candidates[0]),
        ),
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
        f"- accepted harm: `{result['accepted_harm']}`",
        f"- mean sidecar bits: `{result['mean_sidecar_bits']:.3f}`",
        f"- source-necessary IDs: {', '.join(f'`{item}`' for item in result['source_necessary_clean_ids']) or 'none'}",
        "",
    ]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SVAMP70 Source Likelihood Sketch Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- frozen feature: `{payload['frozen_rule'].get('feature')}`",
        f"- frozen direction: `{payload['frozen_rule'].get('direction')}`",
        f"- frozen threshold: `{payload['frozen_rule'].get('threshold')}`",
        f"- max sidecar bits: `{payload['config']['max_sidecar_bits']}`",
        "",
    ]
    lines.extend(_summary_lines("Live CV", payload["live_cv"]))
    lines.extend(_summary_lines("Holdout Frozen", payload["holdout_frozen"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-sketch-jsonl", required=True)
    parser.add_argument("--live-candidate", action="append", required=True)
    parser.add_argument("--live-target-set-json", required=True)
    parser.add_argument("--holdout-sketch-jsonl", required=True)
    parser.add_argument("--holdout-candidate", action="append", required=True)
    parser.add_argument("--holdout-target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--label-shuffle-offset", type=int, default=17)
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
    print(
        json.dumps(
            {"status": payload["status"], "output_json": _display_path(output_json)},
            indent=2,
        )
    )
    return payload


if __name__ == "__main__":
    main()
