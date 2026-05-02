#!/usr/bin/env python3
"""Gate a source-vs-target router using source generation confidence features."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge import evaluate
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


FEATURES = (
    "mean_chosen_logprob",
    "min_chosen_logprob",
    "final_chosen_logprob",
    "mean_entropy",
    "max_entropy",
    "mean_top1_prob",
    "min_top1_prob",
    "final_top1_prob",
    "mean_top1_top2_logit_margin",
    "min_top1_top2_logit_margin",
    "final_top1_top2_logit_margin",
)
CONDITIONS = ("matched", "zero_source", "shuffled_source", "label_shuffle", "target_only")


@dataclass(frozen=True)
class GateConfig:
    outer_folds: int = 5
    accept_penalty: float = 0.10
    min_live_correct: int = 25
    min_live_clean_source_necessary: int = 4
    min_holdout_correct: int = 10
    min_holdout_clean_source_necessary: int = 2
    max_clean_control_union: int = 0
    max_accepted_harm: int = 1


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_by_id(path: pathlib.Path, method: str | None = None) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    if method is not None:
        grouped = syndrome.harness.group_by_method(rows)
        if method in grouped:
            rows = grouped[method]
    return {str(row["example_id"]): row for row in rows}


def _load_target_ids(path: pathlib.Path) -> dict[str, set[str]]:
    ids = _read_json(path).get("ids", {})
    return {
        "clean_residual_targets": {str(value) for value in ids.get("clean_residual_targets", [])},
        "target_self_repair": {str(value) for value in ids.get("target_self_repair", [])},
        "teacher_only": {str(value) for value in ids.get("teacher_only", [])},
    }


def _fold_for_id(example_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % int(folds)


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
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


def _answer_list(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    if isinstance(answer, list):
        return [str(value) for value in answer]
    return [str(answer)]


def _source_correct(source_row: dict[str, Any] | None, target_row: dict[str, Any]) -> bool:
    if source_row is None:
        return False
    return evaluate._generation_match(str(source_row.get("prediction", "")), _answer_list(target_row))


def _target_prediction(target_row: dict[str, Any]) -> str | None:
    pred = syndrome._prediction_numeric(target_row)
    return pred if pred is not None else str(target_row.get("prediction", ""))


def _source_prediction(source_row: dict[str, Any] | None) -> str | None:
    if source_row is None:
        return None
    pred = syndrome._prediction_numeric(source_row)
    return pred if pred is not None else str(source_row.get("prediction", ""))


def _prepare_rows(
    *,
    diagnostics_path: pathlib.Path,
    target_path: pathlib.Path,
    target_method: str,
    target_set_path: pathlib.Path,
    shuffle_offset: int,
    label_shuffle_offset: int,
    outer_folds: int,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    target_by_id = _records_by_id(target_path, target_method)
    source_rows = _read_jsonl(diagnostics_path)
    reference_ids = [str(row["example_id"]) for row in source_rows]
    source_by_id = {str(row["example_id"]): row for row in source_rows}
    target_ids = _load_target_ids(target_set_path)
    prepared: list[dict[str, Any]] = []
    for index, example_id in enumerate(reference_ids):
        target_row = target_by_id[example_id]
        shuffled_id = reference_ids[(index + shuffle_offset) % len(reference_ids)]
        label_id = reference_ids[(index + label_shuffle_offset) % len(reference_ids)]
        prepared.append(
            {
                "index": index,
                "example_id": example_id,
                "fold": _fold_for_id(example_id, outer_folds),
                "gold_answer": _answer_list(target_row),
                "target_prediction": _target_prediction(target_row),
                "target_correct": bool(target_row.get("correct")),
                "source_conditions": {
                    "matched": source_by_id[example_id],
                    "zero_source": None,
                    "shuffled_source": source_by_id[shuffled_id],
                    "label_shuffle": source_by_id[label_id],
                    "target_only": None,
                },
            }
        )
    return prepared, target_ids


def _fit_stump(
    rows: Sequence[dict[str, Any]],
    *,
    train_folds: set[int] | None,
    accept_penalty: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in FEATURES:
        values = [
            row["source_conditions"]["matched"].get(feature)
            for row in rows
            if (train_folds is None or int(row["fold"]) in train_folds)
            and row["source_conditions"]["matched"].get(feature) is not None
        ]
        for threshold in _candidate_thresholds([float(value) for value in values]):
            for direction in ("ge", "le"):
                help_count = harm_count = accept_count = 0
                for row in rows:
                    if train_folds is not None and int(row["fold"]) not in train_folds:
                        continue
                    accepted = _accepts(
                        row["source_conditions"]["matched"].get(feature),
                        threshold=threshold,
                        direction=direction,
                    )
                    if not accepted:
                        continue
                    accept_count += 1
                    sidecar_correct = _source_correct(
                        row["source_conditions"]["matched"],
                        {"answer": row["gold_answer"]},
                    )
                    fallback_correct = bool(row["target_correct"])
                    help_count += int(sidecar_correct and not fallback_correct)
                    harm_count += int((not sidecar_correct) and fallback_correct)
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
    source_row = row["source_conditions"][condition]
    accepted = False
    if rule.get("feature") and source_row is not None:
        accepted = _accepts(
            source_row.get(str(rule["feature"])),
            threshold=float(rule["threshold"]),
            direction=str(rule["direction"]),
        )
    prediction = _source_prediction(source_row) if accepted else row["target_prediction"]
    correct = (
        _source_correct(source_row, {"answer": row["gold_answer"]})
        if accepted
        else bool(row["target_correct"])
    )
    return {
        "prediction": prediction,
        "correct": bool(correct),
        "accepted_source": bool(accepted),
    }


def _summarize_condition(rows: Sequence[dict[str, Any]], condition: str, ids: dict[str, set[str]]) -> dict[str, Any]:
    correct_ids = {
        str(row["example_id"])
        for row in rows
        if bool(row["conditions"][condition]["correct"])
    }
    return {
        "condition": condition,
        "correct_count": len(correct_ids),
        "correct_ids": sorted(correct_ids),
        "clean_correct_ids": sorted(correct_ids & ids["clean_residual_targets"]),
        "target_self_correct_ids": sorted(correct_ids & ids["target_self_repair"]),
        "teacher_only_correct_ids": sorted(correct_ids & ids["teacher_only"]),
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
                "gold_answer": row["gold_answer"],
                "target_prediction": row["target_prediction"],
                "target_correct": row["target_correct"],
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
            row["conditions"]["matched"]["accepted_source"]
            and not row["conditions"]["matched"]["correct"]
            and row["target_correct"]
        )
        for row in routed
    )
    return {
        "condition_summaries": summaries,
        "control_clean_union_ids": sorted(control_union),
        "source_necessary_clean_ids": sorted(source_necessary),
        "accepted_harm": int(accepted_harm),
        "rows": routed,
    }


def _live_cv(rows: Sequence[dict[str, Any]], target_ids: dict[str, set[str]], config: GateConfig) -> dict[str, Any]:
    rules = {}
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


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = GateConfig(
        outer_folds=int(args.outer_folds),
        accept_penalty=float(args.accept_penalty),
        min_live_correct=int(args.min_live_correct),
        min_live_clean_source_necessary=int(args.min_live_clean_source_necessary),
        min_holdout_correct=int(args.min_holdout_correct),
        min_holdout_clean_source_necessary=int(args.min_holdout_clean_source_necessary),
        max_clean_control_union=int(args.max_clean_control_union),
        max_accepted_harm=int(args.max_accepted_harm),
    )
    live_diag = _resolve(args.live_diagnostics_jsonl)
    holdout_diag = _resolve(args.holdout_diagnostics_jsonl)
    live_rows, live_ids = _prepare_rows(
        diagnostics_path=live_diag,
        target_path=_resolve(args.live_target_jsonl),
        target_method=args.target_method,
        target_set_path=_resolve(args.live_target_set_json),
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        outer_folds=config.outer_folds,
    )
    holdout_rows, holdout_ids = _prepare_rows(
        diagnostics_path=holdout_diag,
        target_path=_resolve(args.holdout_target_jsonl),
        target_method=args.target_method,
        target_set_path=_resolve(args.holdout_target_set_json),
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        outer_folds=config.outer_folds,
    )
    live_cv = _live_cv(live_rows, live_ids, config)
    global_rule = _fit_stump(live_rows, train_folds=None, accept_penalty=config.accept_penalty)
    holdout = _evaluate(rows=holdout_rows, target_ids=holdout_ids, global_rule=global_rule)
    live_status, live_failing = _status(live_cv, config=config, holdout=False)
    holdout_status, holdout_failing = _status(holdout, config=config, holdout=True)
    return {
        "date": str(args.date),
        "status": (
            "source_confidence_router_passes_live_and_holdout"
            if live_status == "passes" and holdout_status == "passes"
            else "source_confidence_router_fails_gate"
        ),
        "config": {
            "features": list(FEATURES),
            "outer_folds": config.outer_folds,
            "accept_penalty": config.accept_penalty,
            "min_live_correct": config.min_live_correct,
            "min_live_clean_source_necessary": config.min_live_clean_source_necessary,
            "min_holdout_correct": config.min_holdout_correct,
            "min_holdout_clean_source_necessary": config.min_holdout_clean_source_necessary,
            "max_clean_control_union": config.max_clean_control_union,
            "max_accepted_harm": config.max_accepted_harm,
        },
        "artifacts": {
            "live_diagnostics_jsonl": str(live_diag.relative_to(ROOT)),
            "live_diagnostics_sha256": _sha256_file(live_diag),
            "holdout_diagnostics_jsonl": str(holdout_diag.relative_to(ROOT)),
            "holdout_diagnostics_sha256": _sha256_file(holdout_diag),
        },
        "live_cv": {**live_cv, "status": live_status, "failing_criteria": live_failing},
        "frozen_rule": global_rule,
        "holdout_frozen": {**holdout, "status": holdout_status, "failing_criteria": holdout_failing},
    }


def _summary_lines(label: str, result: dict[str, Any]) -> list[str]:
    matched = result["condition_summaries"]["matched"]
    return [
        f"### {label}",
        "",
        f"- status: `{result['status']}`",
        f"- failing criteria: `{', '.join(result['failing_criteria']) or 'none'}`",
        f"- matched correct: `{matched['correct_count']}`",
        f"- clean source-necessary: `{len(result['source_necessary_clean_ids'])}`",
        f"- clean control union: `{len(result['control_clean_union_ids'])}`",
        f"- accepted harm: `{result['accepted_harm']}`",
        f"- source-necessary IDs: {', '.join(f'`{item}`' for item in result['source_necessary_clean_ids']) or 'none'}",
        "",
    ]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source Confidence Router Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- frozen feature: `{payload['frozen_rule'].get('feature')}`",
        f"- frozen direction: `{payload['frozen_rule'].get('direction')}`",
        f"- frozen threshold: `{payload['frozen_rule'].get('threshold')}`",
        "",
    ]
    lines.extend(_summary_lines("Live CV", payload["live_cv"]))
    lines.extend(_summary_lines("Holdout Frozen", payload["holdout_frozen"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-diagnostics-jsonl", required=True)
    parser.add_argument("--live-target-jsonl", required=True)
    parser.add_argument("--live-target-set-json", required=True)
    parser.add_argument("--holdout-diagnostics-jsonl", required=True)
    parser.add_argument("--holdout-target-jsonl", required=True)
    parser.add_argument("--holdout-target-set-json", required=True)
    parser.add_argument("--target-method", default="target_alone")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--label-shuffle-offset", type=int, default=17)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--accept-penalty", type=float, default=0.10)
    parser.add_argument("--min-live-correct", type=int, default=25)
    parser.add_argument("--min-live-clean-source-necessary", type=int, default=4)
    parser.add_argument("--min-holdout-correct", type=int, default=10)
    parser.add_argument("--min-holdout-clean-source-necessary", type=int, default=2)
    parser.add_argument("--max-clean-control-union", type=int, default=0)
    parser.add_argument("--max-accepted-harm", type=int, default=1)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = analyze(args)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "output_json": str(output_json.relative_to(ROOT))}, indent=2))
    return payload


if __name__ == "__main__":
    main()
