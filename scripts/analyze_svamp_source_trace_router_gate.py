#!/usr/bin/env python3
"""Train a source-trace consistency router on live SVAMP70 and freeze to holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.evaluate import _generation_example_id, load_generation
from scripts import analyze_svamp32_source_only_sidecar_router_gate as base_gate
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


CONDITIONS = (
    "matched",
    "zero_source",
    "shuffled_source",
    "label_shuffle",
    "same_norm_noise",
    "target_only",
    "slots_only",
    "equation_permuted",
)
FEATURES = (
    "source_final_value_matches_last_equation",
    "source_equation_valid_fraction",
    "prompt_number_coverage",
    "source_answer_reused_in_trace",
    "valid_add_count",
    "valid_sub_count",
    "valid_mul_count",
    "valid_div_count",
)


@dataclass(frozen=True)
class RouterConfig:
    moduli: tuple[int, ...]
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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prompt_by_id(eval_file: pathlib.Path | None) -> dict[str, str]:
    if eval_file is None:
        return {}
    return {
        _generation_example_id(example): str(example.prompt)
        for example in load_generation(str(eval_file))
    }


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def _close(left: float, right: float, *, tol: float = 1e-6) -> bool:
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def _equations(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*([+\-*/×x÷])\s*([-+]?\d+(?:\.\d+)?)\s*=\s*([-+]?\d+(?:\.\d+)?)"
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        left = float(match.group(1))
        op = match.group(2)
        right = float(match.group(3))
        result = float(match.group(4))
        if op in ("+",):
            expected = left + right
            op_key = "add"
        elif op in ("-",):
            expected = left - right
            op_key = "sub"
        elif op in ("*", "×", "x"):
            expected = left * right
            op_key = "mul"
        else:
            expected = left / right if right != 0 else math.nan
            op_key = "div"
        valid = math.isfinite(expected) and _close(expected, result)
        rows.append(
            {
                "left": left,
                "op": op_key,
                "right": right,
                "result": result,
                "expected": expected,
                "valid": valid,
            }
        )
    return rows


def _permuted_equation_text(text: str, example_id: str, seed: int) -> str:
    matches = list(
        re.finditer(
            r"([-+]?\d+(?:\.\d+)?\s*[+\-*/×x÷]\s*[-+]?\d+(?:\.\d+)?\s*=\s*)([-+]?\d+(?:\.\d+)?)",
            text,
        )
    )
    if len(matches) < 2:
        return text
    results = [match.group(2) for match in matches]
    offset = (
        int.from_bytes(hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()[:8], "big")
        % (len(results) - 1)
    ) + 1
    rotated = results[offset:] + results[:offset]
    pieces: list[str] = []
    cursor = 0
    for match, replacement in zip(matches, rotated, strict=True):
        pieces.append(text[cursor : match.start(2)])
        pieces.append(replacement)
        cursor = match.end(2)
    pieces.append(text[cursor:])
    return "".join(pieces)


def _feature_values(
    *,
    source_row: dict[str, Any] | None,
    prompt: str,
    example_id: str,
    permute_equations: bool,
    permutation_seed: int,
) -> dict[str, float | None]:
    if source_row is None or syndrome._prediction_numeric(source_row) is None:
        return {feature: None for feature in FEATURES}
    text = str(source_row.get("prediction", "") or "")
    if permute_equations:
        text = _permuted_equation_text(text, example_id, permutation_seed)
    final_numeric = syndrome._prediction_numeric({**source_row, "prediction": text})
    if final_numeric is None:
        return {feature: None for feature in FEATURES}
    try:
        final_value = float(final_numeric)
    except ValueError:
        final_value = math.nan
    equations = _equations(text)
    valid_equations = [row for row in equations if row["valid"]]
    prompt_numbers = set(round(value, 6) for value in _numbers(prompt))
    source_numbers = set(round(value, 6) for value in _numbers(text))
    coverage = (
        len(prompt_numbers & source_numbers) / max(len(prompt_numbers), 1)
        if prompt_numbers
        else None
    )
    last_match = None
    if equations and math.isfinite(final_value):
        last_match = float(_close(final_value, equations[-1]["result"]))
    reused = 0.0
    if math.isfinite(final_value):
        reused = float(any(_close(final_value, row["result"]) for row in valid_equations))
    valid_fraction = len(valid_equations) / max(len(equations), 1) if equations else 0.0
    op_counts = {
        op: float(sum(1 for row in valid_equations if row["op"] == op))
        for op in ("add", "sub", "mul", "div")
    }
    return {
        "source_final_value_matches_last_equation": last_match,
        "source_equation_valid_fraction": float(valid_fraction),
        "prompt_number_coverage": coverage,
        "source_answer_reused_in_trace": reused,
        "valid_add_count": op_counts["add"],
        "valid_sub_count": op_counts["sub"],
        "valid_mul_count": op_counts["mul"],
        "valid_div_count": op_counts["div"],
    }


def _fold_for_id(example_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % int(folds)


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        return []
    thresholds = list(unique)
    if len(unique) > 1:
        thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(thresholds))


def _accepts(value: float | None, *, threshold: float | None, direction: str) -> bool:
    if value is None or threshold is None:
        return False
    if direction == "le":
        return value <= threshold
    if direction == "ge":
        return value >= threshold
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
            row["features"]["matched"].get(feature)
            for row in rows
            if (train_folds is None or int(row["fold"]) in train_folds)
            and row["features"]["matched"].get(feature) is not None
        ]
        for threshold in _candidate_thresholds([float(value) for value in values]):
            for direction in ("le", "ge"):
                help_count = harm_count = accept_count = 0
                for row in rows:
                    if train_folds is not None and int(row["fold"]) not in train_folds:
                        continue
                    accepted = _accepts(
                        row["features"]["matched"].get(feature),
                        threshold=threshold,
                        direction=direction,
                    )
                    if not accepted:
                        continue
                    accept_count += 1
                    sidecar_correct = bool(row["raw_conditions"]["matched"]["correct"])
                    fallback_correct = bool(row["fallback_correct"])
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
                    -candidate["threshold"],
                    candidate["direction"],
                ) > (
                    best["train_score"],
                    -best["train_harm"],
                    best["train_help"],
                    -best["train_accept"],
                    best["feature"],
                    -best["threshold"],
                    best["direction"],
                ):
                    best = candidate
    if best is None:
        return {
            "feature": None,
            "threshold": None,
            "direction": "none",
            "train_help": 0,
            "train_harm": 0,
            "train_accept": 0,
            "train_score": 0.0,
        }
    return best


def _apply_router(row: dict[str, Any], condition: str, rule: dict[str, Any]) -> dict[str, Any]:
    raw_condition = "matched" if condition == "equation_permuted" else condition
    raw = row["raw_conditions"][raw_condition]
    feature = rule.get("feature")
    accepted = False
    if feature:
        accepted = _accepts(
            row["features"][condition].get(str(feature)),
            threshold=float(rule["threshold"]),
            direction=str(rule["direction"]),
        )
    prediction = raw["prediction"] if accepted else row["fallback_prediction"]
    return {
        "prediction": prediction,
        "correct": prediction == row["gold_answer"],
        "accepted_sidecar": bool(accepted),
        "candidate_labels": raw.get("candidate_labels", []),
    }


def _condition_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    target_ids: dict[str, set[str]],
) -> dict[str, Any]:
    return {
        condition: syndrome._summarize_condition(
            list(rows),
            condition=condition,
            clean_ids=target_ids["clean_residual_targets"],
            target_self_ids=target_ids["target_self_repair"],
            teacher_only_ids=target_ids["teacher_only"],
        )
        for condition in CONDITIONS
    }


def _source_rows_by_condition(
    *,
    reference_ids: Sequence[str],
    source_by_id: dict[str, dict[str, Any]],
    index: int,
    shuffle_offset: int,
    label_shuffle_offset: int,
) -> dict[str, dict[str, Any] | None]:
    example_id = reference_ids[index]
    shuffled_id = reference_ids[(index + shuffle_offset) % len(reference_ids)]
    label_shuffled_id = reference_ids[(index + label_shuffle_offset) % len(reference_ids)]
    return {
        "matched": source_by_id[example_id],
        "zero_source": None,
        "shuffled_source": source_by_id[shuffled_id],
        "label_shuffle": source_by_id[label_shuffled_id],
        "same_norm_noise": None,
        "target_only": None,
        "slots_only": None,
        "equation_permuted": source_by_id[example_id],
    }


def _raw_payload(
    *,
    target_spec: syndrome.RowSpec,
    source_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    config: RouterConfig,
    fallback_label: str,
    shuffle_offset: int,
    label_shuffle_offset: int,
    noise_seed: int,
    min_numeric_coverage: int,
    run_date: str,
) -> dict[str, Any]:
    return base_gate.analyze(
        target_spec=target_spec,
        source_spec=source_spec,
        candidate_specs=candidate_specs,
        target_set_path=target_set_path,
        moduli_sets=[config.moduli],
        fallback_label=fallback_label,
        shuffle_offset=shuffle_offset,
        label_shuffle_offset=label_shuffle_offset,
        noise_seed=noise_seed,
        min_correct=0,
        min_target_self=0,
        min_clean_source_necessary=0,
        max_control_clean_union=999999,
        min_numeric_coverage=min_numeric_coverage,
        run_date=run_date,
    )


def _prepare_rows(
    *,
    raw_payload: dict[str, Any],
    target_spec: syndrome.RowSpec,
    source_spec: syndrome.RowSpec,
    target_set_path: pathlib.Path,
    eval_file: pathlib.Path | None,
    shuffle_offset: int,
    label_shuffle_offset: int,
    permutation_seed: int,
    outer_folds: int,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    target_records = syndrome._records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    source_records = syndrome._subset_reference_order(
        syndrome._records_for_method(source_spec),
        reference_ids,
    )
    source_by_id = syndrome._by_id(source_records)
    target_ids = syndrome._load_target_ids(target_set_path)
    prompts = _prompt_by_id(eval_file)
    raw_rows = raw_payload["runs"][0]["rows"]
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        index = int(raw_row["index"])
        example_id = str(raw_row["example_id"])
        source_conditions = _source_rows_by_condition(
            reference_ids=reference_ids,
            source_by_id=source_by_id,
            index=index,
            shuffle_offset=shuffle_offset,
            label_shuffle_offset=label_shuffle_offset,
        )
        features = {
            condition: _feature_values(
                source_row=source_conditions[condition],
                prompt=prompts.get(example_id, ""),
                example_id=example_id,
                permute_equations=(condition == "equation_permuted"),
                permutation_seed=permutation_seed,
            )
            for condition in CONDITIONS
        }
        rows.append(
            {
                "index": index,
                "example_id": example_id,
                "fold": _fold_for_id(example_id, outer_folds),
                "labels": list(raw_row["labels"]),
                "gold_answer": raw_row["gold_answer"],
                "fallback_prediction": raw_row["fallback_prediction"],
                "fallback_correct": raw_row["fallback_prediction"] == raw_row["gold_answer"],
                "raw_conditions": raw_row["conditions"],
                "features": features,
            }
        )
    return rows, target_ids


def _evaluate_with_rules(
    *,
    prepared_rows: Sequence[dict[str, Any]],
    target_ids: dict[str, set[str]],
    rules_by_fold: dict[int, dict[str, Any]],
    global_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    routed_rows: list[dict[str, Any]] = []
    for row in prepared_rows:
        rule = global_rule if global_rule is not None else rules_by_fold[int(row["fold"])]
        conditions = {
            condition: _apply_router(row, condition, rule)
            for condition in CONDITIONS
        }
        routed_rows.append(
            {
                "index": row["index"],
                "example_id": row["example_id"],
                "labels": row["labels"],
                "gold_answer": row["gold_answer"],
                "fallback_prediction": row["fallback_prediction"],
                "router_rule": rule,
                "features": row["features"],
                "conditions": conditions,
            }
        )
    summaries = _condition_summaries(routed_rows, target_ids=target_ids)
    control_names = [
        condition
        for condition in CONDITIONS
        if condition not in ("matched", "equation_permuted")
    ]
    control_clean_union = set().union(
        *[set(summaries[condition]["clean_correct_ids"]) for condition in control_names]
    )
    matched_clean = set(summaries["matched"]["clean_correct_ids"])
    source_necessary = matched_clean - control_clean_union
    accepted_harm = sum(
        int(
            row["conditions"]["matched"]["accepted_sidecar"]
            and not row["conditions"]["matched"]["correct"]
            and row["fallback_prediction"] == row["gold_answer"]
        )
        for row in routed_rows
    )
    equation_permuted_retained = sorted(
        set(summaries["equation_permuted"]["clean_correct_ids"]) & source_necessary
    )
    return {
        "condition_summaries": summaries,
        "control_clean_union_ids": sorted(control_clean_union),
        "source_necessary_clean_ids": sorted(source_necessary),
        "accepted_harm": int(accepted_harm),
        "equation_permuted_retained_source_necessary_ids": equation_permuted_retained,
        "rows": routed_rows,
    }


def _live_cv(
    *,
    prepared_rows: Sequence[dict[str, Any]],
    target_ids: dict[str, set[str]],
    config: RouterConfig,
) -> dict[str, Any]:
    rules_by_fold = {}
    for fold in range(config.outer_folds):
        rule = _fit_stump(
            prepared_rows,
            train_folds=set(range(config.outer_folds)) - {fold},
            accept_penalty=config.accept_penalty,
        )
        rule["fold"] = fold
        rules_by_fold[fold] = rule
    result = _evaluate_with_rules(
        prepared_rows=prepared_rows,
        target_ids=target_ids,
        rules_by_fold=rules_by_fold,
    )
    result["fold_rules"] = [rules_by_fold[idx] for idx in sorted(rules_by_fold)]
    return result


def _status(result: dict[str, Any], *, config: RouterConfig, holdout: bool) -> tuple[str, list[str]]:
    matched = result["condition_summaries"]["matched"]
    min_correct = config.min_holdout_correct if holdout else config.min_live_correct
    min_clean = (
        config.min_holdout_clean_source_necessary
        if holdout
        else config.min_live_clean_source_necessary
    )
    criteria = {
        "min_correct": matched["correct_count"] >= min_correct,
        "min_clean_source_necessary": len(result["source_necessary_clean_ids"]) >= min_clean,
        "max_control_clean_union": len(result["control_clean_union_ids"]) <= config.max_clean_control_union,
        "max_accepted_harm": result["accepted_harm"] <= config.max_accepted_harm,
        "equation_permutation_loses_half": len(
            result["equation_permuted_retained_source_necessary_ids"]
        )
        <= len(result["source_necessary_clean_ids"]) // 2,
    }
    failing = [name for name, passed in criteria.items() if not passed]
    return ("passes" if not failing else "fails", failing)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = RouterConfig(
        moduli=tuple(int(value) for value in args.moduli),
        outer_folds=int(args.outer_folds),
        accept_penalty=float(args.accept_penalty),
        min_live_correct=int(args.min_live_correct),
        min_live_clean_source_necessary=int(args.min_live_clean_source_necessary),
        min_holdout_correct=int(args.min_holdout_correct),
        min_holdout_clean_source_necessary=int(args.min_holdout_clean_source_necessary),
        max_clean_control_union=int(args.max_clean_control_union),
        max_accepted_harm=int(args.max_accepted_harm),
    )
    live_target = syndrome._parse_spec(args.live_target)
    live_source = syndrome._parse_spec(args.live_source)
    holdout_target = syndrome._parse_spec(args.holdout_target)
    holdout_source = syndrome._parse_spec(args.holdout_source)
    live_candidate = [syndrome._parse_spec(args.live_candidate)]
    holdout_candidate = [syndrome._parse_spec(args.holdout_candidate)]
    live_target_set = _resolve(args.live_target_set_json)
    holdout_target_set = _resolve(args.holdout_target_set_json)

    live_raw = _raw_payload(
        target_spec=live_target,
        source_spec=live_source,
        candidate_specs=live_candidate,
        target_set_path=live_target_set,
        config=config,
        fallback_label=str(args.fallback_label),
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        noise_seed=int(args.noise_seed),
        min_numeric_coverage=int(args.live_min_numeric_coverage),
        run_date=str(args.date),
    )
    holdout_raw = _raw_payload(
        target_spec=holdout_target,
        source_spec=holdout_source,
        candidate_specs=holdout_candidate,
        target_set_path=holdout_target_set,
        config=config,
        fallback_label=str(args.fallback_label),
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        noise_seed=int(args.noise_seed),
        min_numeric_coverage=int(args.holdout_min_numeric_coverage),
        run_date=str(args.date),
    )
    live_rows, live_ids = _prepare_rows(
        raw_payload=live_raw,
        target_spec=live_target,
        source_spec=live_source,
        target_set_path=live_target_set,
        eval_file=_resolve(args.live_eval_file) if args.live_eval_file else None,
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        permutation_seed=int(args.permutation_seed),
        outer_folds=config.outer_folds,
    )
    holdout_rows, holdout_ids = _prepare_rows(
        raw_payload=holdout_raw,
        target_spec=holdout_target,
        source_spec=holdout_source,
        target_set_path=holdout_target_set,
        eval_file=_resolve(args.holdout_eval_file) if args.holdout_eval_file else None,
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        permutation_seed=int(args.permutation_seed),
        outer_folds=config.outer_folds,
    )
    live_cv = _live_cv(prepared_rows=live_rows, target_ids=live_ids, config=config)
    global_rule = _fit_stump(
        live_rows,
        train_folds=None,
        accept_penalty=config.accept_penalty,
    )
    holdout = _evaluate_with_rules(
        prepared_rows=holdout_rows,
        target_ids=holdout_ids,
        rules_by_fold={},
        global_rule=global_rule,
    )
    live_status, live_failing = _status(live_cv, config=config, holdout=False)
    holdout_status, holdout_failing = _status(holdout, config=config, holdout=True)
    status = (
        "source_trace_router_passes_live_and_holdout"
        if live_status == "passes" and holdout_status == "passes"
        else "source_trace_router_fails_gate"
    )
    return {
        "date": str(args.date),
        "status": status,
        "config": {
            "moduli": list(config.moduli),
            "outer_folds": config.outer_folds,
            "accept_penalty": config.accept_penalty,
            "features": list(FEATURES),
            "fallback_label": str(args.fallback_label),
            "min_live_correct": config.min_live_correct,
            "min_live_clean_source_necessary": config.min_live_clean_source_necessary,
            "min_holdout_correct": config.min_holdout_correct,
            "min_holdout_clean_source_necessary": config.min_holdout_clean_source_necessary,
            "max_clean_control_union": config.max_clean_control_union,
            "max_accepted_harm": config.max_accepted_harm,
            "permutation_seed": int(args.permutation_seed),
        },
        "artifacts": {
            "live_target_set": str(live_target_set.relative_to(ROOT)),
            "holdout_target_set": str(holdout_target_set.relative_to(ROOT)),
        },
        "live_cv": {
            **live_cv,
            "status": live_status,
            "failing_criteria": live_failing,
        },
        "frozen_rule": global_rule,
        "holdout_frozen": {
            **holdout,
            "status": holdout_status,
            "failing_criteria": holdout_failing,
        },
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
        f"- equation-permuted retained source-necessary: `{len(result['equation_permuted_retained_source_necessary_ids'])}`",
        f"- source-necessary IDs: {', '.join(f'`{item}`' for item in result['source_necessary_clean_ids']) or 'none'}",
        "",
    ]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Trace Self-Consistency Router Gate",
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
    lines.extend(["## Features", ""])
    for feature in payload["config"]["features"]:
        lines.append(f"- `{feature}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-target", required=True)
    parser.add_argument("--live-source", required=True)
    parser.add_argument("--live-candidate", required=True)
    parser.add_argument("--live-target-set-json", required=True)
    parser.add_argument("--live-eval-file")
    parser.add_argument("--holdout-target", required=True)
    parser.add_argument("--holdout-source", required=True)
    parser.add_argument("--holdout-candidate", required=True)
    parser.add_argument("--holdout-target-set-json", required=True)
    parser.add_argument("--holdout-eval-file")
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--moduli", type=syndrome._parse_moduli_set, default=[2, 3, 5, 7])
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--label-shuffle-offset", type=int, default=17)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--permutation-seed", type=int, default=1)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--accept-penalty", type=float, default=0.10)
    parser.add_argument("--min-live-correct", type=int, default=25)
    parser.add_argument("--min-live-clean-source-necessary", type=int, default=4)
    parser.add_argument("--min-holdout-correct", type=int, default=10)
    parser.add_argument("--min-holdout-clean-source-necessary", type=int, default=2)
    parser.add_argument("--max-clean-control-union", type=int, default=0)
    parser.add_argument("--max-accepted-harm", type=int, default=1)
    parser.add_argument("--live-min-numeric-coverage", type=int, default=61)
    parser.add_argument("--holdout-min-numeric-coverage", type=int, default=64)
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
