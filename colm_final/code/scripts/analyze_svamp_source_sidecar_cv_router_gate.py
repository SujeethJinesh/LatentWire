#!/usr/bin/env python3
"""Cross-validated router gate for source-residue sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp32_source_only_sidecar_router_gate as base_gate
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


DEFAULT_FEATURES = [
    "source_prediction_char_count",
    "source_target_len_ratio",
    "source_numeric_count",
    "source_generated_tokens",
    "source_has_final_marker",
]
CONDITIONS = (
    "matched",
    "zero_source",
    "shuffled_source",
    "label_shuffle",
    "same_norm_noise",
    "target_only",
    "slots_only",
)


def _fold_for_id(example_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % int(folds)


def _numeric_count(text: str) -> int:
    return len(re.findall(r"[-+]?\d+(?:\.\d+)?", text))


def _has_final_marker(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            "final answer",
            "answer",
            "therefore",
            "so,",
            "step-by-step explanation",
        )
    )


def _feature_value(
    source_row: dict[str, Any] | None,
    target_row: dict[str, Any],
    feature: str,
) -> float | None:
    if source_row is None or syndrome._prediction_numeric(source_row) is None:
        return None
    source_text = str(source_row.get("prediction", "") or "")
    target_text = str(target_row.get("prediction", "") or "")
    if feature == "source_prediction_char_count":
        return float(len(source_text))
    if feature == "target_prediction_char_count":
        return float(len(target_text))
    if feature == "source_target_len_ratio":
        return float(len(source_text) / max(len(target_text), 1))
    if feature == "source_generated_tokens":
        value = source_row.get("generated_tokens")
        return float(value) if isinstance(value, (int, float)) else None
    if feature == "target_generated_tokens":
        value = target_row.get("generated_tokens")
        return float(value) if isinstance(value, (int, float)) else None
    if feature == "source_numeric_count":
        return float(_numeric_count(source_text))
    if feature == "target_numeric_count":
        return float(_numeric_count(target_text))
    if feature == "source_has_final_marker":
        return float(_has_final_marker(source_text))
    if feature == "target_has_final_marker":
        return float(_has_final_marker(target_text))
    if feature == "prompt_char_count":
        value = source_row.get("prompt_char_count", target_row.get("prompt_char_count"))
        return float(value) if isinstance(value, (int, float)) else None
    if feature == "prompt_byte_count":
        value = source_row.get("prompt_byte_count", target_row.get("prompt_byte_count"))
        return float(value) if isinstance(value, (int, float)) else None
    if feature == "target_prompt_token_count":
        value = target_row.get("target_prompt_token_count")
        return float(value) if isinstance(value, (int, float)) else None
    if feature == "raw_target_token_count":
        value = target_row.get("raw_target_token_count")
        return float(value) if isinstance(value, (int, float)) else None
    raise ValueError(f"Unsupported feature: {feature!r}")


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        return []
    thresholds = list(unique)
    if len(unique) > 1:
        thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(thresholds))


def _accepts(value: float | None, *, feature: str, threshold: float, direction: str) -> bool:
    if value is None:
        return False
    if direction == "le":
        return value <= threshold
    if direction == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported direction: {direction!r}")


def _fit_stump(
    rows: Sequence[dict[str, Any]],
    *,
    train_folds: set[int],
    features: Sequence[str],
    accept_penalty: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feature in features:
        values = [
            float(row["features"]["matched"][feature])
            for row in rows
            if int(row["fold"]) in train_folds and row["features"]["matched"].get(feature) is not None
        ]
        for threshold in _candidate_thresholds(values):
            for direction in ("le", "ge"):
                help_count = 0
                harm_count = 0
                accept_count = 0
                for row in rows:
                    if int(row["fold"]) not in train_folds:
                        continue
                    accepted = _accepts(
                        row["features"]["matched"].get(feature),
                        feature=feature,
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
    raw = row["raw_conditions"][condition]
    feature = rule.get("feature")
    accepted = False
    if feature:
        accepted = _accepts(
            row["features"][condition].get(str(feature)),
            feature=str(feature),
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
    }


def analyze(
    *,
    target_spec: syndrome.RowSpec,
    source_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    moduli_sets: Sequence[Sequence[int]],
    fallback_label: str,
    shuffle_offset: int,
    label_shuffle_offset: int,
    noise_seed: int,
    outer_folds: int,
    features: Sequence[str],
    accept_penalty: float,
    min_correct: int,
    min_target_self: int,
    min_clean_source_necessary: int,
    max_control_clean_union: int,
    min_numeric_coverage: int,
    run_date: str,
) -> dict[str, Any]:
    raw_payload = base_gate.analyze(
        target_spec=target_spec,
        source_spec=source_spec,
        candidate_specs=candidate_specs,
        target_set_path=target_set_path,
        moduli_sets=moduli_sets,
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
    target_records = syndrome._records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    source_records = syndrome._subset_reference_order(
        syndrome._records_for_method(source_spec),
        reference_ids,
    )
    target_by_id = syndrome._by_id(target_records)
    source_by_id = syndrome._by_id(source_records)
    target_ids = syndrome._load_target_ids(target_set_path)

    runs: list[dict[str, Any]] = []
    for raw_run in raw_payload["runs"]:
        prepared_rows: list[dict[str, Any]] = []
        for raw_row in raw_run["rows"]:
            index = int(raw_row["index"])
            example_id = str(raw_row["example_id"])
            source_conditions = _source_rows_by_condition(
                reference_ids=reference_ids,
                source_by_id=source_by_id,
                index=index,
                shuffle_offset=shuffle_offset,
                label_shuffle_offset=label_shuffle_offset,
            )
            feature_values = {
                condition: {
                    feature: _feature_value(
                        source_conditions[condition],
                        target_by_id[example_id],
                        feature,
                    )
                    for feature in features
                }
                for condition in CONDITIONS
            }
            prepared_rows.append(
                {
                    "index": index,
                    "example_id": example_id,
                    "fold": _fold_for_id(example_id, outer_folds),
                    "labels": list(raw_row["labels"]),
                    "gold_answer": raw_row["gold_answer"],
                    "fallback_prediction": raw_row["fallback_prediction"],
                    "fallback_correct": raw_row["fallback_prediction"] == raw_row["gold_answer"],
                    "raw_conditions": raw_row["conditions"],
                    "features": feature_values,
                }
            )

        fold_rules: list[dict[str, Any]] = []
        routed_rows: list[dict[str, Any]] = []
        for fold in range(outer_folds):
            train_folds = set(range(outer_folds)) - {fold}
            rule = _fit_stump(
                prepared_rows,
                train_folds=train_folds,
                features=features,
                accept_penalty=accept_penalty,
            )
            rule["fold"] = fold
            rule["train_folds"] = sorted(train_folds)
            fold_rules.append(rule)
            for row in prepared_rows:
                if int(row["fold"]) != fold:
                    continue
                routed_rows.append(
                    {
                        "index": row["index"],
                        "example_id": row["example_id"],
                        "fold": row["fold"],
                        "labels": row["labels"],
                        "gold_answer": row["gold_answer"],
                        "fallback_prediction": row["fallback_prediction"],
                        "conditions": {
                            condition: _apply_router(row, condition, rule)
                            for condition in CONDITIONS
                        },
                        "router_rule": {
                            key: rule[key]
                            for key in ("feature", "threshold", "direction")
                        },
                        "features": row["features"],
                    }
                )
        routed_rows.sort(key=lambda row: int(row["index"]))
        summaries = _condition_summaries(routed_rows, target_ids=target_ids)
        control_clean_union = set().union(
            *[
                set(summaries[condition]["clean_correct_ids"])
                for condition in CONDITIONS
                if condition != "matched"
            ]
        )
        matched_clean = set(summaries["matched"]["clean_correct_ids"])
        source_necessary_clean = matched_clean - control_clean_union
        accepted_harm = sum(
            int(
                row["conditions"]["matched"]["accepted_sidecar"]
                and not row["conditions"]["matched"]["correct"]
                and row["fallback_prediction"] == row["gold_answer"]
            )
            for row in routed_rows
        )
        criteria = {
            "min_correct": summaries["matched"]["correct_count"] >= min_correct,
            "min_target_self": summaries["matched"]["target_self_correct_count"] >= min_target_self,
            "min_clean_source_necessary": len(source_necessary_clean) >= min_clean_source_necessary,
            "max_control_clean_union": len(control_clean_union) <= max_control_clean_union,
        }
        failing = [name for name, passed in criteria.items() if not passed]
        runs.append(
            {
                "moduli": raw_run["moduli"],
                "syndrome_bytes": int(math.ceil(raw_run["syndrome_bits"] / 8.0)),
                "status": "source_sidecar_cv_router_clears_gate" if not failing else "source_sidecar_cv_router_fails_gate",
                "criteria": criteria,
                "failing_criteria": failing,
                "fold_rules": fold_rules,
                "condition_summaries": summaries,
                "control_clean_union_ids": sorted(control_clean_union),
                "source_necessary_clean_ids": sorted(source_necessary_clean),
                "accepted_harm_count": int(accepted_harm),
                "rows": routed_rows,
            }
        )

    clearing = [run for run in runs if run["status"] == "source_sidecar_cv_router_clears_gate"]
    return {
        "date": run_date,
        "status": "source_sidecar_cv_router_clears_gate"
        if clearing and raw_payload["status"] != "source_only_sidecar_router_fails_gate"
        else "source_sidecar_cv_router_fails_gate",
        "reference_n": raw_payload["reference_n"],
        "reference_ids": raw_payload["reference_ids"],
        "target_ids": raw_payload["target_ids"],
        "artifacts": raw_payload["artifacts"],
        "provenance": raw_payload["provenance"],
        "config": {
            "fallback_label": fallback_label,
            "shuffle_offset": shuffle_offset,
            "label_shuffle_offset": label_shuffle_offset,
            "noise_seed": noise_seed,
            "outer_folds": outer_folds,
            "features": list(features),
            "accept_penalty": float(accept_penalty),
            "min_correct": min_correct,
            "min_target_self": min_target_self,
            "min_clean_source_necessary": min_clean_source_necessary,
            "max_control_clean_union": max_control_clean_union,
            "min_numeric_coverage": min_numeric_coverage,
            "moduli_sets": [list(moduli) for moduli in moduli_sets],
        },
        "runs": runs,
    }


def _selected_run(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        run
        for run in payload["runs"]
        if run["status"] == "source_sidecar_cv_router_clears_gate"
    ] or list(payload["runs"])
    return sorted(
        candidates,
        key=lambda run: (
            -int(run["condition_summaries"]["matched"]["correct_count"]),
            int(run["syndrome_bytes"]),
            len(run["moduli"]),
        ),
    )[0]


def _write_predictions(path: pathlib.Path, payload: dict[str, Any], *, method: str) -> None:
    run = _selected_run(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in run["rows"]:
            matched = row["conditions"]["matched"]
            prediction = matched["prediction"]
            record = {
                "index": row["index"],
                "example_id": row["example_id"],
                "method": method,
                "answer": row["gold_answer"],
                "prediction": "" if prediction is None else str(prediction),
                "normalized_prediction": "" if prediction is None else str(prediction),
                "correct": bool(matched["correct"]),
                "accepted_source_sidecar": bool(matched["accepted_sidecar"]),
                "fallback_prediction": row["fallback_prediction"],
                "router_rule": row["router_rule"],
                "fold": row["fold"],
                "sidecar_moduli": run["moduli"],
                "sidecar_bytes": run["syndrome_bytes"],
            }
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source Sidecar CV Router Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- outer folds: `{payload['config']['outer_folds']}`",
        f"- features: `{', '.join(payload['config']['features'])}`",
        "",
        "## Moduli Sweep",
        "",
        "| Moduli | Bytes | Status | Matched | Clean Necessary | Control Clean Union | Accepted Harm | Source-Necessary IDs | Failing Criteria |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        matched = run["condition_summaries"]["matched"]
        lines.append(
            "| {moduli} | {bytes} | {status} | {matched} | {necessary} | {control} | {harm} | {ids} | {failing} |".format(
                moduli=",".join(str(value) for value in run["moduli"]),
                bytes=run["syndrome_bytes"],
                status=run["status"],
                matched=matched["correct_count"],
                necessary=len(run["source_necessary_clean_ids"]),
                control=len(run["control_clean_union_ids"]),
                harm=run["accepted_harm_count"],
                ids=", ".join(f"`{value}`" for value in run["source_necessary_clean_ids"]) or "none",
                failing=", ".join(run["failing_criteria"]) or "none",
            )
        )
    lines.extend(["", "## Fold Rules", ""])
    for run in payload["runs"]:
        lines.append(f"### Moduli {','.join(str(value) for value in run['moduli'])}")
        lines.append("")
        lines.append("| Fold | Feature | Direction | Threshold | Train Help | Train Harm | Train Accept | Score |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for rule in run["fold_rules"]:
            threshold = rule["threshold"]
            lines.append(
                f"| {rule['fold']} | `{rule['feature']}` | `{rule['direction']}` | "
                f"{threshold if threshold is not None else ''} | {rule['train_help']} | "
                f"{rule['train_harm']} | {rule['train_accept']} | {rule['train_score']:.4f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--source", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--label-shuffle-offset", type=int, default=17)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--moduli-set", action="append", type=syndrome._parse_moduli_set)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--accept-penalty", type=float, default=0.25)
    parser.add_argument("--min-correct", type=int, default=25)
    parser.add_argument("--min-target-self", type=int, default=0)
    parser.add_argument("--min-clean-source-necessary", type=int, default=4)
    parser.add_argument("--max-control-clean-union", type=int, default=0)
    parser.add_argument("--min-numeric-coverage", type=int, default=0)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-predictions-jsonl")
    parser.add_argument("--prediction-method", default="source_sidecar_cv_router")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    features = list(args.feature) or list(DEFAULT_FEATURES)
    moduli_sets = args.moduli_set or [[2, 3], [2, 3, 5], [2, 3, 5, 7], [97]]
    payload = analyze(
        target_spec=args.target,
        source_spec=args.source,
        candidate_specs=list(args.candidate),
        target_set_path=syndrome._resolve(args.target_set_json),
        moduli_sets=moduli_sets,
        fallback_label=args.fallback_label,
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        noise_seed=int(args.noise_seed),
        outer_folds=int(args.outer_folds),
        features=features,
        accept_penalty=float(args.accept_penalty),
        min_correct=int(args.min_correct),
        min_target_self=int(args.min_target_self),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
        max_control_clean_union=int(args.max_control_clean_union),
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
    )
    output_json = syndrome._resolve(args.output_json)
    output_md = syndrome._resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    if args.output_predictions_jsonl:
        _write_predictions(
            syndrome._resolve(args.output_predictions_jsonl),
            payload,
            method=str(args.prediction_method),
        )
    print(json.dumps({"status": payload["status"], "output_json": syndrome._display_path(output_json)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
