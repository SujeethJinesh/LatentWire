#!/usr/bin/env python3
"""Gate a source-only numeric sidecar/router on frozen SVAMP32 rows.

This is a deployability screen, not a final method. It uses only a source-side
numeric prediction to form a compact residue sidecar, then lets the target-side
candidate pool act as decoder side information. Target-only and slots-only
controls never participate in source-signal formation.
"""

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

from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


def _select_by_signature(
    *,
    ordered_values: Sequence[str],
    signature: tuple[int, ...] | None,
    moduli: Sequence[int],
    fallback: str | None,
) -> str | None:
    if signature is None:
        return fallback
    for value in ordered_values:
        if syndrome._signature(value, moduli) == signature:
            return value
    return fallback


def _noise_signature(example_id: str, moduli: Sequence[int], seed: int) -> tuple[int, ...]:
    out: list[int] = []
    for modulus in moduli:
        digest = hashlib.sha256(f"{seed}:{example_id}:{modulus}".encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:8], byteorder="big") % int(modulus))
    return tuple(out)


def _prediction_for_label(
    rows_by_label: dict[str, dict[str, Any]],
    label: str | None,
) -> str | None:
    if not label:
        return None
    if label not in rows_by_label:
        raise ValueError(f"Agreement guard label {label!r} missing from candidate labels")
    return syndrome._prediction_numeric(rows_by_label[label])


def _apply_agreement_guard(
    *,
    candidate: str | None,
    fallback: str | None,
    agreement_prediction: str | None,
) -> str | None:
    if agreement_prediction is not None and fallback is not None and agreement_prediction == fallback:
        return fallback
    return candidate


def _source_quality_passes(row: dict[str, Any] | None, guard: str | None) -> bool:
    if not guard:
        return True
    if row is None:
        return False
    text = str(row.get("prediction", ""))
    lower = text.lower()
    numeric_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", text))
    if guard == "finalish_short_numeric":
        finalish = any(
            marker in lower
            for marker in (
                "final answer",
                "answer",
                "therefore",
                "so,",
                "step-by-step explanation",
            )
        )
        return finalish and len(text) < 240 and numeric_count <= 6
    if guard == "shorter_than_target_numeric":
        raise ValueError("shorter_than_target_numeric requires target row context")
    raise ValueError(f"Unsupported source quality guard: {guard!r}")


def _source_target_quality_passes(
    source_row: dict[str, Any] | None,
    target_row: dict[str, Any],
    guard: str | None,
) -> bool:
    if guard != "shorter_than_target_numeric":
        return _source_quality_passes(source_row, guard)
    if source_row is None or syndrome._prediction_numeric(source_row) is None:
        return False
    return len(str(source_row.get("prediction", "") or "")) < len(
        str(target_row.get("prediction", "") or "")
    )


def _source_quality_score(
    source_row: dict[str, Any] | None,
    target_row: dict[str, Any],
    score_field: str | None,
) -> float | None:
    if not score_field:
        return None
    if source_row is None:
        return None
    source_text = str(source_row.get("prediction", "") or "")
    target_text = str(target_row.get("prediction", "") or "")
    if score_field == "source_prediction_char_count":
        return float(len(source_text))
    if score_field == "target_prediction_char_count":
        return float(len(target_text))
    if score_field == "source_target_len_ratio":
        denominator = max(len(target_text), 1)
        return float(len(source_text) / denominator)
    if score_field == "source_numeric_count":
        return float(len(re.findall(r"[-+]?\d+(?:\.\d+)?", source_text)))
    if score_field == "source_generated_tokens":
        value = source_row.get("generated_tokens")
        if isinstance(value, (int, float)):
            return float(value)
        return None
    raise ValueError(f"Unsupported source quality score field: {score_field!r}")


def _source_quality_threshold_passes(
    *,
    source_row: dict[str, Any] | None,
    target_row: dict[str, Any],
    score_field: str | None,
    min_threshold: float | None,
    max_threshold: float | None,
) -> bool:
    if not score_field:
        return True
    if source_row is None or syndrome._prediction_numeric(source_row) is None:
        return False
    score = _source_quality_score(source_row, target_row, score_field)
    if score is None:
        return False
    if min_threshold is not None and score < min_threshold:
        return False
    if max_threshold is not None and score > max_threshold:
        return False
    return True


def _apply_source_quality_guard(
    *,
    candidate: str | None,
    fallback: str | None,
    source_row: dict[str, Any] | None,
    target_row: dict[str, Any],
    guard: str | None,
    score_field: str | None,
    min_threshold: float | None,
    max_threshold: float | None,
) -> str | None:
    if _source_target_quality_passes(
        source_row, target_row, guard
    ) and _source_quality_threshold_passes(
        source_row=source_row,
        target_row=target_row,
        score_field=score_field,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    ):
        return candidate
    return fallback


def _evaluate_moduli(
    *,
    moduli: Sequence[int],
    reference_ids: Sequence[str],
    target_by_id: dict[str, dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    candidate_by_label: dict[str, dict[str, dict[str, Any]]],
    target_label: str,
    fallback_label: str,
    target_ids: dict[str, set[str]],
    shuffle_offset: int,
    label_shuffle_offset: int,
    noise_seed: int,
    min_correct: int,
    min_target_self: int,
    min_clean_source_necessary: int,
    max_control_clean_union: int,
    preserve_on_agreement_label: str | None,
    source_quality_guard: str | None,
    source_quality_score_field: str | None,
    source_quality_min_threshold: float | None,
    source_quality_max_threshold: float | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_labels = list(candidate_by_label)
    for index, example_id in enumerate(reference_ids):
        rows_by_label = {
            label: records[example_id] for label, records in candidate_by_label.items()
        }
        ordered_values, labels_by_value = syndrome._candidate_pool(rows_by_label, candidate_labels)
        fallback = syndrome._fallback_prediction(
            rows_by_label=rows_by_label,
            fallback_label=fallback_label,
            target_label=target_label,
        )
        agreement_prediction = _prediction_for_label(rows_by_label, preserve_on_agreement_label)
        shuffled_id = reference_ids[(index + shuffle_offset) % len(reference_ids)]
        label_shuffled_id = reference_ids[(index + label_shuffle_offset) % len(reference_ids)]
        gold = syndrome._gold_numeric(target_by_id[example_id])
        source_numeric = syndrome._prediction_numeric(source_by_id[example_id])
        shuffled_source_numeric = syndrome._prediction_numeric(source_by_id[shuffled_id])
        label_shuffled_numeric = syndrome._prediction_numeric(source_by_id[label_shuffled_id])
        source_rows_by_condition = {
            "matched": source_by_id[example_id],
            "zero_source": None,
            "shuffled_source": source_by_id[shuffled_id],
            "label_shuffle": source_by_id[label_shuffled_id],
            "same_norm_noise": None,
        }
        raw_selections = {
            "matched": syndrome._select_by_syndrome(
                ordered_values=ordered_values,
                source_value=source_numeric,
                moduli=moduli,
                fallback=fallback,
            ),
            "zero_source": syndrome._select_by_syndrome(
                ordered_values=ordered_values,
                source_value="0",
                moduli=moduli,
                fallback=fallback,
            ),
            "shuffled_source": syndrome._select_by_syndrome(
                ordered_values=ordered_values,
                source_value=shuffled_source_numeric,
                moduli=moduli,
                fallback=fallback,
            ),
            "label_shuffle": syndrome._select_by_syndrome(
                ordered_values=ordered_values,
                source_value=label_shuffled_numeric,
                moduli=moduli,
                fallback=fallback,
            ),
            "same_norm_noise": _select_by_signature(
                ordered_values=ordered_values,
                signature=_noise_signature(example_id, moduli, noise_seed),
                moduli=moduli,
                fallback=fallback,
            ),
            "target_only": fallback,
            "slots_only": syndrome._select_slots_only(
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
                fallback=fallback,
            ),
        }
        selections = {
            condition: (
                _apply_agreement_guard(
                    candidate=_apply_source_quality_guard(
                        candidate=prediction,
                        fallback=fallback,
                        source_row=source_rows_by_condition.get(condition),
                        target_row=target_by_id[example_id],
                        guard=source_quality_guard,
                        score_field=source_quality_score_field,
                        min_threshold=source_quality_min_threshold,
                        max_threshold=source_quality_max_threshold,
                    ),
                    fallback=fallback,
                    agreement_prediction=agreement_prediction,
                )
                if condition not in ("target_only", "slots_only")
                else prediction
            )
            for condition, prediction in raw_selections.items()
        }
        rows.append(
            {
                "index": index,
                "example_id": example_id,
                "labels": [
                    label for label, ids in target_ids.items() if example_id in ids
                ],
                "gold_answer": gold,
                "source_prediction": source_numeric,
                "fallback_prediction": fallback,
                "agreement_guard_label": preserve_on_agreement_label,
                "agreement_guard_prediction": agreement_prediction,
                "agreement_guard_active": bool(
                    preserve_on_agreement_label
                    and agreement_prediction is not None
                    and fallback is not None
                    and agreement_prediction == fallback
                ),
                "source_quality_guard": source_quality_guard,
                "source_quality_passed": bool(
                    _source_target_quality_passes(
                        source_by_id[example_id],
                        target_by_id[example_id],
                        source_quality_guard,
                    )
                    and _source_quality_threshold_passes(
                        source_row=source_by_id[example_id],
                        target_row=target_by_id[example_id],
                        score_field=source_quality_score_field,
                        min_threshold=source_quality_min_threshold,
                        max_threshold=source_quality_max_threshold,
                    )
                ),
                "source_quality_score_field": source_quality_score_field,
                "source_quality_score": _source_quality_score(
                    source_by_id[example_id],
                    target_by_id[example_id],
                    source_quality_score_field,
                ),
                "candidate_pool_size": len(ordered_values),
                "candidate_pool_values": list(ordered_values),
                "candidate_pool_contains_gold": gold in set(ordered_values),
                "candidate_labels_for_gold": sorted(set(labels_by_value.get(gold, []))),
                "conditions": {
                    condition: {
                        "prediction": prediction,
                        "correct": prediction == gold,
                        "candidate_labels": sorted(
                            set(labels_by_value.get(str(prediction), []))
                        )
                        if prediction is not None
                        else [],
                    }
                    for condition, prediction in selections.items()
                },
            }
        )

    clean_ids = target_ids["clean_residual_targets"]
    target_self_ids = target_ids["target_self_repair"]
    teacher_only_ids = target_ids["teacher_only"]
    conditions = (
        "matched",
        "zero_source",
        "shuffled_source",
        "label_shuffle",
        "same_norm_noise",
        "target_only",
        "slots_only",
    )
    condition_summaries = {
        condition: syndrome._summarize_condition(
            rows,
            condition=condition,
            clean_ids=clean_ids,
            target_self_ids=target_self_ids,
            teacher_only_ids=teacher_only_ids,
        )
        for condition in conditions
    }
    control_clean_union = set().union(
        *[
            set(condition_summaries[condition]["clean_correct_ids"])
            for condition in conditions
            if condition != "matched"
        ]
    )
    matched_clean = set(condition_summaries["matched"]["clean_correct_ids"])
    source_necessary_clean = matched_clean - control_clean_union
    criteria = {
        "min_correct": condition_summaries["matched"]["correct_count"] >= min_correct,
        "min_target_self": (
            condition_summaries["matched"]["target_self_correct_count"] >= min_target_self
        ),
        "min_clean_source_necessary": (
            len(source_necessary_clean) >= min_clean_source_necessary
        ),
        "max_control_clean_union": len(control_clean_union) <= max_control_clean_union,
    }
    failing = [name for name, passed in criteria.items() if not passed]
    status = "source_only_sidecar_router_clears_gate" if not failing else "source_only_sidecar_router_fails_gate"
    return {
        "moduli": list(moduli),
        "syndrome_bits": syndrome._moduli_bits(moduli),
        "syndrome_bytes": int(math.ceil(syndrome._moduli_bits(moduli) / 8.0)),
        "status": status,
        "criteria": criteria,
        "failing_criteria": failing,
        "candidate_pool_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"])) for row in rows
        ),
        "candidate_pool_clean_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"]))
            for row in rows
            if row["example_id"] in clean_ids
        ),
        "condition_summaries": condition_summaries,
        "control_clean_union_ids": sorted(control_clean_union),
        "source_necessary_clean_ids": sorted(source_necessary_clean),
        "rows": rows,
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
    min_correct: int,
    min_target_self: int,
    min_clean_source_necessary: int,
    max_control_clean_union: int,
    min_numeric_coverage: int,
    run_date: str,
    preserve_on_agreement_label: str | None = None,
    source_quality_guard: str | None = None,
    source_quality_score_field: str | None = None,
    source_quality_min_threshold: float | None = None,
    source_quality_max_threshold: float | None = None,
) -> dict[str, Any]:
    target_records = syndrome._records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("target rows contain duplicate example_id values")
    source_records = syndrome._subset_reference_order(
        syndrome._records_for_method(source_spec),
        reference_ids,
    )
    target_by_id = syndrome._by_id(target_records)
    source_by_id = syndrome._by_id(source_records)
    target_ids = syndrome._load_target_ids(target_set_path)

    candidate_by_label: dict[str, dict[str, dict[str, Any]]] = {
        target_spec.label: target_by_id,
    }
    for spec in candidate_specs:
        if spec.label in candidate_by_label:
            raise ValueError(f"Duplicate candidate label {spec.label!r}")
        candidate_by_label[spec.label] = syndrome._by_id(
            syndrome._subset_reference_order(
                syndrome._records_for_method(spec),
                reference_ids,
            )
        )
    if fallback_label not in candidate_by_label:
        raise ValueError(
            f"fallback label {fallback_label!r} not in candidates: {sorted(candidate_by_label)}"
        )
    if preserve_on_agreement_label and preserve_on_agreement_label not in candidate_by_label:
        raise ValueError(
            f"preserve-on-agreement label {preserve_on_agreement_label!r} not in candidates: "
            f"{sorted(candidate_by_label)}"
        )

    source_numeric_coverage = sum(
        int(syndrome._prediction_numeric(row) is not None) for row in source_records
    )
    candidate_numeric_coverage = {
        label: sum(
            int(syndrome._prediction_numeric(records[example_id]) is not None)
            for example_id in reference_ids
        )
        for label, records in candidate_by_label.items()
    }
    provenance_issues: list[str] = []
    if source_numeric_coverage < min_numeric_coverage:
        provenance_issues.append(
            f"source_numeric_coverage={source_numeric_coverage} < {min_numeric_coverage}"
        )
    for label, coverage in candidate_numeric_coverage.items():
        if coverage < min_numeric_coverage:
            provenance_issues.append(
                f"candidate.{label}.numeric_coverage={coverage} < {min_numeric_coverage}"
            )

    runs = [
        _evaluate_moduli(
            moduli=moduli,
            reference_ids=reference_ids,
            target_by_id=target_by_id,
            source_by_id=source_by_id,
            candidate_by_label=candidate_by_label,
            target_label=target_spec.label,
            fallback_label=fallback_label,
            target_ids=target_ids,
            shuffle_offset=shuffle_offset,
            label_shuffle_offset=label_shuffle_offset,
            noise_seed=noise_seed,
            min_correct=min_correct,
            min_target_self=min_target_self,
            min_clean_source_necessary=min_clean_source_necessary,
            max_control_clean_union=max_control_clean_union,
            preserve_on_agreement_label=preserve_on_agreement_label,
            source_quality_guard=source_quality_guard,
            source_quality_score_field=source_quality_score_field,
            source_quality_min_threshold=source_quality_min_threshold,
            source_quality_max_threshold=source_quality_max_threshold,
        )
        for moduli in moduli_sets
    ]
    clearing = [run for run in runs if run["status"] == "source_only_sidecar_router_clears_gate"]
    status = (
        "source_only_sidecar_router_clears_gate"
        if clearing and not provenance_issues
        else "source_only_sidecar_router_fails_gate"
    )
    return {
        "date": run_date,
        "status": status,
        "interpretation": (
            "This is a source-only sidecar/router screen. The source message is "
            "formed from source-side numeric predictions only; target-side rows "
            "are used only as decoder candidate pools and controls."
        ),
        "artifacts": {
            "target": {
                "label": target_spec.label,
                "path": syndrome._display_path(target_spec.path),
                "method": target_spec.method,
            },
            "source": {
                "label": source_spec.label,
                "path": syndrome._display_path(source_spec.path),
                "method": source_spec.method,
            },
            "target_set_json": syndrome._display_path(target_set_path),
            "candidates": [
                {
                    "label": spec.label,
                    "path": syndrome._display_path(spec.path),
                    "method": spec.method,
                }
                for spec in candidate_specs
            ],
        },
        "config": {
            "fallback_label": fallback_label,
            "shuffle_offset": shuffle_offset,
            "label_shuffle_offset": label_shuffle_offset,
            "noise_seed": noise_seed,
            "min_correct": min_correct,
            "min_target_self": min_target_self,
            "min_clean_source_necessary": min_clean_source_necessary,
            "max_control_clean_union": max_control_clean_union,
            "min_numeric_coverage": min_numeric_coverage,
            "preserve_on_agreement_label": preserve_on_agreement_label,
            "source_quality_guard": source_quality_guard,
            "source_quality_score_field": source_quality_score_field,
            "source_quality_min_threshold": source_quality_min_threshold,
            "source_quality_max_threshold": source_quality_max_threshold,
            "moduli_sets": [list(moduli) for moduli in moduli_sets],
        },
        "reference_n": len(reference_ids),
        "reference_ids": reference_ids,
        "target_ids": {key: sorted(value) for key, value in target_ids.items()},
        "provenance": {
            "exact_ordered_id_parity": True,
            "source_numeric_coverage": source_numeric_coverage,
            "candidate_numeric_coverage": candidate_numeric_coverage,
            "issues": provenance_issues,
        },
        "runs": runs,
    }


def _selected_run(payload: dict[str, Any]) -> dict[str, Any]:
    clearing = [
        run
        for run in payload["runs"]
        if run["status"] == "source_only_sidecar_router_clears_gate"
    ]
    candidates = clearing or list(payload["runs"])
    return sorted(
        candidates,
        key=lambda run: (
            -int(run["condition_summaries"]["matched"]["correct_count"]),
            int(run["syndrome_bytes"]),
            len(run["moduli"]),
        ),
    )[0]


def _write_prediction_jsonl(
    path: pathlib.Path,
    payload: dict[str, Any],
    *,
    method: str,
) -> None:
    run = _selected_run(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in run["rows"]:
            matched = row["conditions"]["matched"]
            prediction = matched["prediction"]
            record = {
                "example_id": row["example_id"],
                "method": method,
                "answer": row["gold_answer"],
                "prediction": "" if prediction is None else str(prediction),
                "normalized_prediction": "" if prediction is None else str(prediction),
                "correct": bool(matched["correct"]),
                "accepted_source_sidecar": bool(
                    prediction is not None
                    and row.get("fallback_prediction") is not None
                    and str(prediction) != str(row.get("fallback_prediction"))
                ),
                "fallback_prediction": row.get("fallback_prediction"),
                "source_prediction": row.get("source_prediction"),
                "source_quality_guard": row.get("source_quality_guard"),
                "source_quality_score_field": row.get("source_quality_score_field"),
                "source_quality_score": row.get("source_quality_score"),
                "source_quality_passed": row.get("source_quality_passed"),
                "sidecar_moduli": run["moduli"],
                "sidecar_bytes": run["syndrome_bytes"],
            }
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Only Sidecar Router Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- fallback label: `{payload['config']['fallback_label']}`",
        f"- preserve-on-agreement label: `{payload['config']['preserve_on_agreement_label'] or 'none'}`",
        f"- source quality guard: `{payload['config']['source_quality_guard'] or 'none'}`",
        f"- source quality score field: `{payload['config']['source_quality_score_field'] or 'none'}`",
        f"- source quality min threshold: `{payload['config']['source_quality_min_threshold']}`",
        f"- source quality max threshold: `{payload['config']['source_quality_max_threshold']}`",
        f"- source numeric coverage: `{payload['provenance']['source_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
        "",
        "## Moduli Sweep",
        "",
        "| Moduli | Bytes | Status | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs | Failing Criteria |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        matched = run["condition_summaries"]["matched"]
        lines.append(
            "| {moduli} | {bytes} | {status} | {matched} | {target_self} | {clean} | {necessary} | {control_clean} | {ids} | {failing} |".format(
                moduli=",".join(str(value) for value in run["moduli"]),
                bytes=run["syndrome_bytes"],
                status=run["status"],
                matched=matched["correct_count"],
                target_self=matched["target_self_correct_count"],
                clean=matched["clean_correct_count"],
                necessary=len(run["source_necessary_clean_ids"]),
                control_clean=len(run["control_clean_union_ids"]),
                ids=", ".join(f"`{value}`" for value in run["source_necessary_clean_ids"]) or "none",
                failing=", ".join(run["failing_criteria"]) or "none",
            )
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--source", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target_self_repair")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--label-shuffle-offset", type=int, default=17)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--moduli-set", action="append", type=syndrome._parse_moduli_set)
    parser.add_argument("--min-correct", type=int, default=14)
    parser.add_argument("--min-target-self", type=int, default=3)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--max-control-clean-union", type=int, default=0)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument(
        "--preserve-on-agreement-label",
        default=None,
        help=(
            "If this candidate label's numeric prediction equals the fallback "
            "numeric prediction, keep fallback instead of applying source/control sidecars."
        ),
    )
    parser.add_argument(
        "--source-quality-guard",
        choices=["finalish_short_numeric", "shorter_than_target_numeric"],
        default=None,
        help="Optional text-local source quality guard before applying a source/control sidecar.",
    )
    parser.add_argument(
        "--source-quality-score-field",
        choices=[
            "source_prediction_char_count",
            "target_prediction_char_count",
            "source_target_len_ratio",
            "source_numeric_count",
            "source_generated_tokens",
        ],
        default=None,
        help="Optional source/target-derived quality score to threshold before applying a sidecar.",
    )
    parser.add_argument("--source-quality-min-threshold", type=float)
    parser.add_argument("--source-quality-max-threshold", type=float)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-predictions-jsonl")
    parser.add_argument("--prediction-method", default="source_sidecar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    moduli_sets = args.moduli_set or [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [97],
    ]
    payload = analyze(
        target_spec=args.target,
        source_spec=args.source,
        candidate_specs=args.candidate,
        target_set_path=syndrome._resolve(args.target_set_json),
        moduli_sets=moduli_sets,
        fallback_label=args.fallback_label,
        shuffle_offset=int(args.shuffle_offset),
        label_shuffle_offset=int(args.label_shuffle_offset),
        noise_seed=int(args.noise_seed),
        min_correct=int(args.min_correct),
        min_target_self=int(args.min_target_self),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
        max_control_clean_union=int(args.max_control_clean_union),
        min_numeric_coverage=int(args.min_numeric_coverage),
        preserve_on_agreement_label=args.preserve_on_agreement_label,
        source_quality_guard=args.source_quality_guard,
        source_quality_score_field=args.source_quality_score_field,
        source_quality_min_threshold=args.source_quality_min_threshold,
        source_quality_max_threshold=args.source_quality_max_threshold,
        run_date=str(args.date),
    )
    output_json = syndrome._resolve(args.output_json)
    output_md = syndrome._resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    if args.output_predictions_jsonl:
        _write_prediction_jsonl(
            syndrome._resolve(args.output_predictions_jsonl),
            payload,
            method=str(args.prediction_method),
        )
    print(json.dumps({"status": payload["status"], "output_json": syndrome._display_path(output_json)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
