#!/usr/bin/env python3
"""CPU semantic-predicate decoder over existing SVAMP source artifacts.

This is a bounded artifact probe. It tests whether source-generated reasoning
text can be compressed into simple semantic predicates that disambiguate a
target-side numeric candidate pool while an erasure rule preserves fallback
answers. It is not a model-forward experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome
from scripts import harness_common as harness


CONDITIONS = (
    "matched",
    "zero_source",
    "shuffled_source",
    "label_shuffle",
    "target_only",
    "slots_only",
)
OP_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*([+\-*/x×])\s*([-+]?\d+(?:\.\d+)?)\s*=\s*([-+]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class Surface:
    label: str
    target_set_path: pathlib.Path
    target_set: dict[str, Any]
    records_by_label: dict[str, dict[str, dict[str, Any]]]
    reference_ids: list[str]


@dataclass(frozen=True)
class Candidate:
    value: str
    labels: tuple[str, ...]


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


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _record_spec(label: str, raw: dict[str, Any]) -> syndrome.RowSpec:
    return syndrome.RowSpec(
        label=label,
        path=_resolve(str(raw["path"])),
        method=str(raw["method"]),
    )


def _load_records(raw: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    specs = [_record_spec("target", raw["artifacts"]["target"])]
    specs.append(_record_spec("source", raw["artifacts"]["source"]))
    for baseline in raw["artifacts"].get("baselines", []):
        specs.append(_record_spec(str(baseline["label"]), baseline))
    for control in raw["artifacts"].get("controls", []):
        specs.append(_record_spec(str(control["label"]), control))

    reference_ids = [str(example_id) for example_id in raw["reference_ids"]]
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        rows = syndrome._subset_reference_order(
            syndrome._records_for_method(spec),
            reference_ids,
        )
        records[spec.label] = syndrome._by_id(rows)
    return records


def _load_surface(label: str, path: pathlib.Path) -> Surface:
    raw = _read_json(path)
    return Surface(
        label=label,
        target_set_path=path,
        target_set=raw,
        records_by_label=_load_records(raw),
        reference_ids=[str(example_id) for example_id in raw["reference_ids"]],
    )


def _source_ids(surface: Surface, key: str) -> set[str]:
    return {str(example_id) for example_id in surface.target_set.get("ids", {}).get(key, [])}


def _fold_for_id(example_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % max(int(folds), 1)


def _normal(value: Any) -> str | None:
    return harness._normalize_numeric_string(str(value))


def _prediction_numeric(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return syndrome._prediction_numeric(row)


def _gold_numeric(surface: Surface, example_id: str) -> str:
    row = surface.records_by_label["target"][example_id]
    answer = row.get("answer")
    if isinstance(answer, list):
        for item in answer:
            numeric = harness._extract_reference_numeric_answer(str(item))
            if numeric is not None:
                return numeric
    return syndrome._gold_numeric(row)


def _numeric_mentions(text: Any) -> list[str]:
    out: list[str] = []
    for raw in harness._NUMERIC_TOKEN_RE.findall(str(text or "")):
        numeric = _normal(raw)
        if numeric is not None:
            out.append(numeric)
    return out


def _verified_equation_results(text: str) -> set[str]:
    results: set[str] = set()
    for left, op, right, result in OP_RE.findall(text):
        try:
            a = float(left)
            b = float(right)
            expected = float(result)
        except ValueError:
            continue
        actual: float | None = None
        if op == "+":
            actual = a + b
        elif op == "-":
            actual = a - b
        elif op in {"*", "x", "×"}:
            actual = a * b
        elif op == "/" and b != 0:
            actual = a / b
        if actual is not None and abs(actual - expected) <= 1e-6:
            normalized = _normal(result)
            if normalized is not None:
                results.add(normalized)
    return results


def _pair_results(values: Sequence[str], *, max_values: int = 8) -> set[str]:
    numeric: list[float] = []
    for value in values[:max_values]:
        try:
            numeric.append(float(value))
        except ValueError:
            continue
    out: set[str] = set()
    for i, a in enumerate(numeric):
        for b in numeric[i + 1 :]:
            for result in (a + b, abs(a - b), a * b):
                normalized = _normal(int(result) if result.is_integer() else result)
                if normalized is not None:
                    out.add(normalized)
            if b != 0:
                result = a / b
                normalized = _normal(int(result) if result.is_integer() else result)
                if normalized is not None:
                    out.add(normalized)
            if a != 0:
                result = b / a
                normalized = _normal(int(result) if result.is_integer() else result)
                if normalized is not None:
                    out.add(normalized)
    return out


def _has_final_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in ("answer", "therefore", "so,", "####", "boxed"))


def _operation_names(text: str) -> set[str]:
    lower = text.lower()
    names: set[str] = set()
    if any(token in lower for token in (" total", "sum", "altogether", " in all", "more")):
        names.add("op_add")
    if any(token in lower for token in ("left", "remain", "difference", "less", "fewer", "minus")):
        names.add("op_sub")
    if any(token in lower for token in ("each", "per", "times", "product", " x ", "×", "*")):
        names.add("op_mul")
    if any(token in lower for token in ("divide", "split", "equally", "quotient")):
        names.add("op_div")
    return names


def _source_profile(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "features": set(),
            "final": None,
            "mentions": [],
            "mention_set": set(),
            "last": [],
            "verified": set(),
            "pair": set(),
            "quality": False,
        }
    text = str(row.get("prediction", "") or "")
    mentions = _numeric_mentions(text)
    final = _prediction_numeric(row)
    verified = _verified_equation_results(text)
    feature_names = set(_operation_names(text))
    if _has_final_marker(text):
        feature_names.add("source_has_final_marker")
    if verified:
        feature_names.add("source_has_verified_equation")
    if final is not None:
        feature_names.add("source_has_final_numeric")
    if len(set(mentions[-3:])) == 1 and mentions:
        feature_names.add("source_repeats_final_window")
    quality = bool(verified or ("source_has_final_marker" in feature_names and final in mentions[-3:]))
    return {
        "features": feature_names,
        "final": final,
        "mentions": mentions,
        "mention_set": set(mentions),
        "last": mentions[-3:],
        "verified": verified,
        "pair": _pair_results(mentions),
        "quality": quality,
    }


def _append_value(
    out: list[str],
    labels_by_value: dict[str, list[str]],
    value: Any,
    label: str,
) -> None:
    normalized = _normal(value)
    if normalized is None:
        return
    if normalized not in labels_by_value:
        out.append(normalized)
        labels_by_value[normalized] = []
    labels_by_value[normalized].append(label)


def _candidate_pool(surface: Surface, example_id: str) -> list[Candidate]:
    ordered: list[str] = []
    labels_by_value: dict[str, list[str]] = {}
    for label, rows in surface.records_by_label.items():
        row = rows.get(example_id)
        if row is None:
            continue
        _append_value(ordered, labels_by_value, _prediction_numeric(row), label)
        for value in _numeric_mentions(row.get("prediction", "")):
            _append_value(ordered, labels_by_value, value, label)
    return [
        Candidate(value=value, labels=tuple(labels_by_value[value]))
        for value in ordered
    ]


def _condition_source_row(surface: Surface, index: int, condition: str) -> dict[str, Any] | None:
    example_id = surface.reference_ids[index]
    if condition == "matched":
        return surface.records_by_label["source"].get(example_id)
    if condition == "zero_source":
        return None
    if condition == "shuffled_source":
        other_id = surface.reference_ids[(index + 1) % len(surface.reference_ids)]
        return surface.records_by_label["source"].get(other_id)
    if condition == "label_shuffle":
        other_id = surface.reference_ids[(index + 7) % len(surface.reference_ids)]
        return surface.records_by_label["source"].get(other_id)
    if condition == "target_only":
        return surface.records_by_label["target"].get(example_id)
    if condition == "slots_only":
        return surface.records_by_label.get("t2t", {}).get(example_id)
    raise ValueError(f"Unsupported condition: {condition!r}")


def _features_for_candidate(candidate: Candidate, profile: dict[str, Any], fallback: str | None) -> set[str]:
    features = {"bias", *profile["features"]}
    value = candidate.value
    for label in candidate.labels:
        features.add(f"candidate_label:{label}")
    if fallback is not None and value == fallback:
        features.add("candidate_is_fallback")
    if profile["final"] is not None and value == profile["final"]:
        features.add("candidate_eq_source_final")
    if value in profile["verified"]:
        features.add("candidate_in_verified_equation")
    if value in profile["mention_set"]:
        features.add("candidate_mentioned_by_source")
    if profile["last"] and value == profile["last"][-1]:
        features.add("candidate_is_last_source_number")
    if value in profile["last"][-2:]:
        features.add("candidate_in_last2_source_numbers")
    if value in profile["pair"]:
        features.add("candidate_in_pair_arithmetic_closure")
    try:
        number = abs(float(value))
    except ValueError:
        number = 0.0
    if number < 10:
        features.add("candidate_magnitude:small")
    elif number < 100:
        features.add("candidate_magnitude:medium")
    else:
        features.add("candidate_magnitude:large")
    return features


def _candidate_training_rows(
    surface: Surface,
    train_folds: set[int],
    *,
    folds: int,
) -> list[tuple[set[str], bool]]:
    rows: list[tuple[set[str], bool]] = []
    for index, example_id in enumerate(surface.reference_ids):
        if _fold_for_id(example_id, folds) not in train_folds:
            continue
        source_row = _condition_source_row(surface, index, "matched")
        profile = _source_profile(source_row)
        fallback = _prediction_numeric(surface.records_by_label["target"].get(example_id))
        gold = _gold_numeric(surface, example_id)
        for candidate in _candidate_pool(surface, example_id):
            rows.append((_features_for_candidate(candidate, profile, fallback), candidate.value == gold))
    return rows


def _fit_weights(training_rows: Sequence[tuple[set[str], bool]], *, alpha: float = 1.0) -> dict[str, float]:
    correct = Counter()
    incorrect = Counter()
    feature_names: set[str] = set()
    correct_total = 0
    incorrect_total = 0
    for features, is_correct in training_rows:
        feature_names.update(features)
        if is_correct:
            correct.update(features)
            correct_total += 1
        else:
            incorrect.update(features)
            incorrect_total += 1
    weights: dict[str, float] = {}
    for name in feature_names:
        p_pos = (correct[name] + alpha) / (correct_total + 2.0 * alpha)
        p_neg = (incorrect[name] + alpha) / (incorrect_total + 2.0 * alpha)
        weights[name] = math.log(p_pos / p_neg)
    prior_pos = (correct_total + alpha) / (correct_total + incorrect_total + 2.0 * alpha)
    prior_neg = 1.0 - prior_pos
    weights["__prior__"] = math.log(prior_pos / prior_neg)
    return weights


def _score(features: set[str], weights: dict[str, float]) -> float:
    return float(weights.get("__prior__", 0.0) + sum(weights.get(name, 0.0) for name in features))


def _decode_example(
    *,
    surface: Surface,
    index: int,
    condition: str,
    weights: dict[str, float],
    rule: dict[str, float],
) -> dict[str, Any]:
    example_id = surface.reference_ids[index]
    fallback = _prediction_numeric(surface.records_by_label["target"].get(example_id))
    source_row = _condition_source_row(surface, index, condition)
    profile = _source_profile(source_row)
    candidates = _candidate_pool(surface, example_id)
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        features = _features_for_candidate(candidate, profile, fallback)
        scored.append(
            {
                "value": candidate.value,
                "labels": list(candidate.labels),
                "score": _score(features, weights),
                "features": sorted(features),
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), item["value"]))
    best = scored[0] if scored else {"value": fallback, "score": float("-inf"), "features": []}
    second_score = float(scored[1]["score"]) if len(scored) > 1 else float("-inf")
    margin = float(best["score"]) - second_score
    accepted = (
        profile["quality"]
        and best["value"] != fallback
        and float(best["score"]) >= float(rule["min_score"])
        and margin >= float(rule["min_margin"])
    )
    prediction = str(best["value"]) if accepted else fallback
    gold = _gold_numeric(surface, example_id)
    return {
        "example_id": example_id,
        "prediction": prediction,
        "fallback_prediction": fallback,
        "gold_answer": gold,
        "correct": prediction == gold,
        "accepted_source_sidecar": bool(accepted),
        "best_candidate": best,
        "best_margin": margin,
        "source_quality": bool(profile["quality"]),
        "predicate_names": sorted(profile["features"]),
        "sidecar_bytes": max(1, math.ceil((len(profile["features"]) + 8) / 8)),
    }


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    finite = sorted(set(float(value) for value in values if math.isfinite(float(value))))
    if not finite:
        return [float("inf")]
    mids = [(left + right) / 2.0 for left, right in zip(finite, finite[1:])]
    return sorted(set([min(finite) - 1e-6, *finite, *mids]))


def _fit_rule(
    *,
    surface: Surface,
    train_folds: set[int],
    folds: int,
    weights: dict[str, float],
    accept_penalty: float,
    harm_weight: float,
) -> dict[str, float]:
    train_rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(surface.reference_ids):
        if _fold_for_id(example_id, folds) not in train_folds:
            continue
        raw = _decode_example(
            surface=surface,
            index=index,
            condition="matched",
            weights=weights,
            rule={"min_score": float("-inf"), "min_margin": float("-inf")},
        )
        train_rows.append(raw)
    score_values = [float(row["best_candidate"]["score"]) for row in train_rows]
    margin_values = [float(row["best_margin"]) for row in train_rows]
    best_rule = {"min_score": float("inf"), "min_margin": float("inf"), "train_score": 0.0}
    best_tuple: tuple[float, float, float, float] | None = None
    for min_score in _candidate_thresholds(score_values):
        for min_margin in _candidate_thresholds(margin_values):
            help_count = harm_count = accept_count = 0
            for row in train_rows:
                accepted = (
                    row["source_quality"]
                    and row["best_candidate"]["value"] != row["fallback_prediction"]
                    and float(row["best_candidate"]["score"]) >= min_score
                    and float(row["best_margin"]) >= min_margin
                )
                if not accepted:
                    continue
                accept_count += 1
                fallback_correct = row["fallback_prediction"] == row["gold_answer"]
                help_count += int(row["best_candidate"]["value"] == row["gold_answer"] and not fallback_correct)
                harm_count += int(row["best_candidate"]["value"] != row["gold_answer"] and fallback_correct)
            objective = help_count - float(harm_weight) * harm_count - float(accept_penalty) * accept_count
            ranking = (objective, -harm_count, help_count, -accept_count)
            if best_tuple is None or ranking > best_tuple:
                best_tuple = ranking
                best_rule = {
                    "min_score": float(min_score),
                    "min_margin": float(min_margin),
                    "train_score": float(objective),
                    "train_help": int(help_count),
                    "train_harm": int(harm_count),
                    "train_accept": int(accept_count),
                }
    return best_rule


def _analyze_surface(
    *,
    surface: Surface,
    outer_folds: int,
    accept_penalty: float,
    harm_weight: float,
) -> dict[str, Any]:
    folds = max(int(outer_folds), 1)
    fold_models: dict[int, dict[str, Any]] = {}
    rows_by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for fold in range(folds):
        train_folds = {other for other in range(folds) if other != fold} or {fold}
        weights = _fit_weights(_candidate_training_rows(surface, train_folds, folds=folds))
        rule = _fit_rule(
            surface=surface,
            train_folds=train_folds,
            folds=folds,
            weights=weights,
            accept_penalty=accept_penalty,
            harm_weight=harm_weight,
        )
        fold_models[fold] = {
            "rule": rule,
            "top_positive_weights": sorted(
                ((name, value) for name, value in weights.items() if name != "__prior__"),
                key=lambda item: -item[1],
            )[:12],
            "top_negative_weights": sorted(
                ((name, value) for name, value in weights.items() if name != "__prior__"),
                key=lambda item: item[1],
            )[:12],
        }
        for index, example_id in enumerate(surface.reference_ids):
            if _fold_for_id(example_id, folds) != fold:
                continue
            for condition in CONDITIONS:
                rows_by_condition[condition].append(
                    _decode_example(
                        surface=surface,
                        index=index,
                        condition=condition,
                        weights=weights,
                        rule=rule,
                    )
                )
    clean_ids = _source_ids(surface, "clean_source_only") or _source_ids(surface, "clean_residual_targets")
    target_self_ids = _source_ids(surface, "target_self_repair")
    summaries = {
        condition: _summarize(rows, clean_ids=clean_ids, target_self_ids=target_self_ids)
        for condition, rows in rows_by_condition.items()
    }
    control_union: set[str] = set()
    for condition, summary in summaries.items():
        if condition == "matched":
            continue
        control_union.update(summary["clean_source_necessary_ids"])
    return {
        "label": surface.label,
        "target_set": _display_path(surface.target_set_path),
        "fold_models": fold_models,
        "summaries": summaries,
        "rows_by_condition": rows_by_condition,
        "control_clean_union_ids": sorted(control_union),
    }


def _summarize(
    rows: Sequence[dict[str, Any]],
    *,
    clean_ids: set[str],
    target_self_ids: set[str],
) -> dict[str, Any]:
    correct_ids = {row["example_id"] for row in rows if row["correct"]}
    accepted = [row for row in rows if row["accepted_source_sidecar"]]
    accepted_harm = [
        row["example_id"]
        for row in accepted
        if row["fallback_prediction"] == row["gold_answer"] and not row["correct"]
    ]
    clean_source = sorted(correct_ids & clean_ids)
    target_self_harm = sorted(
        row["example_id"] for row in rows if row["example_id"] in target_self_ids and not row["correct"]
    )
    return {
        "n": len(rows),
        "correct": len(correct_ids),
        "accepted": len(accepted),
        "accepted_harm_count": len(accepted_harm),
        "accepted_harm_ids": sorted(accepted_harm),
        "clean_source_necessary_count": len(clean_source),
        "clean_source_necessary_ids": clean_source,
        "target_self_harm_count": len(target_self_harm),
        "target_self_harm_ids": target_self_harm,
        "mean_sidecar_bytes": (
            sum(float(row["sidecar_bytes"]) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SVAMP Source Semantic Predicate Decoder",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- mode: `{payload['config']['mode']}`",
        "",
        "This CPU-only probe learns fold-local semantic predicate weights over",
        "source generated text and uses an erasure rule to preserve the target",
        "fallback unless the source-supported candidate is sufficiently unique.",
        "",
        "## Surfaces",
        "",
    ]
    for surface in payload["surfaces"]:
        matched = surface["summaries"]["matched"]
        lines.extend(
            [
                f"### {surface['label']}",
                "",
                f"- target set: `{surface['target_set']}`",
                f"- matched correct: `{matched['correct']}/{matched['n']}`",
                f"- matched accepted: `{matched['accepted']}`",
                f"- matched clean source-necessary: `{matched['clean_source_necessary_count']}`",
                f"- matched accepted harm: `{matched['accepted_harm_count']}`",
                f"- control clean union: `{len(surface['control_clean_union_ids'])}`",
                f"- clean IDs: `{matched['clean_source_necessary_ids']}`",
                "",
            ]
        )
    lines.extend(["## Decision", "", payload["decision"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _prediction_rows(surface: dict[str, Any], *, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in surface["rows_by_condition"]["matched"]:
        rows.append(
            {
                "answer": [row["gold_answer"]],
                "correct": bool(row["correct"]),
                "example_id": row["example_id"],
                "fallback_prediction": row["fallback_prediction"],
                "index": len(rows),
                "method": method,
                "normalized_prediction": row["prediction"],
                "prediction": row["prediction"],
                "accepted_source_sidecar": bool(row["accepted_source_sidecar"]),
                "sidecar_bytes": row["sidecar_bytes"],
                "predicate_names": row["predicate_names"],
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-target-set", required=True)
    parser.add_argument("--holdout-target-set", required=True)
    parser.add_argument("--mode", choices=["learned_logodds"], default="learned_logodds")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--accept-penalty", type=float, default=0.25)
    parser.add_argument("--harm-weight", type=float, default=4.0)
    parser.add_argument("--min-live-correct", type=int, default=25)
    parser.add_argument("--min-live-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-holdout-correct", type=int, default=10)
    parser.add_argument("--min-holdout-clean-source-necessary", type=int, default=1)
    parser.add_argument("--max-control-clean-union", type=int, default=0)
    parser.add_argument("--max-accepted-harm", type=int, default=0)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-predictions-jsonl")
    args = parser.parse_args(argv)

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces = [
        _analyze_surface(
            surface=_load_surface("live", _resolve(args.live_target_set)),
            outer_folds=args.outer_folds,
            accept_penalty=args.accept_penalty,
            harm_weight=args.harm_weight,
        ),
        _analyze_surface(
            surface=_load_surface("holdout", _resolve(args.holdout_target_set)),
            outer_folds=args.outer_folds,
            accept_penalty=args.accept_penalty,
            harm_weight=args.harm_weight,
        ),
    ]
    live = surfaces[0]["summaries"]["matched"]
    holdout = surfaces[1]["summaries"]["matched"]
    control_union = set(surfaces[0]["control_clean_union_ids"]) | set(surfaces[1]["control_clean_union_ids"])
    pass_rule = {
        "min_live_correct": live["correct"] >= args.min_live_correct,
        "min_live_clean_source_necessary": live["clean_source_necessary_count"]
        >= args.min_live_clean_source_necessary,
        "min_holdout_correct": holdout["correct"] >= args.min_holdout_correct,
        "min_holdout_clean_source_necessary": holdout["clean_source_necessary_count"]
        >= args.min_holdout_clean_source_necessary,
        "max_control_clean_union": len(control_union) <= args.max_control_clean_union,
        "max_accepted_harm": (
            live["accepted_harm_count"] <= args.max_accepted_harm
            and holdout["accepted_harm_count"] <= args.max_accepted_harm
        ),
    }
    status = (
        "semantic_predicate_decoder_passes_smoke"
        if all(pass_rule.values())
        else "semantic_predicate_decoder_fails_smoke"
    )
    decision = (
        "Promote to strict small gate with frozen IDs and broader controls."
        if status == "semantic_predicate_decoder_passes_smoke"
        else (
            "Do not promote this semantic-predicate decoder on current artifacts. "
            "If revived, it needs stronger source surfaces or model-collected "
            "target-side likelihood/uncertainty features."
        )
    )
    payload = {
        "date": args.date,
        "status": status,
        "config": vars(args),
        "surfaces": surfaces,
        "pass_rule": pass_rule,
        "control_clean_union_ids": sorted(control_union),
        "decision": decision,
    }
    json_path = output_dir / "semantic_predicate_decoder.json"
    md_path = output_dir / "semantic_predicate_decoder.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(md_path, payload)
    if args.output_predictions_jsonl:
        prediction_rows = _prediction_rows(surfaces[0], method="semantic_predicate_decoder_live")
        prediction_rows.extend(_prediction_rows(surfaces[1], method="semantic_predicate_decoder_holdout"))
        _write_jsonl(_resolve(args.output_predictions_jsonl), prediction_rows)
    print(json.dumps({"status": status, "output_json": _display_path(json_path)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
