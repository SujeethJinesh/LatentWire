from __future__ import annotations

"""Evaluate packet-preserving anti-harm vetoes for the strict HellaSwag hybrid.

The live strict packet policy emits one source candidate id using a fixed
hybrid rule. This gate asks whether a shallow source-side veto can avoid the
hybrid's harmful switches while preserving the same receiver-visible packet.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any, Callable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = pathlib.Path(
    "results/source_private_hellaswag_strict_candidate_only_packet_audit_20260503_validation0_9216/"
    "hellaswag_strict_candidate_only_packet_audit.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hybrid_anti_harm_veto_gate_20260503_validation0_9216"
)
DEFAULT_CROSS_FAMILY_SLICES = (
    {
        "slice_start": 1024,
        "slice_end_exclusive": 1536,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/"
            "qwen_strict_packet_predictions_1024_1536.jsonl"
        ),
    },
    {
        "slice_start": 1536,
        "slice_end_exclusive": 2048,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/"
            "qwen_strict_packet_predictions_1536_2048.jsonl"
        ),
    },
)
BOOTSTRAP_SAMPLES = 5000
MIN_VETO_SUPPORT = 3
MIN_ALLOW_SUPPORT = 3
FIT_ROWS = 512
SELECT_ROWS = 512
CROSS_FAMILY_EXCLUDED_PREFIX_ROWS_PER_SLICE = 128


Rule = tuple[str, str, float]


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(input_path: pathlib.Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(input_path)
    rows: list[dict[str, Any]] = []
    for slice_row in payload["slice_rows"]:
        predictions_path = pathlib.Path(slice_row["predictions_path"])
        for within_slice_index, row in enumerate(_read_jsonl(predictions_path)):
            copied = dict(row)
            copied["_slice_start"] = int(slice_row["eval_slice_start"])
            copied["_slice_end_exclusive"] = int(
                slice_row.get("eval_slice_end_exclusive", int(slice_row["eval_slice_start"]) + 1024)
            )
            copied["_within_slice_index"] = int(within_slice_index)
            copied["_predictions_path"] = _display_path(predictions_path)
            rows.append(copied)
    if not rows:
        raise ValueError("no prediction rows loaded")
    required = {
        "answer_index",
        "selected_prediction",
        "vote_prediction",
        "hidden_mean_prediction",
        "score_mean_prediction",
        "score_vote_prediction",
        "trained_label_prediction",
        "selected_margin",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"missing required fields: {missing}")
    return rows, payload


def _load_cross_family_rows(
    *,
    cross_family_slices: tuple[dict[str, Any], ...],
    excluded_prefix_rows_per_slice: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for spec in cross_family_slices:
        predictions_path = pathlib.Path(spec["qwen_predictions"])
        slice_rows = _read_jsonl(predictions_path)
        for within_slice_index, row in enumerate(slice_rows):
            copied = dict(row)
            copied["_slice_start"] = int(spec["slice_start"])
            copied["_slice_end_exclusive"] = int(spec["slice_end_exclusive"])
            copied["_within_slice_index"] = int(within_slice_index)
            copied["_split"] = (
                "excluded_prefix"
                if within_slice_index < int(excluded_prefix_rows_per_slice)
                else "eval"
            )
            copied["_predictions_path"] = _display_path(predictions_path)
            rows.append(copied)
        metadata.append(
            {
                "slice_start": int(spec["slice_start"]),
                "slice_end_exclusive": int(spec["slice_end_exclusive"]),
                "rows": len(slice_rows),
                "excluded_prefix_rows": int(excluded_prefix_rows_per_slice),
                "eval_rows": max(0, len(slice_rows) - int(excluded_prefix_rows_per_slice)),
                "qwen_predictions": _display_path(predictions_path),
                "qwen_predictions_sha256": _sha256_file(predictions_path),
            }
        )
    return rows, metadata


def _hybrid_prediction(row: dict[str, Any]) -> int:
    if int(row["hidden_mean_prediction"]) == int(row["score_mean_prediction"]):
        return int(row["vote_prediction"])
    return int(row["hidden_mean_prediction"])


def _answers(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _selected(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["selected_prediction"]) for row in rows], dtype=np.int64)


def _hybrid(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_hybrid_prediction(row) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    deltas = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(deltas), size=len(deltas))
        draws.append(float(np.mean(deltas[indices])))
    return {
        "delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "helps": int(np.sum(deltas > 0)),
        "harms": int(np.sum(deltas < 0)),
    }


def _feature(row: dict[str, Any], name: str) -> float:
    hybrid = _hybrid_prediction(row)
    selected = int(row["selected_prediction"])
    values = {
        "selected_margin": float(row["selected_margin"]),
        "abs_selected_margin": abs(float(row["selected_margin"])),
        "hidden_eq_score_mean": float(int(row["hidden_mean_prediction"]) == int(row["score_mean_prediction"])),
        "vote_eq_score_vote": float(int(row["vote_prediction"]) == int(row["score_vote_prediction"])),
        "vote_eq_trained_label": float(int(row["vote_prediction"]) == int(row["trained_label_prediction"])),
        "vote_eq_score_mean": float(int(row["vote_prediction"]) == int(row["score_mean_prediction"])),
        "hidden_eq_score_vote": float(int(row["hidden_mean_prediction"]) == int(row["score_vote_prediction"])),
        "hybrid_eq_score_vote": float(hybrid == int(row["score_vote_prediction"])),
        "hybrid_eq_trained_label": float(hybrid == int(row["trained_label_prediction"])),
        "selected_eq_score_vote": float(selected == int(row["score_vote_prediction"])),
        "selected_eq_trained_label": float(selected == int(row["trained_label_prediction"])),
        "hybrid_minus_selected": float(hybrid - selected),
        "selected_id": float(selected),
        "hybrid_id": float(hybrid),
    }
    return values[name]


FEATURE_NAMES = (
    "selected_margin",
    "abs_selected_margin",
    "hidden_eq_score_mean",
    "vote_eq_score_vote",
    "vote_eq_trained_label",
    "vote_eq_score_mean",
    "hidden_eq_score_vote",
    "hybrid_eq_score_vote",
    "hybrid_eq_trained_label",
    "selected_eq_score_vote",
    "selected_eq_trained_label",
    "hybrid_minus_selected",
    "selected_id",
    "hybrid_id",
)


def _rule_applies(rule: Rule, row: dict[str, Any]) -> bool:
    feature_name, op, threshold = rule
    value = _feature(row, feature_name)
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"unknown rule op: {op}")


def _rule_predictions(rows: list[dict[str, Any]], rule: Rule | None) -> np.ndarray:
    predictions: list[int] = []
    for row in rows:
        selected = int(row["selected_prediction"])
        hybrid = _hybrid_prediction(row)
        if hybrid == selected or rule is None:
            predictions.append(hybrid)
            continue
        predictions.append(selected if _rule_applies(rule, row) else hybrid)
    return np.asarray(predictions, dtype=np.int64)


def _rule_key(rule: Rule | None) -> str:
    if rule is None:
        return "no_op"
    return f"{rule[0]} {rule[1]} {rule[2]:.12g}"


def _candidate_rules(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    min_veto_support: int = MIN_VETO_SUPPORT,
    min_allow_support: int = MIN_ALLOW_SUPPORT,
) -> list[Rule]:
    switch_rows = [row for row in rows if _hybrid_prediction(row) != int(row["selected_prediction"])]
    rules: list[Rule] = []
    for feature_name in feature_names:
        values = sorted({_feature(row, feature_name) for row in switch_rows})
        if not values:
            continue
        thresholds = values
        if len(values) > 50:
            thresholds = sorted(
                {
                    values[int(round(quantile * (len(values) - 1)))]
                    for quantile in np.linspace(0.05, 0.95, 19)
                }
            )
        for threshold in thresholds:
            for op in ("<=", ">"):
                rule = (feature_name, op, float(threshold))
                veto_count = sum(_rule_applies(rule, row) for row in switch_rows)
                allow_count = len(switch_rows) - veto_count
                if veto_count >= min_veto_support and allow_count >= min_allow_support:
                    rules.append(rule)
            if set(values).issubset({0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0}):
                for op in ("==", "!="):
                    rule = (feature_name, op, float(threshold))
                    veto_count = sum(_rule_applies(rule, row) for row in switch_rows)
                    allow_count = len(switch_rows) - veto_count
                    if veto_count >= min_veto_support and allow_count >= min_allow_support:
                        rules.append(rule)
    unique = {}
    for rule in rules:
        unique[_rule_key(rule)] = rule
    return list(unique.values())


def _switch_stats(rows: list[dict[str, Any]], predictions: np.ndarray | None = None) -> dict[str, int | float]:
    answers = _answers(rows)
    selected = _selected(rows)
    hybrid = _hybrid(rows)
    if predictions is None:
        predictions = hybrid
    switch_mask = hybrid != selected
    veto_mask = predictions != hybrid
    missed_help_mask = switch_mask & (hybrid == answers) & (predictions != hybrid)
    avoided_harm_mask = switch_mask & (hybrid != answers) & (selected == answers) & (predictions == selected)
    introduced_harm_mask = switch_mask & (hybrid == answers) & (predictions != answers)
    return {
        "switch_count": int(np.sum(switch_mask)),
        "switch_rate": float(np.mean(switch_mask)),
        "hybrid_helps_candidate_only": int(np.sum(switch_mask & (hybrid == answers) & (selected != answers))),
        "hybrid_harms_candidate_only": int(np.sum(switch_mask & (hybrid != answers) & (selected == answers))),
        "hybrid_neutral_switches": int(np.sum(switch_mask & ((hybrid == answers) == (selected == answers)))),
        "veto_count": int(np.sum(veto_mask)),
        "veto_rate": float(np.mean(veto_mask)),
        "avoided_harms": int(np.sum(avoided_harm_mask)),
        "missed_hybrid_helps": int(np.sum(missed_help_mask)),
        "introduced_harms": int(np.sum(introduced_harm_mask)),
    }


def _score_rule(
    *,
    rows: list[dict[str, Any]],
    rule: Rule | None,
    baseline_fn: Callable[[list[dict[str, Any]]], np.ndarray] = _hybrid,
) -> dict[str, Any]:
    answers = _answers(rows)
    predictions = _rule_predictions(rows, rule)
    baseline = baseline_fn(rows)
    deltas = (predictions == answers).astype(np.int64) - (baseline == answers).astype(np.int64)
    stats = _switch_stats(rows, predictions)
    return {
        "rule": _rule_key(rule),
        "rule_tuple": rule,
        "delta_vs_hybrid": float(np.mean(deltas)),
        "net_help_vs_hybrid": int(np.sum(deltas)),
        "accuracy": _accuracy(predictions, answers),
        **stats,
    }


def _train_best_rule(
    *,
    train_rows: list[dict[str, Any]],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[str, Any]:
    candidates: list[Rule | None] = [None]
    candidates.extend(_candidate_rules(train_rows, feature_names=feature_names))
    scored = [_score_rule(rows=train_rows, rule=rule) for rule in candidates]
    return max(
        scored,
        key=lambda item: (
            item["delta_vs_hybrid"],
            item["avoided_harms"] - item["missed_hybrid_helps"],
            -item["veto_count"],
            item["rule"],
        ),
    )


def _select_rule_from_fit_and_selection(
    *,
    fit_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[str, Any]:
    candidates: list[Rule | None] = [None]
    candidates.extend(_candidate_rules(fit_rows, feature_names=feature_names))
    scored = [_score_rule(rows=selection_rows, rule=rule) for rule in candidates]
    best = max(
        scored,
        key=lambda item: (
            item["delta_vs_hybrid"],
            item["avoided_harms"] - item["missed_hybrid_helps"],
            -item["veto_count"],
            item["rule"],
        ),
    )
    best["candidate_rule_count"] = len(candidates)
    best["fit_rows"] = len(fit_rows)
    best["selection_rows"] = len(selection_rows)
    return best


def _random_same_coverage_predictions(
    *,
    rows: list[dict[str, Any]],
    reference_predictions: np.ndarray,
    seed: int,
) -> np.ndarray:
    hybrid = _hybrid(rows)
    selected = _selected(rows)
    switch_indices = np.flatnonzero(hybrid != selected)
    veto_count = int(np.sum(reference_predictions != hybrid))
    rng = np.random.default_rng(seed)
    predictions = hybrid.copy()
    if veto_count > 0:
        chosen = rng.choice(switch_indices, size=min(veto_count, len(switch_indices)), replace=False)
        predictions[chosen] = selected[chosen]
    return predictions


def _oracle_veto_predictions(rows: list[dict[str, Any]]) -> np.ndarray:
    answers = _answers(rows)
    selected = _selected(rows)
    hybrid = _hybrid(rows)
    predictions = hybrid.copy()
    use_selected = (hybrid != answers) & (selected == answers)
    predictions[use_selected] = selected[use_selected]
    return predictions


def _slice_starts(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["_slice_start"]) for row in rows})


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    candidate_only: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
    train_rows: int,
    selector_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = _answers(rows)
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=20260503 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_hybrid = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=20260603 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    slice_rows = _slice_method_rows(
        name=name,
        rows=rows,
        predictions=predictions,
        candidate_only=candidate_only,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    switch_stats = _switch_stats(rows, predictions)
    return {
        "method": name,
        "train_rows": int(train_rows),
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "delta_vs_candidate_only": vs_candidate["delta"],
        "ci95_low_vs_candidate_only": vs_candidate["ci95_low"],
        "ci95_high_vs_candidate_only": vs_candidate["ci95_high"],
        "helps_vs_candidate_only": vs_candidate["helps"],
        "harms_vs_candidate_only": vs_candidate["harms"],
        "delta_vs_fixed_hybrid": vs_hybrid["delta"],
        "ci95_low_vs_fixed_hybrid": vs_hybrid["ci95_low"],
        "ci95_high_vs_fixed_hybrid": vs_hybrid["ci95_high"],
        "helps_vs_fixed_hybrid": vs_hybrid["helps"],
        "harms_vs_fixed_hybrid": vs_hybrid["harms"],
        "improvement_slice_count_vs_fixed_hybrid": sum(
            item["delta_vs_fixed_hybrid"] > 0.0 for item in slice_rows
        ),
        **switch_stats,
        "selector_details": json.dumps(selector_details or {}, sort_keys=True),
    }


def _slice_method_rows(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    candidate_only: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    row_slices = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    answers = _answers(rows)
    out: list[dict[str, Any]] = []
    for slice_start in _slice_starts(rows):
        mask = row_slices == slice_start
        vs_candidate = _paired_ci(
            selected=predictions[mask],
            baseline=candidate_only[mask],
            answers=answers[mask],
            seed=30360503 + slice_start + sum(ord(ch) for ch in name),
            samples=bootstrap_samples,
        )
        vs_hybrid = _paired_ci(
            selected=predictions[mask],
            baseline=fixed_hybrid[mask],
            answers=answers[mask],
            seed=30360603 + slice_start + sum(ord(ch) for ch in name),
            samples=bootstrap_samples,
        )
        out.append(
            {
                "method": name,
                "eval_slice_start": int(slice_start),
                "eval_rows": int(np.sum(mask)),
                "accuracy": _accuracy(predictions[mask], answers[mask]),
                "candidate_only_accuracy": _accuracy(candidate_only[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]),
                "delta_vs_candidate_only": vs_candidate["delta"],
                "ci95_low_vs_candidate_only": vs_candidate["ci95_low"],
                "delta_vs_fixed_hybrid": vs_hybrid["delta"],
                "ci95_low_vs_fixed_hybrid": vs_hybrid["ci95_low"],
                "helps_vs_fixed_hybrid": vs_hybrid["helps"],
                "harms_vs_fixed_hybrid": vs_hybrid["harms"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    main = h["main_fit_select_veto"]
    cross = payload.get("cross_family") or {}
    cross_method = cross.get("method", {})
    lines = [
        "# HellaSwag Hybrid Anti-Harm Veto Gate",
        "",
        f"- positive method pass: `{payload['positive_method_pass']}`",
        f"- total rows: `{h['total_rows']}`",
        f"- fit / selection rows: `{h['fit_rows']}` / `{h['selection_rows']}`",
        f"- heldout eval rows: `{h['heldout_eval_rows']}`",
        f"- candidate-only accuracy: `{h['candidate_only_full_accuracy']:.6f}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_full_accuracy']:.6f}`",
        f"- main veto accuracy: `{main['accuracy']:.6f}`",
        f"- main veto delta vs fixed hybrid: `{main['delta_vs_fixed_hybrid']:.6f}`",
        f"- main veto CI95 low vs fixed hybrid: `{main['ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- main veto avoided harms / missed hybrid helps: `{main['avoided_harms']}` / `{main['missed_hybrid_helps']}`",
        f"- candidate/hybrid oracle accuracy: `{h['candidate_hybrid_oracle_full_accuracy']:.6f}`",
        f"- cross-family pass: `{h['cross_family_pass']}`",
    ]
    if cross_method:
        lines.extend(
            [
                f"- cross-family veto accuracy: `{cross_method['accuracy']:.6f}`",
                f"- cross-family veto delta vs fixed hybrid: `{cross_method['delta_vs_fixed_hybrid']:.6f}`",
            ]
        )
    lines.extend(
        [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
        ]
    )
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    input_path: pathlib.Path | str = DEFAULT_INPUT,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
    cross_family_slices: tuple[dict[str, Any], ...] = DEFAULT_CROSS_FAMILY_SLICES,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, input_payload = _load_rows(input_path)
    slice_starts = _slice_starts(rows)
    fit_slice_start = slice_starts[0]
    first_slice_rows = [row for row in rows if int(row["_slice_start"]) == fit_slice_start]
    fit_rows = [row for row in first_slice_rows if int(row["_within_slice_index"]) < FIT_ROWS]
    selection_rows = [
        row
        for row in first_slice_rows
        if FIT_ROWS <= int(row["_within_slice_index"]) < FIT_ROWS + SELECT_ROWS
    ]
    eval_rows = [row for row in rows if int(row["_slice_start"]) != fit_slice_start]
    if len(fit_rows) != FIT_ROWS or len(selection_rows) != SELECT_ROWS:
        raise ValueError("expected the first strict slice to contain 512 fit rows and 512 selection rows")

    full_answers = _answers(rows)
    full_selected = _selected(rows)
    full_hybrid = _hybrid(rows)
    full_oracle = _oracle_veto_predictions(rows)

    eval_selected = _selected(eval_rows)
    eval_hybrid = _hybrid(eval_rows)

    main_rule = _select_rule_from_fit_and_selection(
        fit_rows=fit_rows,
        selection_rows=selection_rows,
    )
    main_rule_tuple = main_rule["rule_tuple"]
    main_predictions = _rule_predictions(eval_rows, main_rule_tuple)

    margin_rule = _select_rule_from_fit_and_selection(
        fit_rows=fit_rows,
        selection_rows=selection_rows,
        feature_names=("selected_margin", "abs_selected_margin"),
    )
    margin_predictions = _rule_predictions(eval_rows, margin_rule["rule_tuple"])

    loso_predictions_by_row: dict[tuple[int, str], int] = {}
    loso_details = []
    for heldout_start in slice_starts:
        loso_train = [row for row in rows if int(row["_slice_start"]) != heldout_start]
        loso_eval = [row for row in rows if int(row["_slice_start"]) == heldout_start]
        trained = _train_best_rule(train_rows=loso_train)
        predictions = _rule_predictions(loso_eval, trained["rule_tuple"])
        for row, prediction in zip(loso_eval, predictions, strict=True):
            loso_predictions_by_row[(int(row["_slice_start"]), str(row["row_id"]))] = int(prediction)
        loso_details.append(
            {
                "heldout_slice_start": heldout_start,
                "selected_rule": trained["rule"],
                "train_delta_vs_hybrid": trained["delta_vs_hybrid"],
                "train_veto_count": trained["veto_count"],
            }
        )
    loso_predictions = np.asarray(
        [loso_predictions_by_row[(int(row["_slice_start"]), str(row["row_id"]))] for row in rows],
        dtype=np.int64,
    )

    random_predictions = _random_same_coverage_predictions(
        rows=eval_rows,
        reference_predictions=main_predictions,
        seed=20260503,
    )

    method_rows = [
        _method_row(
            name="candidate_only",
            rows=rows,
            predictions=full_selected,
            candidate_only=full_selected,
            fixed_hybrid=full_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=0,
            selector_details={"policy": "emit selected_prediction"},
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=rows,
            predictions=full_hybrid,
            candidate_only=full_selected,
            fixed_hybrid=full_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=0,
            selector_details={
                "policy": (
                    "if hidden_mean_prediction == score_mean_prediction, emit vote_prediction; "
                    "otherwise emit hidden_mean_prediction"
                )
            },
        ),
        _method_row(
            name="fit_select_single_rule_anti_harm_veto",
            rows=eval_rows,
            predictions=main_predictions,
            candidate_only=eval_selected,
            fixed_hybrid=eval_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=len(fit_rows) + len(selection_rows),
            selector_details={
                "fit_slice_start": fit_slice_start,
                "fit_rows": len(fit_rows),
                "selection_rows": len(selection_rows),
                "selected_rule": main_rule["rule"],
                "selection_delta_vs_hybrid": main_rule["delta_vs_hybrid"],
                "selection_accuracy": main_rule["accuracy"],
                "selection_veto_count": main_rule["veto_count"],
                "candidate_rule_count": main_rule["candidate_rule_count"],
            },
        ),
        _method_row(
            name="fit_select_margin_only_anti_harm_veto",
            rows=eval_rows,
            predictions=margin_predictions,
            candidate_only=eval_selected,
            fixed_hybrid=eval_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=len(fit_rows) + len(selection_rows),
            selector_details={
                "fit_slice_start": fit_slice_start,
                "fit_rows": len(fit_rows),
                "selection_rows": len(selection_rows),
                "selected_rule": margin_rule["rule"],
                "selection_delta_vs_hybrid": margin_rule["delta_vs_hybrid"],
                "selection_accuracy": margin_rule["accuracy"],
                "selection_veto_count": margin_rule["veto_count"],
                "candidate_rule_count": margin_rule["candidate_rule_count"],
            },
        ),
        _method_row(
            name="eval_random_same_coverage_veto",
            rows=eval_rows,
            predictions=random_predictions,
            candidate_only=eval_selected,
            fixed_hybrid=eval_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=len(fit_rows) + len(selection_rows),
            selector_details={
                "reference_method": "fit_select_single_rule_anti_harm_veto",
                "seed": 20260503,
            },
        ),
        _method_row(
            name="leave_one_slice_out_single_rule_anti_harm_veto",
            rows=rows,
            predictions=loso_predictions,
            candidate_only=full_selected,
            fixed_hybrid=full_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=len(rows) - 1024,
            selector_details={"diagnostic_only": True, "heldout_rules": loso_details},
        ),
        _method_row(
            name="candidate_hybrid_oracle_veto",
            rows=rows,
            predictions=full_oracle,
            candidate_only=full_selected,
            fixed_hybrid=full_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=0,
            selector_details={"oracle": "choose selected only when fixed hybrid would be wrong and selected is correct"},
        ),
    ]
    method_rows = sorted(method_rows, key=lambda row: (row["accuracy"], row["delta_vs_fixed_hybrid"]), reverse=True)

    slice_method_rows: list[dict[str, Any]] = []
    for row in method_rows:
        name = row["method"]
        if name == "fit_select_single_rule_anti_harm_veto":
            slice_method_rows.extend(
                _slice_method_rows(
                    name=name,
                    rows=eval_rows,
                    predictions=main_predictions,
                    candidate_only=eval_selected,
                    fixed_hybrid=eval_hybrid,
                    bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
                )
            )
        elif name == "fit_select_margin_only_anti_harm_veto":
            slice_method_rows.extend(
                _slice_method_rows(
                    name=name,
                    rows=eval_rows,
                    predictions=margin_predictions,
                    candidate_only=eval_selected,
                    fixed_hybrid=eval_hybrid,
                    bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
                )
            )
        elif name == "leave_one_slice_out_single_rule_anti_harm_veto":
            slice_method_rows.extend(
                _slice_method_rows(
                    name=name,
                    rows=rows,
                    predictions=loso_predictions,
                    candidate_only=full_selected,
                    fixed_hybrid=full_hybrid,
                    bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
                )
            )

    main_method = next(row for row in method_rows if row["method"] == "fit_select_single_rule_anti_harm_veto")
    fixed_hybrid_eval_row = _method_row(
        name="fixed_hybrid_eval_reference",
        rows=eval_rows,
        predictions=eval_hybrid,
        candidate_only=eval_selected,
        fixed_hybrid=eval_hybrid,
        bootstrap_samples=bootstrap_samples,
        train_rows=0,
        selector_details=None,
    )

    cross_family: dict[str, Any] | None = None
    cross_family_pass = True
    if cross_family_slices:
        cross_rows, cross_metadata = _load_cross_family_rows(
            cross_family_slices=cross_family_slices,
            excluded_prefix_rows_per_slice=CROSS_FAMILY_EXCLUDED_PREFIX_ROWS_PER_SLICE,
        )
        cross_eval_rows = [row for row in cross_rows if row["_split"] == "eval"]
        cross_selected = _selected(cross_eval_rows)
        cross_hybrid = _hybrid(cross_eval_rows)
        cross_veto = _rule_predictions(cross_eval_rows, main_rule_tuple)
        cross_main = _method_row(
            name="cross_family_frozen_fit_select_veto",
            rows=cross_eval_rows,
            predictions=cross_veto,
            candidate_only=cross_selected,
            fixed_hybrid=cross_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=len(fit_rows) + len(selection_rows),
            selector_details={
                "frozen_rule": main_rule["rule"],
                "source_training_surface": "strict_qwen_hellaswag_0_1024_fit_select",
            },
        )
        cross_fixed = _method_row(
            name="cross_family_fixed_hybrid_reference",
            rows=cross_eval_rows,
            predictions=cross_hybrid,
            candidate_only=cross_selected,
            fixed_hybrid=cross_hybrid,
            bootstrap_samples=bootstrap_samples,
            train_rows=0,
            selector_details=None,
        )
        cross_slices = _slice_method_rows(
            name="cross_family_frozen_fit_select_veto",
            rows=cross_eval_rows,
            predictions=cross_veto,
            candidate_only=cross_selected,
            fixed_hybrid=cross_hybrid,
            bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
        )
        cross_family_pass = (
            cross_main["delta_vs_fixed_hybrid"] >= 0.0
            and cross_main["ci95_low_vs_candidate_only"] > 0.0
            and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in cross_slices)
        )
        cross_family = {
            "pass_gate": bool(cross_family_pass),
            "metadata": cross_metadata,
            "eval_rows": len(cross_eval_rows),
            "method": cross_main,
            "fixed_hybrid_reference": cross_fixed,
            "slice_rows": cross_slices,
        }

    strict_main_pass = (
        main_method["accuracy"] > fixed_hybrid_eval_row["accuracy"]
        and main_method["ci95_low_vs_fixed_hybrid"] > 0.0
        and main_method["ci95_low_vs_candidate_only"] > 0.0
        and main_method["improvement_slice_count_vs_fixed_hybrid"] >= 7
        and main_method["avoided_harms"] > main_method["missed_hybrid_helps"]
    )
    positive_method_pass = strict_main_pass and cross_family_pass

    full_hybrid_vs_selected = _paired_ci(
        selected=full_hybrid,
        baseline=full_selected,
        answers=full_answers,
        seed=40360503,
        samples=bootstrap_samples,
    )
    payload = {
        "gate": "source_private_hellaswag_hybrid_anti_harm_veto_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "positive_method_pass": bool(positive_method_pass),
        "pass_rule": (
            "Promote the anti-harm veto only if the predeclared fit/select single-rule veto uses the first "
            "512 strict rows to define candidate source-side rules, the next 512 strict rows to select one "
            "rule, and then beats fixed hybrid on strict rows 1024:9216 with positive paired CI, still beats "
            "candidate-only with positive paired CI, improves fixed hybrid on at least seven heldout slices, "
            "avoids more harms than it sacrifices helps, and is nonnegative versus fixed hybrid on cached "
            "Qwen-to-Phi heldout rows. Leave-one-slice-out rows are diagnostic only."
        ),
        "headline": {
            "total_rows": len(rows),
            "fit_rows": len(fit_rows),
            "selection_rows": len(selection_rows),
            "fit_selection_rows": len(fit_rows) + len(selection_rows),
            "heldout_eval_rows": len(eval_rows),
            "candidate_only_full_accuracy": _accuracy(full_selected, full_answers),
            "fixed_hybrid_full_accuracy": _accuracy(full_hybrid, full_answers),
            "fixed_hybrid_delta_vs_candidate_only": full_hybrid_vs_selected["delta"],
            "fixed_hybrid_ci95_low_vs_candidate_only": full_hybrid_vs_selected["ci95_low"],
            "fixed_hybrid_helps_vs_candidate_only": full_hybrid_vs_selected["helps"],
            "fixed_hybrid_harms_vs_candidate_only": full_hybrid_vs_selected["harms"],
            "fixed_hybrid_eval_accuracy": fixed_hybrid_eval_row["accuracy"],
            "strict_main_pass": bool(strict_main_pass),
            "cross_family_pass": bool(cross_family_pass),
            "main_fit_select_veto": main_method,
            "candidate_hybrid_oracle_full_accuracy": _accuracy(full_oracle, full_answers),
            "candidate_hybrid_oracle_delta_vs_fixed_hybrid": next(
                row for row in method_rows if row["method"] == "candidate_hybrid_oracle_veto"
            )["delta_vs_fixed_hybrid"],
        },
        "packet_contract": {
            "receiver_visible_payload": "one final source candidate id after any source-side veto",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "extra_receiver_visible_veto_bit": False,
        },
        "source_artifact": _display_path(input_path),
        "source_artifact_sha256": _sha256_file(input_path),
        "source_candidate_audit_sha256": input_payload.get("source_artifact_sha256"),
        "method_rows": method_rows,
        "slice_method_rows": slice_method_rows,
        "cross_family": cross_family,
        "interpretation": (
            "The anti-harm veto gate tests a selective-classification style source-side accept/fallback rule "
            "for the current fixed hybrid packet. A pass would strengthen the harm-controlled packet method "
            "without changing the 1B raw / 4B framed receiver-visible contract. A failure means the current "
            "shallow packet features do not reliably separate hybrid helps from hybrid harms, so the next "
            "method branch should move to a real receiver/common-basis signal rather than another shallow veto."
        ),
        "lay_explanation": (
            "The hybrid hint sometimes changes the answer choice and occasionally makes it worse. This test "
            "uses the first half-slice to define simple warning rules, the next half-slice to pick one, then "
            "freezes it. When a later switch looks risky, the rule keeps the old hint; otherwise it uses the "
            "hybrid hint."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_hybrid_anti_harm_veto_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "slice_method_rows.csv", slice_method_rows)
    _write_markdown(output_dir / "hellaswag_hybrid_anti_harm_veto_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_hybrid_anti_harm_veto_gate.json",
                "hellaswag_hybrid_anti_harm_veto_gate.md",
                "method_rows.csv",
                "slice_method_rows.csv",
            ],
            "source_artifact": payload["source_artifact"],
            "source_artifact_sha256": payload["source_artifact_sha256"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    parser.add_argument("--skip-cross-family", action="store_true")
    args = parser.parse_args()
    payload = build_gate(
        input_path=args.input,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
        cross_family_slices=() if args.skip_cross_family else DEFAULT_CROSS_FAMILY_SLICES,
    )
    h = payload["headline"]
    main_veto = h["main_fit_select_veto"]
    print(json.dumps({
        "positive_method_pass": payload["positive_method_pass"],
        "candidate_only_full_accuracy": h["candidate_only_full_accuracy"],
        "fixed_hybrid_full_accuracy": h["fixed_hybrid_full_accuracy"],
        "main_veto_accuracy": main_veto["accuracy"],
        "main_veto_delta_vs_fixed_hybrid": main_veto["delta_vs_fixed_hybrid"],
        "main_veto_ci95_low_vs_fixed_hybrid": main_veto["ci95_low_vs_fixed_hybrid"],
        "candidate_hybrid_oracle_full_accuracy": h["candidate_hybrid_oracle_full_accuracy"],
        "cross_family_pass": h["cross_family_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
