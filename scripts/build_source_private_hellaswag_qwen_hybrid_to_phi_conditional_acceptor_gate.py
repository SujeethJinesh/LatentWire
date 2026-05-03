from __future__ import annotations

"""Cached Qwen-hybrid-to-Phi conditional acceptor receiver gate.

This gate tests a tiny target-aware receiver over the public multiple-choice
score simplex. It defaults to the fixed Qwen hybrid packet and may override to
Phi's target prediction only when a frozen fit/select rule fires.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate_20260503_validation1024_2048"
)
DEFAULT_SLICES = (
    {
        "slice_start": 1024,
        "slice_end_exclusive": 1536,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/"
            "qwen_strict_packet_predictions_1024_1536.jsonl"
        ),
        "phi_target_score_cache": (
            "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536/"
            "target_score_cache.json"
        ),
    },
    {
        "slice_start": 1536,
        "slice_end_exclusive": 2048,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/"
            "qwen_strict_packet_predictions_1536_2048.jsonl"
        ),
        "phi_target_score_cache": (
            "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048/"
            "target_score_cache.json"
        ),
    },
)
CONTROL_FIELDS = (
    "source_label_prediction",
    "source_rank_only_bagged_prediction",
    "score_only_bagged_prediction",
    "score_mean_prediction",
    "score_vote_prediction",
    "trained_label_prediction",
    "wrong_example_hidden_prediction",
    "zero_hidden_prediction",
    "candidate_roll_hidden_prediction",
    "score_channel_roll_hidden_prediction",
)
BOOTSTRAP_SAMPLES = 5000
FIT_ROWS_PER_SLICE = 64
SELECT_ROWS_PER_SLICE = 64


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


def _hybrid_prediction(row: dict[str, Any]) -> int:
    if int(row["hidden_mean_prediction"]) == int(row["score_mean_prediction"]):
        return int(row["vote_prediction"])
    return int(row["hidden_mean_prediction"])


def _load_rows(
    *,
    slices: tuple[dict[str, Any], ...],
    fit_rows_per_slice: int,
    select_rows_per_slice: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for spec in slices:
        qwen_path = pathlib.Path(spec["qwen_predictions"])
        phi_path = pathlib.Path(spec["phi_target_score_cache"])
        qwen_rows = _read_jsonl(qwen_path)
        phi_cache = _read_json(phi_path)
        if len(qwen_rows) != int(phi_cache["row_count"]):
            raise ValueError(f"row-count mismatch for slice {spec['slice_start']}")
        for index, (row, row_id, target_pred, target_scores) in enumerate(
            zip(
                qwen_rows,
                phi_cache["row_ids"],
                phi_cache["source_predictions"],
                phi_cache["source_scores"],
                strict=True,
            )
        ):
            if str(row["row_id"]) != str(row_id):
                raise ValueError(f"row-id mismatch at slice {spec['slice_start']} index {index}")
            copied = dict(row)
            copied["phi_target_prediction"] = int(target_pred)
            copied["phi_target_scores"] = [float(value) for value in target_scores]
            copied["qwen_hybrid_prediction"] = _hybrid_prediction(row)
            copied["_slice_start"] = int(spec["slice_start"])
            copied["_slice_end_exclusive"] = int(spec["slice_end_exclusive"])
            copied["_within_slice_index"] = int(index)
            if index < int(fit_rows_per_slice):
                copied["_split"] = "fit"
            elif index < int(fit_rows_per_slice) + int(select_rows_per_slice):
                copied["_split"] = "select"
            else:
                copied["_split"] = "eval"
            rows.append(copied)
        metadata.append(
            {
                "slice_start": int(spec["slice_start"]),
                "slice_end_exclusive": int(spec["slice_end_exclusive"]),
                "rows": len(qwen_rows),
                "fit_rows": int(fit_rows_per_slice),
                "select_rows": int(select_rows_per_slice),
                "eval_rows": max(0, len(qwen_rows) - int(fit_rows_per_slice) - int(select_rows_per_slice)),
                "qwen_predictions": _display_path(qwen_path),
                "qwen_predictions_sha256": _sha256_file(qwen_path),
                "phi_target_score_cache": _display_path(phi_path),
                "phi_target_score_cache_sha256": _sha256_file(phi_path),
            }
        )
    return rows, metadata


def _answers(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


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


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _target_features(row: dict[str, Any], packet_prediction: int | None = None) -> dict[str, float]:
    packet_prediction = _hybrid_prediction(row) if packet_prediction is None else int(packet_prediction)
    target_prediction = int(row["phi_target_prediction"])
    selected = int(row["selected_prediction"])
    scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    order = np.argsort(-scores)
    probs = _softmax(scores)
    centered = scores - np.mean(scores)
    scale = np.std(centered)
    z = centered / (scale if scale > 1e-8 else 1.0)
    return {
        "target_margin": float(scores[order[0]] - scores[order[1]]),
        "target_z_margin": float(z[order[0]] - z[order[1]]),
        "target_max_prob": float(probs[order[0]]),
        "target_entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
        "target_eq_packet": float(target_prediction == packet_prediction),
        "target_eq_selected": float(target_prediction == selected),
        "packet_eq_selected": float(packet_prediction == selected),
        "selected_margin": float(row.get("selected_margin", 0.0)),
        "target_id": float(target_prediction),
        "packet_id": float(packet_prediction),
        "selected_id": float(selected),
        "target_minus_packet": float(target_prediction - packet_prediction),
    }


FEATURE_NAMES = (
    "target_margin",
    "target_z_margin",
    "target_max_prob",
    "target_entropy",
    "target_eq_packet",
    "target_eq_selected",
    "packet_eq_selected",
    "selected_margin",
    "target_id",
    "packet_id",
    "selected_id",
    "target_minus_packet",
)


def _rule_key(rule: Rule | None) -> str:
    if rule is None:
        return "no_op"
    return f"{rule[0]} {rule[1]} {rule[2]:.12g}"


def _rule_applies(rule: Rule, row: dict[str, Any], packet_prediction: int | None = None) -> bool:
    feature, op, threshold = rule
    value = _target_features(row, packet_prediction)[feature]
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"unknown op: {op}")


def _predict_with_rule(
    rows: list[dict[str, Any]],
    rule: Rule | None,
    packet_field: str = "qwen_hybrid_prediction",
) -> np.ndarray:
    predictions: list[int] = []
    for row in rows:
        packet = int(row[packet_field]) if packet_field in row else int(row[packet_field + "_prediction"])
        target = int(row["phi_target_prediction"])
        if rule is not None and target != packet and _rule_applies(rule, row, packet):
            predictions.append(target)
        else:
            predictions.append(packet)
    return np.asarray(predictions, dtype=np.int64)


def _candidate_rules(rows: list[dict[str, Any]]) -> list[Rule | None]:
    rules: list[Rule | None] = [None]
    switch_rows = [row for row in rows if int(row["phi_target_prediction"]) != int(row["qwen_hybrid_prediction"])]
    for feature in FEATURE_NAMES:
        values = sorted({_target_features(row)[feature] for row in switch_rows})
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
        is_discrete = set(values).issubset({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0})
        for threshold in thresholds:
            for op in ("<=", ">") + (("==", "!=") if is_discrete else ()):
                rule = (feature, op, float(threshold))
                fires = sum(
                    int(row["phi_target_prediction"]) != int(row["qwen_hybrid_prediction"])
                    and _rule_applies(rule, row)
                    for row in rows
                )
                if 2 <= fires <= max(2, len(switch_rows) - 2):
                    rules.append(rule)
    return list({_rule_key(rule): rule for rule in rules}.values())


def _select_rule(*, fit_rows: list[dict[str, Any]], select_rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = _answers(select_rows)
    hybrid = _field_array(select_rows, "qwen_hybrid_prediction")
    candidates = []
    for rule in _candidate_rules(fit_rows):
        predictions = _predict_with_rule(select_rows, rule)
        paired = _paired_ci(
            selected=predictions,
            baseline=hybrid,
            answers=answers,
            seed=20260503 + sum(ord(ch) for ch in _rule_key(rule)),
            samples=500,
        )
        candidates.append(
            {
                "rule": rule,
                "rule_name": _rule_key(rule),
                "select_accuracy": _accuracy(predictions, answers),
                "select_delta_vs_fixed_hybrid": paired["delta"],
                "select_helps_vs_fixed_hybrid": paired["helps"],
                "select_harms_vs_fixed_hybrid": paired["harms"],
                "select_override_count": int(np.sum(predictions != hybrid)),
            }
        )
    return max(
        candidates,
        key=lambda item: (
            item["select_accuracy"],
            item["select_delta_vs_fixed_hybrid"],
            item["select_helps_vs_fixed_hybrid"] - item["select_harms_vs_fixed_hybrid"],
            -item["select_override_count"],
            item["rule_name"],
        ),
    )


def _random_same_coverage(
    *,
    rows: list[dict[str, Any]],
    reference_predictions: np.ndarray,
    seed: int,
) -> np.ndarray:
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    target = _field_array(rows, "phi_target_prediction")
    switch_indices = np.flatnonzero(hybrid != target)
    override_count = int(np.sum(reference_predictions != hybrid))
    rng = np.random.default_rng(seed)
    predictions = hybrid.copy()
    if override_count > 0:
        chosen = rng.choice(switch_indices, size=min(override_count, len(switch_indices)), replace=False)
        predictions[chosen] = target[chosen]
    return predictions


def _oracle(rows: list[dict[str, Any]]) -> np.ndarray:
    answers = _answers(rows)
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    target = _field_array(rows, "phi_target_prediction")
    return np.where(hybrid == answers, hybrid, target).astype(np.int64)


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    bootstrap_samples: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = _answers(rows)
    vs_hybrid = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=30360503 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=30360531 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=30360603 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "target_only_accuracy": _accuracy(target_only, answers),
        "delta_vs_fixed_hybrid": vs_hybrid["delta"],
        "ci95_low_vs_fixed_hybrid": vs_hybrid["ci95_low"],
        "ci95_high_vs_fixed_hybrid": vs_hybrid["ci95_high"],
        "helps_vs_fixed_hybrid": vs_hybrid["helps"],
        "harms_vs_fixed_hybrid": vs_hybrid["harms"],
        "delta_vs_candidate_only": vs_candidate["delta"],
        "ci95_low_vs_candidate_only": vs_candidate["ci95_low"],
        "delta_vs_target_only": vs_target["delta"],
        "ci95_low_vs_target_only": vs_target["ci95_low"],
        "override_count": int(np.sum(predictions != fixed_hybrid)),
        "override_rate": float(np.mean(predictions != fixed_hybrid)),
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _slice_rows(
    *,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    answers = _answers(rows)
    starts = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    out = []
    for start in sorted(set(starts.tolist())):
        mask = starts == start
        paired = _paired_ci(
            selected=predictions[mask],
            baseline=fixed_hybrid[mask],
            answers=answers[mask],
            seed=40360503 + int(start),
            samples=bootstrap_samples,
        )
        out.append(
            {
                "slice_start": int(start),
                "eval_rows": int(np.sum(mask)),
                "method_accuracy": _accuracy(predictions[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]),
                "delta_vs_fixed_hybrid": paired["delta"],
                "ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                "helps_vs_fixed_hybrid": paired["helps"],
                "harms_vs_fixed_hybrid": paired["harms"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen Hybrid-To-Phi Conditional Acceptor Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- selected rule: `{h['selected_rule']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- conditional acceptor accuracy: `{h['conditional_acceptor_accuracy']:.6f}`",
        f"- delta vs fixed hybrid: `{h['conditional_delta_vs_fixed_hybrid']:.6f}`",
        f"- CI95 low vs fixed hybrid: `{h['conditional_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- target-or-hybrid oracle accuracy: `{h['target_or_hybrid_oracle_accuracy']:.6f}`",
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
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    slices: tuple[dict[str, Any], ...] = DEFAULT_SLICES,
    fit_rows_per_slice: int = FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = SELECT_ROWS_PER_SLICE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, metadata = _load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    fit_rows = [row for row in rows if row["_split"] == "fit"]
    select_rows = [row for row in rows if row["_split"] == "select"]
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    selected = _select_rule(fit_rows=fit_rows, select_rows=select_rows)
    rule = selected["rule"]

    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    conditional = _predict_with_rule(eval_rows, rule)
    random = _random_same_coverage(rows=eval_rows, reference_predictions=conditional, seed=20260503)
    oracle = _oracle(eval_rows)

    method_rows = [
        _method_row(
            name="conditional_target_acceptor",
            rows=eval_rows,
            predictions=conditional,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            details=selected,
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=eval_rows,
            predictions=fixed_hybrid,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="qwen_candidate_only",
            rows=eval_rows,
            predictions=candidate_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="phi_target_only",
            rows=eval_rows,
            predictions=target_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="random_same_coverage_target_override",
            rows=eval_rows,
            predictions=random,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            details={"seed": 20260503},
        ),
        _method_row(
            name="target_or_hybrid_oracle",
            rows=eval_rows,
            predictions=oracle,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            details={"oracle": True},
        ),
    ]
    for field in CONTROL_FIELDS:
        if field not in eval_rows[0]:
            continue
        control_rows = [dict(row, control_packet_prediction=int(row[field])) for row in eval_rows]
        control_predictions = _predict_with_rule(control_rows, rule, packet_field="control_packet_prediction")
        method_rows.append(
            _method_row(
                name=f"control_{field.removesuffix('_prediction')}",
                rows=eval_rows,
                predictions=control_predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                details={"packet_field": field, "frozen_rule": _rule_key(rule)},
            )
        )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    conditional_row = next(row for row in method_rows if row["method"] == "conditional_target_acceptor")
    control_rows = [row for row in method_rows if row["method"].startswith("control_")]
    best_control = max(control_rows, key=lambda row: row["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=conditional,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    pass_gate = (
        conditional_row["delta_vs_fixed_hybrid"] >= 0.005
        and conditional_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and conditional_row["ci95_low_vs_candidate_only"] > 0.0
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and conditional_row["accuracy"] > best_control["accuracy"]
    )
    payload = {
        "gate": "source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if the frozen fit/select conditional target acceptor beats fixed Qwen hybrid by "
            "at least 0.005 with positive paired CI, still beats Qwen candidate-only with positive paired CI, "
            "is nonnegative on both cached Phi slices, and remains above source-destroying controls."
        ),
        "headline": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
            "eval_rows": len(eval_rows),
            "selected_rule": _rule_key(rule),
            "fixed_hybrid_accuracy": next(
                row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement"
            )["accuracy"],
            "conditional_acceptor_accuracy": conditional_row["accuracy"],
            "conditional_delta_vs_fixed_hybrid": conditional_row["delta_vs_fixed_hybrid"],
            "conditional_ci95_low_vs_fixed_hybrid": conditional_row["ci95_low_vs_fixed_hybrid"],
            "conditional_delta_vs_candidate_only": conditional_row["delta_vs_candidate_only"],
            "conditional_ci95_low_vs_candidate_only": conditional_row["ci95_low_vs_candidate_only"],
            "conditional_overrides": conditional_row["override_count"],
            "conditional_help_vs_fixed_hybrid": conditional_row["helps_vs_fixed_hybrid"],
            "conditional_harm_vs_fixed_hybrid": conditional_row["harms_vs_fixed_hybrid"],
            "best_control_name": best_control["method"],
            "best_control_accuracy": best_control["accuracy"],
            "target_or_hybrid_oracle_accuracy": next(
                row for row in method_rows if row["method"] == "target_or_hybrid_oracle"
            )["accuracy"],
        },
        "packet_contract": {
            "receiver_visible_payload": "one Qwen hybrid source candidate id plus cached Phi target scores used locally",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "target_scores_are_receiver_side_information": True,
        },
        "slice_metadata": metadata,
        "method_rows": method_rows,
        "slice_rows": slice_rows,
        "interpretation": (
            "The conditional acceptor tests whether Phi's own score simplex can safely override the fixed "
            "Qwen hybrid packet. This is the cached target-aware receiver branch recommended after shallow "
            "source-side vetoes failed. A failure means the current target-score acceptor cannot access the "
            "large target-or-hybrid oracle headroom without sacrificing packet utility."
        ),
        "lay_explanation": (
            "Phi sometimes has useful information that could improve the Qwen hint. This test learns a tiny "
            "rule for when Phi should override the hint. On new rows, that rule must beat simply trusting the "
            "Qwen hybrid hint."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.json",
                "hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.md",
                "method_rows.csv",
                "slice_rows.csv",
            ],
            "slice_metadata": metadata,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-rows-per-slice", type=int, default=FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=SELECT_ROWS_PER_SLICE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(json.dumps({
        "pass_gate": payload["pass_gate"],
        "selected_rule": h["selected_rule"],
        "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
        "conditional_acceptor_accuracy": h["conditional_acceptor_accuracy"],
        "conditional_delta_vs_fixed_hybrid": h["conditional_delta_vs_fixed_hybrid"],
        "conditional_ci95_low_vs_fixed_hybrid": h["conditional_ci95_low_vs_fixed_hybrid"],
        "target_or_hybrid_oracle_accuracy": h["target_or_hybrid_oracle_accuracy"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
