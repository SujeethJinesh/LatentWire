from __future__ import annotations

"""HellaSwag source/target complementarity-frontier diagnostic.

This is a no-new-inference diagnostic for the COLM_v2/ICLR loop. It asks
whether existing Qwen source packets expose a learnable frontier where source
top-1/top-2 information can repair Phi/fixed-hybrid mistakes without collapsing
to source-choice or target-cache controls.
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402

DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_complementarity_frontier_diagnostic_20260504_validation1024_2048"
)
DEFAULT_SOURCE_SCORE_CACHE = oracle.DEFAULT_SOURCE_SCORE_CACHE
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES
FIT_ROWS_PER_SLICE = denoise.FIT_ROWS_PER_SLICE
SELECT_ROWS_PER_SLICE = denoise.SELECT_ROWS_PER_SLICE
RIDGE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _answers(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == np.asarray(answers, dtype=np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return oracle._paired_ci(
        selected=np.asarray(selected, dtype=np.int64),
        baseline=np.asarray(baseline, dtype=np.int64),
        answers=np.asarray(answers, dtype=np.int64),
        seed=seed,
        samples=samples,
    )


def _top2_from_scores(scores: Sequence[float]) -> tuple[int, int]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return int(order[0]), int(order[1])


def _entropy(scores: Sequence[float]) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _bin(value: float, thresholds: Sequence[float]) -> int:
    return int(np.digitize(float(value), np.asarray(thresholds, dtype=np.float64), right=False))


def _one_hot(value: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= int(value) < size:
        out[int(value)] = 1.0
    return out


def _source_packet(row: dict[str, Any], *, condition: str, substitute_row: dict[str, Any] | None = None) -> dict[str, Any]:
    source_row = row if substitute_row is None else substitute_row
    q_top1, q_top2 = _top2_from_scores(source_row["qwen_source_scores"])
    scores = np.asarray(source_row["qwen_source_scores"], dtype=np.float64)
    if condition == "candidate_roll":
        q_top1 = (q_top1 + 1) % 4
        q_top2 = (q_top2 + 1) % 4
        scores = np.roll(scores, 1)
    elif condition == "target_derived_packet":
        q_top1, q_top2 = _top2_from_scores(row["phi_target_scores"])
        scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    elif condition == "zero_source":
        q_top1, q_top2 = 0, 1
        scores = np.zeros(4, dtype=np.float64)
    order = np.argsort(-scores)
    margin = float(scores[order[0]] - scores[order[1]])
    return {
        "source_top1": int(q_top1),
        "source_top2": int(q_top2),
        "q_margin_bin": _bin(margin, (0.1, 0.25, 0.5, 1.0)),
        "q_entropy_bin": _bin(_entropy(scores), (0.7, 1.0, 1.2)),
    }


def _packets_for_condition(rows: Sequence[dict[str, Any]], *, condition: str, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if condition == "source_row_shuffle":
        order = rng.permutation(len(rows))
        return [
            _source_packet(row, condition="matched", substitute_row=rows[int(order[index])])
            for index, row in enumerate(rows)
        ]
    if condition == "random_same_byte":
        order = rng.integers(0, len(rows), size=len(rows))
        return [
            _source_packet(row, condition="matched", substitute_row=rows[int(order[index])])
            for index, row in enumerate(rows)
        ]
    return [_source_packet(row, condition=condition) for row in rows]


def _packet_action(row: dict[str, Any], packet: dict[str, Any]) -> int:
    first = int(packet["source_top1"])
    second = int(packet["source_top2"])
    phi_scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    return first if float(phi_scores[first]) >= float(phi_scores[second]) else second


def _feature_vector(row: dict[str, Any], packet: dict[str, Any], *, baseline: int) -> np.ndarray:
    phi_top1, phi_top2 = _top2_from_scores(row["phi_target_scores"])
    phi_scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    phi_margin = float(phi_scores[phi_top1] - phi_scores[phi_top2])
    action = _packet_action(row, packet)
    source_top1 = int(packet["source_top1"])
    source_top2 = int(packet["source_top2"])
    features: list[float] = [1.0]
    features.extend(_one_hot(source_top1, 4))
    features.extend(_one_hot(source_top2, 4))
    features.extend(_one_hot(phi_top1, 4))
    features.extend(_one_hot(int(row["qwen_hybrid_prediction"]), 4))
    features.extend(_one_hot(int(baseline), 4))
    features.extend(_one_hot(int(packet["q_margin_bin"]), 5))
    features.extend(_one_hot(int(packet["q_entropy_bin"]), 4))
    features.extend(_one_hot(_bin(phi_margin, (0.05, 0.1, 0.2, 0.4)), 5))
    features.extend(
        [
            float(source_top1 == phi_top1),
            float(source_top2 == phi_top1),
            float(source_top1 == int(row["qwen_hybrid_prediction"])),
            float(source_top2 == int(row["qwen_hybrid_prediction"])),
            float(source_top1 == int(baseline)),
            float(source_top2 == int(baseline)),
            float(action == phi_top1),
            float(action == int(baseline)),
            float(phi_scores[action] - phi_scores[int(baseline)]),
            float(phi_scores[action] - phi_scores[phi_top1]),
        ]
    )
    return np.asarray(features, dtype=np.float64)


def _feature_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_field: str,
    condition: str = "matched",
    seed: int = 20260504,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    packets = _packets_for_condition(rows, condition=condition, seed=seed)
    baseline = _field_array(rows, baseline_field)
    actions = np.asarray([_packet_action(row, packet) for row, packet in zip(rows, packets, strict=True)], dtype=np.int64)
    x = np.vstack(
        [
            _feature_vector(row, packet, baseline=int(base))
            for row, packet, base in zip(rows, packets, baseline, strict=True)
        ]
    )
    return x, actions, packets


def _fit_utility_model(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_field: str,
    l2: float,
) -> dict[str, Any]:
    x, actions, _ = _feature_matrix(rows, baseline_field=baseline_field)
    answers = _answers(rows)
    baseline = _field_array(rows, baseline_field)
    utility = (actions == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    penalty = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ utility
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    return {"baseline_field": baseline_field, "l2": float(l2), "weights": weights.tolist()}


def _score_rows(
    rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    *,
    condition: str = "matched",
    seed: int = 20260504,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x, actions, packets = _feature_matrix(
        rows,
        baseline_field=str(model["baseline_field"]),
        condition=condition,
        seed=seed,
    )
    weights = np.asarray(model["weights"], dtype=np.float64)
    return x @ weights, actions, packets


def _select_model(
    *,
    fit_rows: Sequence[dict[str, Any]],
    select_rows: Sequence[dict[str, Any]],
    baseline_field: str,
    l2_values: Sequence[float],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answers = _answers(select_rows)
    baseline = _field_array(select_rows, baseline_field)
    config_rows: list[dict[str, Any]] = []
    best_model: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for l2 in l2_values:
        model = _fit_utility_model(fit_rows, baseline_field=baseline_field, l2=float(l2))
        scores, actions, _ = _score_rows(select_rows, model)
        thresholds = sorted(set(float(value) for value in np.quantile(scores, np.linspace(0.0, 1.0, 21))))
        thresholds = [float("inf")] + thresholds
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, actions, baseline).astype(np.int64)
            paired = _paired_ci(
                selected=predictions,
                baseline=baseline,
                answers=answers,
                seed=20260504 + int(float(l2) * 1000) + int(abs(threshold) * 1000 if np.isfinite(threshold) else 17),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            overrides = int(np.sum(predictions != baseline))
            row = {
                "baseline_field": baseline_field,
                "l2": float(l2),
                "threshold": None if not np.isfinite(threshold) else float(threshold),
                "threshold_is_noop": bool(not np.isfinite(threshold)),
                "select_accuracy": _accuracy(predictions, answers),
                "select_delta_vs_baseline": paired["delta"],
                "select_ci95_low_vs_baseline": paired["ci95_low"],
                "select_helps_vs_baseline": paired["helps"],
                "select_harms_vs_baseline": paired["harms"],
                "select_overrides": overrides,
            }
            config_rows.append(row)
            key = (
                float(row["select_ci95_low_vs_baseline"]),
                float(row["select_delta_vs_baseline"]),
                float(row["select_accuracy"]),
                -float(row["select_harms_vs_baseline"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = {
                    **model,
                    "threshold": row["threshold"],
                    "threshold_is_noop": row["threshold_is_noop"],
                    "select_row": row,
                }
    if best_model is None:
        raise RuntimeError("no selected model")
    return best_model, config_rows


def _predict(rows: Sequence[dict[str, Any]], model: dict[str, Any], *, condition: str = "matched") -> tuple[np.ndarray, np.ndarray]:
    baseline = _field_array(rows, str(model["baseline_field"]))
    scores, actions, _ = _score_rows(rows, model, condition=condition, seed=20260504 + sum(ord(ch) for ch in condition))
    threshold = float("inf") if model.get("threshold_is_noop") else float(model["threshold"])
    predictions = np.where(scores >= threshold, actions, baseline).astype(np.int64)
    return predictions, scores


def _method_row(
    *,
    name: str,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    bootstrap_samples: int,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paired = _paired_ci(
        selected=predictions,
        baseline=baseline,
        answers=answers,
        seed=20260504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "accuracy": _accuracy(predictions, answers),
        "baseline_accuracy": _accuracy(baseline, answers),
        "delta_vs_baseline": paired["delta"],
        "ci95_low_vs_baseline": paired["ci95_low"],
        "ci95_high_vs_baseline": paired["ci95_high"],
        "helps_vs_baseline": paired["helps"],
        "harms_vs_baseline": paired["harms"],
        "override_count_vs_baseline": int(np.sum(predictions != baseline)),
        "eval_rows": len(rows),
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "raw_source_scores_or_logits_exposed": False,
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _frontier_counts(rows: Sequence[dict[str, Any]], *, split: str) -> dict[str, Any]:
    answers = _answers(rows)
    target = _field_array(rows, "phi_target_prediction")
    fixed = _field_array(rows, "qwen_hybrid_prediction")
    selected = _field_array(rows, "selected_prediction")
    top1 = []
    top2 = []
    pair_phi = []
    for row in rows:
        packet = _source_packet(row, condition="matched")
        top1.append(packet["source_top1"])
        top2.append(packet["source_top2"])
        pair_phi.append(_packet_action(row, packet))
    top1 = np.asarray(top1, dtype=np.int64)
    top2 = np.asarray(top2, dtype=np.int64)
    pair_phi = np.asarray(pair_phi, dtype=np.int64)
    top1_or_top2_correct = (top1 == answers) | (top2 == answers)
    target_wrong_source_can_help = (target != answers) & top1_or_top2_correct
    fixed_wrong_source_can_help = (fixed != answers) & top1_or_top2_correct
    target_harm_risk = (target == answers) & (pair_phi != answers)
    fixed_harm_risk = (fixed == answers) & (pair_phi != answers)
    return {
        "split": split,
        "rows": len(rows),
        "target_only_accuracy": _accuracy(target, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed, answers),
        "candidate_only_accuracy": _accuracy(selected, answers),
        "source_top1_accuracy": _accuracy(top1, answers),
        "source_pair_phi_accuracy": _accuracy(pair_phi, answers),
        "source_top1_or_top2_oracle_accuracy": float(np.mean(top1_or_top2_correct)),
        "target_wrong_source_top1_or_top2_correct": int(np.sum(target_wrong_source_can_help)),
        "fixed_wrong_source_top1_or_top2_correct": int(np.sum(fixed_wrong_source_can_help)),
        "target_harm_risk_if_pair_phi": int(np.sum(target_harm_risk)),
        "fixed_harm_risk_if_pair_phi": int(np.sum(fixed_harm_risk)),
    }


def _prediction_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: dict[str, Any],
    predictions: np.ndarray,
    scores: np.ndarray,
    baseline_field: str,
) -> list[dict[str, Any]]:
    answers = _answers(rows)
    baseline = _field_array(rows, baseline_field)
    _, actions, packets = _feature_matrix(rows, baseline_field=baseline_field)
    out: list[dict[str, Any]] = []
    for index, (row, packet) in enumerate(zip(rows, packets, strict=True)):
        out.append(
            {
                "row_id": str(row["row_id"]),
                "slice_start": int(row["_slice_start"]),
                "split": str(row["_split"]),
                "answer_index": int(answers[index]),
                "baseline_prediction": int(baseline[index]),
                "packet_action_prediction": int(actions[index]),
                "method_prediction": int(predictions[index]),
                "utility_score": float(scores[index]),
                "method_correct": bool(predictions[index] == answers[index]),
                "baseline_correct": bool(baseline[index] == answers[index]),
                "override_baseline": bool(predictions[index] != baseline[index]),
                "source_top1": int(packet["source_top1"]),
                "source_top2": int(packet["source_top2"]),
                "q_margin_bin": int(packet["q_margin_bin"]),
                "q_entropy_bin": int(packet["q_entropy_bin"]),
                "selected_model_l2": float(model["l2"]),
                "selected_threshold": model["threshold"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Complementarity-Frontier Diagnostic",
        "",
        f"- created UTC: `{payload['created_utc']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        "",
        "## Headline",
        "",
        f"- target-only accuracy: `{h['target_only_accuracy']:.6f}`",
        f"- fixed-hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- source top1/top2 oracle accuracy: `{h['source_top1_or_top2_oracle_accuracy']:.6f}`",
        f"- target-wrong/source-can-help rows: `{h['target_wrong_source_top1_or_top2_correct']}`",
        f"- fixed-wrong/source-can-help rows: `{h['fixed_wrong_source_top1_or_top2_correct']}`",
        f"- selected selector accuracy: `{h['selected_selector_accuracy']:.6f}`",
        f"- selected selector delta vs fixed: `{h['selected_selector_delta_vs_fixed_hybrid']:.6f}`",
        f"- best destructive control: `{h['best_destructive_control_name']}` at `{h['best_destructive_control_accuracy']:.6f}`",
        "",
        "## Method Rows",
        "",
        "| Method | Accuracy | Baseline | Delta | CI95 Low | Helps | Harms | Overrides | Bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["method_rows"]:
        lines.append(
            "| {method} | `{accuracy:.6f}` | `{baseline_accuracy:.6f}` | `{delta_vs_baseline:.6f}` | `{ci95_low_vs_baseline:.6f}` | `{helps_vs_baseline}` | `{harms_vs_baseline}` | `{override_count_vs_baseline}` | `{framed_record_bytes}` |".format(
                **row
            )
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


def build_diagnostic(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    fit_rows_per_slice: int = FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = SELECT_ROWS_PER_SLICE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, slice_metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    fit_rows = [row for row in rows if row["_split"] == "fit"]
    select_rows = [row for row in rows if row["_split"] == "select"]
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    selected_model, config_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        baseline_field="qwen_hybrid_prediction",
        l2_values=RIDGE_LAMBDAS,
        bootstrap_samples=bootstrap_samples,
    )
    eval_predictions, eval_scores = _predict(eval_rows, selected_model)
    answers = _answers(eval_rows)
    fixed = _field_array(eval_rows, "qwen_hybrid_prediction")
    target = _field_array(eval_rows, "phi_target_prediction")
    selected = _field_array(eval_rows, "selected_prediction")
    top1 = np.asarray([_source_packet(row, condition="matched")["source_top1"] for row in eval_rows], dtype=np.int64)
    pair_phi = np.asarray(
        [_packet_action(row, _source_packet(row, condition="matched")) for row in eval_rows],
        dtype=np.int64,
    )
    method_rows = [
        _method_row(
            name="frontier_selector_over_fixed_hybrid",
            rows=eval_rows,
            predictions=eval_predictions,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=4,
            details={"model": {key: value for key, value in selected_model.items() if key != "weights"}},
        ),
        _method_row(
            name="target_only",
            rows=eval_rows,
            predictions=target,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
        ),
        _method_row(
            name="source_top1_choice_control",
            rows=eval_rows,
            predictions=top1,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="source_pair_phi_choice",
            rows=eval_rows,
            predictions=pair_phi,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=2,
            framed_record_bytes=4,
        ),
        _method_row(
            name="candidate_only",
            rows=eval_rows,
            predictions=selected,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
    ]
    for condition in (
        "source_row_shuffle",
        "candidate_roll",
        "random_same_byte",
        "target_derived_packet",
        "zero_source",
    ):
        predictions, _ = _predict(eval_rows, selected_model, condition=condition)
        method_rows.append(
            _method_row(
                name=f"{condition}_frontier_control",
                rows=eval_rows,
                predictions=predictions,
                baseline=fixed,
                answers=answers,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=2 if condition not in {"target_derived_packet", "zero_source"} else 0,
                framed_record_bytes=4 if condition not in {"target_derived_packet", "zero_source"} else 0,
                details={"condition": condition},
            )
        )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    selected_row = next(row for row in method_rows if row["method"] == "frontier_selector_over_fixed_hybrid")
    destructive_rows = [row for row in method_rows if row["method"].endswith("_frontier_control")]
    best_destructive = max(destructive_rows, key=lambda row: row["accuracy"])
    top2_oracle = np.asarray(
        [
            answer if answer in {_source_packet(row, condition="matched")["source_top1"], _source_packet(row, condition="matched")["source_top2"]} else int(row["qwen_hybrid_prediction"])
            for row, answer in zip(eval_rows, answers, strict=True)
        ],
        dtype=np.int64,
    )
    top2_oracle_row = _method_row(
        name="fixed_or_source_top1_top2_oracle_diagnostic",
        rows=eval_rows,
        predictions=top2_oracle,
        baseline=fixed,
        answers=answers,
        bootstrap_samples=bootstrap_samples,
        raw_payload_bytes=0,
        framed_record_bytes=0,
        details={"oracle": True, "not_promotable": True},
    )
    method_rows.insert(0, top2_oracle_row)
    split_rows = [
        _frontier_counts(fit_rows, split="fit"),
        _frontier_counts(select_rows, split="select"),
        _frontier_counts(eval_rows, split="eval"),
    ]
    eval_counts = split_rows[-1]
    pass_gate = (
        selected_row["delta_vs_baseline"] > 0.0
        and selected_row["ci95_low_vs_baseline"] > 0.0
        and selected_row["accuracy"] > best_destructive["accuracy"]
        and selected_row["accuracy"] > next(row for row in method_rows if row["method"] == "source_top1_choice_control")[
            "accuracy"
        ]
        and selected_row["override_count_vs_baseline"] > 0
    )
    payload = {
        "gate": "source_private_hellaswag_complementarity_frontier_diagnostic",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if a train/select-selected frontier selector improves over fixed hybrid with "
            "positive paired CI, beats destructive packet controls, beats source-top1 choice, and performs "
            "nonzero held-out overrides."
        ),
        "headline": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
            "eval_rows": len(eval_rows),
            "target_only_accuracy": _accuracy(target, answers),
            "fixed_hybrid_accuracy": _accuracy(fixed, answers),
            "candidate_only_accuracy": _accuracy(selected, answers),
            "source_top1_accuracy": _accuracy(top1, answers),
            "source_pair_phi_accuracy": _accuracy(pair_phi, answers),
            "source_top1_or_top2_oracle_accuracy": eval_counts["source_top1_or_top2_oracle_accuracy"],
            "fixed_or_source_top1_top2_oracle_accuracy": top2_oracle_row["accuracy"],
            "target_wrong_source_top1_or_top2_correct": eval_counts[
                "target_wrong_source_top1_or_top2_correct"
            ],
            "fixed_wrong_source_top1_or_top2_correct": eval_counts[
                "fixed_wrong_source_top1_or_top2_correct"
            ],
            "selected_selector_accuracy": selected_row["accuracy"],
            "selected_selector_delta_vs_fixed_hybrid": selected_row["delta_vs_baseline"],
            "selected_selector_ci95_low_vs_fixed_hybrid": selected_row["ci95_low_vs_baseline"],
            "selected_selector_overrides": selected_row["override_count_vs_baseline"],
            "best_destructive_control_name": best_destructive["method"],
            "best_destructive_control_accuracy": best_destructive["accuracy"],
            "raw_payload_bytes": 2,
            "framed_record_bytes": 4,
            "native_systems_claim_allowed": False,
        },
        "packet_contract": {
            "receiver_visible_payload": (
                "source top1/top2 candidate IDs plus quantized source margin/entropy bins; "
                "Phi score simplex remains receiver-local side information"
            ),
            "raw_payload_bytes": 2,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "target_scores_are_receiver_side_information": True,
        },
        "source_score_metadata": source_score_metadata,
        "slice_metadata": slice_metadata,
        "split_rows": split_rows,
        "method_rows": method_rows,
        "config_rows": config_rows,
        "interpretation": (
            "The diagnostic tests whether source helpfulness is separable before another receiver is trained. "
            "A large oracle gap with a non-positive selected frontier selector means the benchmark has source "
            "headroom, but the current packet fields do not expose a stable source-causal decision boundary."
        ),
        "lay_explanation": (
            "We looked for questions where Phi is wrong but Qwen's top guesses include the right answer. "
            "Then we asked whether a tiny packet can predict those moments without seeing test answers. "
            "If this fails, another decoder on the same packet is unlikely to become a strong paper result."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_complementarity_frontier_diagnostic.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "split_rows.csv", split_rows)
    _write_jsonl(
        output_dir / "prediction_rows.jsonl",
        _prediction_rows(
            eval_rows,
            model=selected_model,
            predictions=eval_predictions,
            scores=eval_scores,
            baseline_field="qwen_hybrid_prediction",
        ),
    )
    _write_markdown(output_dir / "hellaswag_complementarity_frontier_diagnostic.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_complementarity_frontier_diagnostic.json",
                "hellaswag_complementarity_frontier_diagnostic.md",
                "method_rows.csv",
                "config_rows.csv",
                "split_rows.csv",
                "prediction_rows.jsonl",
            ],
            "slice_metadata": slice_metadata,
            "source_score_metadata": source_score_metadata,
            "pass_gate": bool(pass_gate),
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--fit-rows-per-slice", type=int, default=FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=SELECT_ROWS_PER_SLICE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    payload = build_diagnostic(
        output_dir=args.output_dir,
        source_score_cache=args.source_score_cache,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
