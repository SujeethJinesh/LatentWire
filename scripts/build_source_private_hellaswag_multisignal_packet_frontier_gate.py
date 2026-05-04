from __future__ import annotations

"""HellaSwag Qwen-to-Phi multi-signal packet frontier gate.

This no-new-inference gate tests whether already cached Qwen packet-policy
signals beyond source top1/top2 expose a source-causal repair frontier for Phi.
The source packet sends only discrete candidate IDs from source-private packet
decoders plus a quantized margin; it does not expose text, KV, hidden vectors,
or raw score/logit vectors.
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
    "results/source_private_hellaswag_multisignal_packet_frontier_gate_20260504_validation1024_2048"
)
DEFAULT_SOURCE_SCORE_CACHE = oracle.DEFAULT_SOURCE_SCORE_CACHE
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES
FIT_ROWS_PER_SLICE = denoise.FIT_ROWS_PER_SLICE
SELECT_ROWS_PER_SLICE = denoise.SELECT_ROWS_PER_SLICE
RIDGE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
PACKET_FIELDS = (
    "selected_prediction",
    "hidden_mean_prediction",
    "score_mean_prediction",
    "vote_prediction",
    "score_vote_prediction",
    "source_rank_only_bagged_prediction",
    "score_only_bagged_prediction",
)


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


def _write_csv(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def _softmax(scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _bin(value: float, thresholds: Sequence[float]) -> int:
    return int(np.digitize(float(value), np.asarray(thresholds, dtype=np.float64), right=False))


def _one_hot(value: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= int(value) < size:
        out[int(value)] = 1.0
    return out


def _majority(values: Sequence[int]) -> int:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=4)
    return int(np.argmax(counts))


def _margin_bin(row: dict[str, Any]) -> int:
    return _bin(float(row.get("selected_margin", 0.0)), (0.5, 1.0, 2.0))


def _target_packet(row: dict[str, Any]) -> dict[str, Any]:
    top1, top2 = _top2_from_scores(row["phi_target_scores"])
    predictions = [top1, top1, top2, top1, top2, top1, top2]
    scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    order = np.argsort(-scores)
    margin = float(scores[order[0]] - scores[order[1]])
    return {
        "predictions": predictions,
        "margin_bin": _bin(margin, (0.1, 0.3, 0.6)),
        "condition_source": "target_derived",
    }


def _packet(row: dict[str, Any], *, condition: str, substitute_row: dict[str, Any] | None = None) -> dict[str, Any]:
    if condition == "target_derived_packet":
        return _target_packet(row)
    if condition == "zero_source":
        return {"predictions": [0] * len(PACKET_FIELDS), "margin_bin": 0, "condition_source": "zero"}
    source_row = row if substitute_row is None else substitute_row
    predictions = [int(source_row[field]) for field in PACKET_FIELDS]
    if condition == "candidate_roll":
        predictions = [int((value + 1) % 4) for value in predictions]
    elif condition == "field_shuffle":
        predictions = predictions[1:] + predictions[:1]
    return {
        "predictions": predictions,
        "margin_bin": _margin_bin(source_row),
        "condition_source": condition,
    }


def _packets_for_condition(rows: Sequence[dict[str, Any]], *, condition: str, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if condition in {"source_row_shuffle", "random_same_byte"}:
        order = rng.permutation(len(rows)) if condition == "source_row_shuffle" else rng.integers(0, len(rows), len(rows))
        return [
            _packet(row, condition="matched", substitute_row=rows[int(order[index])])
            for index, row in enumerate(rows)
        ]
    return [_packet(row, condition=condition) for row in rows]


def _candidate_features(row: dict[str, Any], packet: dict[str, Any], candidate: int) -> np.ndarray:
    phi_scores = np.asarray(row["phi_target_scores"], dtype=np.float64)
    phi_probs = _softmax(phi_scores)
    phi_order = np.argsort(-phi_scores)
    phi_ranks = np.empty(4, dtype=np.int64)
    phi_ranks[phi_order] = np.arange(4)
    centered = phi_scores - float(np.mean(phi_scores))
    scale = float(np.std(centered))
    z_scores = centered / (scale if scale > 1e-8 else 1.0)
    predictions = [int(value) for value in packet["predictions"]]
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=4)
    majority = _majority(predictions)
    fixed = int(row["qwen_hybrid_prediction"])
    features: list[float] = [
        1.0,
        float(z_scores[candidate]),
        float(phi_probs[candidate]),
        float(phi_scores[candidate]),
        float(phi_ranks[candidate]),
        float(phi_ranks[candidate] == 0),
        float(phi_scores[phi_order[0]] - phi_scores[phi_order[1]]),
        float(-np.sum(phi_probs * np.log(phi_probs + 1e-12))),
        float(candidate == fixed),
        float(candidate == int(row["selected_prediction"])),
        float(candidate == int(row["hidden_mean_prediction"])),
        float(candidate == int(row["score_mean_prediction"])),
        float(candidate == majority),
        float(counts[candidate]) / float(len(predictions)),
        float(len(set(predictions))),
        float(max(counts)) / float(len(predictions)),
    ]
    for prediction in predictions:
        features.append(float(candidate == prediction))
    features.extend(_one_hot(int(packet["margin_bin"]), 4))
    features.extend(_one_hot(majority, 4))
    return np.asarray(features, dtype=np.float64)


def _feature_tensor(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str = "matched",
    seed: int = 20260504,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    packets = _packets_for_condition(rows, condition=condition, seed=seed)
    rows_x: list[np.ndarray] = []
    for row, packet in zip(rows, packets, strict=True):
        rows_x.append(np.vstack([_candidate_features(row, packet, candidate) for candidate in range(4)]))
    return np.stack(rows_x, axis=0), packets


def _fit_candidate_model(rows: Sequence[dict[str, Any]], *, l2: float) -> dict[str, Any]:
    x, _ = _feature_tensor(rows)
    candidate_ids = np.tile(np.arange(4, dtype=np.int64), len(rows))
    y = (candidate_ids == np.repeat(_answers(rows), 4)).astype(np.float64)
    flat_x = x.reshape(len(rows) * 4, x.shape[-1])
    penalty = float(l2) * np.eye(flat_x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = flat_x.T @ flat_x + penalty
    rhs = flat_x.T @ y
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    return {"l2": float(l2), "weights": weights.tolist()}


def _score_rows(
    rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    *,
    condition: str = "matched",
    seed: int = 20260504,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x, packets = _feature_tensor(rows, condition=condition, seed=seed)
    weights = np.asarray(model["weights"], dtype=np.float64)
    candidate_scores = x @ weights
    actions = np.argmax(candidate_scores, axis=1).astype(np.int64)
    fixed = _field_array(rows, "qwen_hybrid_prediction")
    utility_scores = candidate_scores[np.arange(len(rows)), actions] - candidate_scores[np.arange(len(rows)), fixed]
    return actions, utility_scores, packets


def _select_model(
    *,
    fit_rows: Sequence[dict[str, Any]],
    select_rows: Sequence[dict[str, Any]],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answers = _answers(select_rows)
    fixed = _field_array(select_rows, "qwen_hybrid_prediction")
    config_rows: list[dict[str, Any]] = []
    best_model: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for l2 in RIDGE_LAMBDAS:
        model = _fit_candidate_model(fit_rows, l2=float(l2))
        actions, scores, _ = _score_rows(select_rows, model)
        thresholds = [float("inf")]
        thresholds.extend(sorted(set(float(value) for value in np.quantile(scores, np.linspace(0.0, 1.0, 21)))))
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, actions, fixed).astype(np.int64)
            paired = _paired_ci(
                selected=predictions,
                baseline=fixed,
                answers=answers,
                seed=20260504 + int(float(l2) * 1000) + int(17 if np.isinf(threshold) else abs(threshold) * 1000),
                samples=max(200, min(int(bootstrap_samples), 1000)),
            )
            row = {
                "l2": float(l2),
                "threshold": None if np.isinf(threshold) else float(threshold),
                "threshold_is_noop": bool(np.isinf(threshold)),
                "select_accuracy": _accuracy(predictions, answers),
                "select_delta_vs_fixed": paired["delta"],
                "select_ci95_low_vs_fixed": paired["ci95_low"],
                "select_helps_vs_fixed": paired["helps"],
                "select_harms_vs_fixed": paired["harms"],
                "select_overrides": int(np.sum(predictions != fixed)),
            }
            config_rows.append(row)
            key = (
                float(row["select_ci95_low_vs_fixed"]),
                float(row["select_delta_vs_fixed"]),
                float(row["select_accuracy"]),
                -float(row["select_harms_vs_fixed"]),
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


def _predict(rows: Sequence[dict[str, Any]], model: dict[str, Any], *, condition: str = "matched") -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    fixed = _field_array(rows, "qwen_hybrid_prediction")
    actions, scores, packets = _score_rows(rows, model, condition=condition, seed=20260504 + sum(ord(ch) for ch in condition))
    threshold = float("inf") if model.get("threshold_is_noop") else float(model["threshold"])
    predictions = np.where(scores >= threshold, actions, fixed).astype(np.int64)
    return predictions, scores, packets


def _source_top2_oracle(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    answers = _answers(rows)
    fixed = _field_array(rows, "qwen_hybrid_prediction")
    out = fixed.copy()
    for index, row in enumerate(rows):
        top1, top2 = _top2_from_scores(row["qwen_source_scores"])
        if int(answers[index]) in {top1, top2}:
            out[index] = int(answers[index])
    return out


def _method_row(
    *,
    name: str,
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
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "raw_source_scores_or_logits_exposed": False,
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _prediction_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: dict[str, Any],
    predictions: np.ndarray,
    scores: np.ndarray,
    packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answers = _answers(rows)
    fixed = _field_array(rows, "qwen_hybrid_prediction")
    out: list[dict[str, Any]] = []
    for index, (row, packet) in enumerate(zip(rows, packets, strict=True)):
        out.append(
            {
                "row_id": str(row["row_id"]),
                "slice_start": int(row["_slice_start"]),
                "split": str(row["_split"]),
                "answer_index": int(answers[index]),
                "fixed_hybrid_prediction": int(fixed[index]),
                "method_prediction": int(predictions[index]),
                "utility_score": float(scores[index]),
                "method_correct": bool(predictions[index] == answers[index]),
                "fixed_hybrid_correct": bool(fixed[index] == answers[index]),
                "override_fixed_hybrid": bool(predictions[index] != fixed[index]),
                "packet_predictions": list(map(int, packet["predictions"])),
                "packet_majority": _majority(packet["predictions"]),
                "packet_margin_bin": int(packet["margin_bin"]),
                "selected_model_l2": float(model["l2"]),
                "selected_threshold": model["threshold"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Multi-Signal Packet Frontier Gate",
        "",
        f"- created UTC: `{payload['created_utc']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        "",
        "## Headline",
        "",
        f"- fixed-hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- multi-signal selector accuracy: `{h['multisignal_selector_accuracy']:.6f}`",
        f"- multi-signal delta vs fixed: `{h['multisignal_selector_delta_vs_fixed_hybrid']:.6f}`",
        f"- multi-signal CI95 low vs fixed: `{h['multisignal_selector_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- selected overrides: `{h['multisignal_selector_overrides']}`",
        f"- source top1/top2 oracle accuracy: `{h['source_top1_top2_oracle_accuracy']:.6f}`",
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
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
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
    model, config_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        bootstrap_samples=bootstrap_samples,
    )
    predictions, utility_scores, packets = _predict(eval_rows, model)
    answers = _answers(eval_rows)
    fixed = _field_array(eval_rows, "qwen_hybrid_prediction")
    target = _field_array(eval_rows, "phi_target_prediction")
    candidate = _field_array(eval_rows, "selected_prediction")
    source_top1 = np.asarray([_top2_from_scores(row["qwen_source_scores"])[0] for row in eval_rows], dtype=np.int64)
    source_top2 = np.asarray([_top2_from_scores(row["qwen_source_scores"])[1] for row in eval_rows], dtype=np.int64)
    method_rows = [
        _method_row(
            name="multisignal_packet_frontier",
            predictions=predictions,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=3,
            framed_record_bytes=5,
            details={"model": {key: value for key, value in model.items() if key != "weights"}},
        ),
        _method_row(
            name="target_only",
            predictions=target,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
        ),
        _method_row(
            name="candidate_only",
            predictions=candidate,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="source_top1_choice_control",
            predictions=source_top1,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="source_top2_choice_control",
            predictions=source_top2,
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
        "field_shuffle",
        "random_same_byte",
        "target_derived_packet",
        "zero_source",
    ):
        control_predictions, _, _ = _predict(eval_rows, model, condition=condition)
        method_rows.append(
            _method_row(
                name=f"{condition}_multisignal_control",
                predictions=control_predictions,
                baseline=fixed,
                answers=answers,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=3 if condition not in {"target_derived_packet", "zero_source"} else 0,
                framed_record_bytes=5 if condition not in {"target_derived_packet", "zero_source"} else 0,
            )
        )
    oracle_predictions = _source_top2_oracle(eval_rows)
    method_rows.insert(
        0,
        _method_row(
            name="fixed_or_source_top1_top2_oracle_diagnostic",
            predictions=oracle_predictions,
            baseline=fixed,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
    )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    selected_row = next(row for row in method_rows if row["method"] == "multisignal_packet_frontier")
    destructive_rows = [row for row in method_rows if row["method"].endswith("_multisignal_control")]
    best_destructive = max(destructive_rows, key=lambda row: row["accuracy"])
    best_source_choice = max(
        [row for row in method_rows if row["method"] in {"source_top1_choice_control", "source_top2_choice_control"}],
        key=lambda row: row["accuracy"],
    )
    pass_gate = (
        selected_row["delta_vs_baseline"] > 0.0
        and selected_row["ci95_low_vs_baseline"] > 0.0
        and selected_row["accuracy"] > best_destructive["accuracy"]
        and selected_row["accuracy"] > best_source_choice["accuracy"]
        and selected_row["override_count_vs_baseline"] > 0
    )
    payload = {
        "gate": "source_private_hellaswag_multisignal_packet_frontier_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if the selected multi-signal packet frontier improves over fixed hybrid with "
            "positive paired CI, beats destructive controls, beats source-choice controls, and performs "
            "nonzero held-out overrides."
        ),
        "headline": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
            "eval_rows": len(eval_rows),
            "fixed_hybrid_accuracy": _accuracy(fixed, answers),
            "target_only_accuracy": _accuracy(target, answers),
            "candidate_only_accuracy": _accuracy(candidate, answers),
            "source_top1_accuracy": _accuracy(source_top1, answers),
            "source_top2_accuracy": _accuracy(source_top2, answers),
            "source_top1_top2_oracle_accuracy": next(
                row["accuracy"] for row in method_rows if row["method"] == "fixed_or_source_top1_top2_oracle_diagnostic"
            ),
            "multisignal_selector_accuracy": selected_row["accuracy"],
            "multisignal_selector_delta_vs_fixed_hybrid": selected_row["delta_vs_baseline"],
            "multisignal_selector_ci95_low_vs_fixed_hybrid": selected_row["ci95_low_vs_baseline"],
            "multisignal_selector_overrides": selected_row["override_count_vs_baseline"],
            "best_destructive_control_name": best_destructive["method"],
            "best_destructive_control_accuracy": best_destructive["accuracy"],
            "best_source_choice_control_name": best_source_choice["method"],
            "best_source_choice_control_accuracy": best_source_choice["accuracy"],
            "raw_payload_bytes": 3,
            "framed_record_bytes": 5,
            "native_systems_claim_allowed": False,
        },
        "packet_contract": {
            "packet_fields": list(PACKET_FIELDS),
            "receiver_visible_payload": (
                "Seven source-private packet-policy candidate IDs plus one quantized selected-margin bin."
            ),
            "raw_payload_bytes": 3,
            "framed_record_bytes": 5,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
        },
        "source_score_metadata": source_score_metadata,
        "slice_metadata": slice_metadata,
        "method_rows": method_rows,
        "config_rows": config_rows,
        "interpretation": (
            "This gate tests the strongest no-new-inference source packet available after the top1/top2 frontier failed. "
            "A failure means cached hidden/score/vote source packet policies still do not expose a stable repair frontier "
            "beyond source-choice, routing, and destructive packet controls."
        ),
        "lay_explanation": (
            "Instead of sending only Qwen's top two guesses, this sends several tiny votes from Qwen's existing hidden and score packet decoders. "
            "The test asks whether those extra votes tell Phi when to change its answer safely."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_multisignal_packet_frontier_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_jsonl(
        output_dir / "prediction_rows.jsonl",
        _prediction_rows(
            eval_rows,
            model=model,
            predictions=predictions,
            scores=utility_scores,
            packets=packets,
        ),
    )
    _write_markdown(output_dir / "hellaswag_multisignal_packet_frontier_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_multisignal_packet_frontier_gate.json",
                "hellaswag_multisignal_packet_frontier_gate.md",
                "method_rows.csv",
                "config_rows.csv",
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
    payload = build_gate(
        output_dir=args.output_dir,
        source_score_cache=args.source_score_cache,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
