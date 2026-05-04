from __future__ import annotations

"""Protected Qwen-to-Phi top-2/rival codebook gate for HellaSwag.

This offline gate tests the largest cheap headroom found in the current
telemetry: Qwen's top-2 score pair often contains the answer even when the
fixed Qwen hybrid packet is wrong. The receiver may use Phi-local scores and a
tiny source-side top-2/rival code, but it may not see source text, source KV,
raw hidden vectors, or raw source score vectors.
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
import sys
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_top2_rival_codebook_gate_20260504_validation1024_2048"
)
DEFAULT_SOURCE_SCORE_CACHE = oracle.DEFAULT_SOURCE_SCORE_CACHE
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES

CONDITIONS = (
    "fixed_hybrid",
    "top2_rival_codebook",
    "source_row_shuffle_codebook",
    "candidate_roll_source_codebook",
    "target_derived_codebook",
    "random_same_byte_codebook",
    "candidate_derangement",
    "qwen_candidate_only",
    "phi_target_only",
    "source_top1_label_control",
    "source_top2_label_control",
    "source_top1_or_top2_oracle",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


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
    return denoise._paired_ci(
        selected=np.asarray(selected, dtype=np.int64),
        baseline=np.asarray(baseline, dtype=np.int64),
        answers=np.asarray(answers, dtype=np.int64),
        seed=seed,
        samples=samples,
    )


def _field_array(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _answers(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return _field_array(rows, "answer_index")


def _top2_matrix(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1)
    return order[:, 0].astype(np.int64), order[:, 1].astype(np.int64)


def _digitize(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=np.float64), np.asarray(list(bins), dtype=np.float64)).astype(np.int64)


def _quantile_bins(values: np.ndarray, quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8)) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return []
    return sorted(set(float(value) for value in np.quantile(flat, list(quantiles))))


def _load_rows_with_source_scores(
    *,
    slices: tuple[dict[str, Any], ...],
    fit_rows_per_slice: int,
    select_rows_per_slice: int,
    source_score_cache: pathlib.Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_metadata = oracle._load_source_scores(rows, source_score_cache)
    return rows, metadata, source_metadata


def _fit_bins(rows: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    q_scores = np.asarray([row["qwen_source_scores"] for row in rows], dtype=np.float64)
    p_scores = np.asarray([row["phi_target_scores"] for row in rows], dtype=np.float64)
    q_top1, q_top2 = _top2_matrix(q_scores)
    p_top1, p_top2 = _top2_matrix(p_scores)
    row_ids = np.arange(len(rows))
    q_margin = q_scores[row_ids, q_top1] - q_scores[row_ids, q_top2]
    p_margin = p_scores[row_ids, p_top1] - p_scores[row_ids, p_top2]
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    q_gap = q_scores - q_scores[row_ids, hybrid][:, None]
    p_gap = p_scores - p_scores[row_ids, hybrid][:, None]
    return {
        "q_margin": _quantile_bins(q_margin),
        "p_margin": _quantile_bins(p_margin),
        "q_gap": _quantile_bins(q_gap),
        "p_gap": _quantile_bins(p_gap),
    }


def _action_table(
    rows: Sequence[dict[str, Any]],
    *,
    bins: dict[str, list[float]],
    source_scores_override: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, ...]]]]:
    q_scores = (
        np.asarray(source_scores_override, dtype=np.float64)
        if source_scores_override is not None
        else np.asarray([row["qwen_source_scores"] for row in rows], dtype=np.float64)
    )
    p_scores = np.asarray([row["phi_target_scores"] for row in rows], dtype=np.float64)
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    p_top1, p_top2 = _top2_matrix(p_scores)
    q_top1, q_top2 = _top2_matrix(q_scores)
    row_ids = np.arange(len(rows))
    q_margin_bin = _digitize(q_scores[row_ids, q_top1] - q_scores[row_ids, q_top2], bins["q_margin"])
    p_margin_bin = _digitize(p_scores[row_ids, p_top1] - p_scores[row_ids, p_top2], bins["p_margin"])
    actions = np.stack([hybrid, q_top1, q_top2, p_top1], axis=1).astype(np.int64)
    fields: list[list[tuple[int, ...]]] = []
    for row_index in range(len(rows)):
        row_fields: list[tuple[int, ...]] = []
        for role, action in enumerate(actions[row_index]):
            q_gap_bin = int(_digitize(np.asarray([q_scores[row_index, action] - q_scores[row_index, hybrid[row_index]]]), bins["q_gap"])[0])
            p_gap_bin = int(_digitize(np.asarray([p_scores[row_index, action] - p_scores[row_index, hybrid[row_index]]]), bins["p_gap"])[0])
            row_fields.append(
                (
                    int(role),
                    int(action == p_top1[row_index]),
                    int(action == q_top1[row_index]),
                    int(action == q_top2[row_index]),
                    int(hybrid[row_index] == q_top1[row_index]),
                    int(q_margin_bin[row_index]),
                    int(p_margin_bin[row_index]),
                    q_gap_bin,
                    p_gap_bin,
                )
            )
        fields.append(row_fields)
    return actions, fields


def _fit_bucket_stats(
    *,
    rows: Sequence[dict[str, Any]],
    bins: dict[str, list[float]],
    min_support: int,
    min_mean_delta: float,
    max_harm_rate: float,
) -> dict[tuple[int, ...], dict[str, Any]]:
    answers = _answers(rows)
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    actions, fields = _action_table(rows, bins=bins)
    stats: dict[tuple[int, ...], dict[str, Any]] = {}
    for row_index in range(len(rows)):
        for action_index, action in enumerate(actions[row_index]):
            if int(action) == int(hybrid[row_index]):
                continue
            key = fields[row_index][action_index]
            delta = int(action == answers[row_index]) - int(hybrid[row_index] == answers[row_index])
            entry = stats.setdefault(key, {"support": 0, "delta_sum": 0.0, "helps": 0, "harms": 0})
            entry["support"] += 1
            entry["delta_sum"] += float(delta)
            entry["helps"] += int(delta > 0)
            entry["harms"] += int(delta < 0)
    eligible: dict[tuple[int, ...], dict[str, Any]] = {}
    for key, entry in stats.items():
        support = int(entry["support"])
        if support < int(min_support):
            continue
        helps = int(entry["helps"])
        harms = int(entry["harms"])
        mean_delta = float(entry["delta_sum"]) / float(support)
        harm_rate = harms / float(max(1, helps + harms))
        if mean_delta >= float(min_mean_delta) and harm_rate <= float(max_harm_rate) and helps > harms:
            eligible[key] = {
                **entry,
                "mean_delta": float(mean_delta),
                "harm_rate": float(harm_rate),
                "score": float(mean_delta + 0.01 * (helps - harms) + 0.0001 * support),
            }
    return eligible


def _predict_with_buckets(
    rows: Sequence[dict[str, Any]],
    *,
    bins: dict[str, list[float]],
    buckets: dict[tuple[int, ...], dict[str, Any]],
    source_scores_override: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    hybrid = _field_array(rows, "qwen_hybrid_prediction")
    predictions = hybrid.copy()
    actions, fields = _action_table(rows, bins=bins, source_scores_override=source_scores_override)
    for row_index in range(len(rows)):
        best: tuple[float, int] | None = None
        for action_index, action in enumerate(actions[row_index]):
            entry = buckets.get(fields[row_index][action_index])
            if entry is None:
                continue
            score = float(entry["score"])
            if best is None or score > best[0]:
                best = (score, int(action))
        if best is not None:
            predictions[row_index] = best[1]
    return predictions.astype(np.int64)


def _evaluate(
    *,
    name: str,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> dict[str, Any]:
    answers = _answers(rows)
    paired = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=91060504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "condition": name,
        "accuracy": _accuracy(predictions, answers),
        "delta_vs_fixed_hybrid": float(paired["delta"]),
        "ci95_low_vs_fixed_hybrid": float(paired["ci95_low"]),
        "ci95_high_vs_fixed_hybrid": float(paired["ci95_high"]),
        "helps_vs_fixed_hybrid": int(paired["helps"]),
        "harms_vs_fixed_hybrid": int(paired["harms"]),
        "override_count": int(np.sum(predictions != fixed_hybrid)),
    }


def _select_model(
    *,
    fit_rows: Sequence[dict[str, Any]],
    select_rows: Sequence[dict[str, Any]],
    bins: dict[str, list[float]],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixed_select = _field_array(select_rows, "qwen_hybrid_prediction")
    grid_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, int, int] | None = None
    best_model: dict[str, Any] | None = None
    for min_support in (2, 3, 4, 5, 8, 12, 16, 24):
        for min_mean_delta in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
            for max_harm_rate in (0.0, 0.10, 0.25, 0.50, 1.0):
                buckets = _fit_bucket_stats(
                    rows=fit_rows,
                    bins=bins,
                    min_support=min_support,
                    min_mean_delta=min_mean_delta,
                    max_harm_rate=max_harm_rate,
                )
                if not buckets:
                    continue
                predictions = _predict_with_buckets(select_rows, bins=bins, buckets=buckets)
                metrics = _evaluate(
                    name=f"select_{min_support}_{min_mean_delta}_{max_harm_rate}",
                    rows=select_rows,
                    predictions=predictions,
                    fixed_hybrid=fixed_select,
                    bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
                )
                row = {
                    "min_support": int(min_support),
                    "min_mean_delta": float(min_mean_delta),
                    "max_harm_rate": float(max_harm_rate),
                    "eligible_bucket_count": int(len(buckets)),
                    **metrics,
                }
                grid_rows.append(row)
                key = (
                    float(metrics["accuracy"]),
                    float(metrics["delta_vs_fixed_hybrid"]),
                    float(metrics["ci95_low_vs_fixed_hybrid"]),
                    int(metrics["helps_vs_fixed_hybrid"]) - int(metrics["harms_vs_fixed_hybrid"]),
                    -int(metrics["harms_vs_fixed_hybrid"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_model = {
                        "min_support": int(min_support),
                        "min_mean_delta": float(min_mean_delta),
                        "max_harm_rate": float(max_harm_rate),
                        "eligible_buckets": {"|".join(map(str, key)): value for key, value in buckets.items()},
                    }
    if best_model is None:
        best_model = {
            "min_support": 0,
            "min_mean_delta": 0.0,
            "max_harm_rate": 0.0,
            "eligible_buckets": {},
            "selection": "no eligible buckets",
        }
    return best_model, sorted(grid_rows, key=lambda row: row["accuracy"], reverse=True)


def _decode_buckets(model: dict[str, Any]) -> dict[tuple[int, ...], dict[str, Any]]:
    return {
        tuple(int(part) for part in key.split("|")): value
        for key, value in model.get("eligible_buckets", {}).items()
    }


def _source_top_predictions(rows: Sequence[dict[str, Any]], rank: int) -> np.ndarray:
    q_scores = np.asarray([row["qwen_source_scores"] for row in rows], dtype=np.float64)
    order = np.argsort(-q_scores, axis=1)
    return order[:, int(rank)].astype(np.int64)


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    fit_rows_per_slice: int = denoise.FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = denoise.SELECT_ROWS_PER_SLICE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, slice_metadata, source_metadata = _load_rows_with_source_scores(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
        source_score_cache=source_score_cache,
    )
    fit_rows = [row for row in rows if row["_split"] == "fit"]
    select_rows = [row for row in rows if row["_split"] == "select"]
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    bins = _fit_bins(fit_rows)
    model, grid_rows = _select_model(
        fit_rows=fit_rows,
        select_rows=select_rows,
        bins=bins,
        bootstrap_samples=bootstrap_samples,
    )
    buckets = _decode_buckets(model)
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    phi_target = _field_array(eval_rows, "phi_target_prediction")
    source_top1 = _source_top_predictions(eval_rows, 0)
    source_top2 = _source_top_predictions(eval_rows, 1)
    answers = _answers(eval_rows)
    source_pair_oracle = np.where(source_top1 == answers, source_top1, np.where(source_top2 == answers, source_top2, source_top1))
    rng = random.Random(60504)
    eval_source_scores = [row["qwen_source_scores"] for row in eval_rows]
    random_source_scores = [eval_source_scores[rng.randrange(len(eval_source_scores))] for _ in eval_rows]
    controls = {
        "fixed_hybrid": fixed_hybrid,
        "top2_rival_codebook": _predict_with_buckets(eval_rows, bins=bins, buckets=buckets),
        "source_row_shuffle_codebook": _predict_with_buckets(
            eval_rows,
            bins=bins,
            buckets=buckets,
            source_scores_override=[eval_source_scores[(index + 1) % len(eval_source_scores)] for index in range(len(eval_source_scores))],
        ),
        "candidate_roll_source_codebook": _predict_with_buckets(
            eval_rows,
            bins=bins,
            buckets=buckets,
            source_scores_override=[np.roll(np.asarray(scores, dtype=np.float64), 1).tolist() for scores in eval_source_scores],
        ),
        "target_derived_codebook": _predict_with_buckets(
            eval_rows,
            bins=bins,
            buckets=buckets,
            source_scores_override=[row["phi_target_scores"] for row in eval_rows],
        ),
        "random_same_byte_codebook": _predict_with_buckets(
            eval_rows,
            bins=bins,
            buckets=buckets,
            source_scores_override=random_source_scores,
        ),
        "candidate_derangement": np.roll(_predict_with_buckets(eval_rows, bins=bins, buckets=buckets), 1),
        "qwen_candidate_only": candidate_only,
        "phi_target_only": phi_target,
        "source_top1_label_control": source_top1,
        "source_top2_label_control": source_top2,
        "source_top1_or_top2_oracle": source_pair_oracle,
    }
    method_rows = [
        _evaluate(
            name=condition,
            rows=eval_rows,
            predictions=predictions,
            fixed_hybrid=fixed_hybrid,
            bootstrap_samples=bootstrap_samples,
        )
        for condition, predictions in controls.items()
    ]
    by_condition = {row["condition"]: row for row in method_rows}
    destructive = (
        "source_row_shuffle_codebook",
        "candidate_roll_source_codebook",
        "target_derived_codebook",
        "random_same_byte_codebook",
        "candidate_derangement",
        "qwen_candidate_only",
        "phi_target_only",
        "source_top1_label_control",
        "source_top2_label_control",
    )
    best_destructive = max(destructive, key=lambda name: float(by_condition[name]["accuracy"]))
    paired_vs_source_top1 = _paired_ci(
        selected=controls["top2_rival_codebook"],
        baseline=source_top1,
        answers=answers,
        seed=71160504,
        samples=bootstrap_samples,
    )
    pass_gate = bool(
        by_condition["top2_rival_codebook"]["delta_vs_fixed_hybrid"] > 0.0
        and by_condition["top2_rival_codebook"]["ci95_low_vs_fixed_hybrid"] >= 0.0
        and float(paired_vs_source_top1["delta"]) > 0.0
        and float(paired_vs_source_top1["ci95_low"]) >= 0.0
        and by_condition["top2_rival_codebook"]["accuracy"] >= by_condition[best_destructive]["accuracy"]
    )
    prediction_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(eval_rows):
        for condition in CONDITIONS:
            pred = int(controls[condition][row_index])
            prediction_rows.append(
                {
                    "row_id": row["row_id"],
                    "content_id": row.get("content_id", ""),
                    "condition": condition,
                    "answer_index": int(row["answer_index"]),
                    "prediction_index": pred,
                    "correct": bool(pred == int(row["answer_index"])),
                    "fixed_hybrid_prediction": int(fixed_hybrid[row_index]),
                    "override_fixed_hybrid": bool(pred != int(fixed_hybrid[row_index])),
                }
            )
    headline = {
        "eval_rows": int(len(eval_rows)),
        "fit_rows": int(len(fit_rows)),
        "select_rows": int(len(select_rows)),
        "top2_rival_codebook_accuracy": float(by_condition["top2_rival_codebook"]["accuracy"]),
        "top2_rival_codebook_delta_vs_fixed_hybrid": float(by_condition["top2_rival_codebook"]["delta_vs_fixed_hybrid"]),
        "top2_rival_codebook_ci95_low_vs_fixed_hybrid": float(by_condition["top2_rival_codebook"]["ci95_low_vs_fixed_hybrid"]),
        "top2_rival_codebook_helps_vs_fixed_hybrid": int(by_condition["top2_rival_codebook"]["helps_vs_fixed_hybrid"]),
        "top2_rival_codebook_harms_vs_fixed_hybrid": int(by_condition["top2_rival_codebook"]["harms_vs_fixed_hybrid"]),
        "top2_rival_codebook_override_count": int(by_condition["top2_rival_codebook"]["override_count"]),
        "fixed_hybrid_accuracy": float(by_condition["fixed_hybrid"]["accuracy"]),
        "qwen_candidate_only_accuracy": float(by_condition["qwen_candidate_only"]["accuracy"]),
        "phi_target_accuracy": float(by_condition["phi_target_only"]["accuracy"]),
        "source_top1_label_accuracy": float(by_condition["source_top1_label_control"]["accuracy"]),
        "source_top2_label_accuracy": float(by_condition["source_top2_label_control"]["accuracy"]),
        "source_top1_or_top2_oracle_accuracy": float(by_condition["source_top1_or_top2_oracle"]["accuracy"]),
        "paired_vs_source_top1_label_control": paired_vs_source_top1,
        "best_destructive_control_name": best_destructive,
        "best_destructive_control_accuracy": float(by_condition[best_destructive]["accuracy"]),
        "selected_bucket_count": int(len(buckets)),
        "raw_payload_bytes": 1,
        "framed_record_bytes": 4,
        "native_systems_claim_allowed": False,
    }
    payload = {
        "date": run_date,
        "gate": "source_private_hellaswag_qwen_to_phi_top2_rival_codebook",
        "pass_gate": pass_gate,
        "headline": headline,
        "method_rows": method_rows,
        "grid_rows": grid_rows,
        "selected_model": model,
        "quantization_bins": bins,
        "slice_metadata": slice_metadata,
        "source_score_metadata": source_metadata,
        "packet_contract": {
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "source_top2_rival_code_transmitted": True,
        },
        "interpretation": (
            "The protected top-2/rival codebook passes the larger cached HellaSwag gate."
            if pass_gate
            else "The protected top-2/rival codebook does not pass the larger cached HellaSwag gate; "
            "the top-2 oracle remains headroom rather than a learned receiver."
        ),
        "lay_explanation": (
            "We tested whether Qwen can send a tiny code saying which two candidates look plausible, "
            "and whether Phi can safely use that code to switch away from the old fixed answer. The "
            "oracle says this could help, but the learned protected switch still has to beat the fixed "
            "packet and fake-code controls."
        ),
    }
    json_path = output_dir / "hellaswag_qwen_to_phi_top2_rival_codebook_gate.json"
    md_path = output_dir / "hellaswag_qwen_to_phi_top2_rival_codebook_gate.md"
    _write_json(json_path, payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "grid_rows.csv", grid_rows)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    md_lines = [
        "# Qwen-to-Phi Top-2/Rival Codebook Gate",
        "",
        f"- pass gate: `{pass_gate}`",
        f"- method accuracy: `{headline['top2_rival_codebook_accuracy']:.6f}`",
        f"- fixed hybrid accuracy: `{headline['fixed_hybrid_accuracy']:.6f}`",
        f"- source top-1 accuracy: `{headline['source_top1_label_accuracy']:.6f}`",
        f"- source top-1/top-2 oracle accuracy: `{headline['source_top1_or_top2_oracle_accuracy']:.6f}`",
        f"- best destructive: `{headline['best_destructive_control_name']}` "
        f"({headline['best_destructive_control_accuracy']:.6f})",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
    ]
    _resolve(md_path).write_text("\n".join(md_lines), encoding="utf-8")
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "files": {
                "hellaswag_qwen_to_phi_top2_rival_codebook_gate.json": denoise._sha256_file(json_path),
                "hellaswag_qwen_to_phi_top2_rival_codebook_gate.md": denoise._sha256_file(md_path),
                "method_rows.csv": denoise._sha256_file(output_dir / "method_rows.csv"),
                "grid_rows.csv": denoise._sha256_file(output_dir / "grid_rows.csv"),
                "predictions.jsonl": denoise._sha256_file(output_dir / "predictions.jsonl"),
            },
            "headline": headline,
            "pass_gate": pass_gate,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--fit-rows-per-slice", type=int, default=denoise.FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=denoise.SELECT_ROWS_PER_SLICE)
    parser.add_argument("--run-date", default=dt.date.today().isoformat())
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        source_score_cache=args.source_score_cache,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
