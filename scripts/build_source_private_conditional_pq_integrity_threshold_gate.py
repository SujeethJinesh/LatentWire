from __future__ import annotations

"""Conditional-PQ packet integrity threshold gate.

This gate reuses the conditional-PQ packet and corruption/no-op candidate
receiver, but separates decoding from packet integrity. A scalar integrity rule
is selected on held-out train rows, then evaluated on the n256 held-out-family
surface. If the rule rejects a packet, the receiver no-ops to the target prior.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_source_private_conditional_pq_corruption_noop_receiver_gate import (  # noqa: E402
    CONDITIONS,
    CONTROL_CONDITIONS,
    _fit_receiver,
    _payload_and_permutation_for_condition,
    _predict_condition,
)
from scripts.build_source_private_product_codebook_geometry_gate import _groups_for_variant  # noqa: E402
from scripts.run_source_private_candidate_embedding_receiver import _build_anchor_matrix  # noqa: E402
from scripts.run_source_private_conditional_pq_innovation_gate import (  # noqa: E402
    BASIS_VIEWS,
    CONDITIONING_MODES,
    _dimension_utilities,
    _fit_conditional_encoder,
    _fit_innovation_codebook,
    _oracle_accuracy,
    _representation_dim,
    _sha256_file,
    _unquantized_accuracy,
    _write_jsonl,
)
from scripts.run_source_private_hidden_repair_packet_smoke import Example, _prior_prediction, make_benchmark  # noqa: E402
from scripts.run_source_private_masked_innovation_receiver import _SEMANTIC_CANDIDATE_MATRIX_CACHE  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import _remap_candidate_slots  # noqa: E402
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


TRUST_SCORE_NAMES = (
    "score_margin",
    "chosen_score",
    "max_similarity",
    "negative_min_l2",
    "margin_plus_similarity",
    "margin_minus_min_l2_001",
)


def _answer_labels(rows: Sequence[Example]) -> list[str]:
    return [row.answer_label for row in rows]


def _accuracy(predictions: Sequence[str], answers: Sequence[str]) -> float:
    return float(np.mean([prediction == answer for prediction, answer in zip(predictions, answers, strict=True)]))


def _paired_ci(
    selected: Sequence[str],
    baseline: Sequence[str],
    answers: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    selected_ok = np.asarray(
        [prediction == answer for prediction, answer in zip(selected, answers, strict=True)],
        dtype=np.float64,
    )
    baseline_ok = np.asarray(
        [prediction == answer for prediction, answer in zip(baseline, answers, strict=True)],
        dtype=np.float64,
    )
    diff = selected_ok - baseline_ok
    rng = np.random.default_rng(seed)
    draws = []
    n = len(diff)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        draws.append(float(np.mean(diff[indices])))
    draws.sort()
    return {
        "delta": float(np.mean(diff)),
        "ci95_low": float(draws[int(0.025 * (samples - 1))]),
        "ci95_high": float(draws[int(0.975 * (samples - 1))]),
        "helps": int(np.sum((selected_ok == 1.0) & (baseline_ok == 0.0))),
        "harms": int(np.sum((selected_ok == 0.0) & (baseline_ok == 1.0))),
    }


def _trust_score(row: dict[str, Any], score_name: str) -> float:
    metadata = row["metadata"]
    margin = float(metadata["chosen_score"]) - float(metadata["prior_score"])
    max_similarity = float(metadata["max_similarity"])
    min_l2 = float(metadata["min_l2"])
    if score_name == "score_margin":
        return margin
    if score_name == "chosen_score":
        return float(metadata["chosen_score"])
    if score_name == "max_similarity":
        return max_similarity
    if score_name == "negative_min_l2":
        return -min_l2
    if score_name == "margin_plus_similarity":
        return margin + max_similarity
    if score_name == "margin_minus_min_l2_001":
        return margin - 0.01 * min_l2
    raise ValueError(f"unknown trust score {score_name!r}")


def _fit_models(
    fit_rows: list[Example],
    *,
    train_seed: int,
    eval_seed: int,
    feature_dim: int,
    anchor_count: int,
    basis_view: str,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
    budget_bytes: int,
    variant: str,
    remap_slot_seed: int | None,
    encoder_ridge: float,
    receiver_ridge: float,
    receiver_noop_weight: float,
    fit_intercept: bool,
    mask_repeats: int,
    codebook_iterations: int,
    seed: int,
) -> dict[str, Any]:
    anchor_matrix = (
        _build_anchor_matrix(fit_rows, feature_dim=feature_dim, anchor_count=anchor_count)
        if basis_view == "anchor_relative"
        else None
    )
    representation_dim = _representation_dim(feature_dim=feature_dim, anchor_matrix=anchor_matrix)
    encoder = _fit_conditional_encoder(
        fit_rows,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        ridge=encoder_ridge,
        fit_intercept=fit_intercept,
        mask_repeats=mask_repeats,
        seed=seed * 1009 + train_seed,
    )
    label_shuffle_encoder = _fit_conditional_encoder(
        fit_rows,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        ridge=encoder_ridge,
        fit_intercept=fit_intercept,
        mask_repeats=mask_repeats,
        seed=seed * 1009 + train_seed,
        label_shuffle_seed=train_seed * 5003 + eval_seed,
        constrained_label_shuffle=True,
    )
    utilities = _dimension_utilities(
        fit_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
    )
    groups = _groups_for_variant(
        variant=variant,
        feature_dim=representation_dim,
        budget_bytes=budget_bytes,
        utilities=utilities,
        seed=train_seed * 11003 + eval_seed * 97 + budget_bytes + (remap_slot_seed or 0),
    )
    codebook = _fit_innovation_codebook(
        fit_rows,
        encoder=encoder,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        groups=groups,
        variant=variant,
        utilities=utilities,
        seed=train_seed * 9001 + eval_seed * 17 + budget_bytes,
        iterations=codebook_iterations,
    )
    receiver = _fit_receiver(
        fit_rows,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        representation_dim=representation_dim,
        feature_dim=feature_dim,
        basis_view=basis_view,
        anchor_matrix=anchor_matrix,
        target_topk=target_topk,
        source_topk=source_topk,
        conditioning_mode=conditioning_mode,
        seed=seed,
        ridge=receiver_ridge,
        noop_weight=receiver_noop_weight,
    )
    return {
        "anchor_matrix": anchor_matrix,
        "representation_dim": representation_dim,
        "encoder": encoder,
        "label_shuffle_encoder": label_shuffle_encoder,
        "codebook": codebook,
        "receiver": receiver,
    }


def _raw_condition_rows(
    rows: list[Example],
    *,
    models: dict[str, Any],
    feature_dim: int,
    basis_view: str,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed * 4001 + len(rows))
    out: list[dict[str, Any]] = []
    for row_index, example in enumerate(rows):
        for condition in CONDITIONS:
            out.append(
                _predict_condition(
                    condition=condition,
                    example=example,
                    rows=rows,
                    index=row_index,
                    encoder=models["encoder"],
                    label_shuffle_encoder=models["label_shuffle_encoder"],
                    codebook=models["codebook"],
                    receiver=models["receiver"],
                    representation_dim=models["representation_dim"],
                    feature_dim=feature_dim,
                    basis_view=basis_view,
                    anchor_matrix=models["anchor_matrix"],
                    target_topk=target_topk,
                    source_topk=source_topk,
                    conditioning_mode=conditioning_mode,
                    seed=seed,
                    rng=rng,
                )
            )
    return out


def _apply_integrity(
    raw_rows: Sequence[dict[str, Any]],
    *,
    score_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        condition = str(row["condition"])
        score = _trust_score(row, score_name) if row["payload_bytes"] else float("-inf")
        accepted = bool(condition != "target_only" and score >= threshold)
        prediction = str(row["prediction"]) if accepted else str(row["target_prior_label"])
        out.append(
            {
                **row,
                "raw_prediction": row["prediction"],
                "prediction": prediction,
                "correct": prediction == row["answer_label"],
                "integrity_score_name": score_name,
                "integrity_score": score,
                "integrity_threshold": float(threshold),
                "integrity_accept": accepted,
            }
        )
    return out


def _by_condition(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["condition"]), []).append(row)
    return out


def _condition_predictions(rows: Sequence[dict[str, Any]], condition: str) -> list[str]:
    return [str(row["prediction"]) for row in rows if row["condition"] == condition]


def _condition_answers(rows: Sequence[dict[str, Any]], condition: str) -> list[str]:
    return [str(row["answer_label"]) for row in rows if row["condition"] == condition]


def _condition_accept_rate(rows: Sequence[dict[str, Any]], condition: str) -> float:
    subset = [row for row in rows if row["condition"] == condition]
    if not subset:
        return 0.0
    return float(np.mean([bool(row["integrity_accept"]) for row in subset]))


def _metric(condition_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in condition_rows]
    payloads = [int(row["payload_bytes"]) for row in condition_rows]
    tokens = [int(row["payload_tokens"]) for row in condition_rows]
    return {
        "correct": int(sum(1 for row in condition_rows if row["correct"])),
        "accuracy": float(np.mean([bool(row["correct"]) for row in condition_rows])),
        "accept_rate": float(np.mean([bool(row["integrity_accept"]) for row in condition_rows])),
        "mean_payload_bytes": statistics.fmean(payloads),
        "mean_payload_tokens": statistics.fmean(tokens),
        "p50_latency_ms": statistics.median(latencies),
    }


def _summary_for_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _by_condition(rows)
    answers = _condition_answers(rows, "source")
    source = _condition_predictions(rows, "source")
    target = _condition_predictions(rows, "target_only")
    metrics = {condition: _metric(grouped[condition]) for condition in CONDITIONS}
    best_control_condition = max(CONTROL_CONDITIONS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = _condition_predictions(rows, best_control_condition)
    paired_vs_best = _paired_ci(source, best_control, answers, seed=seed + 31, samples=bootstrap_samples)
    paired_vs_target = _paired_ci(source, target, answers, seed=seed + 17, samples=bootstrap_samples)
    source_correct = {row["example_id"] for row in grouped["source"] if row["correct"]}
    target_correct = {row["example_id"] for row in grouped["target_only"] if row["correct"]}
    return {
        "n": len(grouped["source"]),
        "conditions": CONDITIONS,
        "source_accuracy": metrics["source"]["accuracy"],
        "target_only_accuracy": metrics["target_only"]["accuracy"],
        "source_minus_target": metrics["source"]["accuracy"] - metrics["target_only"]["accuracy"],
        "best_control_condition": best_control_condition,
        "best_control_accuracy": metrics[best_control_condition]["accuracy"],
        "source_minus_best_control": metrics["source"]["accuracy"] - metrics[best_control_condition]["accuracy"],
        "source_accept_rate": metrics["source"]["accept_rate"],
        "best_control_accept_rate": metrics[best_control_condition]["accept_rate"],
        "max_corrupt_accept_rate": max(metrics[condition]["accept_rate"] for condition in CONTROL_CONDITIONS),
        "helps": len(source_correct - target_correct),
        "harms": len(target_correct - source_correct),
        "paired_bootstrap": {
            "source_vs_target": paired_vs_target,
            "source_vs_best_control": paired_vs_best,
        },
        "metrics": metrics,
        "pass_gate": (
            metrics["source"]["accuracy"] >= metrics["target_only"]["accuracy"] + 0.05
            and metrics["source"]["accuracy"] >= metrics[best_control_condition]["accuracy"] + 0.10
            and paired_vs_best["ci95_low"] > 0.0
            and metrics["source"]["accept_rate"] > 0.0
            and max(metrics[condition]["accept_rate"] for condition in CONTROL_CONDITIONS) < metrics["source"]["accept_rate"]
        ),
    }


def _threshold_candidates(rows: Sequence[dict[str, Any]], score_name: str) -> list[float]:
    values = sorted({_trust_score(row, score_name) for row in rows if row["payload_bytes"]})
    if not values:
        return [float("inf")]
    if len(values) > 96:
        values = sorted({values[int(round(q * (len(values) - 1)))] for q in np.linspace(0.0, 1.0, 80)})
    low = values[0] - max(1e-6, abs(values[0]) * 1e-6)
    high = values[-1] + max(1e-6, abs(values[-1]) * 1e-6)
    return [low, *values, high, float("inf")]


def _select_integrity_rule(
    select_raw_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for score_name in TRUST_SCORE_NAMES:
        for threshold in _threshold_candidates(select_raw_rows, score_name):
            integrity_rows = _apply_integrity(select_raw_rows, score_name=score_name, threshold=threshold)
            summary = _summary_for_rows(
                integrity_rows,
                bootstrap_samples=max(100, min(bootstrap_samples, 500)),
                seed=seed + len(threshold_rows),
            )
            row = {
                "score_name": score_name,
                "threshold": None if np.isinf(threshold) else float(threshold),
                "threshold_is_noop": bool(np.isinf(threshold)),
                "source_accuracy": summary["source_accuracy"],
                "target_only_accuracy": summary["target_only_accuracy"],
                "best_control_condition": summary["best_control_condition"],
                "best_control_accuracy": summary["best_control_accuracy"],
                "source_minus_best_control": summary["source_minus_best_control"],
                "ci95_low_vs_best_control": summary["paired_bootstrap"]["source_vs_best_control"]["ci95_low"],
                "source_accept_rate": summary["source_accept_rate"],
                "max_corrupt_accept_rate": summary["max_corrupt_accept_rate"],
                "helps": summary["helps"],
                "harms": summary["harms"],
            }
            threshold_rows.append(row)
            key = (
                float(row["ci95_low_vs_best_control"]),
                float(row["source_minus_best_control"]),
                float(row["source_accuracy"] - row["target_only_accuracy"]),
                float(row["source_accept_rate"] - row["max_corrupt_accept_rate"]),
                -float(row["harms"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = row
    if best is None:
        raise RuntimeError("failed to select integrity rule")
    return best, threshold_rows


def _write_csv(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Conditional PQ Integrity Threshold Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train fit/select/eval: `{h['fit_rows']}/{h['select_rows']}/{h['eval_rows']}`",
        f"- selected score: `{h['selected_score_name']}`",
        f"- selected threshold: `{h['selected_threshold']}`",
        f"- source accuracy: `{h['source_accuracy']:.6f}`",
        f"- target-only accuracy: `{h['target_only_accuracy']:.6f}`",
        f"- best control: `{h['best_control_condition']}` at `{h['best_control_accuracy']:.6f}`",
        f"- source minus best control: `{h['source_minus_best_control']:.6f}`",
        f"- CI95 low vs best control: `{h['ci95_low_vs_best_control']:.6f}`",
        f"- source/max corrupt accept rate: `{h['source_accept_rate']:.6f}/{h['max_corrupt_accept_rate']:.6f}`",
        f"- helps/harms: `{h['helps']}/{h['harms']}`",
        "",
        "## Condition Metrics",
        "",
        "| Condition | Accuracy | Accept Rate | Mean Bytes |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metric = payload["summary"]["metrics"][condition]
        lines.append(
            f"| `{condition}` | `{metric['accuracy']:.6f}` | `{metric['accept_rate']:.6f}` | `{metric['mean_payload_bytes']:.2f}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    integrity_select_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_start_index: int,
    eval_start_index: int,
    train_family_set: str,
    eval_family_set: str,
    diagnostic_table_mode: str,
    candidates: int,
    feature_dim: int,
    anchor_count: int,
    basis_view: str,
    source_topk: int,
    target_topk: int,
    conditioning_mode: str,
    budget_bytes: int,
    variant: str,
    remap_slot_seed: int | None,
    encoder_ridge: float,
    receiver_ridge: float,
    receiver_noop_weight: float,
    fit_intercept: bool,
    mask_repeats: int,
    codebook_iterations: int,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _SEMANTIC_CANDIDATE_MATRIX_CACHE.clear()
    fit_examples = train_examples - integrity_select_examples
    if fit_examples <= 0:
        raise ValueError("train_examples must exceed integrity_select_examples")
    train_rows = make_benchmark(
        examples=train_examples,
        candidates=candidates,
        seed=train_seed,
        family_set=train_family_set,
        start_index=train_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    eval_rows = make_benchmark(
        examples=eval_examples,
        candidates=candidates,
        seed=eval_seed,
        family_set=eval_family_set,
        start_index=eval_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_slot_seed)
    eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_slot_seed)
    fit_rows = train_rows[:fit_examples]
    select_rows = train_rows[fit_examples:]
    models = _fit_models(
        fit_rows,
        train_seed=train_seed,
        eval_seed=eval_seed,
        feature_dim=feature_dim,
        anchor_count=anchor_count,
        basis_view=basis_view,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        budget_bytes=budget_bytes,
        variant=variant,
        remap_slot_seed=remap_slot_seed,
        encoder_ridge=encoder_ridge,
        receiver_ridge=receiver_ridge,
        receiver_noop_weight=receiver_noop_weight,
        fit_intercept=fit_intercept,
        mask_repeats=mask_repeats,
        codebook_iterations=codebook_iterations,
        seed=seed,
    )
    select_raw = _raw_condition_rows(
        select_rows,
        models=models,
        feature_dim=feature_dim,
        basis_view=basis_view,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        seed=seed + 101,
    )
    selected_rule, threshold_rows = _select_integrity_rule(
        select_raw,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 202,
    )
    threshold = float("inf") if selected_rule["threshold_is_noop"] else float(selected_rule["threshold"])
    eval_raw = _raw_condition_rows(
        eval_rows,
        models=models,
        feature_dim=feature_dim,
        basis_view=basis_view,
        source_topk=source_topk,
        target_topk=target_topk,
        conditioning_mode=conditioning_mode,
        seed=seed + 303,
    )
    predictions = _apply_integrity(eval_raw, score_name=str(selected_rule["score_name"]), threshold=threshold)
    summary = _summary_for_rows(predictions, bootstrap_samples=bootstrap_samples, seed=seed + 404)
    unquantized = _unquantized_accuracy(
        eval_rows,
        encoder=models["encoder"],
        feature_dim=feature_dim,
        source_topk=source_topk,
        target_topk=target_topk,
        basis_view=basis_view,
        anchor_matrix=models["anchor_matrix"],
        conditioning_mode=conditioning_mode,
    )
    oracle = _oracle_accuracy(
        eval_rows,
        feature_dim=feature_dim,
        target_topk=target_topk,
        basis_view=basis_view,
        anchor_matrix=models["anchor_matrix"],
        conditioning_mode=conditioning_mode,
    )
    eval_ids = {row.example_id for row in eval_rows}
    train_ids = {row.example_id for row in train_rows}
    payload = {
        "gate": "source_private_conditional_pq_integrity_threshold_gate",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "fit_examples": fit_examples,
        "integrity_select_examples": integrity_select_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_start_index": train_start_index,
        "eval_start_index": eval_start_index,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "diagnostic_table_mode": diagnostic_table_mode,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "anchor_count": anchor_count,
        "basis_view": basis_view,
        "source_topk": source_topk,
        "target_topk": target_topk,
        "conditioning_mode": conditioning_mode,
        "budget_bytes": budget_bytes,
        "variant": variant,
        "remap_slot_seed": remap_slot_seed,
        "encoder_ridge": encoder_ridge,
        "receiver_ridge": receiver_ridge,
        "receiver_noop_weight": receiver_noop_weight,
        "fit_intercept": fit_intercept,
        "mask_repeats": mask_repeats,
        "codebook_iterations": codebook_iterations,
        "selected_integrity_rule": selected_rule,
        "receiver": {
            "feature_names": models["receiver"]["feature_names"],
            "weights": [float(value) for value in models["receiver"]["weights"].tolist()],
            "condition_counts": models["receiver"]["condition_counts"],
            "ridge": models["receiver"]["ridge"],
            "noop_weight": models["receiver"]["noop_weight"],
            "training_diagnostics": models["receiver"]["training_diagnostics"],
        },
        "encoder_sha256": hashlib.sha256(models["encoder"].tobytes()).hexdigest(),
        "label_shuffle_encoder_sha256": hashlib.sha256(models["label_shuffle_encoder"].tobytes()).hexdigest(),
        "codebook_sha256": hashlib.sha256(
            b"".join(centroid.tobytes() for centroid in models["codebook"].centroids)
        ).hexdigest(),
        "receiver_sha256": hashlib.sha256(models["receiver"]["weights"].tobytes()).hexdigest(),
        "systems_accounting": {
            "payload_bytes": budget_bytes,
            "framed_packet_bytes_estimate": budget_bytes + 3,
            "dense_kv_floor_bytes_reference": 21504,
            "native_systems_claim_allowed": False,
        },
        "unquantized_predicted_accuracy": unquantized["accuracy"],
        "target_innovation_oracle_accuracy": oracle["accuracy"],
        "train_eval_id_intersection_count": len(train_ids.intersection(eval_ids)),
        "summary": summary,
        "headline": {
            "fit_rows": fit_examples,
            "select_rows": integrity_select_examples,
            "eval_rows": eval_examples,
            "selected_score_name": selected_rule["score_name"],
            "selected_threshold": selected_rule["threshold"],
            "source_accuracy": summary["source_accuracy"],
            "target_only_accuracy": summary["target_only_accuracy"],
            "best_control_condition": summary["best_control_condition"],
            "best_control_accuracy": summary["best_control_accuracy"],
            "source_minus_best_control": summary["source_minus_best_control"],
            "ci95_low_vs_best_control": summary["paired_bootstrap"]["source_vs_best_control"]["ci95_low"],
            "source_accept_rate": summary["source_accept_rate"],
            "max_corrupt_accept_rate": summary["max_corrupt_accept_rate"],
            "helps": summary["helps"],
            "harms": summary["harms"],
            "unquantized_predicted_accuracy": unquantized["accuracy"],
            "target_innovation_oracle_accuracy": oracle["accuracy"],
            "framed_record_bytes": budget_bytes + 3,
        },
        "pass_gate": bool(summary["pass_gate"] and len(train_ids.intersection(eval_ids)) == 0),
        "interpretation": (
            "This gate tests whether an explicit scalar packet-integrity rule can preserve matched "
            "conditional-PQ packet gains while forcing corrupted packet controls to no-op. A failure "
            "rules out simple thresholded integrity on top of the existing public-zscore receiver."
        ),
        "lay_explanation": (
            "The receiver first scores what the tiny packet wants it to do, then a separate trust rule "
            "decides whether to accept that packet or ignore it and keep the target's original answer."
        ),
    }
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_csv(output_dir / "threshold_rows.csv", threshold_rows)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", "predictions.jsonl", "threshold_rows.csv", "manifest.json"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md", "predictions.jsonl", "threshold_rows.csv"]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_conditional_pq_integrity_threshold_gate_20260504"),
    )
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--integrity-select-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=29)
    parser.add_argument("--train-start-index", type=int, default=10000)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="core")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="holdout")
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="plausible_decoys")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--basis-view", choices=BASIS_VIEWS, default="semantic")
    parser.add_argument("--source-topk", type=int, default=64)
    parser.add_argument("--target-topk", type=int, default=32)
    parser.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default="public_zscore")
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument(
        "--variant",
        choices=["canonical", "utility_balanced", "protected_hadamard", "utility_protected_hadamard"],
        default="utility_protected_hadamard",
    )
    parser.add_argument("--remap-slot-seed", type=int, default=101)
    parser.add_argument("--encoder-ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-noop-weight", type=float, default=0.01)
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--mask-repeats", type=int, default=1)
    parser.add_argument("--codebook-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=output_dir,
        train_examples=args.train_examples,
        integrity_select_examples=args.integrity_select_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        diagnostic_table_mode=args.diagnostic_table_mode,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        anchor_count=args.anchor_count,
        basis_view=args.basis_view,
        source_topk=args.source_topk,
        target_topk=args.target_topk,
        conditioning_mode=args.conditioning_mode,
        budget_bytes=args.budget_bytes,
        variant=args.variant,
        remap_slot_seed=args.remap_slot_seed,
        encoder_ridge=args.encoder_ridge,
        receiver_ridge=args.receiver_ridge,
        receiver_noop_weight=args.receiver_noop_weight,
        fit_intercept=args.fit_intercept,
        mask_repeats=args.mask_repeats,
        codebook_iterations=args.codebook_iterations,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    if args.require_pass and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
