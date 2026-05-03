from __future__ import annotations

"""Cached option-position audit for the HellaSwag fixed hybrid packet policy."""

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_source_private_hellaswag_fixed_hybrid_full_validation_gate import (
    BOOTSTRAP_SAMPLES,
    DEFAULT_AUDIT,
    DEFAULT_TAIL_PREDICTIONS,
    _array,
    _hybrid_array,
    _load_rows,
    _paired_ci,
)


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_fixed_hybrid_option_position_audit_20260503_validation0_10042"
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


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _bootstrap_balanced_delta(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    deltas = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    slots = sorted(set(int(value) for value in answers.tolist()))
    slot_indices = [np.flatnonzero(answers == slot) for slot in slots]
    if len(slot_indices) != 4 or any(len(indices) == 0 for indices in slot_indices):
        raise ValueError("balanced option-position audit requires all four answer positions")
    slot_means = [float(np.mean(deltas[indices])) for indices in slot_indices]
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(samples)):
        sampled_means = []
        for indices in slot_indices:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            sampled_means.append(float(np.mean(deltas[sampled])))
        draws.append(float(np.mean(sampled_means)))
    return {
        "balanced_delta": float(np.mean(slot_means)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "min_slot_delta": float(min(slot_means)),
        "positive_slot_count": int(sum(delta > 0.0 for delta in slot_means)),
    }


def _option_position_rows(
    *,
    answers: np.ndarray,
    candidate_only: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for answer_index in range(4):
        mask = answers == answer_index
        paired = _paired_ci(
            selected=fixed_hybrid[mask],
            baseline=candidate_only[mask],
            answers=answers[mask],
            seed=20260503 + answer_index,
            samples=bootstrap_samples,
        )
        rows.append(
            {
                "answer_index": answer_index,
                "eval_rows": int(np.sum(mask)),
                "answer_share": float(np.mean(mask)),
                "candidate_only_accuracy": _accuracy(candidate_only[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]),
                "hybrid_delta_vs_candidate_only": paired["delta"],
                "hybrid_ci95_low_vs_candidate_only": paired["ci95_low"],
                "hybrid_ci95_high_vs_candidate_only": paired["ci95_high"],
                "hybrid_helps_vs_candidate_only": paired["helps"],
                "hybrid_harms_vs_candidate_only": paired["harms"],
            }
        )
    return rows


def _prediction_distribution_rows(
    *,
    answers: np.ndarray,
    candidate_only: np.ndarray,
    fixed_hybrid: np.ndarray,
) -> list[dict[str, Any]]:
    answer_counts = np.bincount(answers, minlength=4).astype(np.float64)
    answer_share = answer_counts / np.sum(answer_counts)
    rows = []
    for name, predictions in (
        ("answer_gold", answers),
        ("candidate_only", candidate_only),
        ("fixed_hybrid_vote_on_score_agreement", fixed_hybrid),
    ):
        counts = np.bincount(predictions, minlength=4).astype(np.float64)
        share = counts / np.sum(counts)
        rows.append(
            {
                "method": name,
                "slot0_count": int(counts[0]),
                "slot1_count": int(counts[1]),
                "slot2_count": int(counts[2]),
                "slot3_count": int(counts[3]),
                "slot0_share": float(share[0]),
                "slot1_share": float(share[1]),
                "slot2_share": float(share[2]),
                "slot3_share": float(share[3]),
                "max_abs_shift_from_answer_distribution": float(np.max(np.abs(share - answer_share))),
                "total_variation_from_answer_distribution": float(0.5 * np.sum(np.abs(share - answer_share))),
                "max_slot_share": float(np.max(share)),
            }
        )
    return rows


def _roll_control_rows(
    *,
    answers: np.ndarray,
    candidate_only: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for base_name, base_predictions in (
        ("candidate_only", candidate_only),
        ("fixed_hybrid_vote_on_score_agreement", fixed_hybrid),
    ):
        for shift in (1, 2, 3):
            rolled = (base_predictions + shift) % 4
            paired = _paired_ci(
                selected=rolled,
                baseline=fixed_hybrid,
                answers=answers,
                seed=30360503 + 17 * shift + sum(ord(ch) for ch in base_name),
                samples=bootstrap_samples,
            )
            rows.append(
                {
                    "method": f"{base_name}_cyclic_roll_{shift}",
                    "eval_rows": len(answers),
                    "accuracy": _accuracy(rolled, answers),
                    "delta_vs_fixed_hybrid": paired["delta"],
                    "ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                    "ci95_high_vs_fixed_hybrid": paired["ci95_high"],
                    "helps_vs_fixed_hybrid": paired["helps"],
                    "harms_vs_fixed_hybrid": paired["harms"],
                }
            )
    return rows


def _global_packet_permutation_rows(
    *,
    answers: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    identity = tuple(range(4))
    for permutation in itertools.permutations(range(4)):
        if permutation == identity:
            continue
        mapping = np.asarray(permutation, dtype=np.int64)
        permuted = mapping[fixed_hybrid]
        paired = _paired_ci(
            selected=permuted,
            baseline=fixed_hybrid,
            answers=answers,
            seed=50360503 + sum((index + 1) * value for index, value in enumerate(permutation)),
            samples=bootstrap_samples,
        )
        rows.append(
            {
                "method": "fixed_hybrid_global_label_permutation",
                "permutation": " ".join(str(value) for value in permutation),
                "eval_rows": len(answers),
                "accuracy": _accuracy(permuted, answers),
                "delta_vs_fixed_hybrid": paired["delta"],
                "ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                "ci95_high_vs_fixed_hybrid": paired["ci95_high"],
                "helps_vs_fixed_hybrid": paired["helps"],
                "harms_vs_fixed_hybrid": paired["harms"],
            }
        )
    return sorted(rows, key=lambda row: row["accuracy"], reverse=True)


def _rowwise_random_control_rows(
    *,
    answers: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
    seeds: tuple[int, ...] = tuple(range(10)),
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(60360503 + seed)
        deranged_predictions = []
        random_permutation_predictions = []
        for prediction in fixed_hybrid.tolist():
            deranged_choices = [value for value in range(4) if value != int(prediction)]
            deranged_predictions.append(int(rng.choice(deranged_choices)))
            permutation = rng.permutation(4)
            random_permutation_predictions.append(int(permutation[int(prediction)]))
        for name, predictions in (
            ("rowwise_random_derangement", np.asarray(deranged_predictions, dtype=np.int64)),
            ("rowwise_random_permutation", np.asarray(random_permutation_predictions, dtype=np.int64)),
        ):
            paired = _paired_ci(
                selected=predictions,
                baseline=fixed_hybrid,
                answers=answers,
                seed=61360503 + seed + sum(ord(ch) for ch in name),
                samples=bootstrap_samples,
            )
            rows.append(
                {
                    "method": name,
                    "seed": seed,
                    "eval_rows": len(answers),
                    "accuracy": _accuracy(predictions, answers),
                    "delta_vs_fixed_hybrid": paired["delta"],
                    "ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                    "ci95_high_vs_fixed_hybrid": paired["ci95_high"],
                    "helps_vs_fixed_hybrid": paired["helps"],
                    "harms_vs_fixed_hybrid": paired["harms"],
                }
            )
    return sorted(rows, key=lambda row: (row["method"], -row["accuracy"], row["seed"]))


def _equivariance_rows(
    *,
    rows: list[dict[str, Any]],
    answers: np.ndarray,
    fixed_hybrid: np.ndarray,
) -> list[dict[str, Any]]:
    out = []
    base_accuracy = _accuracy(fixed_hybrid, answers)
    for permutation in itertools.permutations(range(4)):
        mapping = np.asarray(permutation, dtype=np.int64)
        permuted_rows = []
        for row in rows:
            copied = dict(row)
            for field in (
                "selected_prediction",
                "vote_prediction",
                "hidden_mean_prediction",
                "score_mean_prediction",
                "score_vote_prediction",
            ):
                if field in copied:
                    copied[field] = int(mapping[int(copied[field])])
            copied["answer_index"] = int(mapping[int(copied["answer_index"])])
            permuted_rows.append(copied)
        permuted_answers = _array(permuted_rows, "answer_index")
        permuted_hybrid = _hybrid_array(permuted_rows)
        permuted_accuracy = _accuracy(permuted_hybrid, permuted_answers)
        out.append(
            {
                "permutation": " ".join(str(value) for value in permutation),
                "base_accuracy": base_accuracy,
                "permuted_accuracy": permuted_accuracy,
                "absolute_accuracy_difference": float(abs(permuted_accuracy - base_accuracy)),
            }
        )
    return out


def _slice_position_rows(
    *,
    rows: list[dict[str, Any]],
    answers: np.ndarray,
    candidate_only: np.ndarray,
    fixed_hybrid: np.ndarray,
) -> list[dict[str, Any]]:
    starts = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    deltas = (fixed_hybrid == answers).astype(np.float64) - (candidate_only == answers).astype(np.float64)
    out = []
    for start in sorted(set(starts.tolist())):
        for answer_index in range(4):
            mask = (starts == start) & (answers == answer_index)
            out.append(
                {
                    "slice_start": int(start),
                    "answer_index": int(answer_index),
                    "eval_rows": int(np.sum(mask)),
                    "hybrid_delta_vs_candidate_only": float(np.mean(deltas[mask])) if np.any(mask) else 0.0,
                    "candidate_only_accuracy": _accuracy(candidate_only[mask], answers[mask])
                    if np.any(mask)
                    else 0.0,
                    "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask])
                    if np.any(mask)
                    else 0.0,
                }
            )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Fixed Hybrid Option-Position Audit",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- candidate-only accuracy: `{h['candidate_only_accuracy']:.6f}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- overall delta vs candidate-only: `{h['overall_delta_vs_candidate_only']:.6f}`",
        f"- overall CI95 low: `{h['overall_ci95_low_vs_candidate_only']:.6f}`",
        f"- answer-balanced delta: `{h['answer_balanced_delta_vs_candidate_only']:.6f}`",
        f"- answer-balanced CI95 low: `{h['answer_balanced_ci95_low_vs_candidate_only']:.6f}`",
        f"- positive answer-position count: `{h['positive_answer_position_count']}` / `4`",
        f"- max fixed-hybrid prediction shift from answer distribution: `{h['fixed_hybrid_max_abs_prediction_shift']:.6f}`",
        f"- best non-identity global packet permutation accuracy: `{h['best_nonidentity_global_permutation_accuracy']:.6f}`",
        f"- best rowwise derangement accuracy: `{h['best_rowwise_derangement_accuracy']:.6f}`",
        f"- max equivariance sanity diff: `{h['max_equivariance_accuracy_difference']:.12f}`",
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


def build_audit(
    *,
    audit_path: pathlib.Path | str = DEFAULT_AUDIT,
    tail_predictions: pathlib.Path | str | None = DEFAULT_TAIL_PREDICTIONS,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    tail_start: int = 9216,
    tail_end_exclusive: int = 10042,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, slice_sources = _load_rows(
        audit_path=audit_path,
        tail_predictions=tail_predictions,
        tail_start=tail_start,
        tail_end_exclusive=tail_end_exclusive,
    )
    answers = _array(rows, "answer_index")
    candidate_only = _array(rows, "selected_prediction")
    fixed_hybrid = _hybrid_array(rows)
    overall = _paired_ci(
        selected=fixed_hybrid,
        baseline=candidate_only,
        answers=answers,
        seed=40360503,
        samples=bootstrap_samples,
    )
    balanced = _bootstrap_balanced_delta(
        selected=fixed_hybrid,
        baseline=candidate_only,
        answers=answers,
        seed=40370503,
        samples=bootstrap_samples,
    )
    option_position_rows = _option_position_rows(
        answers=answers,
        candidate_only=candidate_only,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    distribution_rows = _prediction_distribution_rows(
        answers=answers,
        candidate_only=candidate_only,
        fixed_hybrid=fixed_hybrid,
    )
    fixed_distribution = next(
        row for row in distribution_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement"
    )
    roll_control_rows = _roll_control_rows(
        answers=answers,
        candidate_only=candidate_only,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    global_permutation_rows = _global_packet_permutation_rows(
        answers=answers,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    rowwise_random_rows = _rowwise_random_control_rows(
        answers=answers,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    equivariance_rows = _equivariance_rows(
        rows=rows,
        answers=answers,
        fixed_hybrid=fixed_hybrid,
    )
    slice_position_rows = _slice_position_rows(
        rows=rows,
        answers=answers,
        candidate_only=candidate_only,
        fixed_hybrid=fixed_hybrid,
    )
    fixed_roll_rows = [
        row for row in roll_control_rows if row["method"].startswith("fixed_hybrid_vote_on_score_agreement")
    ]
    best_global_permutation = max(global_permutation_rows, key=lambda row: row["accuracy"])
    derangement_rows = [row for row in rowwise_random_rows if row["method"] == "rowwise_random_derangement"]
    random_permutation_rows = [
        row for row in rowwise_random_rows if row["method"] == "rowwise_random_permutation"
    ]
    best_derangement = max(derangement_rows, key=lambda row: row["accuracy"])
    best_random_permutation = max(random_permutation_rows, key=lambda row: row["accuracy"])
    max_equivariance_diff = max(row["absolute_accuracy_difference"] for row in equivariance_rows)
    pass_gate = (
        overall["delta"] > 0.0
        and overall["ci95_low"] > 0.0
        and balanced["balanced_delta"] > 0.0
        and balanced["ci95_low"] > 0.0
        and balanced["positive_slot_count"] == 4
        and all(row["hybrid_delta_vs_candidate_only"] > 0.0 for row in option_position_rows)
        and max(row["ci95_high_vs_fixed_hybrid"] for row in fixed_roll_rows) < 0.0
        and float(fixed_distribution["max_abs_shift_from_answer_distribution"]) <= 0.02
        and best_global_permutation["accuracy"] <= 0.49
        and best_global_permutation["ci95_high_vs_fixed_hybrid"] < -0.02
        and best_derangement["accuracy"] <= 0.30
        and best_derangement["ci95_high_vs_fixed_hybrid"] < -0.15
        and best_random_permutation["accuracy"] <= 0.30
        and max_equivariance_diff <= 1e-12
    )
    payload = {
        "gate": "source_private_hellaswag_fixed_hybrid_option_position_audit",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "Pass only if fixed hybrid beats candidate-only overall and answer-position-balanced with "
            "positive paired/bootstrap CI, improves in mean for all four gold answer positions, cyclic "
            "rolls of the fixed hybrid packet are significantly worse, and the fixed-hybrid prediction "
            "distribution is not shifted from the gold answer-position distribution by more than 0.02 in "
            "any slot, non-identity global packet-label permutations and rowwise random derangements "
            "collapse, and same-permutation equivariance sanity checks pass. This is a cached "
            "option-position / packet-ID audit, not a substitute for true candidate-text permutation reruns."
        ),
        "headline": {
            "eval_rows": len(rows),
            "candidate_only_accuracy": _accuracy(candidate_only, answers),
            "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
            "overall_delta_vs_candidate_only": overall["delta"],
            "overall_ci95_low_vs_candidate_only": overall["ci95_low"],
            "overall_ci95_high_vs_candidate_only": overall["ci95_high"],
            "overall_helps_vs_candidate_only": overall["helps"],
            "overall_harms_vs_candidate_only": overall["harms"],
            "answer_balanced_delta_vs_candidate_only": balanced["balanced_delta"],
            "answer_balanced_ci95_low_vs_candidate_only": balanced["ci95_low"],
            "answer_balanced_ci95_high_vs_candidate_only": balanced["ci95_high"],
            "min_answer_position_delta": balanced["min_slot_delta"],
            "positive_answer_position_count": balanced["positive_slot_count"],
            "fixed_hybrid_max_abs_prediction_shift": fixed_distribution[
                "max_abs_shift_from_answer_distribution"
            ],
            "fixed_hybrid_total_variation_from_answer_distribution": fixed_distribution[
                "total_variation_from_answer_distribution"
            ],
            "worst_fixed_hybrid_roll_accuracy": max(row["accuracy"] for row in fixed_roll_rows),
            "best_fixed_hybrid_roll_ci95_high_vs_fixed_hybrid": max(
                row["ci95_high_vs_fixed_hybrid"] for row in fixed_roll_rows
            ),
            "best_nonidentity_global_permutation": best_global_permutation["permutation"],
            "best_nonidentity_global_permutation_accuracy": best_global_permutation["accuracy"],
            "best_nonidentity_global_permutation_ci95_high_vs_fixed_hybrid": best_global_permutation[
                "ci95_high_vs_fixed_hybrid"
            ],
            "best_rowwise_derangement_seed": best_derangement["seed"],
            "best_rowwise_derangement_accuracy": best_derangement["accuracy"],
            "best_rowwise_derangement_ci95_high_vs_fixed_hybrid": best_derangement[
                "ci95_high_vs_fixed_hybrid"
            ],
            "best_rowwise_random_permutation_seed": best_random_permutation["seed"],
            "best_rowwise_random_permutation_accuracy": best_random_permutation["accuracy"],
            "best_rowwise_random_permutation_ci95_high_vs_fixed_hybrid": best_random_permutation[
                "ci95_high_vs_fixed_hybrid"
            ],
            "max_equivariance_accuracy_difference": max_equivariance_diff,
        },
        "packet_contract": {
            "receiver_visible_payload": "one final source candidate id emitted by fixed hybrid policy",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "source_artifacts": {
            "candidate_only_audit": _display_path(audit_path),
            "candidate_only_audit_sha256": _sha256_file(audit_path),
            "tail_predictions": _display_path(tail_predictions) if tail_predictions is not None else None,
            "tail_predictions_sha256": _sha256_file(tail_predictions) if tail_predictions is not None else None,
        },
        "slice_sources": slice_sources,
        "option_position_rows": option_position_rows,
        "prediction_distribution_rows": distribution_rows,
        "roll_control_rows": roll_control_rows,
        "global_packet_permutation_rows": global_permutation_rows,
        "rowwise_random_control_rows": rowwise_random_rows,
        "equivariance_rows": equivariance_rows,
        "slice_position_rows": slice_position_rows,
        "interpretation": (
            "The fixed hybrid packet improvement is not concentrated in one answer slot: it is positive "
            "for all four gold answer positions and remains positive under answer-position-balanced "
            "resampling. Direct cyclic rolls, non-identity global label remaps, and rowwise derangements "
            "of the emitted packet collapse, which weakens a simple slot-prior explanation. Same-permutation "
            "equivariance checks catch audit implementation errors. This cached audit cannot prove invariance "
            "to true candidate-text permutations because the source model was not rerun under reordered answer "
            "options."
        ),
        "lay_explanation": (
            "We checked whether the tiny hybrid hint only helps when the correct answer is, for example, "
            "choice A or choice B. It helps a little for every answer position, and if we rotate the hint "
            "to the wrong option number the accuracy collapses. That makes the result less likely to be "
            "just an answer-position trick, though a future stronger test should rerun the model with "
            "the choices physically shuffled."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_fixed_hybrid_option_position_audit.json", payload)
    _write_csv(output_dir / "option_position_rows.csv", option_position_rows)
    _write_csv(output_dir / "prediction_distribution_rows.csv", distribution_rows)
    _write_csv(output_dir / "roll_control_rows.csv", roll_control_rows)
    _write_csv(output_dir / "global_packet_permutation_rows.csv", global_permutation_rows)
    _write_csv(output_dir / "rowwise_random_control_rows.csv", rowwise_random_rows)
    _write_csv(output_dir / "equivariance_rows.csv", equivariance_rows)
    _write_csv(output_dir / "slice_position_rows.csv", slice_position_rows)
    _write_markdown(output_dir / "hellaswag_fixed_hybrid_option_position_audit.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_fixed_hybrid_option_position_audit.json",
                "hellaswag_fixed_hybrid_option_position_audit.md",
                "option_position_rows.csv",
                "prediction_distribution_rows.csv",
                "roll_control_rows.csv",
                "global_packet_permutation_rows.csv",
                "rowwise_random_control_rows.csv",
                "equivariance_rows.csv",
                "slice_position_rows.csv",
            ],
            "source_artifacts": payload["source_artifacts"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=pathlib.Path, default=DEFAULT_AUDIT)
    parser.add_argument("--tail-predictions", type=pathlib.Path, default=DEFAULT_TAIL_PREDICTIONS)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tail-start", type=int, default=9216)
    parser.add_argument("--tail-end-exclusive", type=int, default=10042)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_audit(
        audit_path=args.audit,
        tail_predictions=args.tail_predictions,
        output_dir=args.output_dir,
        tail_start=args.tail_start,
        tail_end_exclusive=args.tail_end_exclusive,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(json.dumps({
        "pass_gate": payload["pass_gate"],
        "eval_rows": h["eval_rows"],
        "overall_delta_vs_candidate_only": h["overall_delta_vs_candidate_only"],
        "overall_ci95_low_vs_candidate_only": h["overall_ci95_low_vs_candidate_only"],
        "answer_balanced_delta_vs_candidate_only": h["answer_balanced_delta_vs_candidate_only"],
        "answer_balanced_ci95_low_vs_candidate_only": h["answer_balanced_ci95_low_vs_candidate_only"],
        "positive_answer_position_count": h["positive_answer_position_count"],
        "worst_fixed_hybrid_roll_accuracy": h["worst_fixed_hybrid_roll_accuracy"],
        "best_nonidentity_global_permutation_accuracy": h["best_nonidentity_global_permutation_accuracy"],
        "best_rowwise_derangement_accuracy": h["best_rowwise_derangement_accuracy"],
        "max_equivariance_accuracy_difference": h["max_equivariance_accuracy_difference"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
