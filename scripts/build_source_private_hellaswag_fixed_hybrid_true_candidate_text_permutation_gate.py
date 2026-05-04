from __future__ import annotations

"""Evaluate full HellaSwag fixed-hybrid predictions under true candidate permutations.

The heavy model rerun is delegated to
``build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py``. This
script prepares a permuted HellaSwag eval slice and, after the hidden pipeline
has produced prediction rows, compares display-coordinate predictions against
the original canonical prediction rows.
"""

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

from scripts import build_source_private_hellaswag_bridge_contract as bridge_contract  # noqa: E402
from scripts import build_source_private_hellaswag_fixed_hybrid_full_validation_gate as fixed_full  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402


DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_ORIGINAL_PREDICTIONS = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_bagged_gate_20260503_"
    "rank_score_channel_controls_qwen05_train512_validation1024/predictions.jsonl"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate_20260503_validation0_128"
)
FIXED8_NONIDENTITY: tuple[tuple[int, int, int, int], ...] = (
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (1, 0, 2, 3),
    (0, 2, 1, 3),
    (3, 2, 1, 0),
    (2, 0, 3, 1),
    (1, 3, 0, 2),
)
CONTROL_FIELDS = (
    "selected_prediction",
    "hidden_mean_prediction",
    "score_mean_prediction",
    "vote_prediction",
    "source_label_prediction",
    "source_rank_only_bagged_prediction",
    "score_only_bagged_prediction",
    "score_vote_prediction",
    "trained_label_prediction",
    "wrong_example_hidden_prediction",
    "zero_hidden_prediction",
    "candidate_roll_hidden_prediction",
    "score_channel_roll_hidden_prediction",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


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


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _permutations(mode: str) -> list[tuple[int, int, int, int]]:
    if mode == "fixed8_nonidentity":
        return list(FIXED8_NONIDENTITY)
    if mode == "all24_nonidentity":
        return [tuple(perm) for perm in itertools.permutations(range(4), 4) if tuple(perm) != (0, 1, 2, 3)]
    raise ValueError(f"unknown permutation mode {mode!r}")


def _inverse(display_to_canonical: tuple[int, ...] | list[int]) -> list[int]:
    inverse = [0 for _ in display_to_canonical]
    for display_index, canonical_index in enumerate(display_to_canonical):
        inverse[int(canonical_index)] = int(display_index)
    return inverse


def _load_eval_slice(eval_full_path: pathlib.Path | str, *, start: int, rows: int) -> list[dict[str, Any]]:
    loaded = _read_jsonl(eval_full_path)
    if start < 0 or rows <= 0:
        raise ValueError("start must be non-negative and rows must be positive")
    selected = loaded[start : start + rows]
    if len(selected) != rows:
        raise ValueError(f"requested {rows} rows from {start}, found {len(selected)}")
    return selected


def prepare_permuted_eval(
    *,
    eval_full_path: pathlib.Path | str = DEFAULT_EVAL,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    eval_slice_start: int = 0,
    eval_rows: int = 128,
    permutation_mode: str = "fixed8_nonidentity",
    seed: int = 20260503,
    run_date: str | None = None,
) -> dict[str, Any]:
    del seed
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    source_rows = _load_eval_slice(eval_full_path, start=eval_slice_start, rows=eval_rows)
    permutations = _permutations(permutation_mode)
    permuted_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for local_index, source in enumerate(source_rows):
        display_to_canonical = permutations[local_index % len(permutations)]
        canonical_to_display = _inverse(display_to_canonical)
        old_answer = int(source["answer_index"])
        new_choices = [source["choices"][canonical_index] for canonical_index in display_to_canonical]
        display_answer = int(canonical_to_display[old_answer])
        original_id = str(source["id"])
        new_id = f"{original_id}_permtext_{eval_slice_start + local_index}_{local_index % len(permutations)}"
        row = dict(source)
        row["id"] = new_id
        row["original_id"] = original_id
        row["original_content_id"] = str(source["content_id"])
        row["original_answer_index"] = old_answer
        row["permutation_id"] = int(local_index % len(permutations))
        row["permutation_display_to_canonical"] = list(display_to_canonical)
        row["permutation_canonical_to_display"] = canonical_to_display
        row["choices"] = new_choices
        row["answer_index"] = display_answer
        row["answer_label"] = row["choice_labels"][display_answer]
        row["content_id"] = bridge_contract._content_id(str(row["question"]), new_choices)
        permuted_rows.append(row)
        mapping_rows.append(
            {
                "permuted_row_id": new_id,
                "original_row_id": original_id,
                "original_content_id": str(source["content_id"]),
                "permuted_content_id": str(row["content_id"]),
                "permutation_id": int(local_index % len(permutations)),
                "permutation_display_to_canonical": " ".join(str(value) for value in display_to_canonical),
                "permutation_canonical_to_display": " ".join(str(value) for value in canonical_to_display),
                "canonical_answer_index": old_answer,
                "display_answer_index": display_answer,
                "candidate_text_sha256_canonical": _sha256_json(source["choices"]),
                "candidate_text_sha256_display_order": _sha256_json(new_choices),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / f"hellaswag_validation_permuted_rows_{eval_slice_start}_{eval_slice_start + eval_rows}.jsonl"
    mapping_path = output_dir / "permutation_mapping_rows.csv"
    _write_jsonl(eval_path, permuted_rows)
    _write_csv(mapping_path, mapping_rows)
    payload = {
        "gate": "source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_prepare",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "eval_full_path": _display_path(eval_full_path),
        "eval_slice_start": int(eval_slice_start),
        "eval_slice_end_exclusive": int(eval_slice_start + eval_rows),
        "eval_rows": int(eval_rows),
        "permutation_mode": permutation_mode,
        "permuted_eval_path": _display_path(eval_path),
        "permuted_eval_sha256": _sha256_file(eval_path),
        "mapping_rows_path": _display_path(mapping_path),
        "all_nonidentity_permutations": all(
            row["permutation_display_to_canonical"] != "0 1 2 3" for row in mapping_rows
        ),
    }
    _write_json(output_dir / "permutation_prepare_manifest.json", payload)
    return payload


def _hybrid_prediction(row: dict[str, Any]) -> int:
    return fixed_full._hybrid_prediction(row)


def _paired_ci(selected: np.ndarray, baseline: np.ndarray, *, seed: int, samples: int) -> dict[str, float | int]:
    deltas = selected.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(deltas), size=len(deltas))
        draws.append(float(np.mean(deltas[indices])))
    return {
        "delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "helps": int(np.sum(deltas > 0.0)),
        "harms": int(np.sum(deltas < 0.0)),
    }


def _accuracy(values: list[bool] | np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if len(values) else 0.0


def _prediction_shift(predictions: list[int], answers: list[int]) -> dict[str, float]:
    answer_counts = np.bincount(np.asarray(answers, dtype=np.int64), minlength=4).astype(np.float64)
    pred_counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=4).astype(np.float64)
    answer_dist = answer_counts / max(1.0, float(np.sum(answer_counts)))
    pred_dist = pred_counts / max(1.0, float(np.sum(pred_counts)))
    shifts = np.abs(pred_dist - answer_dist)
    return {
        "max_abs_prediction_slot_shift_from_answer_distribution": float(np.max(shifts)),
        "total_variation_from_answer_distribution": float(0.5 * np.sum(shifts)),
    }


def _component_row(
    *,
    component: str,
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    original_correct = np.asarray(
        [float(row[f"original_{component}_correct"]) for row in rows],
        dtype=np.float64,
    )
    remapped_correct = np.asarray(
        [float(row[f"remapped_{component}_correct"]) for row in rows],
        dtype=np.float64,
    )
    paired = _paired_ci(remapped_correct, original_correct, seed=70360503 + len(component), samples=bootstrap_samples)
    return {
        "component": component,
        "eval_rows": len(rows),
        "original_accuracy": _accuracy(original_correct),
        "remapped_accuracy": _accuracy(remapped_correct),
        "canonical_consistency_rate": _accuracy(
            [bool(row[f"{component}_canonical_prediction_matches_original"]) for row in rows]
        ),
        "delta_remapped_vs_original": paired["delta"],
        "ci95_low_remapped_vs_original": paired["ci95_low"],
        "ci95_high_remapped_vs_original": paired["ci95_high"],
        "helps_remapped_vs_original": paired["helps"],
        "harms_remapped_vs_original": paired["harms"],
    }


def evaluate_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    original_predictions: pathlib.Path | str = DEFAULT_ORIGINAL_PREDICTIONS,
    permuted_predictions: pathlib.Path | str,
    permuted_eval_path: pathlib.Path | str,
    permuted_run_json: pathlib.Path | str | None = None,
    eval_rows: int | None = None,
    bootstrap_samples: int = 1000,
    min_consistency_for_smoke_pass: float = 0.90,
    max_accuracy_drop_for_smoke_pass: float = 0.05,
    min_eval_rows_for_promotion: int = 512,
    min_consistency_for_promotion: float = 0.95,
    max_accuracy_drop_for_promotion: float = 0.01,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    original_by_id = {str(row["row_id"]): row for row in _read_jsonl(original_predictions)}
    permuted_rows = _read_jsonl(permuted_eval_path)
    permuted_predictions_by_id = {str(row["row_id"]): row for row in _read_jsonl(permuted_predictions)}
    if eval_rows is not None:
        permuted_rows = permuted_rows[:eval_rows]
    comparison_rows: list[dict[str, Any]] = []
    for permuted_eval in permuted_rows:
        permuted_id = str(permuted_eval["id"])
        original_id = str(permuted_eval["original_id"])
        if original_id not in original_by_id:
            raise KeyError(f"missing original prediction row {original_id}")
        if permuted_id not in permuted_predictions_by_id:
            raise KeyError(f"missing permuted prediction row {permuted_id}")
        original = original_by_id[original_id]
        permuted = permuted_predictions_by_id[permuted_id]
        display_to_canonical = [int(value) for value in permuted_eval["permutation_display_to_canonical"]]
        wrong_display_to_canonical = display_to_canonical[1:] + display_to_canonical[:1]
        canonical_answer = int(permuted_eval["original_answer_index"])
        row: dict[str, Any] = {
            "original_row_id": original_id,
            "permuted_row_id": permuted_id,
            "canonical_answer_index": canonical_answer,
            "display_answer_index": int(permuted_eval["answer_index"]),
            "permutation_id": int(permuted_eval["permutation_id"]),
            "permutation_display_to_canonical": " ".join(str(value) for value in display_to_canonical),
            "original_fixed_hybrid_prediction": _hybrid_prediction(original),
            "permuted_fixed_hybrid_display_prediction": _hybrid_prediction(permuted),
        }
        row["remapped_fixed_hybrid_prediction"] = int(display_to_canonical[row["permuted_fixed_hybrid_display_prediction"]])
        row["unremapped_fixed_hybrid_prediction"] = int(row["permuted_fixed_hybrid_display_prediction"])
        row["wrong_remap_fixed_hybrid_prediction"] = int(wrong_display_to_canonical[row["permuted_fixed_hybrid_display_prediction"]])
        row["original_fixed_hybrid_correct"] = bool(row["original_fixed_hybrid_prediction"] == canonical_answer)
        row["remapped_fixed_hybrid_correct"] = bool(row["remapped_fixed_hybrid_prediction"] == canonical_answer)
        row["unremapped_fixed_hybrid_correct"] = bool(row["unremapped_fixed_hybrid_prediction"] == canonical_answer)
        row["wrong_remap_fixed_hybrid_correct"] = bool(row["wrong_remap_fixed_hybrid_prediction"] == canonical_answer)
        row["fixed_hybrid_canonical_prediction_matches_original"] = bool(
            row["remapped_fixed_hybrid_prediction"] == row["original_fixed_hybrid_prediction"]
        )
        for field in CONTROL_FIELDS:
            original_prediction = int(original[field])
            display_prediction = int(permuted[field])
            canonical_prediction = int(display_to_canonical[display_prediction])
            row[f"original_{field}"] = original_prediction
            row[f"remapped_{field}"] = canonical_prediction
            row[f"original_{field}_correct"] = bool(original_prediction == canonical_answer)
            row[f"remapped_{field}_correct"] = bool(canonical_prediction == canonical_answer)
            row[f"{field}_canonical_prediction_matches_original"] = bool(canonical_prediction == original_prediction)
        comparison_rows.append(row)

    if not comparison_rows:
        raise ValueError("no comparison rows produced")
    original_correct = np.asarray(
        [float(row["original_fixed_hybrid_correct"]) for row in comparison_rows],
        dtype=np.float64,
    )
    remapped_correct = np.asarray(
        [float(row["remapped_fixed_hybrid_correct"]) for row in comparison_rows],
        dtype=np.float64,
    )
    unremapped_correct = np.asarray(
        [float(row["unremapped_fixed_hybrid_correct"]) for row in comparison_rows],
        dtype=np.float64,
    )
    wrong_correct = np.asarray(
        [float(row["wrong_remap_fixed_hybrid_correct"]) for row in comparison_rows],
        dtype=np.float64,
    )
    paired = _paired_ci(remapped_correct, original_correct, seed=70370503, samples=bootstrap_samples)
    wrong_paired = _paired_ci(wrong_correct, remapped_correct, seed=70380503, samples=bootstrap_samples)
    run_payload = json.loads(_resolve(permuted_run_json).read_text(encoding="utf-8")) if permuted_run_json else None
    score_cache_hit = (
        bool(run_payload["eval_cache_metadata"]["score_cache_hit"]) if run_payload is not None else None
    )
    hidden_cache_hit = (
        bool(run_payload["eval_cache_metadata"]["hidden_cache_hit"]) if run_payload is not None else None
    )
    shift = _prediction_shift(
        [int(row["remapped_fixed_hybrid_prediction"]) for row in comparison_rows],
        [int(row["canonical_answer_index"]) for row in comparison_rows],
    )
    component_rows = [
        _component_row(component=component, rows=comparison_rows, bootstrap_samples=bootstrap_samples)
        for component in ("fixed_hybrid", *CONTROL_FIELDS)
    ]
    headline = {
        "eval_rows": len(comparison_rows),
        "original_fixed_hybrid_accuracy": _accuracy(original_correct),
        "remapped_fixed_hybrid_accuracy": _accuracy(remapped_correct),
        "unremapped_fixed_hybrid_accuracy": _accuracy(unremapped_correct),
        "wrong_remap_fixed_hybrid_accuracy": _accuracy(wrong_correct),
        "delta_remapped_vs_original": paired["delta"],
        "ci95_low_remapped_vs_original": paired["ci95_low"],
        "ci95_high_remapped_vs_original": paired["ci95_high"],
        "helps_remapped_vs_original": paired["helps"],
        "harms_remapped_vs_original": paired["harms"],
        "fixed_hybrid_canonical_consistency_rate": _accuracy(
            [bool(row["fixed_hybrid_canonical_prediction_matches_original"]) for row in comparison_rows]
        ),
        "wrong_remap_ci95_high_vs_remapped": wrong_paired["ci95_high"],
        "max_prediction_slot_shift_from_answer_distribution": shift[
            "max_abs_prediction_slot_shift_from_answer_distribution"
        ],
        "total_variation_from_answer_distribution": shift["total_variation_from_answer_distribution"],
        "score_cache_hit": score_cache_hit,
        "hidden_cache_hit": hidden_cache_hit,
        "packet_raw_payload_bytes": 1,
        "packet_framed_record_bytes": 4,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_or_score_vector_exposed": False,
        "source_logits_transmitted": False,
        "raw_scores_transmitted": False,
        "mac_local_only": True,
        "native_systems_claim_allowed": False,
    }
    accuracy_drop = headline["original_fixed_hybrid_accuracy"] - headline["remapped_fixed_hybrid_accuracy"]
    smoke_pass = (
        headline["fixed_hybrid_canonical_consistency_rate"] >= min_consistency_for_smoke_pass
        and accuracy_drop <= max_accuracy_drop_for_smoke_pass
        and (score_cache_hit is False if score_cache_hit is not None else True)
        and (hidden_cache_hit is False if hidden_cache_hit is not None else True)
    )
    promotion_pass = (
        len(comparison_rows) >= min_eval_rows_for_promotion
        and headline["fixed_hybrid_canonical_consistency_rate"] >= min_consistency_for_promotion
        and accuracy_drop <= max_accuracy_drop_for_promotion
        and headline["wrong_remap_ci95_high_vs_remapped"] < -0.05
        and (score_cache_hit is False if score_cache_hit is not None else True)
        and (hidden_cache_hit is False if hidden_cache_hit is not None else True)
    )
    headline["smoke_pass_gate"] = bool(smoke_pass)
    headline["promotion_pass_gate"] = bool(promotion_pass)
    headline["promotion_scope"] = (
        f"{len(comparison_rows)}-row one-nonidentity-per-example hidden fixed-hybrid candidate-text hardening; "
        "not all-24 permutation invariance and not full validation"
    )
    payload = {
        "gate": "source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(smoke_pass),
        "smoke_pass_gate": bool(smoke_pass),
        "promotion_pass_gate": bool(promotion_pass),
        "pass_rule": (
            "Mac-local hidden fixed-hybrid permutation smoke passes when remapped canonical predictions "
            "remain close to the original same-row fixed-hybrid predictions, accuracy drop is bounded, "
            "and the permuted hidden pipeline used fresh score/hidden caches when run metadata is supplied. "
            "The stronger promotion flag requires at least 512 rows, >=0.95 canonical consistency, no more "
            "than 0.01 accuracy drop, wrong-remap collapse, and fresh score/hidden caches. This is still "
            "not full-validation or all-24 permutation invariance."
        ),
        "headline": headline,
        "source_artifacts": {
            "original_predictions": _display_path(original_predictions),
            "original_predictions_sha256": _sha256_file(original_predictions),
            "permuted_predictions": _display_path(permuted_predictions),
            "permuted_predictions_sha256": _sha256_file(permuted_predictions),
            "permuted_eval_path": _display_path(permuted_eval_path),
            "permuted_eval_sha256": _sha256_file(permuted_eval_path),
            "permuted_run_json": _display_path(permuted_run_json) if permuted_run_json is not None else None,
            "permuted_run_json_sha256": _sha256_file(permuted_run_json) if permuted_run_json is not None else None,
        },
        "packet_contract": {
            "receiver_visible_payload": "one final fixed-hybrid source candidate id after canonical remapping",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "transferred_source_state_bytes": 0,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "source_hidden_or_score_vector_exposed": False,
            "source_logits_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "component_rows": component_rows,
        "comparison_rows": comparison_rows,
        "interpretation": (
            "This evaluates whether the full fixed-hybrid prediction row follows candidate text under a "
            "fresh hidden-pipeline candidate permutation, rather than only testing cached label remaps. "
            "It is still a bounded Mac-local smoke unless widened to larger slices and more permutations."
        ),
        "lay_explanation": (
            "We shuffled each row's answer endings, reran the hidden hybrid source pipeline, and translated "
            "the displayed answer number back to the original answer number. If the remapped prediction "
            "matches the original same-row prediction, the full hybrid hint is following the answer text."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.json", payload)
    _write_csv(output_dir / "comparison_rows.csv", comparison_rows)
    _write_csv(output_dir / "component_rows.csv", component_rows)
    lines = [
        "# HellaSwag Fixed-Hybrid True Candidate-Text Permutation Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{headline['eval_rows']}`",
        f"- original fixed-hybrid accuracy: `{headline['original_fixed_hybrid_accuracy']:.6f}`",
        f"- remapped fixed-hybrid accuracy: `{headline['remapped_fixed_hybrid_accuracy']:.6f}`",
        f"- canonical consistency: `{headline['fixed_hybrid_canonical_consistency_rate']:.6f}`",
        f"- score cache hit: `{headline['score_cache_hit']}`",
        f"- hidden cache hit: `{headline['hidden_cache_hit']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
    ]
    (output_dir / "hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.json",
                "hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.md",
                "comparison_rows.csv",
                "component_rows.csv",
            ],
            "source_artifacts": payload["source_artifacts"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-full-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-slice-start", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=128)
    parser.add_argument("--permutation-mode", choices=("fixed8_nonidentity", "all24_nonidentity"), default="fixed8_nonidentity")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--original-predictions", type=pathlib.Path, default=DEFAULT_ORIGINAL_PREDICTIONS)
    parser.add_argument("--permuted-predictions", type=pathlib.Path)
    parser.add_argument("--permuted-eval-path", type=pathlib.Path)
    parser.add_argument("--permuted-run-json", type=pathlib.Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-consistency-for-smoke-pass", type=float, default=0.90)
    parser.add_argument("--max-accuracy-drop-for-smoke-pass", type=float, default=0.05)
    parser.add_argument("--min-eval-rows-for-promotion", type=int, default=512)
    parser.add_argument("--min-consistency-for-promotion", type=float, default=0.95)
    parser.add_argument("--max-accuracy-drop-for-promotion", type=float, default=0.01)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-date", default=None)
    args = parser.parse_args()
    prepare = prepare_permuted_eval(
        eval_full_path=args.eval_full_path,
        output_dir=args.output_dir,
        eval_slice_start=args.eval_slice_start,
        eval_rows=args.eval_rows,
        permutation_mode=args.permutation_mode,
        seed=args.seed,
        run_date=args.run_date,
    )
    if args.prepare_only:
        print(json.dumps(prepare, indent=2, sort_keys=True))
        return
    permuted_eval_path = args.permuted_eval_path or _resolve(prepare["permuted_eval_path"])
    if args.permuted_predictions is None:
        raise SystemExit("--permuted-predictions is required unless --prepare-only is set")
    payload = evaluate_gate(
        output_dir=args.output_dir,
        original_predictions=args.original_predictions,
        permuted_predictions=args.permuted_predictions,
        permuted_eval_path=permuted_eval_path,
        permuted_run_json=args.permuted_run_json,
        eval_rows=args.eval_rows,
        bootstrap_samples=args.bootstrap_samples,
        min_consistency_for_smoke_pass=args.min_consistency_for_smoke_pass,
        max_accuracy_drop_for_smoke_pass=args.max_accuracy_drop_for_smoke_pass,
        min_eval_rows_for_promotion=args.min_eval_rows_for_promotion,
        min_consistency_for_promotion=args.min_consistency_for_promotion,
        max_accuracy_drop_for_promotion=args.max_accuracy_drop_for_promotion,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
