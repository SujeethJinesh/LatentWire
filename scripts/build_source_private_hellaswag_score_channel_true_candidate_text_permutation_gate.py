from __future__ import annotations

"""True candidate-text permutation smoke for HellaSwag source score packets.

This gate physically reorders HellaSwag candidate endings before local LM
choice-likelihood scoring, remaps predictions back to canonical candidate IDs,
and reports whether the source candidate-id packet is content-stable under the
display-order perturbation. It is intentionally scoped to the score/candidate-id
channel; it is not a replacement for rerunning the full hidden fixed-hybrid
pipeline under reordered candidate texts.
"""

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import pathlib
import statistics
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_score_packet_headroom as score_headroom  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_score_channel_true_candidate_text_permutation_gate_20260503_validation0_32"
)
DEFAULT_MODEL = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
FIXED8_PERMUTATIONS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (1, 0, 2, 3),
    (0, 2, 1, 3),
    (3, 2, 1, 0),
    (2, 0, 3, 1),
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


def _permutations(mode: str) -> list[tuple[int, int, int, int]]:
    if mode == "fixed8":
        return list(FIXED8_PERMUTATIONS)
    if mode == "all24":
        return list(itertools.permutations(range(4), 4))
    raise ValueError(f"unknown permutation mode {mode!r}")


def _inverse_permutation(display_to_canonical: tuple[int, ...]) -> list[int]:
    inverse = [0 for _ in display_to_canonical]
    for display_index, canonical_index in enumerate(display_to_canonical):
        inverse[int(canonical_index)] = int(display_index)
    return inverse


def _slice_rows(eval_path: pathlib.Path, *, start: int, rows: int) -> list[arc_gate.ArcRow]:
    loaded = arc_gate._load_rows(eval_path)
    if start < 0 or rows <= 0:
        raise ValueError("start must be non-negative and rows must be positive")
    selected = loaded[start : start + rows]
    if len(selected) != rows:
        raise ValueError(f"requested {rows} rows from {start}, found {len(selected)}")
    for row in selected:
        if len(row.choices) != 4:
            raise ValueError("this gate expects exactly four HellaSwag choices per row")
    return selected


def _permuted_row(
    row: arc_gate.ArcRow,
    *,
    permutation_id: int,
    display_to_canonical: tuple[int, ...],
) -> arc_gate.ArcRow:
    display_choices = tuple(row.choices[index] for index in display_to_canonical)
    canonical_to_display = _inverse_permutation(display_to_canonical)
    display_answer_index = canonical_to_display[row.answer_index]
    digest = _sha256_json(
        {
            "canonical_content_id": row.content_id,
            "display_to_canonical": list(display_to_canonical),
            "display_choices": display_choices,
        }
    )
    return arc_gate.ArcRow(
        row_id=f"{row.row_id}::perm{permutation_id}",
        content_id=digest,
        question=row.question,
        choices=display_choices,
        choice_labels=row.choice_labels,
        answer_index=display_answer_index,
        answer_label=row.choice_labels[display_answer_index],
        source_name=row.source_name,
    )


def _expanded_rows(
    rows: list[arc_gate.ArcRow],
    permutations: list[tuple[int, int, int, int]],
) -> tuple[list[arc_gate.ArcRow], list[dict[str, Any]]]:
    expanded: list[arc_gate.ArcRow] = []
    metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for permutation_id, display_to_canonical in enumerate(permutations):
            canonical_to_display = _inverse_permutation(display_to_canonical)
            expanded.append(
                _permuted_row(
                    row,
                    permutation_id=permutation_id,
                    display_to_canonical=display_to_canonical,
                )
            )
            metadata.append(
                {
                    "canonical_row_index": row_index,
                    "canonical_row_id": row.row_id,
                    "canonical_content_id": row.content_id,
                    "permutation_id": permutation_id,
                    "display_to_canonical": list(display_to_canonical),
                    "canonical_to_display": canonical_to_display,
                    "is_identity": list(display_to_canonical) == [0, 1, 2, 3],
                }
            )
    return expanded, metadata


def _accuracy(correct: list[bool] | np.ndarray) -> float:
    if len(correct) == 0:
        return 0.0
    return float(np.mean(np.asarray(correct, dtype=np.float64)))


def _paired_bootstrap(
    selected: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
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


def _balanced_bootstrap(
    *,
    selected_by_example: np.ndarray,
    baseline_by_example: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    deltas = selected_by_example.astype(np.float64) - baseline_by_example.astype(np.float64)
    slot_deltas: list[float] = []
    for answer_index in range(4):
        mask = answers == answer_index
        slot_deltas.append(float(np.mean(deltas[mask])) if np.any(mask) else 0.0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        selected_slots = rng.integers(0, 4, size=4)
        draws.append(float(np.mean([slot_deltas[index] for index in selected_slots])))
    return {
        "balanced_delta": float(np.mean(slot_deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "positive_slot_count": int(sum(delta > 0.0 for delta in slot_deltas)),
        "min_slot_delta": float(min(slot_deltas)),
    }


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


def _build_prediction_rows(
    *,
    canonical_rows: list[arc_gate.ArcRow],
    expanded_rows: list[arc_gate.ArcRow],
    metadata_rows: list[dict[str, Any]],
    source_scores: list[list[float]],
    source_predictions: list[int],
    source_model: dict[str, Any],
    slice_start: int,
    slice_end_exclusive: int,
    scoring_latency_s: float,
    seed: int,
    source_lm_prompt_mode: str,
) -> list[dict[str, Any]]:
    identity_prediction_by_row: dict[int, int] = {}
    for meta, prediction in zip(metadata_rows, source_predictions, strict=True):
        if meta["is_identity"]:
            identity_prediction_by_row[int(meta["canonical_row_index"])] = int(prediction)
    missing_identity = sorted(set(range(len(canonical_rows))) - set(identity_prediction_by_row))
    if missing_identity:
        raise ValueError(f"missing identity permutations for canonical rows: {missing_identity[:5]}")

    avg_wall_time_ms = 1000.0 * scoring_latency_s / max(1, len(expanded_rows))
    out: list[dict[str, Any]] = []
    for expanded, meta, scores, prediction in zip(
        expanded_rows, metadata_rows, source_scores, source_predictions, strict=True
    ):
        canonical_row = canonical_rows[int(meta["canonical_row_index"])]
        display_to_canonical = [int(value) for value in meta["display_to_canonical"]]
        canonical_to_display = [int(value) for value in meta["canonical_to_display"]]
        source_selected_display = int(prediction)
        source_selected_canonical = int(display_to_canonical[source_selected_display])
        wrong_display_to_canonical = display_to_canonical[1:] + display_to_canonical[:1]
        wrong_remap = int(wrong_display_to_canonical[source_selected_display])
        identity_prediction = identity_prediction_by_row[int(meta["canonical_row_index"])]
        prompt = arc_gate._lm_choice_prompt(expanded, prompt_mode=source_lm_prompt_mode)
        display_answer_index = int(canonical_to_display[canonical_row.answer_index])
        out.append(
            {
                "example_id": canonical_row.row_id,
                "hellaswag_ind": canonical_row.row_id,
                "split": "validation",
                "slice_start": int(slice_start),
                "slice_end_exclusive": int(slice_end_exclusive),
                "permutation_id": int(meta["permutation_id"]),
                "permutation_display_to_canonical": display_to_canonical,
                "permutation_canonical_to_display": canonical_to_display,
                "is_identity_permutation": bool(meta["is_identity"]),
                "canonical_answer_index": int(canonical_row.answer_index),
                "display_answer_index": display_answer_index,
                "source_model": source_model.get("model_path", source_model.get("kind", "unknown")),
                "target_model": "not_used_score_channel_audit",
                "method": "score_channel_candidate_id_packet_true_candidate_text_permutation",
                "decode_seed": int(seed),
                "candidate_text_sha256_canonical": _sha256_json(list(canonical_row.choices)),
                "candidate_text_sha256_display_order": _sha256_json(list(expanded.choices)),
                "prompt_sha256": _sha256_json(
                    {
                        "prompt": prompt,
                        "display_choices": list(expanded.choices),
                    }
                ),
                "source_selected_display_id": source_selected_display,
                "source_selected_canonical_id": source_selected_canonical,
                "score_channel_identity_canonical_prediction": int(identity_prediction),
                "score_channel_canonical_prediction": source_selected_canonical,
                "unremapped_display_prediction_as_canonical": source_selected_display,
                "wrong_remap_canonical_prediction": wrong_remap,
                "answer_correct_score_channel_identity": bool(identity_prediction == canonical_row.answer_index),
                "answer_correct_score_channel_remapped": bool(
                    source_selected_canonical == canonical_row.answer_index
                ),
                "answer_correct_unremapped": bool(source_selected_display == canonical_row.answer_index),
                "answer_correct_wrong_remap": bool(wrong_remap == canonical_row.answer_index),
                "canonical_prediction_matches_identity": bool(source_selected_canonical == identity_prediction),
                "source_scores_display_order": [float(value) for value in scores],
                "source_scores_display_order_sha256": _sha256_json([float(value) for value in scores]),
                "packet_raw_payload_bytes": 1,
                "packet_framed_record_bytes": 4,
                "transferred_source_state_bytes": 0,
                "source_text_exposed": False,
                "source_kv_exposed": False,
                "source_hidden_or_score_vector_exposed": False,
                "source_logits_transmitted": False,
                "raw_scores_transmitted": False,
                "parse_failure": False,
                "wall_time_ms": float(avg_wall_time_ms),
                "input_tokens": None,
                "output_tokens": 0,
            }
        )
    return out


def _permutation_accuracy_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for permutation_id in sorted({int(row["permutation_id"]) for row in prediction_rows}):
        rows = [row for row in prediction_rows if int(row["permutation_id"]) == permutation_id]
        out.append(
            {
                "permutation_id": permutation_id,
                "permutation_display_to_canonical": " ".join(
                    str(value) for value in rows[0]["permutation_display_to_canonical"]
                ),
                "eval_rows": len(rows),
                "remapped_accuracy": _accuracy(
                    [bool(row["answer_correct_score_channel_remapped"]) for row in rows]
                ),
                "unremapped_accuracy": _accuracy([bool(row["answer_correct_unremapped"]) for row in rows]),
                "wrong_remap_accuracy": _accuracy([bool(row["answer_correct_wrong_remap"]) for row in rows]),
                "canonical_packet_consistency_rate": _accuracy(
                    [bool(row["canonical_prediction_matches_identity"]) for row in rows]
                ),
            }
        )
    return out


def _answer_position_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for answer_index in range(4):
        rows = [row for row in prediction_rows if int(row["canonical_answer_index"]) == answer_index]
        out.append(
            {
                "answer_index": answer_index,
                "permuted_evaluations": len(rows),
                "canonical_examples": len({row["example_id"] for row in rows}),
                "remapped_accuracy": _accuracy(
                    [bool(row["answer_correct_score_channel_remapped"]) for row in rows]
                ),
                "identity_accuracy": _accuracy(
                    [
                        bool(row["answer_correct_score_channel_identity"])
                        for row in rows
                        if bool(row["is_identity_permutation"])
                    ]
                ),
                "unremapped_accuracy": _accuracy([bool(row["answer_correct_unremapped"]) for row in rows]),
                "consistency_rate": _accuracy([bool(row["canonical_prediction_matches_identity"]) for row in rows]),
            }
        )
    return out


def _summary_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    canonical_rows: list[arc_gate.ArcRow],
    permutations_per_example: int,
    bootstrap_samples: int,
    min_eval_rows_for_smoke_pass: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_example.setdefault(str(row["example_id"]), []).append(row)
    if len(by_example) != len(canonical_rows):
        raise ValueError("prediction rows do not cover the canonical slice exactly")

    identity_correct: list[float] = []
    remapped_mean_correct: list[float] = []
    answers: list[int] = []
    identity_predictions: list[int] = []
    remapped_predictions: list[int] = []
    for canonical_row in canonical_rows:
        rows = by_example[canonical_row.row_id]
        identity_rows = [row for row in rows if bool(row["is_identity_permutation"])]
        if len(identity_rows) != 1:
            raise ValueError(f"expected one identity row for {canonical_row.row_id}")
        identity_row = identity_rows[0]
        identity_correct.append(float(bool(identity_row["answer_correct_score_channel_identity"])))
        remapped_mean_correct.append(
            float(np.mean([bool(row["answer_correct_score_channel_remapped"]) for row in rows]))
        )
        answers.append(int(canonical_row.answer_index))
        identity_predictions.append(int(identity_row["score_channel_identity_canonical_prediction"]))
        remapped_predictions.extend(int(row["score_channel_canonical_prediction"]) for row in rows)

    identity_array = np.asarray(identity_correct, dtype=np.float64)
    remapped_array = np.asarray(remapped_mean_correct, dtype=np.float64)
    paired = _paired_bootstrap(
        selected=remapped_array,
        baseline=identity_array,
        seed=60360504,
        samples=bootstrap_samples,
    )
    balanced = _balanced_bootstrap(
        selected_by_example=remapped_array,
        baseline_by_example=identity_array,
        answers=np.asarray(answers, dtype=np.int64),
        seed=60370504,
        samples=bootstrap_samples,
    )
    permutation_rows = _permutation_accuracy_rows(prediction_rows)
    permutation_accuracies = [float(row["remapped_accuracy"]) for row in permutation_rows]
    identity_accuracy = _accuracy(identity_correct)
    remapped_accuracy = _accuracy(
        [bool(row["answer_correct_score_channel_remapped"]) for row in prediction_rows]
    )
    unremapped_accuracy = _accuracy([bool(row["answer_correct_unremapped"]) for row in prediction_rows])
    wrong_remap_accuracy = _accuracy([bool(row["answer_correct_wrong_remap"]) for row in prediction_rows])
    consistency_rate = _accuracy([bool(row["canonical_prediction_matches_identity"]) for row in prediction_rows])
    wrong_delta = _paired_bootstrap(
        selected=np.asarray([float(bool(row["answer_correct_wrong_remap"])) for row in prediction_rows]),
        baseline=np.asarray([float(bool(row["answer_correct_score_channel_remapped"])) for row in prediction_rows]),
        seed=60380504,
        samples=bootstrap_samples,
    )
    shift = _prediction_shift(remapped_predictions, [int(row["canonical_answer_index"]) for row in prediction_rows])
    parse_failure_rate = _accuracy([bool(row["parse_failure"]) for row in prediction_rows])
    max_abs_delta_from_identity = (
        float(max(abs(float(value) - identity_accuracy) for value in permutation_accuracies))
        if permutation_accuracies
        else 0.0
    )
    accuracy_std = float(statistics.pstdev(permutation_accuracies)) if len(permutation_accuracies) > 1 else 0.0
    smoke_pass = (
        len(canonical_rows) >= min_eval_rows_for_smoke_pass
        and permutations_per_example >= 8
        and parse_failure_rate <= 0.001
        and consistency_rate >= 0.99
        and max_abs_delta_from_identity <= 0.01
        and accuracy_std <= 0.005
        and wrong_delta["ci95_high"] < -0.05
    )
    promotion_pass = (
        len(canonical_rows) >= 1024
        and permutations_per_example == 24
        and paired["delta"] > 0.0
        and paired["ci95_low"] > 0.0
        and balanced["ci95_low"] > 0.0
        and paired["helps"] > paired["harms"]
    )
    summary = {
        "method": "score_channel_candidate_id_packet_true_candidate_text_permutation",
        "eval_rows": len(canonical_rows),
        "permuted_evaluations": len(prediction_rows),
        "permutations_per_example": int(permutations_per_example),
        "identity_accuracy": identity_accuracy,
        "remapped_accuracy": remapped_accuracy,
        "candidate_only_accuracy": identity_accuracy,
        "unremapped_accuracy": unremapped_accuracy,
        "wrong_remap_accuracy": wrong_remap_accuracy,
        "delta_remapped_vs_candidate_only": paired["delta"],
        "ci95_low_remapped_vs_candidate_only": paired["ci95_low"],
        "ci95_high_remapped_vs_candidate_only": paired["ci95_high"],
        "helps_remapped_vs_candidate_only": paired["helps"],
        "harms_remapped_vs_candidate_only": paired["harms"],
        "canonical_packet_consistency_rate": consistency_rate,
        "canonical_packet_flip_rate": 1.0 - consistency_rate,
        "max_abs_accuracy_delta_from_identity": max_abs_delta_from_identity,
        "accuracy_std_across_permutations": accuracy_std,
        "answer_balanced_delta_vs_candidate_only": balanced["balanced_delta"],
        "answer_balanced_ci95_low_vs_candidate_only": balanced["ci95_low"],
        "positive_answer_position_count": balanced["positive_slot_count"],
        "max_prediction_slot_shift_from_answer_distribution": shift[
            "max_abs_prediction_slot_shift_from_answer_distribution"
        ],
        "best_nonidentity_wrong_remap_accuracy": wrong_remap_accuracy,
        "best_wrong_remap_ci95_high_vs_remapped": wrong_delta["ci95_high"],
        "parse_failure_rate": parse_failure_rate,
        "packet_raw_payload_bytes": 1,
        "packet_framed_record_bytes": 4,
        "transferred_source_state_bytes": 0,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_or_score_vector_exposed": False,
        "mac_local_only": True,
        "native_systems_claim_allowed": False,
        "smoke_pass_gate": bool(smoke_pass),
        "promotion_pass_gate": bool(promotion_pass),
    }
    headline = {
        **summary,
        "source_score_channel_scope": (
            "component-level physical candidate-text permutation smoke; not full fixed-hybrid rerun"
        ),
        "total_variation_from_answer_distribution": shift["total_variation_from_answer_distribution"],
    }
    return [summary], headline


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Score-Channel True Candidate-Text Permutation Gate",
        "",
        f"- smoke pass gate: `{payload['smoke_pass_gate']}`",
        f"- promotion pass gate: `{payload['promotion_pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- permuted evaluations: `{h['permuted_evaluations']}`",
        f"- permutations per example: `{h['permutations_per_example']}`",
        f"- identity accuracy: `{h['identity_accuracy']:.6f}`",
        f"- remapped accuracy: `{h['remapped_accuracy']:.6f}`",
        f"- canonical packet consistency: `{h['canonical_packet_consistency_rate']:.6f}`",
        f"- max accuracy delta from identity: `{h['max_abs_accuracy_delta_from_identity']:.6f}`",
        f"- accuracy std across permutations: `{h['accuracy_std_across_permutations']:.6f}`",
        f"- wrong-remap accuracy: `{h['wrong_remap_accuracy']:.6f}`",
        f"- wrong-remap CI95 high vs remapped: `{h['best_wrong_remap_ci95_high_vs_remapped']:.6f}`",
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
    eval_path: pathlib.Path | str = DEFAULT_EVAL,
    slice_start: int = 0,
    eval_rows: int = 32,
    permutation_mode: str = "fixed8",
    score_cache: pathlib.Path | str | None = None,
    source_lm_model: str = DEFAULT_MODEL,
    source_lm_device: str = "auto_cpu",
    source_lm_dtype: str = "float32",
    source_lm_max_length: int = 256,
    source_lm_normalization: str = "mean",
    source_lm_prompt_mode: str = "continuation",
    local_files_only: bool = True,
    bootstrap_samples: int = 1000,
    min_eval_rows_for_smoke_pass: int = 16,
    seed: int = 20260504,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    eval_path = _resolve(eval_path)
    cache_path = _resolve(score_cache) if score_cache is not None else None
    permutations = _permutations(permutation_mode)
    canonical_rows = _slice_rows(eval_path, start=slice_start, rows=eval_rows)
    expanded, metadata = _expanded_rows(canonical_rows, permutations)
    source_scores, source_predictions, source_model, score_cache_sha256 = score_headroom._source_scores(
        rows=expanded,
        score_cache=cache_path,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    prediction_rows = _build_prediction_rows(
        canonical_rows=canonical_rows,
        expanded_rows=expanded,
        metadata_rows=metadata,
        source_scores=source_scores,
        source_predictions=source_predictions,
        source_model=source_model,
        slice_start=slice_start,
        slice_end_exclusive=slice_start + eval_rows,
        scoring_latency_s=float(source_model.get("latency_s", 0.0)),
        seed=seed,
        source_lm_prompt_mode=source_lm_prompt_mode,
    )
    summary_rows, headline = _summary_rows(
        prediction_rows,
        canonical_rows=canonical_rows,
        permutations_per_example=len(permutations),
        bootstrap_samples=bootstrap_samples,
        min_eval_rows_for_smoke_pass=min_eval_rows_for_smoke_pass,
    )
    permutation_rows = _permutation_accuracy_rows(prediction_rows)
    answer_rows = _answer_position_rows(prediction_rows)
    smoke_pass = bool(headline["smoke_pass_gate"])
    promotion_pass = bool(headline["promotion_pass_gate"])
    payload = {
        "gate": "source_private_hellaswag_score_channel_true_candidate_text_permutation_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "smoke_pass_gate": smoke_pass,
        "promotion_pass_gate": promotion_pass,
        "pass_gate": smoke_pass,
        "pass_rule": (
            "Component smoke passes when a Mac-local true candidate-text rerun covers the requested "
            "frozen slice with at least eight permutations per example, remapped canonical source "
            "candidate IDs match the identity packet at >=0.99 consistency, per-permutation accuracy "
            "does not drift from identity by more than 0.01, accuracy std across permutations is <=0.005, "
            "and intentionally wrong remaps are significantly worse. Promotion to positive full HellaSwag "
            "packet evidence remains false unless a >=1024 row all-24-permutation rerun shows positive "
            "paired/answer-balanced lift. This gate is source score/candidate-id only, not a full hidden "
            "fixed-hybrid rerun."
        ),
        "headline": headline,
        "source_artifacts": {
            "eval_path": _display_path(eval_path),
            "eval_sha256": _sha256_file(eval_path),
            "score_cache": _display_path(cache_path) if cache_path is not None else None,
            "score_cache_sha256": score_cache_sha256,
        },
        "source_model": {
            **source_model,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
        },
        "packet_contract": {
            "receiver_visible_payload": "one source-selected candidate id after canonical remapping",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "transferred_source_state_bytes": 0,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "source_hidden_or_score_vector_exposed": False,
            "source_logits_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "systems_trace": {
            "measurement_scope": "Mac-local offline Transformers score-channel rerun; no native GPU serving claim.",
            "mac_local_only": True,
            "native_systems_claim_allowed": False,
            "permuted_evaluations": len(prediction_rows),
            "wall_time_s": float(sum(float(row["wall_time_ms"]) for row in prediction_rows) / 1000.0),
            "packet_raw_payload_bytes_per_request": 1,
            "packet_framed_record_bytes_per_request": 4,
            "transferred_source_state_bytes_per_request": 0,
            "unavailable_native_metrics": [
                "ttft_ms_p50",
                "ttft_ms_p95",
                "tpot_ms_p50",
                "tpot_ms_p95",
                "goodput_requests_per_s",
                "peak_gpu_memory_gb",
                "hbm_read_bytes_per_request",
                "hbm_write_bytes_per_request",
            ],
        },
        "summary_rows": summary_rows,
        "permutation_accuracy_rows": permutation_rows,
        "answer_position_rows": answer_rows,
        "interpretation": (
            "This physically reruns the Qwen HellaSwag continuation-likelihood source scorer after "
            "reordering candidate endings, then maps display predictions back to canonical candidate IDs. "
            "A pass means the score-channel candidate-id packet is content-stable under true display-order "
            "perturbation on this frozen slice. It does not prove the learned hidden fixed-hybrid row is "
            "candidate-order invariant; that requires rerunning the hidden-innovation pipeline under "
            "the same physical permutations."
        ),
        "lay_explanation": (
            "We actually shuffled the answer endings before asking the local source model to score them. "
            "Then we translated the displayed option number back to the original option number. If the "
            "same original answer is selected after shuffling, the score-channel hint is following the "
            "candidate text rather than the slot it appeared in."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_score_channel_true_candidate_text_permutation_gate.json", payload)
    _write_jsonl(output_dir / "permutation_prediction_rows.jsonl", prediction_rows)
    _write_csv(output_dir / "permutation_summary_rows.csv", summary_rows)
    _write_csv(output_dir / "permutation_accuracy_rows.csv", permutation_rows)
    _write_csv(output_dir / "answer_position_rows.csv", answer_rows)
    _write_markdown(output_dir / "hellaswag_score_channel_true_candidate_text_permutation_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_score_channel_true_candidate_text_permutation_gate.json",
                "hellaswag_score_channel_true_candidate_text_permutation_gate.md",
                "permutation_prediction_rows.jsonl",
                "permutation_summary_rows.csv",
                "permutation_accuracy_rows.csv",
                "answer_position_rows.csv",
            ],
            "source_artifacts": payload["source_artifacts"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--slice-start", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=32)
    parser.add_argument("--permutation-mode", choices=("fixed8", "all24"), default="fixed8")
    parser.add_argument("--score-cache", type=pathlib.Path)
    parser.add_argument("--source-lm-model", default=DEFAULT_MODEL)
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-eval-rows-for-smoke-pass", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--run-date", default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        slice_start=args.slice_start,
        eval_rows=args.eval_rows,
        permutation_mode=args.permutation_mode,
        score_cache=args.score_cache,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=args.local_files_only,
        bootstrap_samples=args.bootstrap_samples,
        min_eval_rows_for_smoke_pass=args.min_eval_rows_for_smoke_pass,
        seed=args.seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
