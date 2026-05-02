from __future__ import annotations

"""Build an ARC source-family/source-cache falsification gate.

The gate keeps the ARC Fourier/anchor-syndrome receiver and public basis, but
replaces the Qwen source-choice cache with an independently materialized source
cache, defaulting to TinyLlama-1.1B.  It reports both the full ARC slice and the
stricter slice where the alternate source and Qwen source choose different
candidates.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_fourier_anchor_syndrome_gate as fourier_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_seed_stability as seed_stability  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_source_family_cache_falsification_20260502")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_QWEN_VALIDATION_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/"
    "source_prediction_cache.jsonl"
)
DEFAULT_QWEN_TEST_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/"
    "source_prediction_cache.jsonl"
)
DEFAULT_SOURCE_MODEL = pathlib.Path(
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
    "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)
QWEN_SUBSTITUTED_CONDITION = "qwen_substituted_packet"


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "mean_payload_bytes": 0.0}
    return {
        "n": len(rows),
        "correct": int(sum(1 for row in rows if row["correct"])),
        "accuracy": float(sum(1 for row in rows if row["correct"]) / len(rows)),
        "mean_payload_bytes": float(statistics.fmean(float(row["payload_bytes"]) for row in rows)),
    }


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    conditions = sorted({str(row["condition"]) for row in rows})
    return {condition: _condition_summary([row for row in rows if row["condition"] == condition]) for condition in conditions}


def _source_accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    if not rows:
        return 0.0
    return float(
        sum(int(prediction == row.answer_index) for row, prediction in zip(rows, predictions, strict=True))
        / len(rows)
    )


def _materialize_lm_source_cache(
    *,
    rows: list[arc_gate.ArcRow],
    output_path: pathlib.Path,
    source_family: str,
    source_model: pathlib.Path,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
) -> dict[str, Any]:
    scores, predictions, lm_state = arc_gate._lm_choice_loglikelihood_scores(
        rows,
        model_path=str(source_model),
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        local_files_only=local_files_only,
        normalization=source_lm_normalization,
        prompt_mode=source_lm_prompt_mode,
    )
    cache_rows = []
    for row, prediction in zip(rows, predictions, strict=True):
        cache_rows.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "source_family": source_family,
                "source_model": _display_path(source_model),
                "source_score_mode": "lm_choice_loglikelihood",
                "source_lm_prompt_mode": source_lm_prompt_mode,
                "source_lm_normalization": source_lm_normalization,
                "source_selected_index": int(prediction),
                "source_selected_choice_sha256": arc_gate._sha256_text(row.choices[int(prediction)]),
                "source_visible_fields": ["question", "choices"],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, cache_rows)
    return {
        **lm_state,
        "source_family": source_family,
        "source_model": _display_path(source_model),
        "source_score_mode": "lm_choice_loglikelihood",
        "source_eval_accuracy_before_packet": _source_accuracy(rows, predictions),
        "source_score_digest": hashlib.sha256(json.dumps(scores, sort_keys=True).encode("utf-8")).hexdigest(),
        "source_cache": _display_path(output_path),
        "source_cache_sha256": _sha256_file(output_path),
        "rows": len(rows),
    }


def _load_source_predictions(rows: list[arc_gate.ArcRow], cache_path: pathlib.Path) -> list[int]:
    return fourier_gate._source_predictions(rows, fourier_gate._read_source_cache(cache_path))


def _audit_source_cache(
    *,
    rows: list[arc_gate.ArcRow],
    cache_path: pathlib.Path,
    label: str,
) -> dict[str, Any]:
    row_by_content = {row.content_id: row for row in rows}
    seen: set[str] = set()
    duplicate_content_ids: set[str] = set()
    extra_content_ids: set[str] = set()
    missing_forbidden_contract: list[str] = []
    leaked_payload_keys: list[str] = []
    invalid_selected_indices: list[str] = []
    mismatched_choice_hashes: list[str] = []
    source_families: set[str] = set()
    source_models: set[str] = set()
    source_score_modes: set[str] = set()
    row_count = 0
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            cache_row = json.loads(line)
            content_id = str(cache_row.get("content_id", ""))
            if content_id in seen:
                duplicate_content_ids.add(content_id)
            seen.add(content_id)
            if content_id not in row_by_content:
                extra_content_ids.add(content_id)
                continue
            leaked = sorted(set(cache_row) & set(arc_gate.FORBIDDEN_SOURCE_KEYS))
            if leaked:
                leaked_payload_keys.append(f"{content_id}:{','.join(leaked)}")
            forbidden = set(cache_row.get("forbidden_source_fields", ()))
            if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
                missing_forbidden_contract.append(content_id)
            selected_index = int(cache_row.get("source_selected_index", -1))
            eval_row = row_by_content[content_id]
            if selected_index < 0 or selected_index >= len(eval_row.choices):
                invalid_selected_indices.append(content_id)
                continue
            expected_hash = arc_gate._sha256_text(eval_row.choices[selected_index])
            actual_hash = cache_row.get("source_selected_choice_sha256")
            if actual_hash is not None and str(actual_hash) != expected_hash:
                mismatched_choice_hashes.append(content_id)
            if cache_row.get("source_family"):
                source_families.add(str(cache_row["source_family"]))
            if cache_row.get("source_model"):
                source_models.add(str(cache_row["source_model"]))
            if cache_row.get("source_score_mode"):
                source_score_modes.add(str(cache_row["source_score_mode"]))
    missing_content_ids = sorted(set(row_by_content) - seen)
    errors = {
        "missing_forbidden_contract": missing_forbidden_contract,
        "leaked_payload_keys": leaked_payload_keys,
        "invalid_selected_indices": invalid_selected_indices,
        "mismatched_choice_hashes": mismatched_choice_hashes,
        "duplicate_content_ids": sorted(duplicate_content_ids),
        "missing_content_ids": missing_content_ids,
        "extra_content_ids": sorted(extra_content_ids),
    }
    if any(errors.values()):
        raise ValueError(f"{label} source cache audit failed: {errors}")
    return {
        "label": label,
        "path": _display_path(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "rows": row_count,
        "expected_rows": len(rows),
        "forbidden_payload_keys_absent": True,
        "forbidden_contract_declared": True,
        "source_families": sorted(source_families),
        "source_models": sorted(source_models),
        "source_score_modes": sorted(source_score_modes),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    baseline: str,
    seed: int,
    samples: int,
) -> dict[str, float]:
    return arc_gate._paired_bootstrap(rows, condition=condition, baseline=baseline, seed=seed, samples=samples)


def _summarize_disagreement_seed(
    *,
    seed: int,
    alt_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    disagreement_ids: set[str],
    bootstrap_samples: int,
    min_disagreement_count: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    min_gap_over_qwen: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filtered_alt = [row for row in alt_rows if row["content_id"] in disagreement_ids]
    qwen_substituted = [
        {**row, "condition": QWEN_SUBSTITUTED_CONDITION}
        for row in qwen_rows
        if row["content_id"] in disagreement_ids and row["condition"] == arc_gate.MATCHED_CONDITION
    ]
    combined = [*filtered_alt, *qwen_substituted]
    metrics = _condition_metrics(combined)
    matched = metrics.get(arc_gate.MATCHED_CONDITION, {"accuracy": 0.0})["accuracy"]
    target = metrics.get("target_only", {"accuracy": 0.0})["accuracy"]
    same_byte_text = metrics.get("same_byte_structured_text", {"accuracy": 0.0})["accuracy"]
    qwen_sub = metrics.get(QWEN_SUBSTITUTED_CONDITION, {"accuracy": 0.0})["accuracy"]
    destructive = {
        condition: metrics.get(condition, {"accuracy": 0.0})["accuracy"]
        for condition in arc_gate.STRICT_DESTRUCTIVE_CONTROLS
    }
    best_control_name = max(destructive, key=destructive.get)
    best_control = destructive[best_control_name]
    target_ci = _paired_bootstrap(
        combined,
        condition=arc_gate.MATCHED_CONDITION,
        baseline="target_only",
        seed=seed + 2001,
        samples=bootstrap_samples,
    )
    qwen_ci = _paired_bootstrap(
        combined,
        condition=arc_gate.MATCHED_CONDITION,
        baseline=QWEN_SUBSTITUTED_CONDITION,
        seed=seed + 2002,
        samples=bootstrap_samples,
    )
    pass_gate = bool(
        len(disagreement_ids) >= min_disagreement_count
        and matched >= target + min_lift_over_target
        and matched >= best_control + min_gap_over_control
        and matched >= same_byte_text + min_gap_over_text
        and matched >= qwen_sub + min_gap_over_qwen
        and target_ci["ci95_low"] > 0.0
        and qwen_ci["ci95_low"] > 0.0
    )
    summary = {
        "seed": seed,
        "n": len(disagreement_ids),
        "pass_gate": pass_gate,
        "matched_accuracy": matched,
        "target_accuracy": target,
        "same_byte_structured_text_accuracy": same_byte_text,
        "qwen_substituted_packet_accuracy": qwen_sub,
        "best_destructive_control": best_control_name,
        "best_destructive_control_accuracy": best_control,
        "matched_minus_target": matched - target,
        "matched_minus_same_byte_text": matched - same_byte_text,
        "matched_minus_qwen_substituted": matched - qwen_sub,
        "matched_minus_best_destructive": matched - best_control,
        "paired_ci95_vs_target": target_ci,
        "paired_ci95_vs_qwen_substituted": qwen_ci,
    }
    disagreement_prediction_rows = [
        row
        for row in combined
        if row["condition"] in {arc_gate.MATCHED_CONDITION, QWEN_SUBSTITUTED_CONDITION}
    ]
    return summary, disagreement_prediction_rows


def _aggregate_disagreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pass_count": 0,
            "seed_count": 0,
            "all_seeds_pass": False,
            "n": 0,
        }
    return {
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "seed_count": len(rows),
        "all_seeds_pass": all(row["pass_gate"] for row in rows),
        "n": rows[0]["n"],
        "matched_accuracy_mean": float(statistics.fmean(row["matched_accuracy"] for row in rows)),
        "matched_accuracy_min": min(row["matched_accuracy"] for row in rows),
        "target_accuracy": rows[0]["target_accuracy"],
        "same_byte_structured_text_accuracy_mean": float(
            statistics.fmean(row["same_byte_structured_text_accuracy"] for row in rows)
        ),
        "qwen_substituted_packet_accuracy_mean": float(
            statistics.fmean(row["qwen_substituted_packet_accuracy"] for row in rows)
        ),
        "matched_minus_target_min": min(row["matched_minus_target"] for row in rows),
        "matched_minus_same_byte_text_min": min(row["matched_minus_same_byte_text"] for row in rows),
        "matched_minus_qwen_substituted_min": min(row["matched_minus_qwen_substituted"] for row in rows),
        "matched_minus_best_destructive_min": min(row["matched_minus_best_destructive"] for row in rows),
        "paired_ci95_low_vs_target_min": min(row["paired_ci95_vs_target"]["ci95_low"] for row in rows),
        "paired_ci95_low_vs_qwen_substituted_min": min(
            row["paired_ci95_vs_qwen_substituted"]["ci95_low"] for row in rows
        ),
    }


def _evaluate_split(
    *,
    split_name: str,
    rows: list[arc_gate.ArcRow],
    train_rows: list[arc_gate.ArcRow],
    alt_source_cache: pathlib.Path,
    qwen_source_cache: pathlib.Path,
    seeds: list[int],
    budget_bytes: int,
    anchor_count: int,
    spectral_dim: int,
    code_dim: int,
    bootstrap_samples: int,
    min_disagreement_count: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    min_gap_over_qwen: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_texts = arc_gate._choice_pair_texts(train_rows)
    alt_predictions = _load_source_predictions(rows, alt_source_cache)
    qwen_predictions = _load_source_predictions(rows, qwen_source_cache)
    disagreement_ids = {
        row.content_id
        for row, alt_prediction, qwen_prediction in zip(rows, alt_predictions, qwen_predictions, strict=True)
        if alt_prediction != qwen_prediction
    }
    agreement_rows = [
        {
            "split": split_name,
            "row_id": row.row_id,
            "content_id": row.content_id,
            "alt_source_selected_index": int(alt_prediction),
            "qwen_source_selected_index": int(qwen_prediction),
            "agree": bool(alt_prediction == qwen_prediction),
            "answer_index": row.answer_index,
            "alt_source_correct": bool(alt_prediction == row.answer_index),
            "qwen_source_correct": bool(qwen_prediction == row.answer_index),
        }
        for row, alt_prediction, qwen_prediction in zip(rows, alt_predictions, qwen_predictions, strict=True)
    ]
    source_features, receiver_features, basis_metadata = fourier_gate._fourier_pair_features_for_variant(
        eval_rows=rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        variant=fourier_gate.MATCHED_VARIANT,
    )
    index_prior = arc_gate._index_prior(train_rows)
    alt_residuals = arc_gate._candidate_residuals(rows, source_features)
    receiver_residuals = arc_gate._candidate_residuals(rows, receiver_features)
    full_seed_rows: list[dict[str, Any]] = []
    disagreement_seed_rows: list[dict[str, Any]] = []
    matched_prediction_rows: list[dict[str, Any]] = []
    disagreement_prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        projection = arc_gate._projection_matrix(source_features.shape[1], code_dim, seed=seed + 171)
        alt_prediction_rows = arc_gate._rows_for_predictions(
            eval_rows=rows,
            residuals=alt_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=alt_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=index_prior,
            seed=seed + 911,
        )
        qwen_prediction_rows = arc_gate._rows_for_predictions(
            eval_rows=rows,
            residuals=alt_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=qwen_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=index_prior,
            seed=seed + 911,
        )
        full_seed_rows.append(
            seed_stability._summarize_seed(
                seed=seed,
                rows=alt_prediction_rows,
                bootstrap_samples=bootstrap_samples,
                min_lift_over_target=min_lift_over_target,
                min_gap_over_control=min_gap_over_control,
                min_gap_over_text=min_gap_over_text,
                has_overlap=False,
            )
        )
        disagreement_summary, disagreement_rows = _summarize_disagreement_seed(
            seed=seed,
            alt_rows=alt_prediction_rows,
            qwen_rows=qwen_prediction_rows,
            disagreement_ids=disagreement_ids,
            bootstrap_samples=bootstrap_samples,
            min_disagreement_count=min_disagreement_count,
            min_lift_over_target=min_lift_over_target,
            min_gap_over_control=min_gap_over_control,
            min_gap_over_text=min_gap_over_text,
            min_gap_over_qwen=min_gap_over_qwen,
        )
        disagreement_seed_rows.append(disagreement_summary)
        matched_prediction_rows.extend(
            {
                **row,
                "split": split_name,
                "seed": seed,
                "source_cache_family": "alternate",
            }
            for row in alt_prediction_rows
            if row["condition"] == arc_gate.MATCHED_CONDITION
        )
        disagreement_prediction_rows.extend(
            {**row, "split": split_name, "seed": seed}
            for row in disagreement_rows
        )
    full_aggregate = seed_stability._aggregate(full_seed_rows)
    disagreement_aggregate = _aggregate_disagreement(disagreement_seed_rows)
    split_pass = bool(full_aggregate["all_seeds_pass"] and disagreement_aggregate["all_seeds_pass"])
    return {
        "split_name": split_name,
        "rows": len(rows),
        "alt_source_cache": _display_path(alt_source_cache),
        "alt_source_cache_sha256": _sha256_file(alt_source_cache),
        "qwen_source_cache": _display_path(qwen_source_cache),
        "qwen_source_cache_sha256": _sha256_file(qwen_source_cache),
        "basis_metadata": basis_metadata,
        "source_cache_agreement": {
            "agreement_count": sum(1 for row in agreement_rows if row["agree"]),
            "disagreement_count": len(disagreement_ids),
            "agreement_rate": float(sum(1 for row in agreement_rows if row["agree"]) / len(agreement_rows)),
            "alt_source_accuracy": _source_accuracy(rows, alt_predictions),
            "qwen_source_accuracy": _source_accuracy(rows, qwen_predictions),
        },
        "full_slice": {
            "per_seed": full_seed_rows,
            "aggregate": full_aggregate,
        },
        "qwen_disagreement_slice": {
            "per_seed": disagreement_seed_rows,
            "aggregate": disagreement_aggregate,
        },
        "headline": {
            "pass_gate": split_pass,
            "full_slice_pass": full_aggregate["all_seeds_pass"],
            "qwen_disagreement_slice_pass": disagreement_aggregate["all_seeds_pass"],
            "full_slice": full_aggregate,
            "qwen_disagreement_slice": disagreement_aggregate,
            "source_cache_agreement": {
                "agreement_rate": float(sum(1 for row in agreement_rows if row["agree"]) / len(agreement_rows)),
                "disagreement_count": len(disagreement_ids),
            },
        },
    }, matched_prediction_rows, disagreement_prediction_rows, agreement_rows


def _write_seed_csv(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fields = [
        "split",
        "surface",
        "seed",
        "pass_gate",
        "n",
        "matched_accuracy",
        "target_accuracy",
        "same_byte_structured_text_accuracy",
        "qwen_substituted_packet_accuracy",
        "best_destructive_control",
        "best_destructive_control_accuracy",
        "matched_minus_target",
        "matched_minus_same_byte_text",
        "matched_minus_qwen_substituted",
        "matched_minus_best_destructive",
        "paired_ci95_low_vs_target",
        "paired_ci95_low_vs_qwen_substituted",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split_name, split in payload["splits"].items():
            for row in split["full_slice"]["per_seed"]:
                writer.writerow(
                    {
                        "split": split_name,
                        "surface": "full_slice",
                        "seed": row["seed"],
                        "pass_gate": row["pass_gate"],
                        "n": split["rows"],
                        "matched_accuracy": row["matched_accuracy"],
                        "target_accuracy": row["target_accuracy"],
                        "same_byte_structured_text_accuracy": row["same_byte_structured_text_accuracy"],
                        "qwen_substituted_packet_accuracy": "",
                        "best_destructive_control": row["best_destructive_control"],
                        "best_destructive_control_accuracy": row["best_destructive_control_accuracy"],
                        "matched_minus_target": row["matched_minus_target"],
                        "matched_minus_same_byte_text": row["matched_minus_same_byte_text"],
                        "matched_minus_qwen_substituted": "",
                        "matched_minus_best_destructive": row["matched_minus_best_destructive"],
                        "paired_ci95_low_vs_target": row["paired_ci95_vs_target"]["ci95_low"],
                        "paired_ci95_low_vs_qwen_substituted": "",
                    }
                )
            for row in split["qwen_disagreement_slice"]["per_seed"]:
                writer.writerow(
                    {
                        "split": split_name,
                        "surface": "qwen_disagreement_slice",
                        "seed": row["seed"],
                        "pass_gate": row["pass_gate"],
                        "n": row["n"],
                        "matched_accuracy": row["matched_accuracy"],
                        "target_accuracy": row["target_accuracy"],
                        "same_byte_structured_text_accuracy": row["same_byte_structured_text_accuracy"],
                        "qwen_substituted_packet_accuracy": row["qwen_substituted_packet_accuracy"],
                        "best_destructive_control": row["best_destructive_control"],
                        "best_destructive_control_accuracy": row["best_destructive_control_accuracy"],
                        "matched_minus_target": row["matched_minus_target"],
                        "matched_minus_same_byte_text": row["matched_minus_same_byte_text"],
                        "matched_minus_qwen_substituted": row["matched_minus_qwen_substituted"],
                        "matched_minus_best_destructive": row["matched_minus_best_destructive"],
                        "paired_ci95_low_vs_target": row["paired_ci95_vs_target"]["ci95_low"],
                        "paired_ci95_low_vs_qwen_substituted": row[
                            "paired_ci95_vs_qwen_substituted"
                        ]["ci95_low"],
                    }
                )


def _write_agreement_csv(path: pathlib.Path, agreement_rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "row_id",
        "content_id",
        "alt_source_selected_index",
        "qwen_source_selected_index",
        "agree",
        "answer_index",
        "alt_source_correct",
        "qwen_source_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(agreement_rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    test = payload["splits"]["test"]["headline"]
    lines = [
        "# ARC Source-Family/Source-Cache Falsification Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- alternate source family: `{payload['alternate_source_family']}`",
        f"- packet budget: `{payload['basis']['budget_bytes']}B`",
        f"- test full-slice pass: `{test['full_slice_pass']}`",
        f"- test Qwen-disagreement pass: `{test['qwen_disagreement_slice_pass']}`",
        f"- test Qwen disagreement rows: `{test['source_cache_agreement']['disagreement_count']}`",
        "",
        "| Split | Surface | Pass seeds | Matched mean | Target | Text | Qwen-sub | Min CI target | Min CI Qwen |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, split in payload["splits"].items():
        full = split["full_slice"]["aggregate"]
        dis = split["qwen_disagreement_slice"]["aggregate"]
        lines.append(
            f"| {split_name} | full | {full['pass_count']}/{full['seed_count']} | "
            f"{full['matched_accuracy_mean']:.3f} | {full['target_accuracy']:.3f} | "
            f"{full['same_byte_structured_text_accuracy']:.3f} | - | "
            f"{full['paired_ci95_low_vs_target_min']:.3f} | - |"
        )
        lines.append(
            f"| {split_name} | Qwen-disagreement | {dis['pass_count']}/{dis['seed_count']} | "
            f"{dis['matched_accuracy_mean']:.3f} | {dis['target_accuracy']:.3f} | "
            f"{dis['same_byte_structured_text_accuracy_mean']:.3f} | "
            f"{dis['qwen_substituted_packet_accuracy_mean']:.3f} | "
            f"{dis['paired_ci95_low_vs_target_min']:.3f} | "
            f"{dis['paired_ci95_low_vs_qwen_substituted_min']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay description: we ask a different local source model to choose answers, encode those choices "
            "with the same tiny Fourier/anchor packet, and then focus on examples where the Qwen source chose "
            "something else. If the alternate packet still wins there, the result is less likely to be a Qwen "
            "cache artifact.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_source_family_cache_falsification(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    qwen_validation_cache: pathlib.Path,
    qwen_test_cache: pathlib.Path,
    alt_validation_cache: pathlib.Path,
    alt_test_cache: pathlib.Path,
    alternate_source_family: str,
    alternate_source_model: pathlib.Path,
    materialize_alt_caches: bool,
    force_rematerialize: bool,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
    seeds: list[int],
    budget_bytes: int,
    anchor_count: int,
    spectral_dim: int,
    code_dim: int,
    bootstrap_samples: int,
    min_disagreement_count: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    min_gap_over_qwen: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = arc_gate._load_rows(train_path)
    validation_rows = arc_gate._load_rows(validation_path)
    test_rows = arc_gate._load_rows(test_path)

    source_cache_audit: dict[str, Any] = {
        "alternate_source_family": alternate_source_family,
        "alternate_source_model": _display_path(alternate_source_model),
        "materialization_requested": materialize_alt_caches,
        "materialized_this_run": False,
        "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
    }
    if materialize_alt_caches:
        if force_rematerialize or not alt_validation_cache.exists():
            source_cache_audit["validation_materialization"] = _materialize_lm_source_cache(
                rows=validation_rows,
                output_path=alt_validation_cache,
                source_family=alternate_source_family,
                source_model=alternate_source_model,
                source_lm_device=source_lm_device,
                source_lm_dtype=source_lm_dtype,
                source_lm_max_length=source_lm_max_length,
                source_lm_normalization=source_lm_normalization,
                source_lm_prompt_mode=source_lm_prompt_mode,
                local_files_only=local_files_only,
            )
            source_cache_audit["materialized_this_run"] = True
        if force_rematerialize or not alt_test_cache.exists():
            source_cache_audit["test_materialization"] = _materialize_lm_source_cache(
                rows=test_rows,
                output_path=alt_test_cache,
                source_family=alternate_source_family,
                source_model=alternate_source_model,
                source_lm_device=source_lm_device,
                source_lm_dtype=source_lm_dtype,
                source_lm_max_length=source_lm_max_length,
                source_lm_normalization=source_lm_normalization,
                source_lm_prompt_mode=source_lm_prompt_mode,
                local_files_only=local_files_only,
            )
            source_cache_audit["materialized_this_run"] = True

    qwen_validation_audit = _audit_source_cache(
        rows=validation_rows,
        cache_path=qwen_validation_cache,
        label="qwen_validation",
    )
    qwen_test_audit = _audit_source_cache(
        rows=test_rows,
        cache_path=qwen_test_cache,
        label="qwen_test",
    )
    alt_validation_audit = _audit_source_cache(
        rows=validation_rows,
        cache_path=alt_validation_cache,
        label="alternate_validation",
    )
    alt_test_audit = _audit_source_cache(
        rows=test_rows,
        cache_path=alt_test_cache,
        label="alternate_test",
    )
    for split_name, alt_audit, qwen_audit in (
        ("validation", alt_validation_audit, qwen_validation_audit),
        ("test", alt_test_audit, qwen_test_audit),
    ):
        if alt_audit["cache_sha256"] == qwen_audit["cache_sha256"]:
            raise ValueError(f"{split_name} alternate and Qwen source caches have identical sha256")

    validation_payload, validation_matched, validation_disagreement, validation_agreement = _evaluate_split(
        split_name="validation",
        rows=validation_rows,
        train_rows=train_rows,
        alt_source_cache=alt_validation_cache,
        qwen_source_cache=qwen_validation_cache,
        seeds=seeds,
        budget_bytes=budget_bytes,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_disagreement_count=min_disagreement_count,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        min_gap_over_qwen=min_gap_over_qwen,
    )
    test_payload, test_matched, test_disagreement, test_agreement = _evaluate_split(
        split_name="test",
        rows=test_rows,
        train_rows=train_rows,
        alt_source_cache=alt_test_cache,
        qwen_source_cache=qwen_test_cache,
        seeds=seeds,
        budget_bytes=budget_bytes,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_disagreement_count=min_disagreement_count,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        min_gap_over_qwen=min_gap_over_qwen,
    )
    pass_gate = bool(validation_payload["headline"]["pass_gate"] and test_payload["headline"]["pass_gate"])
    payload = {
        "gate": "source_private_arc_challenge_source_family_cache_falsification",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "alternate_source_family": alternate_source_family,
        "source_families": ["qwen2.5_0.5b", alternate_source_family],
        "basis": {
            "parent_gate": "results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502",
            "anchor_count": anchor_count,
            "spectral_dim": spectral_dim,
            "code_dim": code_dim,
            "budget_bytes": budget_bytes,
            "seeds": seeds,
            "bootstrap_samples": bootstrap_samples,
            "basis": "public train-anchor relative coordinates followed by orthonormal low-frequency DCT-II",
        },
        "source_cache_audit": {
            **source_cache_audit,
            "qwen_validation_cache": _display_path(qwen_validation_cache),
            "qwen_test_cache": _display_path(qwen_test_cache),
            "alt_validation_cache": _display_path(alt_validation_cache),
            "alt_test_cache": _display_path(alt_test_cache),
            "cache_sha256": {
                "qwen_validation": _sha256_file(qwen_validation_cache),
                "qwen_test": _sha256_file(qwen_test_cache),
                "alt_validation": _sha256_file(alt_validation_cache),
                "alt_test": _sha256_file(alt_test_cache),
            },
            "cache_row_audits": {
                "qwen_validation": qwen_validation_audit,
                "qwen_test": qwen_test_audit,
                "alternate_validation": alt_validation_audit,
                "alternate_test": alt_test_audit,
            },
        },
        "splits": {
            "validation": validation_payload,
            "test": test_payload,
        },
        "headline": {
            "validation_pass": validation_payload["headline"]["pass_gate"],
            "test_pass": test_payload["headline"]["pass_gate"],
            "test_full_slice": test_payload["headline"]["full_slice"],
            "test_qwen_disagreement_slice": test_payload["headline"]["qwen_disagreement_slice"],
            "test_source_cache_agreement": test_payload["headline"]["source_cache_agreement"],
        },
        "pass_rule": (
            "Pass requires validation and test full-slice alternate-source packets to pass all seeds, plus a "
            "Qwen-disagreement slice with enough rows where the alternate-source packet beats target-only, "
            "same-byte text, the best destructive control, and a Qwen-substituted packet with positive paired "
            "uncertainty."
        ),
        "interpretation": (
            "This gate directly tests whether the ARC Fourier/anchor-syndrome result depends on the original "
            "Qwen source-choice cache. A pass would promote the method to source-family-general packet "
            "communication. A failure is still useful: it says the current ARC positive row remains source-cache "
            "specific and must be framed below ICLR headline strength until a stronger cross-family source "
            "endpoint lands."
        ),
    }
    json_path = output_dir / "source_family_cache_falsification.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_seed_csv(output_dir / "per_source_split_seed_metrics.csv", payload)
    _write_agreement_csv(output_dir / "source_cache_agreement.csv", [*validation_agreement, *test_agreement])
    _write_jsonl(output_dir / "matched_predictions.jsonl", [*validation_matched, *test_matched])
    _write_jsonl(
        output_dir / "qwen_disagreement_predictions.jsonl",
        [*validation_disagreement, *test_disagreement],
    )
    (output_dir / "source_cache_audit.json").write_text(
        json.dumps(payload["source_cache_audit"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "source_family_cache_falsification.md", payload)
    manifest = {
        "artifacts": [
            "source_family_cache_falsification.json",
            "source_family_cache_falsification.md",
            "source_cache_audit.json",
            "per_source_split_seed_metrics.csv",
            "source_cache_agreement.csv",
            "matched_predictions.jsonl",
            "qwen_disagreement_predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "source_family_cache_falsification.json",
                "source_family_cache_falsification.md",
                "source_cache_audit.json",
                "per_source_split_seed_metrics.csv",
                "source_cache_agreement.csv",
                "matched_predictions.jsonl",
                "qwen_disagreement_predictions.jsonl",
            )
        },
        "pass_gate": pass_gate,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# ARC Source-Family/Source-Cache Falsification Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- validation pass: `{validation_payload['headline']['pass_gate']}`",
                f"- test pass: `{test_payload['headline']['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--qwen-validation-cache", type=pathlib.Path, default=DEFAULT_QWEN_VALIDATION_CACHE)
    parser.add_argument("--qwen-test-cache", type=pathlib.Path, default=DEFAULT_QWEN_TEST_CACHE)
    parser.add_argument("--alt-validation-cache", type=pathlib.Path, default=None)
    parser.add_argument("--alt-test-cache", type=pathlib.Path, default=None)
    parser.add_argument("--alternate-source-family", default="tinyllama_1.1b")
    parser.add_argument("--alternate-source-model", type=pathlib.Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--skip-cache-materialization", action="store_true")
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--source-lm-device", default="mps")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=192)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="qa")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--seeds", type=_parse_int_list, default="47,53,59,61,67")
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--anchor-count", type=int, default=384)
    parser.add_argument("--spectral-dim", type=int, default=96)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-disagreement-count", type=int, default=150)
    parser.add_argument("--min-lift-over-target", type=float, default=0.05)
    parser.add_argument("--min-gap-over-control", type=float, default=0.03)
    parser.add_argument("--min-gap-over-text", type=float, default=0.02)
    parser.add_argument("--min-gap-over-qwen", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    alt_validation_cache = (
        _resolve(args.alt_validation_cache)
        if args.alt_validation_cache is not None
        else output_dir / "tinyllama_validation" / "source_prediction_cache.jsonl"
    )
    alt_test_cache = (
        _resolve(args.alt_test_cache)
        if args.alt_test_cache is not None
        else output_dir / "tinyllama_test" / "source_prediction_cache.jsonl"
    )
    build_source_family_cache_falsification(
        output_dir=output_dir,
        train_path=_resolve(args.train_path),
        validation_path=_resolve(args.validation_path),
        test_path=_resolve(args.test_path),
        qwen_validation_cache=_resolve(args.qwen_validation_cache),
        qwen_test_cache=_resolve(args.qwen_test_cache),
        alt_validation_cache=alt_validation_cache,
        alt_test_cache=alt_test_cache,
        alternate_source_family=args.alternate_source_family,
        alternate_source_model=_resolve(args.alternate_source_model),
        materialize_alt_caches=not args.skip_cache_materialization,
        force_rematerialize=args.force_rematerialize,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=not args.allow_downloads,
        seeds=args.seeds,
        budget_bytes=args.budget_bytes,
        anchor_count=args.anchor_count,
        spectral_dim=args.spectral_dim,
        code_dim=args.code_dim,
        bootstrap_samples=args.bootstrap_samples,
        min_disagreement_count=args.min_disagreement_count,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
        min_gap_over_qwen=args.min_gap_over_qwen,
    )


if __name__ == "__main__":
    main()
